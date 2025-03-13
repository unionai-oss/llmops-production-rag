"""Use LLM critic to filter Q&A dataset for quality."""

import json
from dataclasses import dataclass, asdict
from functools import partial
from typing import Annotated, Optional

import flytekit as fl
import pandas as pd

from flytekit.deck import TopFrameRenderer
from flytekit.types.file import FlyteFile

from llmops_rag.image import image
from llmops_rag.create_qa_dataset import QuestionAndAnswers
from llmops_rag.utils import openai_env_secret


QuestionAndAnswerDataset = fl.Artifact(name="question_and_answer_dataset")
EvalDatasetArtifact = fl.Artifact(name="eval-dataset", partition_keys=["dataset_type"])

N_RETRIES = 10
SCORE_THRESHOLD = 3


@dataclass
class Score:
    evaluation: str
    rating: int


@dataclass
class QualityScores:
    groundedness: Optional[Score]
    relevance: Optional[Score]
    correctness_per_answer: list[Optional[Score]]


@dataclass
class ReferenceAnswer:
    question_id: int
    question: str
    reference_answer: str
    is_user_generated: bool
    groundedness_score: int
    groundedness_evaluation: str
    relevance_score: int
    relevance_evaluation: str
    correctness_score: int
    correctness_evaluation: str


@fl.task(
    container_image=image,
    requests=fl.Resources(cpu="2", mem="4Gi"),
    secret_requests=[fl.Secret(key="openai_api_key")],
    cache=True,
    cache_version="2",
)
@openai_env_secret
def llm_critic(dataset: FlyteFile, dataset_index: int) -> QualityScores:
    """Returns critic scores per question-answer pair."""
    # implement prompts for each quality score filed
    from langchain.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser


    groundedness_prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""
        You will be given a context and a question.
        Your task is to provide a rating representing how well
        one can answer the given question unambiguously with
        the given context. Give your response on a scale of 1 to 5,
        where 1 means that the question is not answerable at all
        given the context, and 5 means that the question is clearly
        and unambiguously answerable with the context.

        Provide your response as follows:

        Response:::
        Evaluation: (your rationale for the rating, as text)
        Rating: (your rating, as a number between 1 and 5)

        You MUST provide values for 'Evaluation:' and 'Rating:'
        in your response.

        Now here are the question and context.

        Question: {question}\n
        Context: {context}\n
        Response::: 
        """,
    )

    relevance_prompt = PromptTemplate(
        input_variables=["question"],
        template="""
        You will be given a question.
        Your task is to provide a rating representing how
        useful this question can be to data scientists, analysts,
        and ML engineers using the pandas python data analysis library.

        Give your response on a scale of 1 to 5, where 1 means that
        the question is not useful at all, and 5 means that the
        question is extremely useful.

        Provide your response as follows:

        Response:::
        Evaluation: (your rationale for the rating, as text)
        Rating: (your rating, as a number between 1 and 5)

        You MUST provide values for 'Evaluation:' and 'Rating:'
        in your response.

        Now here is the question.

        Question: {question}\n
        Response::: 
        """,
    )

    correctness_prompt = PromptTemplate(
        input_variables=["question", "context", "answer"],
        template="""
        You will be given a context, a question, and an answer
        to the question. Your task is to provide a 'correctness rating'
        scoring the correctness of the given answer to the question,
        based on the context.
        
        Give your answer on a scale of 1 to 5, referring to the
        rubric below for the exact definition of each rating.

        ### Rating Rubric:
        [Is the response correct, accurate, and factual based on the reference answer?]
        Rating 1: The response is completely incorrect, inaccurate, and/or not factual.
        Rating 2: The response is mostly incorrect, inaccurate, and/or not factual.
        Rating 3: The response is somewhat correct, accurate, and/or factual.
        Rating 4: The response is mostly correct, accurate, and factual.
        Rating 5: The response is completely correct, accurate, and factual.

        Provide your answer as follows:

        Response:::
        Evaluation: (your rationale for the rating, as text)
        Rating: (your rating, as a number between 1 and 5)

        You MUST provide values for 'Evaluation:' and 'Rating:' in your answer.

        Now here are the question and context.

        Question: {question}\n
        Context: {context}\n
        Answer: {answer}\n
        Response::: 
        """,
    )

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.9, presence_penalty=1.5)

    class ScoreParser(StrOutputParser):
        def parse(self, text: str) -> Score:
            """Returns the input text with no changes."""
            assert text.startswith("Evaluation:") and "Rating:" in text
            evaluation, rating = text.split("Rating:")
            return Score(evaluation=evaluation.replace("Evaluation: ", ""), rating=int(rating.strip()))
        
    def retry_handler(chain, input):
        for i in range(N_RETRIES):
            try:
                return chain.invoke(input)
            except Exception as e:
                print(f"[Retry {i}] error: {e}")
        return None

    groundedness_chain = groundedness_prompt | llm | ScoreParser()
    relevance_chain = relevance_prompt | llm | ScoreParser()
    correctness_chain = correctness_prompt | llm | ScoreParser()

    with open(dataset, "r") as f:
        qa_dataset = [QuestionAndAnswers(**x) for x in json.load(f)]

    qa_datapoint = qa_dataset[dataset_index]
    groundedness_score = retry_handler(groundedness_chain, {"question": qa_datapoint.question, "context": qa_datapoint.context})
    relevance_score = retry_handler(relevance_chain, {"question": qa_datapoint.question})
    correctness_scores = [
        retry_handler(correctness_chain, {"question": qa_datapoint.question, "context": qa_datapoint.context, "answer": answer})
        for answer in qa_datapoint.answers
    ]
    return QualityScores(groundedness=groundedness_score, relevance=relevance_score, correctness_per_answer=correctness_scores)



@fl.dynamic(
    container_image=image,
    cache=True,
    cache_version="2",
    retries=5,
)
def apply_llm_critic(dataset: FlyteFile) -> list[QualityScores]:
    """Applies the LLM critic to a dataset of question-answer pairs."""
    with open(dataset, "r") as f:
        dataset_index = [*range(len(json.load(f)))]

    llm_critic_partial = partial(llm_critic, dataset=dataset)
    quality_scores = fl.map_task(llm_critic_partial)(dataset_index=dataset_index)
    return quality_scores


@fl.task(
    container_image=image,
    requests=fl.Resources(cpu="2", mem="4Gi"),
    cache=True,
    cache_version="2",
)
def filter_dataset(dataset: FlyteFile, scores: list[QualityScores]) -> list[ReferenceAnswer]:
    """Filters out low-quality question-answer pairs."""
    reference_answers = []
    with open(dataset, "r") as f:
        qa_dataset = [QuestionAndAnswers(**x) for x in json.load(f)]

    for qa_datapoint, score in zip(qa_dataset, scores):
        if (
            any(x is None for x in [score.groundedness, score.relevance])
            or any(x is None for x in score.correctness_per_answer)
        ):
            continue

        if score.groundedness.rating < SCORE_THRESHOLD or score.relevance.rating < SCORE_THRESHOLD:
            continue

        for answer, correctness_score in zip(qa_datapoint.answers, score.correctness_per_answer):
            if correctness_score.rating < SCORE_THRESHOLD:
                continue
                
            reference_answers.append(
                ReferenceAnswer(
                    question_id=qa_datapoint.id,
                    question=qa_datapoint.question,
                    reference_answer=answer,
                    is_user_generated=False,
                    groundedness_score=score.groundedness.rating,
                    groundedness_evaluation=score.groundedness.evaluation,
                    relevance_score=score.relevance.rating,
                    relevance_evaluation=score.relevance.evaluation,
                    correctness_score=correctness_score.rating,
                    correctness_evaluation=correctness_score.evaluation,
                )
            )
            # pick the first answer that is of high correctness quality
            break

    return reference_answers


@fl.task(
    container_image=image,
    requests=fl.Resources(cpu="2", mem="4Gi"),
    enable_deck=True,
    deck_fields=[],
    cache=True,
    cache_version="2",
)
def prepare_dataset(reference_answers: list[ReferenceAnswer]) -> Annotated[pd.DataFrame, EvalDatasetArtifact]:
    """Prepare and visualize dataset."""
    dataset = pd.DataFrame.from_records([asdict(x) for x in reference_answers])
    fl.Deck("filtered dataset", TopFrameRenderer().to_html(dataset))
    return EvalDatasetArtifact.create_from(dataset, dataset_type="llm_filtered")


@fl.workflow
def create_llm_filtered_dataset(
    dataset: FlyteFile = QuestionAndAnswerDataset.query(),
) -> pd.DataFrame:
    scores = apply_llm_critic(dataset)
    reference_answers = filter_dataset(dataset, scores)
    return prepare_dataset(reference_answers)
