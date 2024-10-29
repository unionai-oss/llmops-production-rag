"""Evaluate a RAG workflow."""

from dataclasses import dataclass, asdict
from functools import partial
from typing import Annotated, Optional

import pandas as pd
import flytekit as fk
from flytekit.deck import TopFrameRenderer
from flytekit.types.directory import FlyteDirectory

from llmops_rag.config import RAGConfig
from llmops_rag.image import image as rag_image
from llmops_rag.vector_store import create as create_vector_store
from llmops_rag.rag_basic import rag_basic
from llmops_rag.utils import openai_env_secret, convert_fig_into_html


EvalDatasetArtifact = fk.Artifact(name="eval-dataset", partition_keys=["dataset_type"])

image = rag_image.with_packages(["nltk", "rouge-score", "seaborn"])


@dataclass
class HPOConfig(RAGConfig):
    condition_name: str = ""


@dataclass
class Question:
    question_id: int
    question: str
    reference_answer: str
    is_user_generated: bool


@dataclass
class RAGInput:
    question: Question
    vector_store: FlyteDirectory
    prompt_template: str


@dataclass
class Answer:
    answer: str
    question_id: int


@fk.task(
    container_image=image,
    cache=True,
    cache_version="1",
)
def prepare_questions(dataset: pd.DataFrame, n_answers: int) -> list[Question]:
    questions = (
        dataset.loc[dataset.index.repeat(n_answers)][
            ["question_id", "question", "reference_answer", "is_user_generated"]
        ]
        .astype({"question_id": int})
        .to_dict(orient="records")
    )
    return [Question(**record) for record in questions]


@fk.task(
    container_image=image,
    cache=True,
    cache_version="1",
)
def prepare_answers(answers: list[str], questions: list[Question]) -> list[Answer]:
    return [
        Answer(
            answer=answer,
            question_id=question.question_id,
        )
        for answer, question in zip(answers, questions)
    ]


@fk.dynamic(container_image=image, cache=True, cache_version="4")
def generate_answers(
    questions: list[Question],
    root_url_tags_mapping: Optional[dict] = None,
    splitter: str = "character",
    chunk_size: int = 2048,
    prompt_template: str = "",
    limit: Optional[int | float] = None,
    embedding_type: Optional[str] = "openai",
    exclude_patterns: Optional[list[str]] = None,
) -> list[Answer]:
    vector_store = create_vector_store(
        root_url_tags_mapping=root_url_tags_mapping,
        splitter=splitter,
        chunk_size=chunk_size,
        limit=limit,
        embedding_type=embedding_type,
        exclude_patterns=exclude_patterns,
    )
    answers = rag_basic(
        questions=[question.question for question in questions],
        vector_store=vector_store,
        prompt_template=prompt_template,
    )
    return prepare_answers(answers, questions)


@fk.dynamic(container_image=image, cache=True, cache_version="4")
def gridsearch(
    questions: list[Question],
    hpo_configs: list[HPOConfig],
    root_url_tags_mapping: Optional[dict] = None,
) -> list[list[Answer]]:
    answers = []
    for config in hpo_configs:
        _answers = generate_answers(
            questions=questions,
            root_url_tags_mapping=root_url_tags_mapping,
            splitter=config.splitter,
            chunk_size=config.chunk_size,
            exclude_patterns=config.exclude_patterns,
            prompt_template=config.prompt_template,
            limit=config.limit,
            embedding_type=config.embedding_type,
        )
        answers.append(_answers)
    return answers


@fk.task(
    container_image=image,
    cache=True,
    cache_version="1",
    enable_deck=True,
)
def combine_answers(
    answers: list[list[Answer]],
    eval_configs: list[HPOConfig],
    questions: list[Question],
) -> Annotated[pd.DataFrame, TopFrameRenderer(10)]:
    # TODO: concatenate all answers into a single dataframe
    combined_answers = []
    for _answers, config in zip(answers, eval_configs):
        for answer, question in zip(_answers, questions):
            assert answer.question_id == question.question_id
            combined_answers.append(
                {
                    "question_id": question.question_id,
                    "question": question.question,
                    "answer": answer.answer,
                    "reference_answer": question.reference_answer,
                    "is_user_generated": question.is_user_generated,
                    **asdict(config),
                }
            )

    return pd.DataFrame(combined_answers)


def traditional_nlp_eval(answers_dataset: pd.DataFrame) -> pd.DataFrame:
    from nltk.translate.bleu_score import sentence_bleu
    from rouge_score import rouge_scorer

    bleu_scores, rouge1_scores = [], []
    rouge_scorer = rouge_scorer.RougeScorer(["rouge1"])
    for row in answers_dataset.itertuples():
        bleu_scores.append(
            sentence_bleu([row.reference_answer.split()], row.answer.split())
        )
        _rouge_scores = rouge_scorer.score(row.reference_answer, row.answer)
        rouge1_scores.append(_rouge_scores["rouge1"].fmeasure)

    return answers_dataset.assign(
        bleu_score=bleu_scores,
        rouge1_f1=rouge1_scores,
    )


DEFAULT_EVAL_PROMPT_TEMPLATE = """### Task Description:
You are an expert in judging the correctness of answers relating
to pandas, a python data analysis library. Given a question and a
reference answer, determine if the candidate answer is equivalent or
better than the reference answer in terms of correctness.

### Question:
{question}

### Reference Answer:
{reference_answer}

### Candidate Answer:
{candidate_answer}

### Judgement:
Is the candidate answer equivalent or better than the reference answer
in terms of correctness? You MUST answer "Yes" or "No".
"""


def llm_judge_eval(
    answers_dataset: pd.DataFrame, eval_prompt_template: Optional[str] = None
) -> pd.DataFrame:
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model_name="gpt-4o", temperature=0.9)
    prompt = PromptTemplate.from_template(
        eval_prompt_template or DEFAULT_EVAL_PROMPT_TEMPLATE
    )

    llm_correctness_scores = []

    for _, row in answers_dataset.iterrows():
        chain = prompt | model
        result = chain.invoke(
            {
                "question": row["question"],
                "reference_answer": row["reference_answer"],
                "candidate_answer": row["answer"],
            }
        )

        result = result.content.lower().strip().strip(".").strip("'").strip('"')
        if result not in ["yes", "no"]:
            score = 0.0
        elif result == "yes":
            score = 1.0
        else:
            score = 0.0
        llm_correctness_scores.append(score)

    return answers_dataset.assign(llm_correctness_score=llm_correctness_scores)


@fk.task(
    container_image=image,
    enable_deck=True,
    secret_requests=[fk.Secret(key="openai_api_key")],
    requests=fk.Resources(cpu="4", mem="8Gi"),
    cache=True,
    cache_version="6",
)
@openai_env_secret
def evaluate(
    answers_dataset: pd.DataFrame,
    eval_prompt_template: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import seaborn as sns

    evaluation = traditional_nlp_eval(answers_dataset)
    evaluation = llm_judge_eval(evaluation, eval_prompt_template)

    evaluation_summary = (
        evaluation
        .astype({"exclude_patterns": str})
        .groupby([*HPOConfig.__dataclass_fields__])[
            ["bleu_score", "rouge1_f1", "llm_correctness_score"]
        ]
        .mean()
        .reset_index()
    )

    analysis_df = evaluation_summary.melt(
        id_vars=["condition_name"],
        value_vars=["bleu_score", "rouge1_f1", "llm_correctness_score"],
        var_name="metric",
        value_name="score",
    )

    g = sns.FacetGrid(analysis_df, col="metric", sharey=False)
    g.map_dataframe(sns.barplot, y="condition_name", x="score", orient="h")
    g.add_legend()

    decks = fk.current_context().decks
    decks.insert(0, fk.Deck("Evaluation", TopFrameRenderer(10).to_html(evaluation)))
    decks.insert(0, fk.Deck("Evaluation Summary", TopFrameRenderer(10).to_html(evaluation_summary)))
    decks.insert(0, fk.Deck("Benchmarking Results", convert_fig_into_html(g.figure)))

    return evaluation, evaluation_summary


@fk.workflow
def optimize_rag(
    hpo_configs: list[HPOConfig],
    root_url_tags_mapping: Optional[dict] = None,
    eval_dataset: Annotated[pd.DataFrame, EvalDatasetArtifact] = EvalDatasetArtifact.query(dataset_type="llm_filtered"),
    eval_prompt_template: Optional[str] = None,
    n_answers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    questions = prepare_questions(eval_dataset, n_answers)
    answers = gridsearch(questions, hpo_configs, root_url_tags_mapping)
    answers_dataset = combine_answers(answers, hpo_configs, questions)
    evaluation, evalution_summary = evaluate(
        answers_dataset, eval_prompt_template
    )
    return evaluation, evalution_summary
