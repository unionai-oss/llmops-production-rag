"""Generate a question and answer dataset."""

import functools
import json
from dataclasses import dataclass
from typing import List, Annotated

import flytekit as fk
from flytekit.deck import MarkdownRenderer
from flytekit.types.file import FlyteFile

from llmops_rag.document import CustomDocument
from llmops_rag.image import image
from llmops_rag.vector_store import KnowledgeBase
from llmops_rag.utils import openai_env_secret


DEFAULT_ANNOTATION_SET_NAME = "default"

QuestionAndAnswerDataset = fk.Artifact(name="question_and_answer_dataset")


@dataclass
class QuestionAndAnswers:
    question: str
    answers: list[str]
    context: str
    url: str
    id: int | None = None


@fk.task(
    container_image=image,
    requests=fk.Resources(cpu="2", mem="8Gi"),
    secret_requests=[fk.Secret(key="openai_api_key")],
    cache=True,
    cache_version="0",
    environment={"OPENAI_API_TYPE": "chat"},
)
@openai_env_secret
def generate_qa_datapoints(
    flyte_doc: CustomDocument, n_questions_per_doc: int, n_answers_per_question: int
) -> List[QuestionAndAnswers]:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    document = flyte_doc.to_document()

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.9, presence_penalty=1.5)

    question_prompt = PromptTemplate(
        input_variables=["context", "n_questions"],
        template="""Given the following context, generate {n_questions} relevant and specific
        questions that require thoughtful, nuanced answers.

        Context: {context}

        Requirements:
        1. Each question should be clear but encourage deep thinking or inference.
        2. Avoid questions that can be answered with a simple factual statement.
        3. Incorporate "what if" scenarios or potential challenges based on the context.
        4. Ensure questions cover different aspects and perspectives of the context.
        5. The question should relate to the overarching theme or concepts in the context but not be
           directly answerable by it. Think of the question as a follow-up from an attentive student
           seeking to explore the topic further or clarify a complex point.
        6. If the context contains code, include a mix of questions where the answer is the code
           and higher level questions that require reasoning about the code.
        7. IMPORTANT: Place each question on a new line, with no numbering or prefixes.

        Questions:""",
    )

    answer_prompt = PromptTemplate(
        input_variables=["context", "question", "n_answers"],
        template="""Given the following context and question, provide {n_answers} concise, thoughtful, and
        distinct answers. Each answer should be provide a balanced perspective, analysis, or inference based
        on the given context.

        Context: {context}

        Question: {question}

        Requirements:
        1. Provide {n_answers} distinct answers. Each answer should be 1 sentence long.
        2. While each answer will be different, individual answers should still be as comprehensive as
           possible and address all angles of the question.
        3. Focus on delivering concise and impactful responses. Do not restate the question in the answers.
        4. If the answer is not directly in the context, use reasonable inferences or analysis to provide
           possible answers.
        5. IMPORTANT: Place each answer on a new line, with no numbering or prefixes.

        Answers:""",
    )

    question_chain = question_prompt | llm | StrOutputParser()
    answer_chain = answer_prompt | llm | StrOutputParser()

    # Generate multiple questions
    questions_text = question_chain.invoke(
        {"context": document.page_content, "n_questions": n_questions_per_doc}
    )
    questions = [q.strip() for q in questions_text.strip().split("\n") if q.strip()]

    qa_pairs = []
    for question in questions:
        # Generate multiple answers for each question
        answers_text = answer_chain.invoke(
            {"context": document.page_content, "question": question, "n_answers": n_answers_per_question}
        )
        answers = [
            ans.strip() for ans in answers_text.strip().split("\n") if ans.strip()
        ]
        qa_pairs.append(
            QuestionAndAnswers(
                question=question,
                answers=answers,
                context=document.page_content,
                url=flyte_doc.metadata["source"],
            )
        )

    return qa_pairs


@fk.task(
    container_image=image,
    requests=fk.Resources(cpu="2", mem="4Gi"),
    cache=True,
    cache_version="0",
    enable_deck=True,
    deck_fields=[],
)
def create_dataset(questions_and_answers: List[List[QuestionAndAnswers]]) -> FlyteFile:
    questions_and_answers_flat = [
        qa for qa_sublist in questions_and_answers for qa in qa_sublist
    ]
    qa_dataset = []
    for i, qa in enumerate(questions_and_answers_flat, start=1):
        qa_dataset.append(
            {
                "id": i,
                "question": qa.question,
                "answers": qa.answers,
                "context": qa.context,
                "url": qa.url,
            }
        )

    file_path = "qa_dataset.json"
    with open(file_path, "w") as f:
        json.dump(qa_dataset, f, indent=4)

    fk.Deck("QA Dataset", MarkdownRenderer().to_html(f"Number of questions: {len(qa_dataset)}"))
    return FlyteFile(path=file_path)


@fk.workflow
def create_qa_dataset(
    documents: List[CustomDocument] = KnowledgeBase.query(),
    n_questions_per_doc: int = 1,
    n_answers_per_question: int = 5,
) -> Annotated[FlyteFile, QuestionAndAnswerDataset]:
    partial_task = functools.partial(
        generate_qa_datapoints,
        n_questions_per_doc=n_questions_per_doc,
        n_answers_per_question=n_answers_per_question,
    )
    questions_and_answers = fk.map_task(partial_task)(flyte_doc=documents)
    return create_dataset(questions_and_answers=questions_and_answers)
