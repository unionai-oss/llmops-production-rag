"""Generate a question and answer dataset."""

import functools
import json
from dataclasses import dataclass
from typing import List, Annotated

import union
from flytekit.deck import MarkdownRenderer
from flytekit.types.file import FlyteFile

from llmops_rag.document import CustomDocument
from llmops_rag.image import image
from llmops_rag.vector_store import KnowledgeBase
from llmops_rag.utils import openai_env_secret


DEFAULT_ANNOTATION_SET_NAME = "default"

QuestionAndAnswerDataset = union.Artifact(name="question_and_answer_dataset")


@dataclass
class QuestionAndAnswers:
    question: str
    answers: list[str]
    context: str
    url: str
    id: int | None = None


@union.task(
    container_image=image,
    requests=union.Resources(cpu="2", mem="8Gi"),
    secret_requests=[union.Secret(key="openai_api_key")],
    # cache=True,
    # cache_version="3",
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
        8. IMPORTANT: Write the question as if you were a data scientist or analyst asking a question
           on an online Q&A site like Stack Exchange.

        Questions:""",
    )

    answer_prompt = PromptTemplate(
        input_variables=["context", "question", "n_answers"],
        template="""
        You are an assistant for question-answering tasks in the pandas python data analysis library.
        Use only the context to answer the question. Make the answer as detailed but as concise as
        possible.

        Given the following context and question, provide {n_answers} distinct answers.
        Each answer should be provide a balanced perspective, analysis, or inference based
        on the given context.

        Context: {context}

        Question: {question}

        Requirements:
        1. Provide {n_answers} distinct answers. Each answer can be as long as needed.
        2. While each answer will be different, individual answers should still be as comprehensive as
           possible and address all angles of the question.
        3. Focus on delivering the best answer possible given the context. Do not restate the question in the answers.
        4. Include code snippets in the answers if they are relevant.
        5. If the answer is not directly in the context, use reasonable inferences or analysis to provide
           possible answers.
        6. Important: delimit each answer with a new line containing only the characters "---".

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
            ans.strip() for ans in answers_text.strip().split("---") if ans.strip()
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


QA_DATASET_TEMPLATE = """
# ❓ Question and Answer Dataset

This dataset contains questions and answers generated from the provided documents.

- Number of questions: {n_questions}
- Answers per question: {n_answers_per_question}

## Preview

{preview}
"""


@union.task(
    container_image=image,
    requests=union.Resources(cpu="2", mem="4Gi"),
    # cache=True,
    # cache_version="2",
    enable_deck=True,
    deck_fields=[],
)
def create_dataset(questions_and_answers: List[List[QuestionAndAnswers]], n_answers_per_question: int) -> FlyteFile:
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

    preview = "\n\n".join([f"```json\n{json.dumps(x, indent=4)}\n```" for x in qa_dataset[:5]])
    union.Deck("QA Dataset", MarkdownRenderer().to_html(
        QA_DATASET_TEMPLATE.format(
            n_questions=len(qa_dataset),
            n_answers_per_question=n_answers_per_question,
            preview=preview)
        )
    )
    return FlyteFile(path=file_path)


@union.workflow
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
    questions_and_answers = union.map(partial_task)(flyte_doc=documents)
    return create_dataset(questions_and_answers, n_answers_per_question)
