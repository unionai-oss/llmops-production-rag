import functools
import itertools
import json
import os
from dataclasses import dataclass
from typing import List, Annotated

from flytekit import (
    task,
    workflow,
    Resources,
    Artifact,
    Secret,
    map_task,
    current_context,
)
from flytekit.types.file import FlyteFile

from llmops_rag.document import CustomDocument
from llmops_rag.image import image
from llmops_rag.vector_store import KnowledgeBase


DEFAULT_ANNOTATION_SET_NAME = "default"

QuestionAndAnswerDataset = Artifact(name="question_and_answer_dataset")


@dataclass
class QuestionAndAnswers:
    question: str
    answers: List[str]
    url: str


@task(
    container_image=image,
    requests=Resources(cpu="2", mem="8Gi"),
    secret_requests=[Secret(key="openai_api_key")],
    cache=True,
    cache_version="v10",
)
def generate_qa_pairs(
    flyte_doc: CustomDocument, n_questions: int, n_answers: int
) -> List[QuestionAndAnswers]:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain

    os.environ["OPENAI_API_KEY"] = current_context().secrets.get(key="openai_api_key")
    os.environ["OPENAI_API_TYPE"] = "chat"

    document = flyte_doc.to_document()

    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.9, presence_penalty=1.5)

    question_prompt = PromptTemplate(
        input_variables=["context", "n_questions"],
        template="""Given the following context, generate {n_questions} relevant and specific questions that require thoughtful, nuanced answers.

        Context: {context}

        Requirements:
        1. Each question should be clear but encourage deep thinking or inference.
        2. Avoid questions that can be answered with a simple factual statement.
        3. Incorporate "what if" scenarios or potential challenges based on the context.
        4. Ensure questions cover different aspects and perspectives of the context.
        5. The question should relate to the overarching theme or concepts in the context but not be directly answerable by it. Think of the question as a follow-up from an attentive student seeking to explore the topic further or clarify a complex point.
        6. IMPORTANT: Place each question on a new line, with no numbering or prefixes.

        Questions:""",
    )

    answer_prompt = PromptTemplate(
        input_variables=["context", "question", "n_answers"],
        template="""Given the following context and question, provide {n_answers} concise, thoughtful, and distinct answers. Each answer should be provide a balanced perspective, analysis, or inference based on the given context.

        Context: {context}

        Question: {question}

        Requirements:
        1. Provide {n_answers} distinct answers. Each answer should be 1 sentence long.
        2. While each answer will be different, individual answers should still be as comprehensive as possible and address all angles of the question.
        3. Focus on delivering concise and impactful responses. Do not restate the question in the answers.
        4. If the answer is not directly in the context, use reasonable inferences or analysis to provide possible answers.
        5. IMPORTANT: Place each answer on a new line, with no numbering or prefixes.

        Answers:""",
    )

    question_chain = LLMChain(llm=llm, prompt=question_prompt)
    answer_chain = LLMChain(llm=llm, prompt=answer_prompt)

    # Generate multiple questions
    questions_text = question_chain.run(
        context=document.page_content, n_questions=n_questions
    )
    questions = [q.strip() for q in questions_text.strip().split("\n") if q.strip()]

    qa_pairs = []
    for question in questions:
        # Generate multiple answers for each question
        answers_text = answer_chain.run(
            context=document.page_content, question=question, n_answers=n_answers
        )
        answers = [
            ans.strip() for ans in answers_text.strip().split("\n") if ans.strip()
        ]
        print("question:")
        print(question)
        print("answers:")
        print(answers)
        qa_pairs.append(
            QuestionAndAnswers(
                question=question, answers=answers, url=flyte_doc.metadata["source"]
            )
        )

    return qa_pairs


@task(
    container_image=image,
    requests=Resources(cpu="2", mem="4Gi"),
    cache=True,
    cache_version="v2",
)
def create_dataset(questions_and_answers: List[List[QuestionAndAnswers]]) -> FlyteFile:
    questions_and_answers_flat = [
        qa for qa_sublist in questions_and_answers for qa in qa_sublist
    ]
    qa_triples_list = []
    id_counter = 1
    for i, qa in enumerate(questions_and_answers_flat):
        question = qa.question
        answers = qa.answers
        answer_combinations = list(itertools.combinations(answers, 2))

        for j, combo in enumerate(answer_combinations):
            qa_triples_list.append(
                {
                    "id": id_counter,
                    "question": question,
                    "answers": list(combo),
                }
            )
            id_counter += 1

    file_path = "qa_triples_list.json"
    with open(file_path, "w") as f:
        json.dump(qa_triples_list, f, indent=4)

    return FlyteFile(path=file_path)


@workflow
def run(
    documents: List[CustomDocument] = KnowledgeBase.query(),
    n_questions: int = 1,
    n_answers: int = 5,
) -> Annotated[FlyteFile, QuestionAndAnswerDataset]:
    partial_task = functools.partial(
        generate_qa_pairs, n_questions=n_questions, n_answers=n_answers
    )
    questions_and_answers = map_task(partial_task)(flyte_doc=documents)
    return create_dataset(questions_and_answers=questions_and_answers)
