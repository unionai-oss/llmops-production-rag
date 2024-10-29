"""Union and Flyte chat assistant workflow."""

from typing import Optional

from flytekit import (
    workflow,
    Artifact,
    Deck,
    Secret,
    Resources,
)
from flytekit.deck import MarkdownRenderer
from flytekit.types.directory import FlyteDirectory
from union.actor import ActorEnvironment

from llmops_rag.image import image
from llmops_rag.utils import openai_env_secret


VectorStore = Artifact(name="vector-store")

DEFAULT_PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks in the biomedical domain.
Use only the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Make the answer as
detailed as possible. If the answer contains acronyms, make sure to expand on them.

Question: {question}
Context: {context}
Answer:
"""


actor = ActorEnvironment(
    name="simple-rag",
    ttl_seconds=120,
    container_image=image,
    requests=Resources(cpu="2", mem="8Gi"),
    secret_requests=[Secret(key="openai_api_key")],
)


@actor.task(enable_deck=True, deck_fields=[])
@openai_env_secret
def retrieve(
    question: str,
    vector_store: FlyteDirectory,
) -> str:
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings

    vector_store.download()
    vector_store = FAISS.load_local(
        vector_store.path,
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True,
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8},
    )
    context = "\n\n".join(doc.page_content for doc in retriever.invoke(question))
    Deck("Context", MarkdownRenderer().to_html(context))
    return context


@actor.task(enable_deck=True, deck_fields=[])
@openai_env_secret
def generate(
    question: str,
    context: str,
    prompt_template: Optional[str] = None,
) -> str:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = PromptTemplate.from_template(prompt_template or DEFAULT_PROMPT_TEMPLATE)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.9)

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context})
    Deck("Answer", MarkdownRenderer().to_html(answer))
    return answer


@workflow
def run(
    question: str,
    vector_store: FlyteDirectory = VectorStore.query(),  # 👈 this uses the vector store artifact by default
    prompt_template: Optional[str] = None,
) -> str:
    context = retrieve(question, vector_store)
    return generate(
        question=question,
        context=context,
        prompt_template=prompt_template,
    )
