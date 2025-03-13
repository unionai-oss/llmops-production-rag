"""Union and Flyte chat assistant workflow."""

from typing import Optional

import flytekitplugins.inference as fl_inference
import union

from flytekit.deck import MarkdownRenderer
from flytekit.extras.accelerators import L4
from flytekit.types.directory import FlyteDirectory
from union.actor import ActorEnvironment

from llmops_rag.image import image
from llmops_rag.utils import openai_env_secret
from llmops_rag.config import DEFAULT_PROMPT_TEMPLATE


VectorStore = union.Artifact(name="vector-store")


OLLAMA_MODEL_NAME = "llama3.1"
ollama_instance = fl_inference.Ollama(
    model=fl_inference.Model(OLLAMA_MODEL_NAME),
    gpu="1",
)


actor = ActorEnvironment(
    name="rag-actor",
    ttl_seconds=180,
    container_image=image,
    replica_count=8,
    requests=union.Resources(cpu="2", mem="8Gi"),
    secret_requests=[union.Secret(key="openai_api_key")],
)


ollama_actor = ActorEnvironment(
    name="ollama-actor",
    ttl_seconds=180,
    container_image=image,
    replica_count=1,
    requests=union.Resources(gpu="1", mem="8Gi"),
    accelerator=L4,
    pod_template=ollama_instance.pod_template,
)


@actor.task(enable_deck=True, deck_fields=[])
@openai_env_secret
def retrieve(
    questions: list[str],
    vector_store: FlyteDirectory,
    embedding_model: Optional[str] = None,
    search_type: str = "similarity",
    rerank: bool = False,
    num_retrieved_docs: int = 20,
    num_docs_final: int = 5,
) -> list[str]:
    from langchain.retrievers.document_compressors import FlashrankRerank
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
    from flashrank import Ranker

    assert num_retrieved_docs >= num_docs_final
    embedding_model = embedding_model or "text-embedding-ada-002"

    vector_store.download()
    vector_store = FAISS.load_local(
        vector_store.path,
        OpenAIEmbeddings(model=embedding_model),
        allow_dangerous_deserialization=True,
    )
    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": num_retrieved_docs},
    )
    if rerank:
        retriever = ContextualCompressionRetriever(
            base_retriever=retriever,
            base_compressor=FlashrankRerank(
                client=Ranker(model_name="ms-marco-MultiBERT-L-12"),
                top_n=num_docs_final
            ),
        )

    contexts = []
    for question in questions:
        relevant_docs = [doc.page_content for doc in retriever.invoke(question)]
        context = "\n\n".join(relevant_docs[:num_docs_final])
        contexts.append(context)

    union.Deck("Context", MarkdownRenderer().to_html(contexts[0]))
    return contexts


@actor.task(enable_deck=True, deck_fields=[])
@openai_env_secret
def generate(
    questions: list[str],
    contexts: list[str],
    generation_model: Optional[str] = None,
    prompt_template: Optional[str] = None,
) -> list[str]:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI

    model_name = generation_model or "gpt-4o"
    prompt = PromptTemplate.from_template(prompt_template or DEFAULT_PROMPT_TEMPLATE)
    llm = ChatOpenAI(model_name=model_name, temperature=0.9)

    chain = prompt | llm | StrOutputParser()
    answers = []
    for question, context in zip(questions, contexts):
        answer = chain.invoke({"question": question, "context": context})
        answers.append(answer)

    union.Deck("Answer", MarkdownRenderer().to_html(answers[0]))
    return answers


@union.workflow
def rag_basic(
    questions: list[str],
    vector_store: FlyteDirectory = VectorStore.query(),  # 👈 this uses the vector store artifact by default
    embedding_model: str = "text-embedding-ada-002",
    generation_model: str = "gpt-4o-mini",
    prompt_template: Optional[str] = None,
    search_type: str = "similarity",
    rerank: bool = False,
    num_retrieved_docs: int = 20,
    num_docs_final: int = 5,
) -> list[str]:
    contexts = retrieve(
        questions,
        vector_store,
        embedding_model,
        search_type,
        rerank,
        num_retrieved_docs,
        num_docs_final,
    )
    return generate(
        questions,
        contexts,
        generation_model,
        prompt_template,
    )


@ollama_actor.task(enable_deck=True, deck_fields=[])
def generate_ollama(
    questions: list[str],
    contexts: list[str],
    generation_model: Optional[str] = None,
    prompt_template: Optional[str] = None,
) -> list[str]:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = PromptTemplate.from_template(prompt_template or DEFAULT_PROMPT_TEMPLATE)
    assert generation_model in [OLLAMA_MODEL_NAME]

    llm = ChatOpenAI(
        model_name=generation_model,
        base_url=f"{ollama_instance.base_url}/v1",
        api_key="ollama",
        temperature=0.9
    )

    chain = prompt | llm | StrOutputParser()
    answers = []
    for question, context in zip(questions, contexts):
        answer = chain.invoke({"question": question, "context": context})
        answers.append(answer)

    union.Deck("Answer", MarkdownRenderer().to_html(answers[0]))
    return answers


@union.workflow
def rag_basic_ollama(
    questions: list[str],
    vector_store: FlyteDirectory = VectorStore.query(),  # 👈 this uses the vector store artifact by default
    embedding_model: str = "text-embedding-ada-002",
    generation_model: str = "llama3.1",
    search_type: str = "similarity",
    rerank: bool = False,
    num_retrieved_docs: int = 20,
    num_docs_final: int = 5,
    prompt_template: Optional[str] = None,
) -> list[str]:
    contexts = retrieve(
        questions=questions,
        vector_store=vector_store,
        embedding_model=embedding_model,
        search_type=search_type,
        rerank=rerank,
        num_retrieved_docs=num_retrieved_docs,
        num_docs_final=num_docs_final,
    )
    return generate_ollama(
        questions=questions,
        contexts=contexts,
        generation_model=generation_model,
        prompt_template=prompt_template,
    )
