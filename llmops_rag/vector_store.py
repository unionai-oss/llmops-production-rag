"""Union and Flyte chat assistant workflow."""

import asyncio
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

from flytekit import (
    task,
    workflow,
    Artifact,
    Secret,
    Resources,
)
from flytekit.core.artifact import Inputs
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from union.artifacts import DataCard

from llmops_rag.image import image
from llmops_rag.document import CustomDocument
from llmops_rag.utils import openai_env_secret



KnowledgeBase = Artifact(name="knowledge-base")
VectorStore = Artifact(name="vector-store")

DEFAULT_PROMPT_TEMPLATE = """You are a helpful chat assistant that is an expert in Flyte and the flytekit sdk.
Create a final answer with references ("SOURCES").
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
If the QUESTION is not relevant to Flyte or flytekit, just say that you're not able
to answer any questions that are not related to Flyte or flytekit.
ALWAYS return a "SOURCES" part in your answer.

SOURCES:

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:

"""


@task(
    container_image=image,
    cache=True,
    cache_version="4",
    requests=Resources(cpu="2", mem="8Gi"),
    enable_deck=True,
)
def create_knowledge_base(
    root_url_tags_mapping: Optional[dict] = None,
    include_union: bool | None = None,
    limit: Optional[int | float] = None,
    exclude_patterns: Optional[list[str]] = None,
) -> Annotated[list[CustomDocument], KnowledgeBase]:
    """
    Get the documents to create the knowledge base.
    """
    from langchain_community.document_loaders import AsyncHtmlLoader
    from llmops_rag.document import get_links, HTML2MarkdownTransformer

    include_union = include_union if isinstance(include_union, bool) else False

    if root_url_tags_mapping is None:
        root_url_tags_mapping = {
            "https://docs.flyte.org/en/latest/": ("article", {"class": "bd-article"}),
        }
    if include_union:
        root_url_tags_mapping.update(
            {
                "https://docs.union.ai/byoc/": ("article", {"class": "bd-article"}),
                "https://docs.union.ai/serverless/": (
                    "article",
                    {"class": "bd-article"},
                ),
            }
        )

    exclude_patterns = exclude_patterns or ["/api/", "/_tags/"]
    page_transformer = HTML2MarkdownTransformer(root_url_tags_mapping)
    urls = list(
        itertools.chain(
            *(get_links(url, limit, exclude_patterns) for url in root_url_tags_mapping)
        )
    )
    loader = AsyncHtmlLoader(urls)
    html = loader.lazy_load()

    md_transformed = page_transformer.transform_documents(
        html,
        unwanted_tags=[
            "script",
            "style",
            ("a", {"class": "headerlink"}),
            ("button", {"class": "CopyButton"}),
            ("div", {"class": "codeCopied"}),
            ("span", {"class": "lang"}),
        ],
        remove_lines=False,
    )

    root_path = Path("./docs")
    root_path.mkdir(exist_ok=True)
    documents = []
    for i, doc in enumerate(md_transformed):
        if doc.page_content == "":
            print(f"Skipping empty document {doc}")
            continue
        path = root_path / f"doc_{i}.md"
        print(f"Writing document {doc.metadata['source']} to {path}")
        with path.open("w") as f:
            f.write(doc.page_content)
        documents.append(
            CustomDocument(page_filepath=FlyteFile(str(path)), metadata=doc.metadata)
        )

    # create a new event loop to avoid asyncio RuntimeError. Somehow langchain
    # will close the event loop and not allow further async operations
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    return documents


def generate_data_md(
    document_chunks,
    embedding_type: str,
    n_chunks: int = 5,
) -> str:
    document_chunks_preview = document_chunks[:n_chunks]
    document_preview_str = ""
    for i, doc in enumerate(document_chunks_preview):
        document_preview_str += f"""\n\n---

### 📖 Chunk {i}

**Page metadata:**

{doc.metadata}

**Content:**

```
{doc.page_content.replace("```", "")}
```
"""

    return f"""# 📚 Vector store knowledge base.

This artifact is a vector store of {len(document_chunks)} document chunks using {embedding_type} embeddings.

## Preview

{document_preview_str}
"""


@task(
    container_image=image,
    cache=True,
    cache_version="3",
    requests=Resources(cpu="2", mem="8Gi"),
    secret_requests=[Secret(key="openai_api_key")],
    enable_deck=True,
)
@openai_env_secret
def create_vector_store(
    documents: list[CustomDocument] = KnowledgeBase.query(),
    splitter: str | None = None,
    chunk_size: int | float | None = None,
    embedding_type: str | None = None,
) -> Annotated[FlyteDirectory, VectorStore]:
    """
    Create the search index.
    """
    from langchain_community.vectorstores import FAISS
    from langchain.docstore.document import Document
    from langchain.text_splitter import (
        CharacterTextSplitter,
        RecursiveCharacterTextSplitter,
        MarkdownHeaderTextSplitter,
    )

    splitter = splitter or "recursive"
    chunk_size = int(chunk_size or 1024)
    embedding_type = embedding_type or "openai"

    documents = [flyte_doc.to_document() for flyte_doc in documents]
    if splitter == "character":
        splitter = CharacterTextSplitter(
            separator=" ", chunk_size=int(chunk_size), chunk_overlap=0
        )
    elif splitter == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.2),
            length_function=len,
            is_separator_regex=False,
        )
    elif splitter == "markdown":
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        )
    else:
        raise ValueError(f"Invalid splitter: {splitter}")

    if embedding_type == "openai":
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings()
    elif embedding_type == "huggingface":
        from langchain_huggingface.embeddings import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    document_chunks = [
        Document(page_content=chunk, metadata=doc.metadata)
        for doc in documents
        for chunk in splitter.split_text(doc.page_content)
    ]

    vector_store = FAISS.from_documents(
        documents=document_chunks,
        embedding=embeddings,
    )
    path = "./vector_store"
    vector_store.save_local(path)
    return VectorStore.create_from(
        FlyteDirectory(path=path),
        DataCard(generate_data_md(document_chunks, embedding_type)),
    )


@workflow
def create(
    splitter: str = "character",
    chunk_size: int = 2048,
    include_union: bool = False,
    limit: Optional[int | float] = None,
    embedding_type: str = "openai",
    exclude_patterns: Optional[list[str]] = None,
) -> FlyteDirectory:
    """
    Workflow for creating the vector store knowledge base.
    """
    docs = create_knowledge_base(
        include_union=include_union,
        limit=limit,
        exclude_patterns=exclude_patterns,
    )
    vector_store = create_vector_store(
        documents=docs,
        splitter=splitter,
        chunk_size=chunk_size,
        embedding_type=embedding_type,
    )
    return vector_store
