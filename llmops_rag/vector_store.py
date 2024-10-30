"""Union and Flyte chat assistant workflow."""

import asyncio
import itertools
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Optional

import flytekit as fk
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from union.artifacts import DataCard

from llmops_rag.image import image
from llmops_rag.document import CustomDocument
from llmops_rag.utils import openai_env_secret



KnowledgeBase = fk.Artifact(name="knowledge-base")
VectorStore = fk.Artifact(name="vector-store")


@fk.task(
    container_image=image,
    cache=True,
    cache_version="6",
    requests=fk.Resources(cpu="2", mem="8Gi"),
    enable_deck=True,
)
def create_knowledge_base(
    root_url_tags_mapping: Optional[dict] = None,
    limit: Optional[int | float] = None,
    exclude_patterns: Optional[list[str]] = None,
    _random_state: Optional[int] = None,
) -> Annotated[list[CustomDocument], KnowledgeBase]:
    """
    Get the documents to create the knowledge base.
    """
    from langchain_community.document_loaders import AsyncHtmlLoader
    from llmops_rag.document import get_links, HTML2MarkdownTransformer

    root_url_tags_mapping = root_url_tags_mapping or {
        "https://pandas.pydata.org/docs/user_guide/": ("div", {"class": "bd-article-container"}),
    }

    exclude_patterns = exclude_patterns or [
        "docs/getting_started", "/docs/reference/", "/docs/development/", "/docs/whatsnew/",
    ]
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
    embedding_model: str,
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

This artifact is a vector store of {len(document_chunks)} document chunks using {embedding_model} embeddings.

## Preview

{document_preview_str}
"""


@fk.task(
    container_image=image,
    cache=True,
    cache_version="3",
    requests=fk.Resources(cpu="2", mem="8Gi"),
    secret_requests=[fk.Secret(key="openai_api_key")],
    enable_deck=True,
)
@openai_env_secret
def chunk_and_embed_documents(
    documents: list[CustomDocument],
    splitter: str,
    chunk_size: int,
    embedding_model: Optional[str] = None,
) -> Annotated[FlyteDirectory, VectorStore]:
    """
    Create the search index.
    """
    from langchain_community.vectorstores import FAISS
    from langchain.docstore.document import Document
    from langchain_openai import OpenAIEmbeddings
    from langchain.text_splitter import (
        CharacterTextSplitter,
        RecursiveCharacterTextSplitter,
        MarkdownHeaderTextSplitter,
    )

    splitter = splitter or "recursive"
    chunk_size = int(chunk_size or 1024)
    embedding_model = embedding_model or "text-embedding-ada-002"

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


    embeddings = OpenAIEmbeddings(model=embedding_model)

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
        DataCard(generate_data_md(document_chunks, embedding_model)),
    )


@fk.workflow
def create_vector_store(
    root_url_tags_mapping: Optional[dict] = None,
    splitter: str = "character",
    chunk_size: int = 2048,
    limit: Optional[int | float] = None,
    embedding_model: Optional[str] = "text-embedding-ada-002",
    exclude_patterns: Optional[list[str]] = None,
) -> FlyteDirectory:
    """
    Workflow for creating the vector store knowledge base.
    """
    docs = create_knowledge_base(
        root_url_tags_mapping=root_url_tags_mapping,
        limit=limit,
        exclude_patterns=exclude_patterns,
    )
    vector_store = chunk_and_embed_documents(
        documents=docs,
        splitter=splitter,
        chunk_size=chunk_size,
        embedding_model=embedding_model,
    )
    return vector_store


create_vector_store_lp = fk.LaunchPlan.get_or_create(
    name="create_vector_store_lp",
    workflow=create_vector_store,
    default_inputs={
        "limit": 10,
    },
    schedule=fk.FixedRate(
        duration=timedelta(minutes=3)
    )
)
