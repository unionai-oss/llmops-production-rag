from dataclasses import dataclass


DEFAULT_PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks in the pandas python data analysis library.
Use only the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Make the answer as
detailed as possible. When providing the answer, DO NOT make any explicit references
to the context, e.g. "According to the context..." or "The provided code snippets demostrates...".

Write the answer in markdown format. Make sure that Python code snippets are formatted with ```python

## Question:
{question}

## Context:
{context}

## Answer:
"""


@dataclass
class RAGConfig:
    splitter: str = "character"
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    chunk_size: int = 2048
    embedding_model: str = "text-embedding-ada-002"
    generation_model: str = "gpt-4o-mini"
    search_type: str = "similarity"  # or "mmr"
    rerank: bool = False
    num_retrieved_docs: int = 20
    num_docs_final: int = 5
