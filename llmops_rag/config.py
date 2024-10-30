from dataclasses import dataclass
from typing import Optional


@dataclass
class RAGConfig:
    splitter: str = "character"
    exclude_patterns: Optional[list[str]] = None
    prompt_template: str = ""
    chunk_size: int = 2048
    limit: int = 100
    embedding_model: str = "text-embedding-ada-002"
    search_type: str = "similarity"  # or "mmr"
    rerank: bool = False
    num_retrieved_docs: int = 20
    num_docs_final: int = 5
