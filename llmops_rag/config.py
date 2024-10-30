from dataclasses import dataclass
from typing import Optional


@dataclass
class RAGConfig:
    splitter: str = "character"
    exclude_patterns: Optional[list[str]] = None
    prompt_template: str = ""
    chunk_size: int = 2048
    limit: Optional[int] = None
    embedding_model: str = "text-embedding-ada-002"
    rerank: bool = False
