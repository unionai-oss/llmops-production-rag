from dataclasses import dataclass
from typing import Optional

from mashumaro.mixins.json import DataClassJSONMixin


@dataclass
class RAGConfig(DataClassJSONMixin):
    splitter: str = "character"
    exclude_patterns: Optional[list[str]] = None
    prompt_template: str = ""
    chunk_size: int = 2048
    include_union: bool = False
    limit: Optional[int | float] = None
    embedding_type: str = "openai"
