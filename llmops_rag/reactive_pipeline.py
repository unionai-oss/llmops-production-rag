"""A reactive pipeline to trigger RAG optimization when new eval dataset is available."""

from typing import Optional
from datetime import timedelta

import union
import pandas as pd
from union.artifacts import OnArtifact

from llmops_rag.document import CustomDocument
from llmops_rag.create_qa_dataset import create_qa_dataset
from llmops_rag.create_llm_filtered_dataset import create_llm_filtered_dataset, EvalDatasetArtifact
from llmops_rag.optimize_rag import optimize_rag, GridSearchConfig
from llmops_rag.vector_store import create_knowledge_base, KnowledgeBase


@union.workflow
def knowledge_base_workflow(
    root_url_tags_mapping: Optional[dict] = None,
    limit: Optional[int | float] = None,
    exclude_patterns: Optional[list[str]] = None,
) -> list[CustomDocument]:
    return create_knowledge_base(
        root_url_tags_mapping=root_url_tags_mapping,
        limit=limit,
        exclude_patterns=exclude_patterns,
    ).with_overrides(cache=False)


@union.workflow
def create_eval_dataset(
    documents: list[CustomDocument],
    n_questions_per_doc: int = 1,
    n_answers_per_question: int = 5,
) -> pd.DataFrame:
    qa_dataset = create_qa_dataset(
        documents=documents,
        n_questions_per_doc=n_questions_per_doc,
        n_answers_per_question=n_answers_per_question,
    )
    return create_llm_filtered_dataset(dataset=qa_dataset)


knowledge_base_lp = union.LaunchPlan.get_or_create(
    knowledge_base_workflow,
    name="knowledge_base_lp",
    default_inputs={"limit": 10},
    schedule=union.FixedRate(duration=timedelta(minutes=3))
)

create_eval_dataset_lp = union.LaunchPlan.get_or_create(
    create_eval_dataset,
    name="create_eval_dataset_lp",
    trigger=OnArtifact(
        trigger_on=KnowledgeBase,
        inputs={"documents": KnowledgeBase.query()},
    )
)

optimize_rag_lp = union.LaunchPlan.get_or_create(
    optimize_rag,
    name="optimize_rag_lp",
    default_inputs={
        "gridsearch_config": GridSearchConfig(
            embedding_model=[
                "text-embedding-ada-002",
                "text-embedding-3-small",
                "text-embedding-3-large",
            ],
            chunk_size=[256],
            splitter=["recursive"],
        ),
    },
    trigger=OnArtifact(
        trigger_on=EvalDatasetArtifact,
        inputs={"eval_dataset": EvalDatasetArtifact.query()},
    )
)


GridSearchConfig(
    embedding_model=[
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "text-embedding-3-large",
    ],
    chunk_size=[256],
    splitter=["recursive"],
)
