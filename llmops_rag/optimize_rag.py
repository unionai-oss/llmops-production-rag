"""Evaluate a RAG workflow."""

import itertools
from dataclasses import dataclass, asdict
from typing import Annotated, Optional

import pandas as pd
import flytekit as fk
from flytekit.deck import TopFrameRenderer
from flytekit.types.directory import FlyteDirectory

from llmops_rag.config import RAGConfig
from llmops_rag.image import image
from llmops_rag.vector_store import create_vector_store
from llmops_rag.rag_basic import rag_basic
from llmops_rag.utils import openai_env_secret, convert_fig_into_html


EvalDatasetArtifact = fk.Artifact(name="eval-dataset", partition_keys=["dataset_type"])


@dataclass
class HPOConfig(RAGConfig):
    condition_name: str = ""


@dataclass
class GridSearchConfig:
    splitter: Optional[list[str]] = None
    exclude_patterns: Optional[list[str]] = None
    prompt_template: Optional[list[str]] = None
    chunk_size: Optional[list[int]] = None
    limit: Optional[list[int]] = None
    embedding_model: Optional[list[str]] = None
    generation_model: Optional[list[str]] = None
    search_type: Optional[list[str]] = None
    rerank: Optional[list[bool]] = None
    num_retrieved_docs: Optional[list[int]] = None
    num_docs_final: Optional[list[int]] = None


@dataclass
class Question:
    question_id: int
    question: str
    reference_answer: str
    is_user_generated: bool


@dataclass
class RAGInput:
    question: Question
    vector_store: FlyteDirectory
    prompt_template: str


@dataclass
class Answer:
    answer: str
    question_id: int


@fk.task(
    container_image=image,
    cache=True,
    cache_version="1",
)
def prepare_hpo_configs(gridsearch_config: GridSearchConfig) -> list[HPOConfig]:
    gridsearch_config = asdict(gridsearch_config)
    gridsearch_config = {k: v for k, v in gridsearch_config.items() if v is not None}
    keys = gridsearch_config.keys()

    hpo_configs = []
    for values in itertools.product(*gridsearch_config.values()):
        name = []
        config = {}
        for key, value in zip(keys, values):
            config[key] = value
            _key = "_".join(x[:3] for x in key.split("_"))
            if len(str(value)) > 10:
                value = str(value)[:10]
                value = value.replace(" ", "-")
            name.append(f"{_key}={value}")
        name = ":".join(name)
        hpo_configs.append(HPOConfig(condition_name=name, **config))

    return hpo_configs


@fk.task(
    container_image=image,
    cache=True,
    cache_version="1",
)
def prepare_questions(dataset: pd.DataFrame, n_answers: int) -> list[Question]:
    questions = (
        dataset.loc[dataset.index.repeat(n_answers)][
            ["question_id", "question", "reference_answer", "is_user_generated"]
        ]
        .astype({"question_id": int})
        .to_dict(orient="records")
    )
    return [Question(**record) for record in questions]


@fk.task(
    container_image=image,
    cache=True,
    cache_version="1",
)
def prepare_answers(answers: list[str], questions: list[Question]) -> list[Answer]:
    return [
        Answer(
            answer=answer,
            question_id=question.question_id,
        )
        for answer, question in zip(answers, questions)
    ]


@fk.dynamic(container_image=image, cache=True, cache_version="5")
def generate_answers(
    questions: list[Question],
    root_url_tags_mapping: Optional[dict] = None,
    exclude_patterns: Optional[list[str]] = None,
    splitter: str = "character",
    chunk_size: int = 2048,
    prompt_template: str = "",
    limit: Optional[int] = None,
    embedding_model: Optional[str] = None,
    generation_model: Optional[str] = None,
    search_type: str = "similarity",
    rerank: bool = False,
    num_retrieved_docs: int = 20,
    num_docs_final: int = 5,
) -> list[Answer]:
    vector_store = create_vector_store(
        root_url_tags_mapping=root_url_tags_mapping,
        splitter=splitter,
        chunk_size=chunk_size,
        limit=limit,
        embedding_model=embedding_model,
        exclude_patterns=exclude_patterns,
    )
    answers = rag_basic(
        questions=[question.question for question in questions],
        vector_store=vector_store,
        embedding_model=embedding_model,
        generation_model=generation_model,
        prompt_template=prompt_template,
        search_type=search_type,
        rerank=rerank,
        num_retrieved_docs=num_retrieved_docs,
        num_docs_final=num_docs_final,
    )
    return prepare_answers(answers, questions)


@fk.dynamic(container_image=image, cache=True, cache_version="5")
def gridsearch(
    questions: list[Question],
    hpo_configs: list[HPOConfig],
    root_url_tags_mapping: Optional[dict] = None,
    exclude_patterns: Optional[list[str]] = None,
    limit: Optional[int] = None,
) -> list[list[Answer]]:
    answers = []
    for config in hpo_configs:
        _answers = generate_answers(
            questions=questions,
            root_url_tags_mapping=root_url_tags_mapping,
            exclude_patterns=exclude_patterns,
            splitter=config.splitter,
            chunk_size=config.chunk_size,
            prompt_template=config.prompt_template,
            limit=limit,
            embedding_model=config.embedding_model,
            generation_model=config.generation_model,
            rerank=config.rerank,
        )
        answers.append(_answers)
    return answers


@fk.task(
    container_image=image,
    cache=True,
    cache_version="1",
    enable_deck=True,
)
def combine_answers(
    answers: list[list[Answer]],
    eval_configs: list[HPOConfig],
    questions: list[Question],
) -> Annotated[pd.DataFrame, TopFrameRenderer(10)]:
    # TODO: concatenate all answers into a single dataframe
    combined_answers = []
    for _answers, config in zip(answers, eval_configs):
        for answer, question in zip(_answers, questions):
            assert answer.question_id == question.question_id
            combined_answers.append(
                {
                    "question_id": question.question_id,
                    "question": question.question,
                    "answer": answer.answer,
                    "reference_answer": question.reference_answer,
                    "is_user_generated": question.is_user_generated,
                    **asdict(config),
                }
            )

    return pd.DataFrame(combined_answers)


def traditional_nlp_eval(answers_dataset: pd.DataFrame) -> pd.DataFrame:
    from nltk.translate.bleu_score import sentence_bleu
    from rouge_score import rouge_scorer

    bleu_scores, rouge1_scores = [], []
    rouge_scorer = rouge_scorer.RougeScorer(["rouge1"])
    for row in answers_dataset.itertuples():
        bleu_scores.append(
            sentence_bleu([row.reference_answer.split()], row.answer.split())
        )
        _rouge_scores = rouge_scorer.score(row.reference_answer, row.answer)
        rouge1_scores.append(_rouge_scores["rouge1"].fmeasure)

    return answers_dataset.assign(
        bleu_score=bleu_scores,
        rouge1_f1=rouge1_scores,
    )


DEFAULT_EVAL_PROMPT_TEMPLATE = """### Task Description:
You are an expert in judging the correctness of answers relating
to pandas, a python data analysis library. Given a question and a
reference answer, determine if the candidate answer is equivalent or
better than the reference answer in terms of correctness.

### Question:
{question}

### Reference Answer:
{reference_answer}

### Candidate Answer:
{candidate_answer}

### Judgement:
Is the candidate answer equivalent or better than the reference answer
in terms of correctness? You MUST answer "Yes" or "No".
"""


def llm_judge_eval(
    answers_dataset: pd.DataFrame, eval_prompt_template: Optional[str] = None
) -> pd.DataFrame:
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model_name="gpt-4o", temperature=0.9)
    prompt = PromptTemplate.from_template(
        eval_prompt_template or DEFAULT_EVAL_PROMPT_TEMPLATE
    )

    llm_correctness_scores = []

    for _, row in answers_dataset.iterrows():
        chain = prompt | model
        result = chain.invoke(
            {
                "question": row["question"],
                "reference_answer": row["reference_answer"],
                "candidate_answer": row["answer"],
            }
        )

        result = result.content.lower().strip().strip(".").strip("'").strip('"')
        if result not in ["yes", "no"]:
            score = 0.0
        elif result == "yes":
            score = 1.0
        else:
            score = 0.0
        llm_correctness_scores.append(score)

    return answers_dataset.assign(llm_correctness_score=llm_correctness_scores)


@fk.task(
    container_image=image,
    secret_requests=[fk.Secret(key="openai_api_key")],
    requests=fk.Resources(cpu="4", mem="8Gi"),
    cache=True,
    cache_version="7",
)
@openai_env_secret
def evaluate(
    answers_dataset: pd.DataFrame,
    metric: Optional[str] = None,
    eval_prompt_template: Optional[str] = None,
) -> tuple[RAGConfig, pd.DataFrame, pd.DataFrame]:

    evaluation = traditional_nlp_eval(answers_dataset)
    evaluation = llm_judge_eval(evaluation, eval_prompt_template)

    metric = metric or "llm_correctness_score"
    evaluation_summary = (
        evaluation
        .groupby([*HPOConfig.__dataclass_fields__])[
            ["bleu_score", "rouge1_f1", "llm_correctness_score"]
        ]
        .mean()
        .reset_index()
    )

    sorted_df = evaluation_summary.sort_values(by=metric, ascending=False)
    best_config = sorted_df[[*RAGConfig.__dataclass_fields__]].iloc[0].to_dict()
    return RAGConfig(**best_config), evaluation, evaluation_summary


@fk.task(
    container_image=image,
    enable_deck=True,
    deck_fields=[],
    requests=fk.Resources(cpu="4", mem="8Gi"),
)
def report(evaluation: pd.DataFrame, evaluation_summary: pd.DataFrame):
    import seaborn as sns

    analysis_df = evaluation_summary.melt(
        id_vars=["condition_name"],
        value_vars=["bleu_score", "rouge1_f1", "llm_correctness_score"],
        var_name="metric",
        value_name="score",
    )

    g = sns.FacetGrid(analysis_df, col="metric", sharex=False)
    g.map_dataframe(sns.barplot, y="condition_name", x="score", orient="h")
    g.add_legend()

    decks = fk.current_context().decks
    decks.insert(0, fk.Deck("Evaluation", TopFrameRenderer(10).to_html(evaluation)))
    decks.insert(0, fk.Deck("Evaluation Summary", TopFrameRenderer(10).to_html(evaluation_summary)))
    decks.insert(0, fk.Deck("Benchmarking Results", convert_fig_into_html(g.figure)))


@fk.workflow
def optimize_rag(
    gridsearch_config: GridSearchConfig,
    root_url_tags_mapping: Optional[dict] = None,
    exclude_patterns: Optional[list[str]] = None,
    limit: Optional[int] = 10,
    eval_dataset: Annotated[pd.DataFrame, EvalDatasetArtifact] = EvalDatasetArtifact.query(dataset_type="llm_filtered"),
    eval_prompt_template: Optional[str] = None,
    n_answers: int = 5,
) -> RAGConfig:
    hpo_configs = prepare_hpo_configs(gridsearch_config)
    questions = prepare_questions(eval_dataset, n_answers)
    answers = gridsearch(questions, hpo_configs, root_url_tags_mapping, exclude_patterns, limit)
    answers_dataset = combine_answers(answers, hpo_configs, questions)
    best_config, evaluation, evalution_summary = evaluate(
        answers_dataset, eval_prompt_template
    )
    report(evaluation, evalution_summary)
    return best_config
