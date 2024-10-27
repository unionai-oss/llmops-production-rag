"""Create an evaluation dataset based on human annotations."""

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd
from collections import defaultdict
from itertools import groupby
from typing import Annotated

from flytekit import task, workflow, Artifact, ImageSpec, Secret, current_context, Deck
from flytekit.deck import TopFrameRenderer
from flytekit.models.filters import ValueIn
from flytekit.remote.remote import MOST_RECENT_FIRST
from union.remote import UnionRemote


image = ImageSpec(packages=["pandas", "pyarrow"])


EvalDatasetArtifact = Artifact(name="test-eval-dataset")


@dataclass
class Annotation:
    question_id: int
    question: str
    answers: list[str]
    label: str
    correct_answer_text: Optional[str]


@dataclass
class RawRanking:
    id: int
    question: str
    answer: str
    elo_rating: float


@dataclass
class ReferenceAnswer:
    question_id: int
    question: str
    reference_answer: str
    is_user_generated: bool


def create_comparison_data(
    annotations: list[Annotation],
) -> tuple[list[Annotation], list[ReferenceAnswer]]:
    """
    Create a list of comparisons between annotations.
    """
    comparisons = []
    user_reference_answers = []
    for annotation in annotations:
        if annotation.label == "question_incoherent":
            continue
        if annotation.correct_answer_text:
            user_reference_answers.append(
                ReferenceAnswer(
                    question_id=annotation.question_id,
                    question=annotation.question,
                    reference_answer=annotation.correct_answer_text,
                    is_user_generated=True,
                )
            )
        comparisons.append(annotation)
    return comparisons, user_reference_answers


def calculate_expected_score(rating_a: float, rating_b: float) -> float:
    """
    Calculate the expected score of item A against item B.
    """
    expected_score_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    return expected_score_a


def update_ratings(
    rating_a: float, rating_b: float, score_a: float, K: int = 32
) -> tuple[float, float]:
    """
    Update ratings for item A and item B after a comparison.
    """
    expected_score_a = calculate_expected_score(rating_a, rating_b)
    expected_score_b = 1 - expected_score_a  # Since expected scores sum to 1

    new_rating_a = rating_a + K * (score_a - expected_score_a)
    new_rating_b = rating_b + K * ((1 - score_a) - expected_score_b)

    return new_rating_a, new_rating_b


def calculate_elo_rankings(
    comparisons: list[Annotation], K: int = 32
) -> tuple[defaultdict, dict]:
    """
    Calculate the ELO rankings for the comparisons.

    K-factor, can be adjusted based on the volatility desired
    """
    ratings = defaultdict(float)
    answer_to_question = {}

    # initialize ratings for all answers
    for annotation in comparisons:
        for answer in annotation.answers:
            if answer not in ratings:
                ratings[answer] = 1500.0
            if answer not in answer_to_question:
                answer_to_question[answer] = annotation.question_id, annotation.question

    # update the ratings based on the comparisons
    for annotation in comparisons:
        score = 0.0
        if annotation.label in ("both", "neither"):
            score = 0.5
        elif annotation.label == "answer_1":
            score = 1.0
        elif annotation.label == "answer_2":
            score = 0.0

        answer_1, answer_2 = annotation.answers[0], annotation.answers[1]
        rating_1, rating_2 = ratings[answer_1], ratings[answer_2]
        new_rating_1, new_rating_2 = update_ratings(rating_1, rating_2, score, K=K)
        ratings[answer_1] = new_rating_1
        ratings[answer_2] = new_rating_2

    return ratings, answer_to_question


def rank_annotations_per_question(
    annotations: list[Annotation],
) -> tuple[list[ReferenceAnswer], Optional[list[RawRanking]]]:
    comparisons, user_reference_answers = create_comparison_data(annotations)
    if not comparisons:
        return user_reference_answers, None

    rankings, answer_to_question_id = calculate_elo_rankings(comparisons)

    annotated_reference_answers = []
    raw_rankings = []
    sorted_rankings = sorted(rankings.items(), key=lambda x: x[1], reverse=True)

    # collect raw rankings
    for answer, rating in sorted_rankings:
        question_id, question = answer_to_question_id[answer]
        raw_rankings.append(
            RawRanking(
                id=question_id,
                question=question,
                answer=answer,
                elo_rating=rating,
            )
        )

    # collect best answer for this question
    best_answer = sorted_rankings[0][0]
    question_id, question = answer_to_question_id[best_answer]
    annotated_reference_answers.append(
        ReferenceAnswer(
            question_id=question_id,
            question=question,
            reference_answer=best_answer,
            is_user_generated=False,
        )
    )

    return [*annotated_reference_answers, *user_reference_answers], raw_rankings


LABEL_KEY = "union_annotator"
LABEL_VALUE = "llmops_workshop_v0"


@task(
    container_image=image,
    secret_requests=[Secret(key="helpabot_app_key")],
)
def collect_annotations() -> list[Annotation]:
    os.environ["UNION_API_KEY"] = current_context().secrets.get(key="helpabot_app_key")
    remote = UnionRemote(default_project="flytesnacks", default_domain="development")
    token = None
    _executions = []
    while True:
        executions, token = remote.client.list_executions_paginated(
            project="flytesnacks",
            domain="development",
            limit=2,
            filters=[
                ValueIn("execution_tag.key", [LABEL_KEY]),
                ValueIn("execution_tag.value", [LABEL_VALUE]),
                ValueIn("phase", ["SUCCEEDED"]),
            ],
            sort_by=MOST_RECENT_FIRST,
            token=token,
        )
        _executions.extend(executions)
        if not token:
            break

    annotation_data = []
    for ex in _executions:
        execution = remote.fetch_execution(name=ex.id.name)
        output, *_ = execution.outputs.values()
        data = json.loads(output)
        for _, row in data.items():
            annotation_data.append(
                Annotation(
                    question_id=int(row["question_id"]),
                    question=row["question"],
                    answers=row["answers"],
                    label=row["label"],
                    correct_answer_text=row["correct_answer_text"],
                )
            )

    return annotation_data


@task(container_image=image, enable_deck=True)
def create_dataset(
    annotations: list[Annotation],
    min_annotations_per_question: int,
) -> tuple[Annotated[pd.DataFrame, EvalDatasetArtifact], Optional[pd.DataFrame]]:
    sorted_annotations = sorted(annotations, key=lambda x: x.question_id)

    all_reference_answers = []
    all_raw_rankings = []

    for _, annotations_by_question in groupby(
        sorted_annotations, key=lambda x: x.question_id
    ):
        annotations_by_question = list(annotations_by_question)
        if len(annotations_by_question) < min_annotations_per_question:
            print(
                f"Skipping question {annotations_by_question[0].question_id} "
                f"because it has less than {min_annotations_per_question} annotations"
            )
            continue
        reference_answers, raw_rankings = rank_annotations_per_question(
            annotations_by_question
        )
        all_reference_answers.extend(reference_answers)
        if raw_rankings is not None:
            all_raw_rankings.extend(raw_rankings)

    if all_raw_rankings is not None:
        all_raw_rankings = pd.DataFrame(all_raw_rankings)

    reference_answers_df = pd.DataFrame([asdict(a) for a in all_reference_answers])
    current_context().decks.insert(
        0, Deck("Eval Dataset", TopFrameRenderer(10).to_html(reference_answers_df))
    )
    return reference_answers_df, all_raw_rankings


@workflow
def create(min_annotations_per_question: int = 1) -> pd.DataFrame:
    annotations = collect_annotations()
    out, _ = create_dataset(annotations, min_annotations_per_question)
    return out
