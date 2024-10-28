"""Use LLM critic to filter Q&A dataset for quality."""

from dataclasses import dataclass

import flytekit as fk
import pandas as pd

from flytekit.types.file import FlyteFile

from llmops_rag.image import image
from llmops_rag.create_qa_dataset import QuestionAndAnswers


QuestionAndAnswerDataset = fk.Artifact(name="question_and_answer_dataset")


@dataclass
class QualityScore:
    groundedness: int
    relevance: int
    standalone: int
    correctness_per_answer: list[int]


@dataclass
class ReferenceAnswer:
    question_id: int
    question: str
    reference_answer: str
    is_user_generated: bool


@fk.task(
    container_image=image,
    requests=fk.Resources(cpu="2", mem="4Gi"),
)
def llm_critic(qa_datapoint: QuestionAndAnswers) -> QualityScore:
    """Returns critic scores per question-answer pair."""
    # implement prompts for each quality score filed


@fk.dynamic(container_image=image)
def apply_llm_critic(dataset: FlyteFile) -> tuple[list[QuestionAndAnswers], list[QualityScore]]:
    """Applies the LLM critic to a dataset of question-answer pairs."""
    # load dataset
    # map question-answer pairs over llm_critic


@fk.task(
    container_image=image,
    requests=fk.Resources(cpu="2", mem="4Gi"),
)
def filter_dataset(qa_dataset: list[QuestionAndAnswers], scores: list[QualityScore]) -> list[ReferenceAnswer]:
    """Filters out low-quality question-answer pairs."""
    # filter out poor scoring question-answer pairs
    # for qa datapoints with more than one answer: LLM to choose best answer
    # return list of ReferenceAnswer


@fk.task(
    container_image=image,
    requests=fk.Resources(cpu="2", mem="4Gi"),
)
def prepare_dataset(reference_answers: list[ReferenceAnswer]) -> pd.DataFrame:
    """Prepare and visualize dataset."""


@fk.workflow
def create_llm_filtered_dataset(
    dataset: FlyteFile = QuestionAndAnswerDataset.query(),
) -> pd.DataFrame:
    qa_dataset, scores = apply_llm_critic(dataset)
    reference_answers = filter_dataset(qa_dataset, scores)
    return prepare_dataset(reference_answers)
