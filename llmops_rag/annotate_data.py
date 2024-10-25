import json
import random
from datetime import timedelta

from flytekit import workflow, Resources, Artifact, wait_for_input
from flytekit.types.file import FlyteFile
from union.actor import ActorEnvironment


from llmops_rag.image import image

DEFAULT_ANNOTATION_SET_NAME = "default"

QuestionAndAnswerDataset = Artifact(name="question_and_answer_dataset")
AnnotatedDataset = Artifact(name="annotated_dataset", partition_keys=["name"])


actor = ActorEnvironment(
    name="data-annotation-sampler",
    ttl_seconds=720,
    container_image=image,
    requests=Resources(cpu="2", mem="4Gi"),
)


@actor.task(
    cache=True,
    cache_version="1",
    timeout=timedelta(minutes=30),
)
def sample_triple(random_seed: int, n_samples: int, dataset: FlyteFile) -> list[dict]:
    dataset.download()
    with open(dataset.path, "r") as f:
        qa_triples_list = json.load(f)

    assert (
        len(qa_triples_list) >= n_samples
    ), f"Not enough samples in the dataset. Required: {n_samples}, Available: {len(qa_triples_list)}"

    random.seed(random_seed)
    sampled_qa = random.sample(qa_triples_list, k=n_samples)

    return sampled_qa


@workflow
def create_annotation_set(
    random_seed: int,
    n_samples: int = 10,
    dataset: FlyteFile = QuestionAndAnswerDataset.query(),
) -> str:
    triple_sample = sample_triple(
        random_seed=random_seed, n_samples=n_samples, dataset=dataset
    )
    feedback = wait_for_input("feedback", timeout=timedelta(hours=1), expected_type=str)
    triple_sample >> feedback
    return feedback
