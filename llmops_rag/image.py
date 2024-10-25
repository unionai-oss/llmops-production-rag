from flytekit import ImageSpec


image = ImageSpec(
    apt_packages=["git", "wget"],
    requirements="requirements.lock.txt",
    env={"GIT_PYTHON_REFRESH": "quiet"},
)
