from flytekit import ImageSpec


image = ImageSpec(
    apt_packages=["git", "wget"],
    requirements="requirements.txt",
    env={"GIT_PYTHON_REFRESH": "quiet"},
    builder="union",
)
