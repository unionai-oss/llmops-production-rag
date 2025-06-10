from union import ImageSpec


image = ImageSpec(
    name="llmops-rag",
    apt_packages=["git", "wget"],
    requirements="requirements.txt",
    env={"GIT_PYTHON_REFRESH": "quiet"},
    builder="union",
    copy=["llmops_rag"],
)
