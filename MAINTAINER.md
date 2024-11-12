# Maintainer Guide

This document contains instructions for running the workflows in the `llmops_rag` module.

Create the vector store:

```bash
union run --remote llmops_rag/vector_store.py create_vector_store
```

Run the RAG workflow:

```bash
union run --remote llmops_rag/rag_basic.py rag_basic --questions '["How do I read and write a pandas dataframe to csv format?"]'
```

Create the QA dataset:

```bash
union run --remote llmops_rag/create_qa_dataset.py create_qa_dataset --n_questions_per_doc 5 --n_answers_per_question 5
```

Filter the dataset with an LLM critic:

```bash
union run --remote llmops_rag/create_llm_filtered_dataset.py create_llm_filtered_dataset
```

Run a RAG experiment:

```bash
union run --remote llmops_rag/optimize_rag.py optimize_rag --gridsearch_config config/embedding_model_experiment.yaml
```
