# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a personal learning repository following the [HuggingFace LLM Course](https://huggingface.co/learn/llm-course). All content consists of Jupyter notebooks.

## Structure

Each `chapter-0{N}/` directory contains notebooks named `section{N}.ipynb` with framework-specific variants:
- `section{N}_pt.ipynb` — PyTorch implementation
- `section{N}_tf.ipynb` — TensorFlow implementation
- `section{N}.ipynb` — framework-agnostic content

## Chapter Overview

| Chapter | Topic |
|---------|-------|
| 1 | Transformer pipelines (sentiment analysis, NER, summarization, translation) |
| 2 | Using models and tokenizers (AutoModel, AutoTokenizer) |
| 3 | Fine-tuning pretrained models with Trainer API |
| 4 | Sharing models and datasets on HuggingFace Hub |
| 5 | HuggingFace Datasets library |
| 6 | Tokenizers in depth (BPE, WordPiece, training custom tokenizers) |
| 7 | Main NLP tasks (token classification, masked LM, translation, summarization, causal LM) |
| 8 | Debugging and asking for help |
| 9 | Building demos with Gradio |

## Running Notebooks

Notebooks are designed to run in Google Colab or a local Jupyter environment. Each notebook starts with an install cell:

```bash
pip install datasets evaluate transformers[sentencepiece]
# Some chapters also need:
pip install accelerate
```

To run locally:
```bash
jupyter notebook
# or
jupyter lab
```

## Communication Language

所有对话使用中文，包括解释、回复和说明。以下内容保留英文：
- 专业术语（如 attention、tokenizer、fine-tuning、embedding 等）
- 代码、命令、文件名、变量名、API 名称

## Workflow Rules

- **修改后必须展示 diff**：每次修改文件后，用 Python 对比前后版本并以可读格式展示变更内容。对于 `.ipynb` 文件，`git diff` 输出的是原始 JSON，无法直观看出代码变化，必须用 `difflib` 提取 cell 源码后再做对比展示。
- **不主动 commit/push**：不要在修改完文件后主动执行 `git commit` 或 `git push`，等待用户明确要求再操作。

## Key Patterns in Notebooks

- Chapter 7 notebooks that push to Hub require `notebook_login()` and git-lfs setup
- PyTorch notebooks use `Trainer` API or manual training loops with `accelerate`
- TensorFlow notebooks use `model.fit()` with Keras
- Notebooks include pre-computed cell outputs so they can be read without re-running
