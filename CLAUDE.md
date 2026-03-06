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

## Key Patterns in Notebooks

- Chapter 7 notebooks that push to Hub require `notebook_login()` and git-lfs setup
- PyTorch notebooks use `Trainer` API or manual training loops with `accelerate`
- TensorFlow notebooks use `model.fit()` with Keras
- Notebooks include pre-computed cell outputs so they can be read without re-running
