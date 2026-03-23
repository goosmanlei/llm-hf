# Repository Guidelines

## Project Structure & Module Organization
This repository is a notebook-first learning workspace for the HuggingFace LLM Course. Content is organized by chapter directories: `chapter-01/` through `chapter-13/`.

- Main learning artifacts: `chapter-*/section-*.ipynb`
- Optional variants: `section-*_pt.ipynb`, `section-*_tf.ipynb`, and `*.summary.ipynb`
- Chapter assets/examples: e.g., `chapter-05/github-issues-classifier/`, `chapter-06/code-search-net-tokenizer/`
- Project-level guidance: `CLAUDE.md`

There is no central `src/` package; notebooks are the primary unit of work.

## Build, Test, and Development Commands
Use a local virtual environment and run notebooks via Jupyter.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install datasets evaluate "transformers[sentencepiece]" accelerate
jupyter lab
```

Common extras by chapter:
- Demos: `python -m pip install gradio`
- Evaluation tasks: `python -m pip install seqeval sacrebleu rouge_score nltk`

## Coding Style & Naming Conventions
- Keep notebook names consistent with existing patterns: `section-NN.ipynb` and optional suffixes (`.summary`, `.exercise`, `.from-video-01`).
- Prefer small, focused cell edits instead of large notebook rewrites.
- Preserve existing markdown explanation style and output-rich notebooks when possible.
- Follow `CLAUDE.md`: communicate in Chinese for collaboration text; keep technical terms/code in English.

## Testing Guidelines
There is no unified automated test suite yet. Validate changes by:
- Running edited notebook cells end-to-end (or affected sections only for heavy training notebooks).
- Verifying imports, dataset loading, and metric computation cells.
- For training chapters, confirm at least one successful forward/eval path before submitting.

## Commit & Pull Request Guidelines
Recent history favors concise, imperative commit messages, often with scope prefixes, e.g.:
- `fix(ch12): align GRPO length reward with token count`
- `chapter-12: add GRPOConfig use_vllm compatible option`

PRs should include:
- What changed and why (chapter/section specific)
- Any environment or dependency requirements
- Screenshots/output snippets for UI/demo notebook changes (e.g., Gradio)
- Notes on large model/GPU assumptions when relevant
