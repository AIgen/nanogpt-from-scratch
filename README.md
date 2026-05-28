# nanogpt-from-scratch

A character-level GPT built piece by piece in PyTorch, as a learning exercise.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick check

```bash
python nanogpt.py
```

Builds a tiny model, runs one forward pass, asserts the output shape.

## Data

`input.txt` is the Tiny Shakespeare corpus (~1 MB, public domain), pre-downloaded. 