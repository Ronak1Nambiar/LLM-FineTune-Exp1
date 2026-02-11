# LLM-FineTune-Exp1

A clean starter project for local supervised fine-tuning (SFT) with **Unsloth + LoRA**.

## What this project provides

- A production-friendly training entrypoint: `scripts/train.py`
- A compatibility checker: `scripts/check_env.py`
- Example training config: `configs/train.example.yaml`
- Tiny sample dataset format: `examples/train.sample.jsonl`

## 1) Environment setup (Fish + CachyOS friendly)

```fish
sudo pacman -Syu
sudo pacman -S --needed pyenv base-devel openssl xz tk
```

Add to `~/.config/fish/config.fish`:

```fish
set -Ux PYENV_ROOT $HOME/.pyenv
fish_add_path $PYENV_ROOT/bin
status --is-interactive; and pyenv init - | source
```

Restart Fish, then:

```fish
pyenv install 3.11.11
pyenv local 3.11.11
python -m venv .venv
source .venv/bin/activate.fish
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

## 2) Verify your setup

```fish
python scripts/check_env.py
```

## 3) Prepare config + dataset

```fish
cp configs/train.example.yaml configs/train.yaml
```

Edit `configs/train.yaml`:

- `model_name`: Hugging Face repo id (or full local model folder containing `config.json`)
- `dataset_path`: path to JSONL with `{"text": "..."}` rows
- `output_dir`: where adapters/checkpoints should be saved

You can test shape using the provided sample file:

```fish
cp examples/train.sample.jsonl data/train.jsonl
```

## 4) Run training

```fish
python scripts/train.py --config configs/train.yaml
```

## Common errors and fixes

### `ImportError: cannot import name 'LoraConfig' from 'unsloth'`
Use:

```python
from unsloth import FastLanguageModel
from peft import LoraConfig
```

### `No config file found - are you sure the model_name is correct?`
Your `model_name` is wrong or local path is not a full HF model folder. Ensure `config.json` exists.

### Python 3.14 issues
Use Python **3.11**. Unsloth and related packages are often unstable on very new Python versions.

## Notes

- This repo focuses on LoRA adapter training (not full-parameter fine-tuning).
- Keep secrets/token auth out of git.
