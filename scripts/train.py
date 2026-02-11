#!/usr/bin/env python3
"""Minimal, robust Unsloth fine-tuning entrypoint.

Expected dataset format (JSONL):
{"text": "..."}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel


REQUIRED_CONFIG_KEYS = {
    "model_name",
    "dataset_path",
    "output_dir",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune an instruction model with Unsloth + LoRA")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Path to YAML config file (default: configs/train.yaml)",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing config file at {path}. Copy configs/train.example.yaml to configs/train.yaml first."
        )
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping/object.")

    missing = REQUIRED_CONFIG_KEYS - set(config.keys())
    if missing:
        raise ValueError(f"Config missing required keys: {sorted(missing)}")
    return config


def _is_local_path(model_name: str) -> bool:
    return model_name.startswith("/") or model_name.startswith("./") or model_name.startswith("../")


def validate_model_name(model_name: str) -> None:
    if _is_local_path(model_name):
        local = Path(model_name)
        if not local.exists():
            raise FileNotFoundError(f"Local model path does not exist: {local}")
        if not (local / "config.json").exists():
            raise FileNotFoundError(
                f"No config.json found under {local}. This must be a full Hugging Face model directory."
            )
        return

    # Remote repo id check for clearer errors.
    try:
        HfApi().model_info(model_name)
    except Exception as exc:  # noqa: BLE001 - we want to surface any hub-side issue clearly.
        raise RuntimeError(
            "Unable to resolve model on Hugging Face Hub. Check model_name spelling and auth access. "
            f"model_name={model_name!r}"
        ) from exc


def preview_dataset(dataset_path: str, max_preview: int = 2) -> None:
    data_file = Path(dataset_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with data_file.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= max_preview:
                break
            row = json.loads(line)
            if "text" not in row:
                raise ValueError("Dataset rows must contain a 'text' key.")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model_name = str(config["model_name"])
    dataset_path = str(config["dataset_path"])
    output_dir = str(config["output_dir"])

    validate_model_name(model_name)
    preview_dataset(dataset_path)

    max_seq_length = int(config.get("max_seq_length", 2048))
    load_in_4bit = bool(config.get("load_in_4bit", True))
    use_gradient_checkpointing = config.get("use_gradient_checkpointing", "unsloth")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    lora_config = LoraConfig(
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 16)),
        lora_dropout=float(config.get("lora_dropout", 0.0)),
        bias=str(config.get("bias", "none")),
        target_modules=list(
            config.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        ),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config.r,
        target_modules=lora_config.target_modules,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=int(config.get("seed", 42)),
    )

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    training_args = TrainingArguments(
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 2)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 4)),
        warmup_steps=int(config.get("warmup_steps", 5)),
        max_steps=int(config.get("max_steps", 60)),
        learning_rate=float(config.get("learning_rate", 2e-4)),
        logging_steps=int(config.get("logging_steps", 1)),
        optim=str(config.get("optim", "adamw_8bit")),
        weight_decay=float(config.get("weight_decay", 0.01)),
        lr_scheduler_type=str(config.get("lr_scheduler_type", "linear")),
        seed=int(config.get("seed", 42)),
        output_dir=output_dir,
        save_steps=int(config.get("save_steps", 30)),
        save_total_limit=int(config.get("save_total_limit", 2)),
        fp16=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete. Adapter + tokenizer saved to {output_dir}")


if __name__ == "__main__":
    main()
