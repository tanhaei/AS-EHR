"""Optional LoRA supervised fine-tuning for a real MedGemma checkpoint."""
from __future__ import annotations

import json
from pathlib import Path

from .config import deep_get, validate_for_training
from .prompting import SYSTEM_PROMPT, build_user_prompt
from .schemas import load_records


def train_lora(config: dict) -> None:  # pragma: no cover - requires large model/GPU
    validate_for_training(config)
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("install the 'model' optional dependencies") from exc

    model_id = deep_get(config, "model.model_id")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=getattr(torch, deep_get(config, "model.torch_dtype", "bfloat16")),
        device_map=deep_get(config, "model.device_map", "auto"),
    )
    lora = LoraConfig(
        r=int(deep_get(config, "lora.rank")),
        lora_alpha=int(deep_get(config, "lora.alpha")),
        lora_dropout=float(deep_get(config, "lora.dropout", 0.0)),
        target_modules=list(deep_get(config, "lora.target_modules")),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.config.use_cache = False

    max_length = int(deep_get(config, "training.max_length", 8192))

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, path: str):
            self.records = load_records(path)
            if any(not record.reference_summary for record in self.records):
                raise ValueError("all training records require reference_summary")

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            record = self.records[index]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(record, record.features)},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            target = json.dumps(
                {"summary": record.reference_summary, "included_feature_ids": record.required_feature_ids},
                ensure_ascii=False,
                sort_keys=True,
            )
            target_ids = tokenizer(target + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
            input_ids = (prompt_ids + target_ids)[:max_length]
            prompt_length = min(len(prompt_ids), len(input_ids))
            labels = [-100] * prompt_length + input_ids[prompt_length:]
            return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}

    def collate(rows):
        width = max(len(row["input_ids"]) for row in rows)
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in rows:
            pad = width - len(row["input_ids"])
            result["input_ids"].append(row["input_ids"] + [tokenizer.pad_token_id] * pad)
            result["attention_mask"].append(row["attention_mask"] + [0] * pad)
            result["labels"].append(row["labels"] + [-100] * pad)
        return {key: torch.tensor(value) for key, value in result.items()}

    arguments = TrainingArguments(
        output_dir=deep_get(config, "training.output_dir"),
        num_train_epochs=float(deep_get(config, "training.epochs")),
        learning_rate=float(deep_get(config, "training.learning_rate")),
        per_device_train_batch_size=int(deep_get(config, "training.batch_size")),
        per_device_eval_batch_size=int(deep_get(config, "training.eval_batch_size", 1)),
        gradient_accumulation_steps=int(deep_get(config, "training.gradient_accumulation_steps")),
        warmup_ratio=float(deep_get(config, "training.warmup_ratio", 0.0)),
        weight_decay=float(deep_get(config, "training.weight_decay", 0.0)),
        logging_steps=int(deep_get(config, "training.logging_steps", 10)),
        save_strategy="epoch",
        evaluation_strategy="epoch",
        bf16=bool(deep_get(config, "training.bf16", True)),
        seed=int(deep_get(config, "training.seed", 2026)),
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=Dataset(deep_get(config, "data.train_jsonl")),
        eval_dataset=Dataset(deep_get(config, "data.validation_jsonl")),
        data_collator=collate,
    )
    trainer.train()
    output = Path(deep_get(config, "training.output_dir")) / "final_adapter"
    trainer.save_model(output)
    tokenizer.save_pretrained(output)
