from pathlib import Path
import mlflow
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from repo_architect.config import settings


def train_lora(data_path: str = "data/raw_commits.jsonl", output_dir: str = "lora_code_reviewer") -> None:
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Training dataset not found: {data_path}")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("repo-architect-lora")
    tokenizer = AutoTokenizer.from_pretrained(settings.llm_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(settings.llm_model_id, torch_dtype=settings.torch_dtype, device_map="auto" if settings.device == "cuda" else None)
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    dataset = load_dataset("json", data_files=data_path, split="train")

    def tokenize_fn(example):
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Diff:\n{example['input']}\n\n### Response:\n{example['output']}{tokenizer.eos_token}"
        tokenized = tokenizer(prompt, truncation=True, max_length=1024)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_ds = dataset.map(tokenize_fn, remove_columns=dataset.column_names)
    use_bf16 = settings.device == "cuda" and torch.cuda.is_bf16_supported()
    with mlflow.start_run():
        args = TrainingArguments(output_dir=output_dir, per_device_train_batch_size=2, gradient_accumulation_steps=4, num_train_epochs=3, learning_rate=2e-4, bf16=use_bf16, fp16=(settings.device == "cuda" and not use_bf16), logging_steps=5, save_strategy="epoch", report_to="mlflow", dataloader_num_workers=0)
        trainer = Trainer(model=model, args=args, train_dataset=tokenized_ds, data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True))
        trainer.train()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        mlflow.log_artifacts(output_dir, artifact_path="lora_adapter")
