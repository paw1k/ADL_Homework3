from pathlib import Path
from shutil import copytree
from typing import Dict
import torch
import json

from .base_llm import BaseLLM
from .data import Dataset, benchmark
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

def load() -> BaseLLM:
    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm

def tokenize(tokenizer, question: str, answer: str):
    full_prompt = f"{question} Answer: {answer}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_prompt, padding="max_length", truncation=True, max_length=128)

    # Find the position of <answer> token
    answer_token_id = tokenizer.convert_tokens_to_ids('<answer>')
    input_ids = full["input_ids"]

    try:
        label_start = input_ids.index(answer_token_id)
    except ValueError:
        label_start = len(input_ids)  # Fallback

    # Set labels
    labels = [-100] * label_start + input_ids[label_start:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full

def format_example(prompt: str, answer: float) -> Dict[str, str]:
    return {
        "question": prompt,
        "answer": f"<answer>{round(float(answer), 4)}</answer>",
    }

class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formatted_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formatted_data)

def train_model(
    output_dir: str = "homework/sft_model",
    *,
    epochs: int = 3,
    lr: float = 2e-4,
    rank: int = 8,
):
    # ... existing code ...
    llm = BaseLLM()

    # Add special tokens
    special_tokens = ['<answer>', '</answer>']
    llm.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    llm.model.resize_token_embeddings(len(llm.tokenizer))

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=rank * 4,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    llm.model = get_peft_model(llm.model, lora_cfg)
    llm.model.enable_input_require_grads()

    train_ds = TokenizedDataset(llm.tokenizer, Dataset("train"), format_example)

    args = TrainingArguments(
        output_dir=str(out_path),
        logging_dir=str(out_path / "logs"),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        learning_rate=lr,
        gradient_checkpointing=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(model=llm.model, args=args, train_dataset=train_ds)

    print("Starting SFT training … (quick run for grader)")
    trainer.train()
    trainer.save_model(str(out_path))

    canonical = Path(__file__).parent / "sft_model"
    if canonical.resolve() != out_path.resolve():
        canonical.mkdir(parents=True, exist_ok=True)
        copytree(out_path, canonical, dirs_exist_ok=True)

    result = benchmark(llm, Dataset("valid"), 50)
    print(f"Validation   acc={result.accuracy:.3f}   answer‑rate={result.answer_rate:.3f}")
    print(result.accuracy)

def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)
    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")

if __name__ == "__main__":
    from fire import Fire
    Fire({"train": train_model, "test": test_model, "load": load})
