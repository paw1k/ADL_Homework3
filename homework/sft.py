from .base_llm import BaseLLM
from .data import Dataset, benchmark
from pathlib import Path
from shutil import copytree
from typing import Dict

import torch
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel


def load() -> BaseLLM:
    model_path = Path(__file__).parent / "sft_model"
    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()
    return llm


def format_example(prompt: str, answer: str) -> dict[str, str]:
    return {
        "question": prompt,
        "answer": f"<answer>{round(answer, 4)}</answer>"
    }

def tokenize(tokenizer, question: str, answer: str):
    full_text = f"{question} {answer}{tokenizer.eos_token}"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    q_ids = tokenizer(question, truncation=True, max_length=128)["input_ids"]
    q_len = len(q_ids)

    labels = [-100] * q_len + input_ids[q_len:]
    labels = labels[:128] + [-100] * max(0, 128 - len(labels))  # pad/truncate to match

    for i, a in enumerate(full["attention_mask"]):
        if a == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        self.tokenizer = tokenizer
        self.data = data
        self.format_fn = format_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def train_model(
    output_dir: str = "homework/sft_model",
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    rank: int = 8,
):
    """Fine-tune SmolLM2 using LoRA for direct answer generation."""
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    llm = BaseLLM()

    # Attach LoRA
    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=rank * 4,
        bias="none",
        target_modules="all-linear",
        task_type=TaskType.CAUSAL_LM,
    )
    llm.model = get_peft_model(llm.model, lora_cfg)
    llm.model.enable_input_require_grads()

    # Tokenized training dataset
    train_ds = TokenizedDataset(llm.tokenizer, Dataset("train"), format_example)

    args = TrainingArguments(
        output_dir=str(out_path),
        logging_dir=str(out_path / "logs"),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        learning_rate=lr,
        gradient_checkpointing=True,
        report_to="tensorboard",
        fp16=torch.cuda.is_available(),
        save_total_limit=1
    )

    trainer = Trainer(model=llm.model, args=args, train_dataset=train_ds)

    print("Starting SFT training … (quick run for grader)")
    trainer.train()
    trainer.save_model(str(out_path))

    # Save also to homework/sft_model for grader compatibility
    canonical = Path(__file__).parent / "sft_model"
    if canonical.resolve() != out_path.resolve():
        canonical.mkdir(parents=True, exist_ok=True)
        copytree(out_path, canonical, dirs_exist_ok=True)

    # Validate
    val = benchmark(llm, Dataset("valid"), 50)
    print(f"Validation   acc={val.accuracy:.3f}   answer‑rate={val.answer_rate:.3f}")
    print(val.accuracy)


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)
    val = benchmark(llm, testset, 100)
    print(f"{val.accuracy=:.3f}  {val.answer_rate=:.3f}")


if __name__ == "__main__":
    from fire import Fire
    Fire({"train": train_model, "test": test_model, "load": load})
