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


def format_example(prompt: str, answer: float) -> dict[str, str]:
    return {
        "question": prompt,
        "answer": f"<answer>{round(float(answer), 4)}</answer>"
    }

def tokenize(tokenizer, question: str, answer: str):
    """Tokenize the input/output pair and mask the input when computing loss."""
    full_text = f"{question} {answer}{tokenizer.eos_token}"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Tokenize without special tokens to make sure <answer> appears as-is
    full = tokenizer(full_text, padding="max_length", truncation=True,
                     max_length=128, add_special_tokens=False)

    input_ids = full["input_ids"]
    question_ids = tokenizer(question, add_special_tokens=False)["input_ids"]

    # Locate start of <answer>
    label_start = len(question_ids)
    labels = [-100] * label_start + input_ids[label_start:]

    # Mask out padded tokens as well
    for i, attn in enumerate(full["attention_mask"]):
        if attn == 0:
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
    epochs: int = 5,
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
        per_device_train_batch_size=32,
        learning_rate=lr,
        gradient_checkpointing=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
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
