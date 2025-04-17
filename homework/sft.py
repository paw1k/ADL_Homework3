from pathlib import Path
from shutil import copytree
from typing import Dict

import torch
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

from .base_llm import BaseLLM
from .data import Dataset, benchmark

# ---------------------------------------------------------------------------
# Helper: prepare plain Q/A text – no chat template for SFT
# ---------------------------------------------------------------------------

def format_example(prompt: str, answer: float) -> Dict[str, str]:
    rounded = round(float(answer), 4)
    return {
        "question": prompt,
        "answer": f"<answer>{rounded}</answer>",
    }


# ---------------------------------------------------------------------------
# Tokenisation utilities re‑used by RFT
# ---------------------------------------------------------------------------

def tokenize(tokenizer, question: str, answer: str):
    full_text = f"{question} {answer}{tokenizer.eos_token}"
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    # mask out the prompt tokens in labels
    q_len = len(tokenizer(question)["input_ids"])
    labels = [-100] * q_len + enc["input_ids"][q_len:]
    # also mask any padded positions
    labels = [lab if mask == 1 else -100 for lab, mask in zip(labels, enc["attention_mask"])]
    enc["labels"] = labels
    return enc


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, fmt_fn):
        self.tokenizer = tokenizer
        self.data = data
        self.fmt_fn = fmt_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q, a = self.data[idx]
        item = self.fmt_fn(q, a)
        return tokenize(self.tokenizer, **item)


# ---------------------------------------------------------------------------
# Training entry‑point
# ---------------------------------------------------------------------------

def train_model(output_dir: str = "homework/sft_model", **kwargs):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1) load base model
    llm = BaseLLM()

    # 2) attach LoRA adapter
    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    llm.model = get_peft_model(llm.model, lora_cfg)

    # 3) prepare dataset
    train_ds = TokenizedDataset(llm.tokenizer, Dataset("train"), format_example)

    # 4) training args (very small for fast completion)
    args = TrainingArguments(
        output_dir=str(output_path),
        logging_dir=str(output_path / "logs"),
        num_train_epochs=1,
        per_device_train_batch_size=8,
        learning_rate=2e-4,
        gradient_checkpointing=True,
        report_to="none",
        fp16=False,
    )

    trainer = Trainer(model=llm.model, args=args, train_dataset=train_ds)

    print("Starting SFT training (tiny epoch)…")
    trainer.train()

    # 5) save adapter
    trainer.save_model(str(output_path))

    # also copy to canonical folder if different
    canonical = Path(__file__).parent / "sft_model"
    if canonical.resolve() != output_path.resolve():
        canonical.mkdir(exist_ok=True, parents=True)
        copytree(output_path, canonical, dirs_exist_ok=True)

    # quick sanity benchmark
    val_acc = benchmark(llm, Dataset("valid"), 50).accuracy
    print(f"Validation accuracy after SFT: {val_acc:.3f}")


# ---------------------------------------------------------------------------
# Loader expected by the grader
# ---------------------------------------------------------------------------

def load() -> BaseLLM:  # noqa: D401
    from peft import PeftModel

    path = Path(__file__).parent / "sft_model"
    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, path).to(llm.device)
    llm.model.eval()
    return llm


# quick CLI helpers
if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "load": load})


def test_model(ckpt_path: str | None = None):  # noqa: D401
    """Load the saved adapter (or one provided path) and print validation acc."""
    path = Path(ckpt_path) if ckpt_path else Path(__file__).parent / "sft_model"
    from peft import PeftModel
    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, path).to(llm.device)
    from .data import Dataset, benchmark
    result = benchmark(llm, Dataset("valid"), max_question=50)
    print(f"{result.accuracy=:.3f}  {result.answer_rate=:.3f}")

