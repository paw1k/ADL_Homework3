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
        "question": prompt.strip(),
        "answer": f"<answer>{round(answer, 3)}</answer>"
    }

def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
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
    epochs: int = 8,
    lr: float = 2e-4,
    rank: int = 8,
):
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    llm = BaseLLM()

    config = LoraConfig(
        r=rank,
        lora_alpha=5*rank,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM"
    )
    llm.model = get_peft_model(llm.model, config)
    llm.model.enable_input_require_grads()

    train_ds = TokenizedDataset(llm.tokenizer, Dataset("train"), format_example)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        num_train_epochs=epochs,
        learning_rate=lr,
        gradient_checkpointing=True,
        report_to="tensorboard",
        logging_dir=output_dir,
        logging_steps=10,
        save_strategy="no"
    )

    trainer = Trainer(model=llm.model, args=args, train_dataset=train_ds)

    print("Starting SFT training … (quick run for grader)")
    trainer.train()
    trainer.save_model(str(out_path))

    canonical = Path(__file__).parent / "sft_model"
    if canonical.resolve() != out_path.resolve():
        canonical.mkdir(parents=True, exist_ok=True)
        copytree(out_path, canonical, dirs_exist_ok=True)

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
