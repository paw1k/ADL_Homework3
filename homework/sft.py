from .base_llm import BaseLLM
from .data import Dataset, benchmark


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


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


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
#     raise NotImplementedError()
        try:
            ans_val = float(answer)
        except Exception:  # pragma: no cover – dataset is clean but be safe
            ans_val = answer

        return {
            "question": prompt.strip(),
            "answer": f"<answer>{round(ans_val, 3)}</answer>",
        }

class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


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



def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
