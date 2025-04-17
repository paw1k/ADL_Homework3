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


def train_model(
    output_dir: str,
    **kwargs,
):
#     raise NotImplementedError()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        # 1. model + LoRA
        llm = _create_lora_model()

        # 2. dataset (tokenised on‑the‑fly)
        train_data = TokenizedDataset(llm.tokenizer, Dataset("train"), format_example)

        # 3. trainer args – super tiny to keep runtime minimal
        args = TrainingArguments(
            output_dir=str(output),
            per_device_train_batch_size=32,
            num_train_epochs=1,
            learning_rate=5e-4,
            logging_dir=str(output / "logs"),
            report_to="none",
            gradient_checkpointing=True,
            save_total_limit=1,
        )

        trainer = Trainer(model=llm.model, args=args, train_dataset=train_data)
        trainer.train()

        # Save adapter in *output_dir* and in fixed path for the grader
        trainer.save_model(str(output))

        default_path = Path(__file__).parent / "sft_model"
        default_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(default_path))

        # Quick self‑check (prints accuracy but does not gate execution)
        print("Running quick validation after SFT …")
        val = benchmark(llm, Dataset("valid"), 100)
        print(f"accuracy={val.accuracy:.3f}  answer_rate={val.answer_rate:.3f}")

        test_model(output_dir)


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
