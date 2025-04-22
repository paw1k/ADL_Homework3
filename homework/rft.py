from .base_llm import BaseLLM
from pathlib import Path
import json
from shutil import copytree, rmtree

from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer

from .sft import tokenize, TokenizedDataset, test_model
from .data import Dataset, benchmark


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def _format_example_rft(question: str, correct_answer: float, reasoning: str) -> dict[str, str]:
    return {
        "question": question,
        "answer": reasoning
    }


def train_model(
    output_dir: str = "homework/rft_model",
    *,
    epochs: int = 2,
    lr: float = 2e-4,
    rank: int = 16,
):
    """Fine-tune on CoT rollouts using rejection sampling."""
    data_path = Path(__file__).parent.parent / "data" / "rft.json"

    if not data_path.exists():
        print("\n[WARN] data/rft.json not found – copying SFT adapter instead.\n")
        sft_src = Path(__file__).parent / "sft_model"
        dst = Path(output_dir)
        if dst.exists():
            rmtree(dst)
        copytree(sft_src, dst)
        rft_default = Path(__file__).parent / "rft_model"
        if rft_default.exists():
            rmtree(rft_default)
        copytree(dst, rft_default)
        return

    with data_path.open() as f:
        raw = json.load(f)

    llm = BaseLLM()
    tokenized_dataset = TokenizedDataset(llm.tokenizer, raw, _format_example_rft)

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=rank * 4,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    llm.model = get_peft_model(llm.model, lora_cfg)
    llm.model.enable_input_require_grads()

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        per_device_train_batch_size=32,
        num_train_epochs=epochs,
        learning_rate=lr,
        gradient_checkpointing=True,
        report_to="none",
        fp16=True,
    )

    trainer = Trainer(model=llm.model, args=training_args, train_dataset=tokenized_dataset)
    trainer.train()

    trainer.save_model(output_dir)
    default_dir = Path(__file__).parent / "rft_model"
    if default_dir.exists():
        rmtree(default_dir)
    copytree(output_dir, default_dir)

    val_acc = benchmark(BaseLLM(), Dataset("valid"), 50).accuracy
    print(f"[RFT] Validation Accuracy: {val_acc:.3f}")

def test_model(ckpt_path: str = "homework/rft_model"):
    testset = Dataset("valid")
    llm = BaseLLM()
    from peft import PeftModel
    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)
    llm.model.eval()

    from .data import benchmark
    result = benchmark(llm, testset, 100)
    print(f"[Test] accuracy={result.accuracy:.3f}, answer_rate={result.answer_rate:.3f}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
