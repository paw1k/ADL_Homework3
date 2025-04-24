# homework/rft.py
from pathlib import Path
import json
from shutil import copytree

from peft import LoraConfig, get_peft_model, PeftModel
from transformers import Trainer, TrainingArguments

from .base_llm import BaseLLM
from .sft import tokenize, TokenizedDataset, test_model
from .data import Dataset, benchmark

def load() -> BaseLLM:
    ckpt = Path(__file__).parent / "rft_model"
    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, ckpt).to(llm.device)
    llm.model.eval()
    return llm

class _RFTData:
    def __init__(self, path="data/rft.json"):
        with open(path) as f:
            self.data = json.load(f)

    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        q, _gt, reasoning = self.data[i]
        return q, reasoning

def _format(q: str, reasoning: str):
    return {"question": q.strip(), "answer": reasoning.strip()}

def train_model(
    output_dir: str = "homework/rft_model",
    epochs: int = 6,
    lr: float = 2e-4,
    rank: int = 16,
):
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    llm = BaseLLM()

    cfg = LoraConfig(
        r=rank,
        lora_alpha=32,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    llm.model = get_peft_model(llm.model, cfg)
    llm.model.enable_input_require_grads()

    train_ds = TokenizedDataset(llm.tokenizer, _RFTData(), _format)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        num_train_epochs=epochs,
        learning_rate=lr,
        gradient_checkpointing=True,
        report_to="tensorboard",
        logging_steps=10,
        save_strategy="no",
    )

    Trainer(model=llm.model, args=args, train_dataset=train_ds).train()

    llm.model.save_pretrained(str(out))
    canonical = Path(__file__).parent / "rft_model"
    if canonical.resolve() != out.resolve():
        canonical.mkdir(parents=True, exist_ok=True)
        copytree(out, canonical, dirs_exist_ok=True)

    val = benchmark(llm, Dataset("valid"), 50)
    print(f"validation  acc={val.accuracy:.3f}   ans-rate={val.answer_rate:.3f}")

if __name__ == "__main__":
    from fire import Fire
    Fire({"train": train_model, "test": test_model, "load": load})
