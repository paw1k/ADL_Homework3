from typing import overload
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<answer>', '</answer>']})
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.device = device

    def format_prompt(self, question: str) -> str:
        return f"{question} Answer:"

    def parse_answer(self, answer: str) -> float:
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
            return float("nan")

    def generate(self, prompt: str) -> str:
        return self.batched_generate([prompt])[0]

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]: ...

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]: ...

    def batched_generate(
        self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
    ) -> list[str] | list[list[str]]:
        from tqdm import tqdm

        micro_batch_size = 32
        if len(prompts) > micro_batch_size:
            return [
                r
                for idx in tqdm(range(0, len(prompts), micro_batch_size),
                                desc=f"LLM micro-batches (size={micro_batch_size})")
                for r in self.batched_generate(
                    prompts[idx:idx + micro_batch_size], num_return_sequences, temperature
                )
            ]

        # Generation setup
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        n_return = num_return_sequences or 1
        do_sample = temperature > 0.0

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                num_return_sequences=n_return,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Slice out only the newly generated part
        prompt_lens = attention_mask.sum(dim=1)
        decoded: list[str] = []
        for i, out in enumerate(outputs):
            start = prompt_lens[i % len(prompts)]
            new_tokens = out[start:]
            decoded.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))

        if n_return == 1:
            return decoded
        grouped = [decoded[i * n_return:(i + 1) * n_return] for i in range(len(prompts))]
        return grouped

    def answer(self, *questions) -> list[float]:
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
        return [self.parse_answer(g) for g in generations]


def test_model():
    model = BaseLLM()
    testset = ["The cat went up", "The dog went down"]
    for t in testset:
        print("input:", t)
        output = model.generate(t)
        print("output:", output)
    print("Batched:", model.batched_generate(testset))


if __name__ == "__main__":
    from fire import Fire
    Fire({"test": test_model})
