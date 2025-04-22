from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    """LLM that uses two strong in‑context examples and forces <answer> tags."""

def format_prompt(self, question: str) -> str:
        messages = [
            {"role": "system", "content": "Convert units accurately. Use step-by-step reasoning and put the final answer within <answer> tags."},
            {"role": "user", "content": "How many gram are there per 6 kg?"},
            {"role": "assistant", "content": "1 kg = 1000 grams. So 6 kg = 6 * 1000 = <answer>6000</answer> grams."},
            {"role": "user", "content": question}
        ]
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
#         raise NotImplementedError()


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
