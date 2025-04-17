from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:  # noqa: D401 – public API
        """Return a chat‑style prompt following SmolLM2’s template."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert unit‑conversion assistant. "
                    "Show concise working and wrap the final numeric value in "
                    "<answer> tags."
                ),
            },
            {"role": "user", "content": self._EXAMPLE_Q},
            {"role": "assistant", "content": self._EXAMPLE_A},
            {"role": "user", "content": question},
        ]

        # Convert chat list → single string (don’t tokenize here)
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

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
