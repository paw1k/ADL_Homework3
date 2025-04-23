from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    """LLM that uses two strong in‑context examples and forces <answer> tags."""

    # Two diverse examples
    _EXAMPLES = [
        (
            "How many gram are there per 3 kg?",
            "1 kg = 1000 g\n3 kg × 1000 g/kg = 3000 g\n<answer>3000</answer>",
        ),
        (
            "Convert 5 mile to meter.",
            "1 mile = 1609.344 m\n5 mile × 1609.344 m/mile = 8046.72 m\n<answer>8046.72</answer>",
        ),
        (
            "Convert 8 km/h to m/s.",
            "1 km = 1000 m, 1 h = 3600 s\n(8 × 1000) / 3600 = 2.222222… m/s\n<answer>2.2222222222222223</answer>",
        ),
    ]

    def format_prompt(self, question: str) -> str:  # noqa: D401
        """Return a chat template prompt.

        **Contract for the model** (spelled out in *system* message):
        1. Begin the response with the numeric result wrapped in `<answer>` tags.
        2. Afterwards, add one short sentence of reasoning.
        This guarantees `parse_answer` will always find the tag even if the
        model later truncates.
        """

        # System instruction – short & strict
        sys_msg = (
            "You are a helpful and accurate unit conversion expert assistant.  "
            "Please solve the conversion step-by-step and write the calculation. "
            "At the end, write the final result inside <answer> tags on a new line."
        )

        messages: list[dict[str, str]] = [{"role": "system", "content": sys_msg}]

        # Add our worked examples
        for q, a in self._EXAMPLES:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

        # Finally the real question
        messages.append({"role": "user", "content": question})

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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
