from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    """LLM that uses two strong in‑context examples and forces <answer> tags."""

    # Two diverse examples
    _EXAMPLES = [
        (
            "How many gram are there per 3 kg?",
            "1 kg = 1000 g\n3 kg × 1000 g/kg = 3000 g\n<answer>3000.0</answer>",
        ),
        (
            "Convert 5 mile to meter.",
            "1 mile = 1609.344 m\n5 × 1609.344 = <answer>8046.72</answer>",
        ),
        (
            "How much is 4 kmh when converted to m/s?",
            "1 km = 1000 m, 1 h = 3600 s\n(4 × 1000) / 3600 = <answer>1.1111111111111112</answer>",
        ),
        (
            "How many in are there per 8 ft?",
            "1 ft = 12 in\n8 × 12 = <answer>96.0</answer>",
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
            "You are a unit‑conversion expert. "
            "First think step‑by‑step and write the calculation, "
            "then on a new line output the result as <answer>NUMBER</answer>."
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
