from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    """LLM that uses two strong in‑context examples and forces <answer> tags."""

    # Two diverse examples
    _EXAMPLES = [
        (
            "Can you change 2 hour to its equivalent in min?",
            "1 hour = 60 minutes. 2 * 60 = <answer>120.0</answer>"
        ),
        (
            "What is the conversion of 3 kg to ounce?",
            "1 kg = 35.27396195 ounces. 3 * 35.27396195 = <answer>105.82188585</answer>"
        ),
        (
            "Convert 5 mi/h to m/s?",
            "1 mi = 1609.344 m, 1 h = 3600 s. So 5 mi/h = (5 * 1609.344) / 3600 = <answer>2.2352</answer>"
        ),
        (
            "What is 6 litre in millilitre?",
            "1 litre = 1000 millilitres. 6 * 1000 = <answer>6000.0</answer>"
        )
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
