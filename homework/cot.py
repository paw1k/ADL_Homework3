from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    """LLM that uses two strong in‑context examples and forces <answer> tags."""

    # Two diverse examples
    _EXAMPLES = [
        (
            "Could you provide the value of 4 kmh in mph?",
            "To convert km/h to mi/h, use the factor 1 km/h ≈ 0.621371. Multiply the speed by 0.621371.\n<answer>2.4854847689493362</answer>"
        ),
        (
            "What is the equivalent of 7 pound in kg?",
            "1 pound = 0.45359237 kg. Multiply 7 * 0.45359237 to convert to kilograms.\n<answer>3.17514659</answer>"
        ),
        (
            "What is the measurement of 6 decades when converted into year?",
            "1 decade = 10 years. Multiply 6 * 10.\n<answer>60.0</answer>"
        ),
        (
            "Tell me how many fluid ounce are there in 2 l.",
            "1 liter = 33.814 fluid ounces. Multiply 2 * 33.814.\n<answer>67.62804540368597</answer>"
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

        sys_msg = (
            "You are a helpful and accurate unit conversion assistant. "
            "For each question, calculate step-by-step, then give the final answer "
            "wrapped in <answer> tags on a new line. Be concise and precise."
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
