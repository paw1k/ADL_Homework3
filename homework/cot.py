from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    """LLM that uses two strong in‑context examples and forces <answer> tags."""

    # Two diverse examples
    _EXAMPLES = [
        (
            "How many gram are there per 3 kg?",
            "<answer>3000</answer> Because 1 kg = 1000 g, so 3 kg × 1000 g/kg = 3000 g.",
        ),
        (
            "Convert 2 hours to seconds.",
            "<answer>7200</answer> Because 1 h = 3600 s, so 2 h × 3600 s/h = 7200 s.",
        ),
        (
            "Convert 8 km/h to m/s.",
            "<answer>2.2222222222222223</answer> Because 1 km = 1000 m and 1 h = 3600 s, so 8 ×1000/3600 ≈ 2.2222 m/s.",
        ),
        (
            "How many pounds are there in 5 kg?",
            "<answer>11.023113109</answer> Because 1 kg = 2.2046226218487757 lb, so 5 × 2.2046226218487757 = 11.023113109 lb.",
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
            "You are an expert unit‑conversion assistant. "
            "Start your reply with the numeric result wrapped exactly as <answer>NUMBER</answer>. "
            "Then give one concise sentence showing the calculation."
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
