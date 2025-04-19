from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
    """
    Build a chat prompt that encourages step‑by‑step reasoning **and**
    forces the model to wrap the final number in <answer> tags.
    """

    # 1️⃣ One worked example – short, explicit, shows the tag early
    ex_q = "How many gram are there per 3 kg?"
    ex_a = (
        "1 kg = 1000 g → 3 kg × 1000 g/kg = <answer>3000</answer> g."
        " The final answer is 3000 grams."
    )

    # 2️⃣ Put everything into SmolLM2’s chat‑template
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert **unit‑conversion assistant**. "
                "Think step‑by‑step but *be concise*. "
                "ALWAYS finish with the result wrapped in <answer> tags."
            ),
        },
        {"role": "user", "content": ex_q},
        {"role": "assistant", "content": ex_a},
        {"role": "user", "content": question},
    ]

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
