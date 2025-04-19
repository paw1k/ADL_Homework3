from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        example_q = "Convert 2 kg to g."
        example_a = "2 kg = 2000 g. <answer>2000</answer>"

        messages = [
            {"role": "system", "content": "You are a unit conversion assistant. Wrap the final answer in <answer> tags."},
            {"role": "user", "content": example_q},
            {"role": "assistant", "content": example_a},
            {"role": "user", "content": question}
        ]

        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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
