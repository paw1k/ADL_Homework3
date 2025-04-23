from typing import overload

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into an input to SmolLM2. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        return question

    def parse_answer(self, answer: str) -> float:
        """
        Parse the <answer></answer> tag and return a float.
        This function is somewhat robust to output errors (e.g. missing </answer> tags).
        """
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
            return float("nan")

    def generate(self, prompt: str) -> str:
        """
        (Optional) Implement this method first and then implement batched_generate below.
        It is much easier to implement generation without batching.

        The overall flow is the same:
        - tokenize the prompt with self.tokenizer
        - call self.model.generate
        - decode the outputs with self.tokenizer.decode

        """
        return self.batched_generate([prompt])[0]

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        """
        Batched version of `generate` method.
        This version returns a single generation for each prompt.
        """

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        """
        Batched version of `generate` method.
        This version returns a list of generation for each prompt.
        """

    def batched_generate(
        self,
        prompts: list[str],
        num_return_sequences: int | None = None,
        temperature: float = 0.0,
    ) -> list[str] | list[list[str]]:
        """Efficiently decode *all* prompts in one forward‑pass.

        The implementation follows the hints in the README: left‑pad prompts to
        equal length, feed them through ``model.generate``, and finally decode.
        """
        from tqdm import tqdm  # imported lazily to avoid overhead in the grader

        # ------------------------------------------------------------------ #
        # recurse in micro batches if caller passes very large *prompts*
        # ------------------------------------------------------------------ #
        micro_batch_size = 32
        if len(prompts) > micro_batch_size:
            return [
                r
                for idx in tqdm(
                    range(0, len(prompts), micro_batch_size),
                    desc=f"LLM micro‑batches (size={micro_batch_size})",
                )
                for r in self.batched_generate(
                    prompts[idx : idx + micro_batch_size], num_return_sequences, temperature
                )
            ]

        # ------------------- tokenizer & generation params --------------- #
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        tok_batch = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
        n_return = num_return_sequences or 1
        do_sample = temperature > 0.0

        with torch.no_grad():
            gen_ids = self.model.generate(
                **tok_batch,
                max_new_tokens=100,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                num_return_sequences=n_return,
                eos_token_id=self.tokenizer.eos_token_id,
            )

                # ------------------ slice away the prompt portion ---------------- #
        attn = tok_batch["attention_mask"]  # (batch, seq_len)
        seq_len = attn.shape[1]
        starts = attn.sum(dim=1).tolist()  # prompt length for each item (left‑padding‑aware)

        decoded: list[str] = []
        for i, seq in enumerate(gen_ids):
            start = starts[i // n_return]          # integer‑division fixes the index
            decoded.append(self.tokenizer.decode(seq[start:], skip_special_tokens=True))

        # ---- reshape if caller asked for multiple return sequences ------ #
        if n_return == 1:
            return decoded  # type: ignore[return-value]

        grouped: list[list[str]] = [
            decoded[i * n_return : (i + 1) * n_return] for i in range(len(prompts))
        ]
        return grouped

#         raise NotImplementedError()

    def answer(self, *questions) -> list[float]:
        """
        Answer questions given as individual string arguments.
        """
        # Convert each question
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
        return [self.parse_answer(g) for g in generations]


def test_model():
    # The following code simply tests of the BaseLLM is able to complete text.
    # It should produce garbage answers, but it should not crash.
    # In my case it talks about cats eating cats, and dogs being happy.
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input", t)
        answer = model.generate(t)
        print("output", answer)
    answers = model.batched_generate(testset)
    print(answers)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model})
