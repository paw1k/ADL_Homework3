from pathlib import Path
from typing import list

from .cot import CoTModel
from .data import Dataset, is_answer_valid

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
#     raise NotImplementedError()
    def generate_dataset(
        output_json: str | None = None,
        *,
        oversample: int = 10,
        temperature: float = 0.6,
    ):
        """Create an RFT dataset via Chain‑of‑Thought sampling.

        Parameters
        ----------
        output_json : str | None
            Target file.  If *None*, defaults to ``data/rft.json`` under the
            homework package root.
        oversample : int, default **10**
            How many completions to sample per question.
        temperature : float, default **0.6**
            Sampling temperature – anything >0 enables nucleus sampling.
        """

        # -------------------------------------------------------------------
        # Prep paths & model
        # -------------------------------------------------------------------
        pkg_root = Path(__file__).parent.parent
        if output_json is None:
            output_json = pkg_root / "data" / "rft.json"
        else:
            output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)

        cot_model = CoTModel()

        # -------------------------------------------------------------------
        # Iterate over the *training* split
        # -------------------------------------------------------------------
        rollouts: list[list] = []
        train = Dataset("train")

        for q, true_answer in train:
            # sample *oversample* responses
            generations = cot_model.batched_generate(
                [cot_model.format_prompt(q)], num_return_sequences=oversample, temperature=temperature
            )[0]

            # find first correct generation
            picked: str | None = None
            for g in generations:
                if is_answer_valid(cot_model.parse_answer(g), true_answer):
                    picked = g
                    break

            if picked is not None:
                rollouts.append([q, true_answer, picked])

        # -------------------------------------------------------------------
        # Save
        # -------------------------------------------------------------------
        with output_json.open("w") as f:
            json.dump(rollouts, f, indent=2)
        print(f"[datagen] wrote {len(rollouts)} RFT entries to {output_json}")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
