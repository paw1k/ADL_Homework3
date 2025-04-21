from pathlib import Path
from typing import Optional
import json

from .cot import CoTModel
from .data import Dataset, is_answer_valid

def generate_dataset(
    output_json: Optional[str] = None,
    *,
    oversample: int = 10,
    temperature: float = 0.6,
):
    """Create an RFT dataset via Chain‑of‑Thought sampling."""
    pkg_root = Path(__file__).parent.parent
    if output_json is None:
        output_json = pkg_root / "data" / "rft.json"
    else:
        output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    cot_model = CoTModel()

    rollouts: list[list[str]] = []
    train = Dataset("train")

    for q, true_answer in train:
        generations = cot_model.batched_generate(
            [cot_model.format_prompt(q)], num_return_sequences=oversample, temperature=temperature
        )[0]

        picked: Optional[str] = None
        for g in generations:
            if is_answer_valid(cot_model.parse_answer(g), true_answer):
                picked = g
                break

        if picked is not None:
            rollouts.append([q, true_answer, picked])

    with output_json.open("w") as f:
        json.dump(rollouts, f, indent=2)
    print(f"[datagen] wrote {len(rollouts)} RFT entries to {output_json}")


if __name__ == "__main__":
    from fire import Fire
    Fire(generate_dataset)
