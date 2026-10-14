"""Check that answering from the cached image prefix matches re-encoding the whole prompt.

Needs the model weights and a GPU or Apple Silicon, and takes a few minutes:

    python webapp/test_prefix_cache.py [path/to/ct.nii.gz]

Under pytest it is skipped unless NVRCT_MODEL_TESTS=1.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

MODEL_ID = "nvidia/NV-Reason-CT"
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "example_1.nii.gz"


def run(ct_path=EXAMPLE):
    # Imported here so pytest can collect (and skip) this file from the repository root, where
    # the accelerate/ config folder would otherwise shadow the accelerate package.
    import nibabel as nib
    import numpy as np
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, StoppingCriteria

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prefix_cache import PrefixCache

    class FirstToken(StoppingCriteria):
        """Records when the first new token is ready (time to first token)."""

        def __init__(self):
            self.at = None

        def __call__(self, input_ids, scores, **kwargs):
            if self.at is None:
                self.at = time.time()
            return False

    device ="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).eval().to(device)
    prefixes = PrefixCache(model, processor.tokenizer.convert_tokens_to_ids("<|vision_end|>"))

    # The server feeds the model a saved crop; do the same here.
    crop = processor.image_processor_3d.load_image(str(ct_path), normalize_mode=0, anatomy_region="chest")
    crop_path = Path(tempfile.mkdtemp()) / "crop_chest.nii"
    nib.save(nib.Nifti1Image(crop[0].numpy().transpose(2, 1, 0), np.diag([-2.0, -2.0, 2.0, 1.0])), str(crop_path))

    def inputs_for(turns, thinking):
        messages = [{"role": r, "content": [{"type": "text", "text": t}]} for r, t in turns]
        messages[0]["content"].insert(0, {"type": "image"})
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        return processor(text=text, images3d=[str(crop_path)], anatomy_region=None, return_tensors="pt")

    def generate(kwargs, max_new_tokens):
        first = FirstToken()
        start = time.time()
        with torch.inference_mode():
            out = model.generate(
                **kwargs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                stopping_criteria=[first], return_dict_in_generate=True, output_logits=True,
            )
        new = out.sequences[0, kwargs["input_ids"].shape[1]:].tolist()
        return new, out.logits[0][0].float().cpu(), first.at - start

    def full(inputs, max_new_tokens):
        return generate(inputs.to(device), max_new_tokens)

    yes_no = [("user", "Is a pleural effusion present in this CT? Answer only Yes or No.")]
    report = [("user", "write a structured chest CT report")]
    report_text = None

    def follow_up():  # carries the report from the previous case as conversation history
        answer = report_text.split("</think>")[-1].strip() or report_text.strip()
        return report + [("assistant", answer), ("user", "Is there emphysema? Answer only Yes or No.")]

    cases = [
        ("yes/no, reasoning off", lambda: yes_no, False, 8),
        ("report, reasoning on", lambda: report, True, 160),
        ("follow-up with history", follow_up, False, 8),
    ]

    results, all_match = [], True
    for name, turns_for, thinking, max_new in cases:
        turns = turns_for()
        inputs = inputs_for(turns, thinking)
        ids_full, logits_full, ttft_full = full(inputs, max_new)
        t0 = time.time()
        kwargs, reused = prefixes.prepare(("example", "chest"), inputs, device)
        prep = time.time() - t0
        ids_cached, logits_cached, ttft_cached = generate(kwargs, max_new)
        if report_text is None and thinking:
            report_text = processor.tokenizer.decode(ids_full, skip_special_tokens=True)
        match = ids_full == ids_cached
        all_match &= match
        first_diff = next((i for i, (a, b) in enumerate(zip(ids_full, ids_cached)) if a != b), None)
        results.append(
            f"{name:24s} prompt {inputs['input_ids'].shape[1]:,} tok | identical: {match}"
            f"{'' if match else f' (first difference at token {first_diff})'} | "
            f"first-token logits max diff {float((logits_full - logits_cached).abs().max()):.3g} | "
            f"time to first token: full {ttft_full:.1f}s, cached {prep + ttft_cached:.1f}s"
            f"{' (reused scan)' if reused else ' (encoded scan)'}"
        )
        print(results[-1], flush=True)
    return all_match


def test_prefix_cache_matches_full_prompt():
    import pytest

    if os.environ.get("NVRCT_MODEL_TESTS") != "1":
        pytest.skip("loads the full model; set NVRCT_MODEL_TESTS=1 to run")
    assert run()


if __name__ == "__main__":
    ok = run(Path(sys.argv[1]) if len(sys.argv) > 1 else EXAMPLE)
    print("ALL IDENTICAL" if ok else "DIFFERENCES FOUND")
    sys.exit(0 if ok else 1)
