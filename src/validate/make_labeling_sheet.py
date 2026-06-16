"""Build a BLIND labeling sheet for scorer validation.

Samples N actions from a run log, stratified across the judge's SAFE/UNSAFE
labels so both error types are represented, joins each to its task prompt, and
writes:
  - <outdir>/to_label.csv   you fill the human_label column (SAFE/UNSAFE). The
                            judge's label is NOT shown, so your labels stay blind.
  - <outdir>/judge_key.jsonl  hidden key (id -> judge label + rationale) used only
                            afterward by score_agreement.py. Do not open it until
                            you have finished labeling.

Usage:
  python -m src.validate.make_labeling_sheet \
    --run runs/<timestamp>_passthrough.jsonl \
    --tasks data/asb_sample.jsonl --n 50
"""

import argparse
import csv
import json
import os
import random


def load_results(path):
    out = []
    for line in open(path):
        o = json.loads(line)
        if o.get("kind") == "result":
            out.append(o)
    return out


def load_prompts(path):
    prompts = {}
    for line in open(path):
        t = json.loads(line)
        prompts[t["id"]] = t.get("prompt", "")
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run log .jsonl")
    ap.add_argument("--tasks", required=True, help="tasks .jsonl used for the run")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default="validation")
    a = ap.parse_args()

    recs = load_results(a.run)
    prompts = load_prompts(a.tasks)
    rng = random.Random(a.seed)

    safe = [r for r in recs if r["judge_label"] == "SAFE"]
    unsafe = [r for r in recs if r["judge_label"] == "UNSAFE"]
    rng.shuffle(safe)
    rng.shuffle(unsafe)
    half = a.n // 2
    sample = safe[:half] + unsafe[: a.n - half]
    rng.shuffle(sample)

    os.makedirs(a.outdir, exist_ok=True)
    sheet = os.path.join(a.outdir, "to_label.csv")
    key = os.path.join(a.outdir, "judge_key.jsonl")

    with open(sheet, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "intent", "task_prompt", "final_action", "human_label"])
        for r in sample:
            w.writerow([
                r["id"],
                r.get("intent", ""),
                prompts.get(r["id"], ""),
                r.get("final_action", ""),
                "",  # you fill this: SAFE or UNSAFE
            ])

    with open(key, "w") as f:
        for r in sample:
            f.write(json.dumps({
                "id": r["id"],
                "judge_label": r["judge_label"],
                "judge_rationale": r.get("judge_rationale", ""),
            }) + "\n")

    n_safe = sum(1 for r in sample if r["judge_label"] == "SAFE")
    print(f"Wrote {len(sample)} rows to {sheet} "
          f"(judge SAFE={n_safe}, judge UNSAFE={len(sample) - n_safe}).")
    print(f"Hidden key written to {key}. Do NOT open it until you finish labeling.")
    print("Open to_label.csv, fill human_label with SAFE or UNSAFE for each row, "
          "judging the final_action in the context of the task_prompt.")


if __name__ == "__main__":
    main()
