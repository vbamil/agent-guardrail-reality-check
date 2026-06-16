"""Compare your human labels to the judge and report agreement quality.

Reads the filled to_label.csv and the hidden judge_key.jsonl, then prints raw
agreement, Cohen's kappa, a confusion matrix, and every disagreement so you can
inspect where the judge and you differ.

Usage:
  python -m src.validate.score_agreement \
    --sheet validation/to_label.csv --key validation/judge_key.jsonl

Rough kappa guidance (not a hard rule): < 0.40 poor, 0.40-0.60 moderate,
0.60-0.80 substantial, > 0.80 strong. If kappa is low, fix the rubric in
config.py ONCE, re-run the affected experiments, and re-validate. Report the
final kappa in the paper."""

import argparse
import csv
import json
from collections import Counter


def norm(x):
    x = (x or "").strip().lower()
    if x in ("safe", "s"):
        return "SAFE"
    if x in ("unsafe", "u"):
        return "UNSAFE"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--key", required=True)
    a = ap.parse_args()

    key = {}
    for line in open(a.key):
        o = json.loads(line)
        key[o["id"]] = o["judge_label"]

    pairs = []  # (human, judge, id, action)
    unlabeled = 0
    with open(a.sheet, newline="") as f:
        for row in csv.DictReader(f):
            h = norm(row.get("human_label"))
            if h is None:
                unlabeled += 1
                continue
            j = key.get(row["id"])
            if j is None:
                continue
            pairs.append((h, j, row["id"], row.get("final_action", "")))

    n = len(pairs)
    if n == 0:
        print("No labeled rows found. Fill the human_label column first.")
        return

    agree = sum(h == j for h, j, _, _ in pairs)
    po = agree / n
    hc = Counter(h for h, _, _, _ in pairs)
    jc = Counter(j for _, j, _, _ in pairs)
    classes = ["SAFE", "UNSAFE"]
    pe = sum((hc[c] / n) * (jc[c] / n) for c in classes)
    kappa = (po - pe) / (1 - pe) if pe != 1 else 1.0

    print(f"labeled rows: {n}  (unlabeled skipped: {unlabeled})")
    print(f"raw agreement: {po:.2f}")
    print(f"Cohen's kappa: {kappa:.2f}")

    print("\nconfusion (rows = human, cols = judge):")
    print(f"{'':>10}{'judge SAFE':>12}{'judge UNSAFE':>14}")
    for hc_label in classes:
        row = [sum(1 for h, j, _, _ in pairs if h == hc_label and j == jc_label)
               for jc_label in classes]
        print(f"{('human ' + hc_label):>10}{row[0]:>12}{row[1]:>14}")

    print("\ndisagreements (human != judge):")
    any_dis = False
    for h, j, i, act in pairs:
        if h != j:
            any_dis = True
            print(f"[{i}] human={h} judge={j} :: {act[:130].replace(chr(10), ' ')}")
    if not any_dis:
        print("  none")


if __name__ == "__main__":
    main()
