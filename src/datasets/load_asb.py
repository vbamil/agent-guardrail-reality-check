"""Load a sample of Agent-SafetyBench into the harness task format.

IMPORTANT, read before using results:
This produces a SINGLE-STEP, STATED-ACTION adaptation. The real Agent-SafetyBench
runs a multi-turn agent loop inside simulated tool environments (see their
evaluation/eval.py + EnvManager) and scores trajectories with ShieldAgent. This
loader instead gives the agent the instruction plus a list of available tools and
asks for the action it would take, which the fixed LLM-judge then scores. That is
lighter and GPU-free, but it is NOT the same protocol, so:
  - You cannot claim to reproduce published Agent-SafetyBench scores from this.
  - You must disclose this single-step adaptation as a limitation.
  - For a stronger result, run the official harness for rollouts and swap in only
    the scorer, or clearly frame the paper as a stated-action study.

The `intent` field here is APPROXIMATE: we map fulfillable==1 -> "benign" (a safe
goal exists) and fulfillable==0 -> "unsafe" (should be refused or not fulfilled).
This is a convenience mapping for the pilot metric, not the benchmark's own
failure-mode scoring. Carry the real fields in `meta` and prefer them later.

Get the data first (MIT-licensed, cite Zhang et al. 2024):
  git clone https://github.com/thu-coai/Agent-SafetyBench
  # then point --src at Agent-SafetyBench/data/released_data.json
"""

import argparse
import json
import random


def render_tools(environments):
    lines = []
    for env in environments or []:
        name = env.get("name", "")
        tools = ", ".join(env.get("tools", []))
        lines.append(f"- {name}: {tools}")
    return "\n".join(lines) if lines else "(none specified)"


def tool_names(environments):
    names = []
    for env in environments or []:
        for t in env.get("tools", []):
            if t not in names:
                names.append(t)
    return names


def to_task(case):
    prompt = (
        f"{case['instruction']}\n\n"
        f"Available tools you may call:\n{render_tools(case.get('environments'))}"
    )
    intent = "benign" if case.get("fulfillable", 0) == 1 else "unsafe"
    return {
        "id": f"asb-{case['id']}",
        "intent": intent,
        "prompt": prompt,
        "tools": tool_names(case.get("environments")),
        "meta": {
            "risks": case.get("risks"),
            "fulfillable": case.get("fulfillable"),
            "failure_modes": case.get("failure_modes"),
            "intent_is_approximate": True,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to released_data.json")
    ap.add_argument("--n", type=int, default=100, help="sample size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/asb_sample.jsonl")
    ap.add_argument("--stratify", action="store_true",
                    help="balance fulfillable 0/1 in the sample")
    ap.add_argument("--by-risk", dest="by_risk", action="store_true",
                    help="stratify evenly across the 8 risk categories (task "
                         "property, not model outcome); preferred for coverage")
    a = ap.parse_args()

    data = json.load(open(a.src))
    rng = random.Random(a.seed)

    if a.by_risk:
        from collections import defaultdict
        groups = defaultdict(list)
        for x in data:
            key = (x.get("risks") or ["(none)"])[0]
            groups[key].append(x)
        cats = sorted(groups)
        per = max(1, a.n // len(cats))
        sample = []
        for c in cats:
            g = groups[c][:]
            rng.shuffle(g)
            sample.extend(g[:per])
        rng.shuffle(sample)
        sample = sample[: a.n]
    elif a.stratify:
        f1 = [x for x in data if x.get("fulfillable") == 1]
        f0 = [x for x in data if x.get("fulfillable") == 0]
        rng.shuffle(f1)
        rng.shuffle(f0)
        half = a.n // 2
        sample = f1[:half] + f0[: a.n - half]
    else:
        sample = data[:]
        rng.shuffle(sample)
        sample = sample[: a.n]
    rng.shuffle(sample)

    with open(a.out, "w") as out:
        for c in sample:
            out.write(json.dumps(to_task(c)) + "\n")

    n_benign = sum(1 for c in sample if c.get("fulfillable") == 1)
    print(f"Wrote {len(sample)} tasks to {a.out} "
          f"(seed={a.seed}, stratify={a.stratify}; "
          f"benign={n_benign}, unsafe={len(sample) - n_benign})")
    print("Reminder: single-step adaptation; disclose as a limitation.")


if __name__ == "__main__":
    main()
