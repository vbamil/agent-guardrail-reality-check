"""Structured runner: the agent emits a tool call (name + args), the guard inspects
the structured call, and the judge scores the resulting final action. This is the
mode used to test real guard methods (rule-based, reasoning-based, learned).

Usage:
  python -m src.run_structured --tasks data/asb_sample_s13.jsonl --guard passthrough
  python -m src.run_structured --tasks data/asb_sample_s13.jsonl --guard rule_based_ref

Note: rule_based_ref is a labeled REFERENCE guard, not AgentSpec. Do not report it
as a published method's result."""

import argparse
import datetime
import json
import os
import platform
from openai import APIError

from src.agent import propose_structured_action
from src.guards.structured import get_structured_guard, StructuredAction, GuardDecision, render_action
from src.scorer import score_action
from src.run_pilot import summarize
from config import AGENT_MODEL, JUDGE_MODEL, TEMPERATURE, TOP_P


def load_tasks(path, limit=None):
    tasks = []
    for line in open(path):
        line = line.strip()
        if line:
            tasks.append(json.loads(line))
    return tasks[:limit] if limit else tasks


def run(tasks_path, guard_name, limit, out_dir):
    guard = get_structured_guard(guard_name)
    tasks = load_tasks(tasks_path, limit)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ts}_structured_{guard_name}.jsonl")

    provenance = {
        "kind": "provenance", "mode": "structured", "timestamp": ts,
        "agent_model": AGENT_MODEL, "judge_model": JUDGE_MODEL,
        "guard": guard_name, "temperature": TEMPERATURE, "top_p": TOP_P,
        "tasks_file": tasks_path, "python": platform.python_version(),
        "note": "Single-step structured adaptation; disclose as a limitation.",
    }

    records = []
    with open(out_path, "w") as out:
        out.write(json.dumps(provenance) + "\n")
        for t in tasks:
            a = propose_structured_action(t["prompt"], t.get("tools", []))
            action = StructuredAction(a["refused"], a["tool"], a["args"], a["rationale"])
            guard_error_type = None
            guard_error_code = None
            try:
                decision = guard.review(t["prompt"], action)
                judged = score_action(t["prompt"], decision.final_action)
            except APIError as e:
                err = getattr(e, "body", {}) or {}
                guard_error_code = (err.get("error") or {}).get("code")
                guard_error_type = type(e).__name__
                decision = GuardDecision(
                    allow=None,
                    final_action="API_BLOCKED: guard request blocked by API policy.",
                    guard_name=guard_name,
                    note="api_blocked_guard",
                )
                judged = {
                    "label": "API_BLOCKED",
                    "rationale": "guard request failed or was blocked by API policy",
                    "error_type": guard_error_type,
                    "error_code": guard_error_code,
                }
            rec = {
                "kind": "result", "id": t.get("id"), "intent": t.get("intent"),
                "meta": t.get("meta"),
                "proposed_tool": action.tool, "proposed_args": action.args,
                "agent_refused": action.refused,
                "guard_allow": decision.allow,
                "final_action": decision.final_action,
                "guard_note": decision.note,
                "judge_label": judged["label"], "judge_rationale": judged["rationale"],
                "judge_error_type": judged.get("error_type"),
                "judge_error_code": judged.get("error_code"),
                "guard_error_type": guard_error_type,
                "guard_error_code": guard_error_code,
            }
            records.append(rec)
            out.write(json.dumps(rec) + "\n")
            print(f"[{rec['id']}] intent={rec['intent']} tool={action.tool} "
                  f"allow={rec['guard_allow']} judge={rec['judge_label']}")

    summarize(records)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True)
    p.add_argument("--guard", default="passthrough",
                   help="passthrough (baseline) or rule_based_ref (reference, NOT AgentSpec)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="runs")
    a = p.parse_args()
    run(a.tasks, a.guard, a.limit, a.out)
