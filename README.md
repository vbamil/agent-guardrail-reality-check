# Agent Guardrail Reality-Check: Pilot Harness

A minimal, honest starting point for an independent, standardized evaluation of
LLM-agent guardrail methods. This is **Phase 0 infrastructure**: it lets you run
the full pipeline (agent -> guard -> fixed scorer -> logged result) end to end on
a handful of built-in smoke-test tasks, using only an OpenAI API key and no GPU.
Once the pipeline works, you swap the smoke-test tasks for real benchmark data
and the demo guard for a real method.

## What this is and is not

- It **is** a working skeleton: a fixed LLM-judge scorer, an OpenAI agent runner,
  a guard interface, a no-op baseline guard, and a runner that scores and logs.
- It is **not** a finished experiment, and it ships with **no results**. Every
  number you eventually report must come from a real run you actually execute.
  See `PROVENANCE_AND_HONESTY.md`.

## Honest scope for a local machine + OpenAI + a few days

- **Base/agent model:** OpenAI (e.g. `gpt-4o-mini`). No local GPU needed.
- **Scorer:** a single fixed OpenAI LLM-judge with a frozen rubric (this repo).
  The Agent-SafetyBench official scorer (ShieldAgent) is a GPU model; using an
  LLM-judge instead is a deliberate, disclosable substitution. Validate it
  against a small hand-labeled sample (see `PROVENANCE_AND_HONESTY.md`).
- **Methods:** start with rule-based (AgentSpec) and/or reasoning-based
  (GuardAgent), which can use an OpenAI core. Defer ToolSafe/TS-Guard (trained
  model, needs a GPU).
- **Data:** sample a subset (e.g. 100-200 cases), not all 2,000.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...   # your key; never commit it
```

## Run the smoke test (no benchmark download needed)

```bash
python -m src.run_pilot --tasks data/sample_tasks.jsonl --limit 6
```

You should see each task: the agent's proposed action, the guard decision, the
judge's safe/unsafe label, and a final summary with a safety score and a benign
completion rate. Results are written to `runs/<timestamp>.jsonl`.

## Day 1: load a real Agent-SafetyBench sample

```bash
git clone https://github.com/thu-coai/Agent-SafetyBench   # MIT; cite Zhang et al. 2024
python src/datasets/load_asb.py \
  --src Agent-SafetyBench/data/released_data.json \
  --n 320 --by-risk --seed 42 --out data/asb_sample.jsonl
# baseline (no guard), then the demo guard, on real tasks:
python -m src.run_pilot --tasks data/asb_sample.jsonl --guard passthrough
```

Use `--by-risk` (even coverage across the 8 risk categories) for the baseline; it
is unbiased because it stratifies on a task property, not on whether the model
failed. The runner now prints a per-category safety breakdown and the count of
"live" unsafe cases (those not already handled safely), which is the set any real
guard can actually affect. Do NOT build samples by selecting tasks the base model
failed; that is selection-on-outcome and biases results.

Read the header of `src/datasets/load_asb.py` first. This is a single-step,
stated-action adaptation of the benchmark (the agent describes the action it would
take, rather than running the full multi-turn tool environment). It is GPU-free
and good for a pilot, but it is NOT the official protocol, so you cannot claim to
reproduce published Agent-SafetyBench scores, and you must disclose the adaptation.
The `intent` label is approximated from the `fulfillable` flag; the real fields
are preserved in each task's `meta`.

## Structured mode (for testing real guard methods)

The prose runner (`run_pilot`) was used to validate the scorer. To test real guard
methods, use the structured runner: the agent emits a tool call (name + args) the
guard can inspect.

```bash
# unguarded baseline (structured):
python -m src.run_structured --tasks data/asb_sample_s13.jsonl --guard passthrough
# reference rule-based guard (NOT AgentSpec; pipeline validation only):
python -m src.run_structured --tasks data/asb_sample_s13.jsonl --guard rule_based_ref
```

`rule_based_ref` is a labeled REFERENCE enforcer to confirm the harness can detect
a guard's effect. It is not AgentSpec and must never be reported as AgentSpec or as
a published method's result. Integrating the real AgentSpec means wiring in its own
rule engine from its repository. This structured mode is still a single-step
adaptation (one tool call per task, no multi-turn environment execution); disclose
that as a limitation.

## Phase plan

0. **Pipeline smoke test** (this kit). Prove agent -> guard -> scorer -> log works.
1. **Real scorer validation.** Hand-label ~50 actions; measure judge agreement.
2. **Baselines.** Run the agent with the no-op guard on a real task sample to get
   an unguarded safety/utility baseline.
3. **One method.** Plug AgentSpec or GuardAgent into `src/guards/` and run all
   conditions (replication, cross-benchmark transfer, model substitution).
4. **Expand** only to method/model/benchmark combinations that actually integrate;
   document what does not.

## Real data and methods (to plug in)

- Agent-SafetyBench: https://github.com/thu-coai/Agent-SafetyBench
  (data also on HF: `thu-coai/Agent-SafetyBench`; official scorer `thu-coai/ShieldAgent`)
- AgentHarm (transfer set): from Andriushchenko et al., 2024 (arXiv:2410.09024)
- AgentSpec: https://arxiv.org/abs/2503.18666
- GuardAgent: https://github.com/guardagent/code
- ToolSafe/TS-Guard: https://github.com/MurrayTom/ToolSafe (defer; needs GPU)

Verify each repo's license and intended use before integrating.
