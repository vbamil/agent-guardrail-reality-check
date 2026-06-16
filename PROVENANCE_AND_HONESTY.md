# Provenance and honesty notes

This study's credibility rests entirely on every reported number being a real,
recorded measurement. Read this before you collect anything you intend to report.

## Hard rules
1. No number goes in the paper unless it came from a run logged under `runs/`.
2. The frozen rubric in `config.py` must not change once data collection starts.
   If you change it, re-run everything; partial mixes are invalid.
3. Record, for every reported table: agent model, judge model, guard + version,
   benchmark + split, sample size, date, and the run file it came from.
4. If you substitute the LLM-judge for ShieldAgent, say so explicitly in the
   paper and report the judge's agreement with a hand-labeled sample.

## Scorer validation (do this before trusting any safety number)
1. Sample ~50 (task, final_action) pairs from a real baseline run.
2. Hand-label each SAFE/UNSAFE yourself, blind to the judge.
3. Compute agreement (and ideally Cohen's kappa) between you and the judge.
4. Report that agreement in the paper. If it is low, fix the rubric BEFORE
   collecting results, then re-validate.

## Things to disclose in the paper
- Base model is an OpenAI API model, not a local open model.
- Scorer is an LLM-judge, not the benchmark's native ShieldAgent scorer.
- Results are on a sampled subset of size N, not the full benchmark.
- Any method you could not integrate, and why (this is a legitimate finding).
- The TechRxiv preprint and the earlier desk-rejected submission, in the cover
  letter, if you submit to a venue.

## What "done" looks like for a credible pilot
- Unguarded baseline (passthrough) safety + utility on a real sample.
- At least one real method (AgentSpec or GuardAgent) on the same sample.
- A transfer condition (e.g., AgentHarm subset) under the same fixed judge.
- A validated scorer with reported agreement.
- A public repo containing the code and the run logs, with the link in the paper.
