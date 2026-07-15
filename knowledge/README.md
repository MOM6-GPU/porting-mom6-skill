# MOM6 GPU knowledge base

The knowledge the `skills/` in this repo read from. `KNOWLEDGE.md` is the operational playbook
(porting procedure, decision rules, symptom→fix index, prioritized work queue, proven-works and
never-do lists); `gpu-knowledge/00-14` are the deep-dive docs behind it, plus standalone bug notes.

Read `KNOWLEDGE.md` first — it is self-contained, and every load-bearing rule is inlined with its
citation so you can act without opening the deep docs.

## Provenance — read this before trusting a line number

The knowledge base cites MOM6 source by `file:line`, plus commit hashes and branch names. Those are
only meaningful against a specific tree:

| | |
|---|---|
| `file:line` refs valid against | `dev/gpu` @ **`c82e1254a`** |
| upstream baseline | `dev-gfdl` @ `c3237e27f` |
| built | 2026-07-14, from source + git only — no builds or runs |

**Line numbers rot**, and they rot invisibly here: there is no MOM6 source in this repo to
contradict a stale reference. Verify any anchor before acting on it. Commit hashes and branch names
do not rot — prefer them where both are available.

## The confidence markers are load-bearing

`KNOWLEDGE.md` §8a and §8b record, per claim, which were **verified from source**, which are
**advanced but need a run or a profile to close**, and which are **open maintainer decisions**. The
same distinction appears in the bug notes. That grading is the most valuable thing in here — please
preserve it when editing rather than flattening everything to assertion.

Several findings are explicitly **unconfirmed** and say so. They are recorded because they are
worth checking, not because they are established.

## Status

Current best practice, not a finished spec. The port is a work in progress and only the code paths
exercised by the `benchmark` / `benchmark_ALE` configurations have been validated. When you hit a
pattern these docs do not cover, make a decision consistent with the philosophy, flag it, and add it
here once it is confirmed.
