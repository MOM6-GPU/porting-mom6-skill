---
name: mom6-gpu-programmer
description: End-to-end driver for MOM6 GPU porting work. Takes a routine or module from "should we port this?" through design, bitwise-safe refactor, port, verification and merge review — or evaluates an existing port, branch or diff. Use when asked to "port <routine>", "start a port", "what would it take to port X", "evaluate/audit this port", "review this GPU diff", "is this port ready to merge", or when debugging wrong GPU answers. Orchestrates the mom6-gpu-architect and gpu-data-residency skills and owns the gates between them.
---

# MOM6 GPU programmer

The driver for a whole piece of GPU porting work. It **sequences** the other skills and **owns the
gates between them**; it does not restate what they say.

| Need | Where it lives |
|---|---|
| shape, call map, refactor-or-port triage | the **`mom6-gpu-architect`** skill |
| where to map arrays, where to copy back | the **`gpu-data-residency`** skill |
| porting mechanics (k-blocking, loop forms, halos, verification) | `knowledge/KNOWLEDGE.md` §3, §4 |
| symptom → fix | `knowledge/KNOWLEDGE.md` §5 |
| merge review rubric (8 gates) | `knowledge/gpu-knowledge/10-inflight-ports.md:455` |

Invoke a child skill with the **Skill tool** at the phase that needs it (`mom6-gpu-architect` at
Phase 1, `gpu-data-residency` at Phase 3). They are separate skills, so calling them from here is
fine: the Skill tool's only stated prohibition is re-invoking a skill that is *already running* —
parent→child yes, self-recursion no.

Two practical notes:

- **Invoke at the phase that needs it, not all up front.** A child's full SKILL.md enters the
  conversation on invocation and stays there for the session. Phase 1 and Phase 3 are far apart;
  loading both at the start just burns context.
- If a child's content is already in this conversation, **use it directly — do not re-invoke**.
  Nested skill-to-skill invocation is not spelled out in the public Skills documentation, so if it
  ever misbehaves the fallback is identical in effect: `Read skills/<child>/SKILL.md`.

The user can also chain these directly from the command line (`/mom6-gpu-architect /gpu-data-residency
<task>`) — this skill is for when the *sequencing and the gates between* the phases are the point.

## Prime directive: never claim a gate you did not run

Ground rule 6 (`KNOWLEDGE.md:60`): **study source + git only unless explicitly told to build.**
The acceptance gate (§3 Step 9) requires a build *and* a ≥2-GPU run.

Those two facts collide, and the collision is the most important thing about this skill:

> **By default you can produce a port. You cannot accept one.**

Check the boundary before promising anything: `which nvfortran mpirun`. On a typical session here
they are absent — Phase 4 is then a **hand-back**, not a step you perform.

So: never write "verified", "bitwise-identical", "confirmed", or "passes" about a run you did not
execute. Write **"unverified — here is the exact experiment"** and name the config, the ranks, and
the fields to checksum. `knowledge/KNOWLEDGE.md` §8a's OPEN list is open precisely because source-reading
cannot close those questions; quietly adding false certainty to that pile is the worst available
contribution. §8a's own header models the standard: *"source+git only; no builds/runs."*

## Mode: PORT

### Phase 0 — Orient  ⟨STOP-GATE: is this even the right target?⟩

1. Read `knowledge/KNOWLEDGE.md` (§2 architecture, §6 work queue) if not already in context.
2. **In-flight check** (§2.4): is there a branch carrying groundwork? If yes → **STOP and report**.
   Restarting work a branch already did is the most expensive mistake available.
3. **Dependency check** (§6): Tier 1 is ordered — EOS `_loc` coverage → N²/density inputs →
   `set_diffusivity` → KPP/EPBL → `kappa_shear`. If a prerequisite is unported → **STOP and report
   the chain**, don't design on sand.
4. **EOS check**: does the target need a form on device other than buggy-Wright / Roquet_rho? The
   default `WRIGHT_FULL` is still polymorphic and device-fatal. That form is then the real work item.

### Phase 1 — Design  ⟨STOP-GATE: verdict recorded⟩

Invoke **`mom6-gpu-architect`**. It returns: shape (LINES/MAXLOOP), the loop-interior call list,
and a **port-in-place vs refactor-first** verdict.

If the verdict is refactor-first and the extraction is substantial → **STOP and confirm scope with
the user** before writing code. A refactor is a separate piece of work with its own risk.

### Phase 2 — Refactor (only if triaged)  ⟨GATE: CPU bitwise identical⟩

The only legal refactor is **verbatim extraction** into `pure`/`elemental` (doc 08 §4) — move code,
never reorder it. Recipe in the architect skill, Step 5.

**This is its own commit, with zero directives in it, gated on CPU alone.** A verbatim extraction
*must* be bit-identical on CPU; if it isn't, the extraction wasn't verbatim — and you learn that
from a cheap CPU run instead of from a GPU checksum mismatch tangled up with mapping bugs.

If you cannot run the CPU gate → the extraction diff is still deliverable, but it is **unverified**.
Say so and stop; do not stack the port on top of an unverified refactor.

### Phase 3 — Port

Follow `knowledge/KNOWLEDGE.md` §3 Steps 3–8 for the mechanics. For **every data-mapping decision** — where
`enter data` goes, `to` vs `alloc`, which host consumer forces an `update from` — invoke
**`gpu-data-residency`** rather than improvising. Mapping bugs are the silent class.

Keep the §7.2 never-do list open while writing. The ones that bite during a port:
no early `exit`/`return`/`cycle` in device loops; no shared-scalar writes without `reduce`;
no `map(delete:)` on an object this scope doesn't own; every `enter data` gets a mirrored `exit data`.

### Phase 4 — Verify  ⟨GATE: §3 Step 9 — usually a hand-back⟩

The gate is: bit-identical `MOM_checksums` field checksums + EFP `write_energy`, block-size
invariance (`nkblock` 0/1/nz agree), and **≥2 ranks/GPUs** — single-GPU correctness does not prove
a port; the `alloc`-vs-`to` and missing-`reduce` bugs are latent until multi-device.

If you cannot build: produce the diff plus **the experiment** — exact config, rank count, fields to
checksum, and what a failure would mean. Then **STOP and hand back**. This is a legitimate,
expected outcome, not a failure.

### Phase 5 — Merge review  ⟨GATE: all 8⟩

Run doc 10's 8-gate rubric (`10-inflight-ports.md:455-508`) against your own diff *verbatim* —
don't paraphrase it, it cites the evidence for each gate. Summary of what it checks: block-size CS
params `#ifdef`-gated; one persistent data region per hot path; `do concurrent` as the default
idiom; EOS via the 2D/3D `_loc` interface with the v-table resolved host-side; an explicit bitwise
argument naming which runtime path you hit; no debug prints; CPU defaults actually benchmarked (not
copy-pasted); k-recurrences nested `do concurrent(j) → serial do k → do concurrent(i)`.

Gate 7 (CPU tuning benchmarked) is another one you probably cannot close in-session. Say so.

## Mode: EVALUATE

For "audit this port", "is this ready", "review this diff", or "why are the answers wrong".

1. **Scope it** — a diff, a branch, a module, or a symptom. For a branch: `git diff dev-gfdl...<branch>`.
2. **If a symptom is reported**, go to `knowledge/KNOWLEDGE.md` §5's symptom index *first*, and check **row 0
   first, always**: your own diff (missing map, misplaced accumulation, unbalanced enter/exit)
   before blaming nvfortran. Rows marked `SILENT` are the dangerous class.
3. **Map/transfer audit** → invoke **`gpu-data-residency`** (its Step 5 balance checks and Step 6
   report format).
4. **Shape/call audit** (if the question is "how hard is this?") → invoke **`mom6-gpu-architect`**.
5. **Merge readiness** → doc 10's 8 gates.
6. **Report** with findings separated into **confirmed** (both sites read, branch structure checked),
   **suspected** (needs a run), and **needs a maintainer decision**. Give each finding: the two
   sites, the predicate that reaches the bad path, the symptom it produces, and the one-line fix.

Known latent items worth checking against before reporting something as new: §8b finding A (six
early-`exit`-under-DC sites still at HEAD), finding B (the `ADp` mapping lifecycle), and
`gpu-data-residency`'s `khdt_x` worked example.

## Stop and ask — do not push through

- An in-flight branch already covers the target (§2.4).
- A prerequisite is unported (EOS form, density inputs).
- The triage says refactor-first and the extraction is large.
- **Naive vs blessed**: ground rule 2 requires one source form serving CPU and GPU. `epbl-3d` is
  fast and GPU-only. That trade is a maintainer decision — surface it, don't make it.
- A gate needs a build/run you cannot do.
- **You find a bug in existing code.** Report it with its reaching predicate; do not silently fix it
  inside an unrelated port — it belongs in its own diff with its own gate.
- The question is genuinely unclosable from source (the §8a OPEN shape).

Don't commit or push unless asked. When you do, the sequence is: refactor commit (CPU gate) →
port commit (GPU gate) — separately auditable, never squashed together.

## Never

1. Claim a verification you didn't run, or let "compiles" stand in for "bitwise".
2. Accept a nonzero checksum diff as rounding. One differing bit means the port is wrong.
3. Reorder floating-point arithmetic — including "tidying" parentheses (§7.2 #1).
4. Port past a `STOP-GATE` because the next phase looks easy.
5. Restate the child skills' content here instead of invoking them.
6. Write outside the repo; temp artifacts → `tmp_local_artifacts/`.
