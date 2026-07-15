---
name: mom6-gpu-architect
description: Map a MOM6 routine or module's structure and call graph, then design its GPU port — including whether the first step should be a bitwise-safe refactor (verbatim extraction into pure/elemental procedures) rather than attacking a 500-line monolithic loop directly. Use when picking up a new port target, asking "how should I port this", "what does this call", "is this portable as-is", "should I refactor first", "map out this module", or when sizing/sequencing porting work. Produces a port design; hand the mapping details to the gpu-data-residency skill and the mechanics to KNOWLEDGE.md §3.
---

# MOM6 GPU architect: map the routine, then design the port

Produces a **port design** for one routine or module: a call map, a shape verdict, an explicit
refactor-or-port-in-place decision, a loop-form plan, and a sequenced work plan with gates.

**This skill decides *what* to do and in *what order*. It does not restate *how*.** Hand off:

| Question | Goes to |
|---|---|
| the porting mechanics (k-blocking, loop forms, verification) | `knowledge/KNOWLEDGE.md` §3, §4 |
| where to map arrays / where to copy back | the **`gpu-data-residency`** skill |
| device-callable helper rules | `knowledge/KNOWLEDGE.md` §3 Step 4, doc 08 |
| is this bitwise-safe? | `knowledge/KNOWLEDGE.md` §7.2, doc 07 |

## Step 1 — Orient before measuring

1. Read `knowledge/KNOWLEDGE.md` §6 (work queue) — **is this target already in flight?** §2.4 lists branches
   carrying groundwork (j-blocking, `declare target` prep). Build on it; do not restart. Several
   Tier-1 items have branches that already answer half the design.
2. Read §6's dependency ordering. Tier 1 is not a menu: **EOS `_loc` coverage → N²/density inputs
   → `set_diffusivity` → KPP/EPBL → `kappa_shear`**. Designing `set_diffusivity` before the EOS
   chain exists is designing on sand.
3. Note the module's status in §2.4 (ported / in-flight / untouched).

## Step 2 — Measure the shape

```bash
scripts/routine-map.sh <file.F90>
```

One row per procedure, sorted by **MAXLOOP** — the longest single loop body. That is the metric
that matters. A 600-line routine of small loops ports fine; a 600-line routine whose *one* loop
body is 450 lines is the monolith problem, and `LINES` alone cannot tell them apart.

Calibration from this tree (`references/calibration.md` has the full table and how to regenerate):

| Routine | LINES | MAXLOOP | Verdict |
|---|---|---|---|
| `find_uv_at_h` | 123 | 65 | **the blessed template** — ported as-is |
| `set_diffusivity` | 624 | 269 | refactor first |
| `KPP_compute_BLD` | 522 | 369 | research-grade |
| `applyBoundaryFluxesInOut` | 664 | 452 | refactor first |
| `ePBL_column` | 1061 | 701 | naive port only (see below) |

## Step 3 — Map the callees (the triage input)

```bash
scripts/callees.sh <file.F90> <routine-name>
```

**`MAXDEP` is the design-critical column.** A callee at `MAXDEP 0` is host orchestration outside
the loops — free, irrelevant to the port. A callee at `MAXDEP > 0` is *inside the loop nest*: to
port that loop, it must become device-callable, or be hoisted out first.

So: **the loop-interior call list is the triage.** The blessed template has an empty one —

```
find_uv_at_h:              every callee MAXDEP 0   -> portable as-is
applyBoundaryFluxesInOut:  mom_error MAXDEP 4, forcing_SinglePointPrint MAXDEP 4,
                           post_data MAXDEP 1, 2 generic interfaces MAXDEP 2  -> refactor first
```

`callees.sh` only catches `call` statements — Fortran function references are indistinguishable
from array indexing without a symbol table. **Read the loop bodies for function references**
(EOS elementals, `ratio_max`, `cuberoot`); they carry the same constraint.

## Step 4 — The refactor triage

Classify every callee with `MAXDEP > 0`:

| Class | Consequence | Action before porting |
|---|---|---|
| **HOST-SINK** (`MOM_error`, `post_data`, chksum, halo, I/O) | **Blocks the port outright** — never-do #10: no allocate, I/O or `post_data` in a device loop | Hoist out, or return status via `intent(out)` flags and fold host-side after (the `efp_decompose` model) |
| **GENERIC-INTERFACE** / `class(...)` dispatch | Device-fatal or silently wrong (doc 06; never-do #2) | Port the `_loc` free function first (doc 06 §6.3) |
| **External pkg** (`cvmix_*`, FMS) | Largest `declare target` surface; may be unportable | Size it *before* committing to the port |
| **In-file, not `pure`** | Silent wrong numbers if it fails to inline (`3cb184edd`) | Make `pure` + `declare target`, or `FORCEINLINE` |
| **In-file, already `pure`** | Cheap | `declare target` |
| *(list is empty)* | — | **Port in place** — go to `knowledge/KNOWLEDGE.md` §3 |

**Refactor first if any of:** a HOST-SINK is called inside the loop nest; the loop body writes
module state (blocks `pure`); polymorphic dispatch happens inside the loop; or MAXLOOP is large
*and* the body mixes ≥2 distinct concerns. Otherwise port in place.

Treat the thresholds as calibration, not law. The judgement is **"can one reviewer hold this loop
body and its bitwise argument in their head at once?"** — 65 lines with no interior calls, yes;
452 lines with `MOM_error` at depth 4, no.

The counter-example that keeps this honest: **`ePBL_column`** (1061/701) has 8 loop-interior
callees but *zero* host sinks and *zero* polymorphic dispatch — all in-file, all extractable. That
self-contained call graph is why `origin/epbl-3d`'s naive whole-column `pure` + `do concurrent`
worked at all ("100x… ~2ms/step"). Shape alone would have condemned it. It still needed manual
inlining of `get_Langmuir_Number` and `find_mstar` — **the exact two callees `callees.sh` flags at
MAXDEP 2** — and it has no CPU-preserving story, so it fails ground rule 2. Big MAXLOOP means
*"read the call list before judging"*, not *"refuse"*.

## Step 5 — The only legal refactor: verbatim extraction

Ground rule 1 forbids reordering floating-point arithmetic. So "refactor to make porting easier"
has exactly **one** blessed form (doc 08 §4): **extract the innermost side-effect-free arithmetic
into a `pure`/`elemental` procedure, moving code, never reordering it.**

1. Find the innermost side-effect-free arithmetic span inside the hot loop.
2. Move it **verbatim** into a new `pure`/`elemental` procedure. Copy expressions
   character-for-character. Do not "tidy" parentheses — Fortran parens pin evaluation order.
3. Route side effects out as `intent(out)` flags; the host caller folds them into module state
   *after* the loop. This is exactly why `efp_decompose` is `pure`: it reports `is_nan`/`is_ovf`
   rather than touching the module error flags (`MOM_coms.F90:779`). It is also the answer to a
   `MOM_error(FATAL)` at depth 4.
4. If callers need the old API, keep the original as a **thin wrapper** delegating to the new
   procedure — the EOS `_loc` model (`52a1b3954`).
5. Add `!$omp declare target` after all declarations, before the first executable.
6. **Gate the refactor on CPU alone, bitwise, as its own commit — before any directive exists.**

Step 6 is the whole argument for refactoring first. A verbatim extraction **must** be bit-identical
on CPU; if it isn't, the extraction wasn't verbatim, and you learn that from a cheap CPU run
instead of from a GPU checksum mismatch tangled up with mapping bugs. It splits a scary 664-line
port into two separately-auditable diffs:

```
commit 1: verbatim extraction, zero directives   -> gate: CPU bitwise identical
commit 2: the port (KNOWLEDGE.md §3)             -> gate: §3 Step 9 (GPU bitwise + ≥2 GPUs)
```

**Not refactors — these change bits and are forbidden** (§7.2): re-associating arithmetic;
distributing parentheses; splitting a producer loop from the reduction that consumes it
(`5f413739b` — even ifort re-associates, even at `-O0`); fusing loops that don't meet doc 05 §4's
conditions. And note declaration order of large stack arrays measurably moves CPU performance
(`28eb296f4`, "Move with caution!") — reordering declarations is not free either.

## Step 6 — Hazard audit

Run `knowledge/KNOWLEDGE.md` §3 Step 2 against the module — do not re-derive it here. It covers pointer
members and `associated()` flow, restart-registered fields, EOS forms exercised, recurrences, halo
calls, diagnostics, and float reductions. Two that most often change the *design* rather than the
code:

- **Recurrences** (`x(k)` depends on `x(k±1)`): these pick the loop form for you (§4.1 branch 5,
  teams-loop + serial k) and disqualify k-blocking outright (doc 05 §7.0).
- **EOS forms**: only buggy-Wright and Roquet_rho have GPU-safe direct kernels. The **default
  `WRIGHT_FULL` is still polymorphic and device-fatal**. If your target needs another form on
  device, that form is a prerequisite port, not a detail.

## Step 7 — Loop-form plan

For each loop, take the first matching branch of the `knowledge/KNOWLEDGE.md` §4.1 decision tree and record
the choice plus its reason. Do not invent forms. Note where k-blocking applies (§3 Step 3) and
whether the module needs its own `nkblock` CS parameter.

## Step 8 — Emit the design

Lead with the verdict — **port in place** or **refactor first** — and the one fact that decided it.
Then:

1. **Shape** — LINES/MAXLOOP table for the target routines.
2. **Call map** — loop-interior callees only, each classified per Step 4, each with its action.
3. **Verdict + reasoning**, naming the blocking callee or hazard if refactoring.
4. **Refactor plan** (if any) — what gets extracted verbatim, what flags replace what side effects,
   what the CPU gate is.
5. **Port plan** — loop-form per loop; k-blocking yes/no; data-mapping handed to
   `gpu-data-residency`; halo strategy (§3 Step 7).
6. **Prerequisites** — EOS forms, in-flight branches to merge first, dependency-order items.
7. **Sequenced commits, each with its gate.**
8. **Open questions** — things needing a run, a profile, or a maintainer decision. Say so rather
   than guessing; §8a/§8b show which questions source-reading genuinely cannot close.

## Anti-patterns

- **Designing against `LINES`.** MAXLOOP and the loop-interior call list decide; total length does not.
- **"Refactor" that isn't verbatim.** If you retyped an expression, you changed it. Copy it.
- **Skipping §2.4.** Restarting work an in-flight branch already did is the most expensive mistake available.
- **Porting a routine whose EOS form isn't ported.** The prerequisite is the work.
- **Naive-vs-blessed by default.** `epbl-3d` is fast and GPU-only; ground rule 2 requires one source
  form serving both targets. Choose explicitly and say which you chose (doc 10's 8-gate rubric).
- **Treating a big MAXLOOP as a refusal.** Read the call list first — `ePBL_column` is the lesson.
