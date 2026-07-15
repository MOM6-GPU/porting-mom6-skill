# Worked example: `khdt_x` in `tracer_hordiff`

`src/tracer/MOM_tracer_hor_diff.F90`, subroutine `tracer_hordiff` (from `:122`). This one array
exercises every state transition, every classification trap, and the highest-yield bug shape. Work
through it once before running the procedure on something new.

Raw material:

```bash
skills/gpu-data-residency/scripts/residency-scan.sh \
  src/tracer/MOM_tracer_hor_diff.F90 khdt_x
```

## Step 1 — Identity

`khdt_x` is declared at `:162` as **subroutine-scope scratch** (a local automatic, `Khtr*dt` times
face width). So its map belongs at routine entry / return, not in a `_init`/`_end`. It is:

- `:209` `enter data map(alloc: khdt_x, khdt_y, kh_u, kh_v)` — entry, `alloc`
- `:723` `exit data map(release: khdt_x, khdt_y, Kh_u, Kh_v)` — return, `release` not `delete` ✓

Both correct: `alloc` because the first device touch is a write, `release` because this scope's map
is per-call and `delete` would zero the refcount on anything an outer scope had mapped.

## Step 2 — The ledger

Four mutually exclusive branches produce `khdt_x`. Walk each to the host read at `:394`.

| # | Line | Event | H/D | State after | Verdict |
|---|---|---|---|---|---|
| 1 | `:209` | `map(alloc: khdt_x)` | — | `HOST_FRESH` (garbage) | ok |
| | | **Branch A — `do_online .and. use_VarMix`** | | | |
| 2 | `:290` | `do concurrent (j,I)`, `khdt_x(I,j) = ...` | **DEV** | `DEV_FRESH` | ok |
| 3 | `:338` | `if (CS%max_diff_CFL > 0.0)` | — | — | ⚠ predicate |
| 4 | `:339` | `update from(khdt_x, ...)` | — | `SYNCED` | **only inside #3** |
| 5 | `:341` | `!$OMP parallel do` limiter, reads+writes `khdt_x` | **HOST** | `HOST_FRESH` | ok (dominated by #4) |
| 6 | `:374` | `update to(khdt_x, ...)` | — | `SYNCED` | ok |
| | | **Branch B — `Resoln_scaled`** | | | |
| 7 | `:297` | `!$OMP parallel do`, writes `khdt_x` | **HOST** | `HOST_FRESH` | ok |
| 8 | `:309` | `update to(khdt_x, ...)` | — | `SYNCED` | ok |
| | | **Branch C — constant diffusivity** | | | |
| 9 | `:312-333` | `!$OMP parallel do`, writes `khdt_x` | **HOST** | `HOST_FRESH` | ok |
| 10 | `:335` | `update to(khdt_x, ...)` | — | `SYNCED` | ok |
| | | **Branch D — `.not. do_online`** | | | |
| 11 | `:378` | `!$OMP parallel do`, `khdt_x = read_khdt_x` | **HOST** | `HOST_FRESH` | ok |
| 12 | `:386` | `call pass_vector(khdt_x, khdt_y, ...)` | **HOST-SINK** | `HOST_FRESH` | ok — host array, host halo |
| 13 | `:387` | `update to(khdt_x, ...)` | — | `SYNCED` | ok — refresh after host halo |
| | | **Join** | | | |
| 14 | `:390` | `if (CS%check_diffusive_CFL)` | — | — | ⚠ predicate |
| 15 | `:394` | `CFL(i,j) = 2.0*((khdt_x(I-1,j) + ...` in a plain `do` | **HOST** | — | **BUG on branch A** |
| 16 | `:438` | `update from(khdt_x, khdt_y)` under `use_hor_bnd_diffusion` | — | `SYNCED` | ok — dominates `:441+` |
| 17 | `:722` | `update from(khdt_x, khdt_y) if (CS%debug .or. CS%id_khdt_x>0 .or. ...)` | — | `SYNCED` | ok — the decoupled-transfer idiom |
| 18 | `:726/:731` | `uvchksum` / `post_data(CS%id_khdt_x, ...)` | **HOST-SINK** | — | ok (dominated by #17, guard matches) |

## Step 3 — The finding

Rows 2 → 15. On **branch A only**, `khdt_x` is written on device and the host copy is never
written at all. The copy-back at `:339` is guarded by `CS%max_diff_CFL > 0.0`; the host read at
`:394` is guarded by `CS%check_diffusive_CFL`. **These are independent runtime parameters** — the
transfer's predicate does not dominate the read's:

```
CHECK_DIFFUSIVE_CFL   default .false.   (:1745)
MAX_TR_DIFFUSION_CFL  default -1.0      (:1750)
```

Reaching predicate for the stale read:

```
do_online .and. use_VarMix .and. .not.(CS%max_diff_CFL > 0.0) .and. CS%check_diffusive_CFL
```

i.e. `CHECK_DIFFUSIVE_CFL = True` with the **default** `MAX_TR_DIFFUSION_CFL`, and variable mixing
on — a natural configuration ("iterate to respect the CFL limit" without local diffusivity
limiting), not an exotic one.

Consequence: `max_CFL` is computed from an uninitialized host `khdt_x`, `max_across_PEs` at `:399`
spreads it, and `num_itts = max(1, ceiling(max_CFL - ...))` at `:401` picks a garbage iteration
count for the tracer diffusion — wrong answers, or a huge `num_itts` and a hang, depending on what
the uninitialized memory holds. `CFL` is also posted as a diagnostic at `:403`.

Note the symptom this would *not* produce: it is invisible to a default-config checksum run, and
invisible on branches B/C/D. That is why the ledger is worth writing down rather than eyeballed.

One-line fix — move the copy-back to the consumer's predicate:

```fortran
  if (CS%check_diffusive_CFL) then
    !$omp target update from(khdt_x, khdt_y)      ! <-- add
    if (CS%show_call_tree) call callTree_waypoint("Checking diffusive CFL (tracer_hordiff)")
```

(Redundant with `:339` when both predicates hold — a second `update from` of a `SYNCED` array is a
wasted copy, not a bug. If that matters, guard it `.and. .not.(CS%max_diff_CFL > 0.0)`, at the cost
of a predicate that has to be maintained in lockstep with `:338`. Prefer the simple version.)

## What this example teaches

1. **Branches are the analysis.** A device write and a host read in the same routine with a
   transfer *somewhere* between them proves nothing. Only a transfer whose predicate is implied by
   the read's predicate dominates it.
2. **`!$OMP parallel do` is a host loop.** Eight of them here, interleaved with `do concurrent`
   device regions in the same `if/elseif` chain. Rows 5, 7, 9, 11 are host; row 2 is device. Read
   the directive, not the indentation.
3. **`map(alloc:)` means the host copy is garbage** until someone writes it. Branch A never does.
4. **The right idiom is right there in the same file**, at `:722` — one guarded `update from`
   whose condition is the disjunction of every consumer's condition
   (`CS%debug .or. CS%id_khdt_x>0 .or. CS%id_khdt_y>0`), then the individual consumers each behind
   their own guard. Copy that shape.
5. `pass_vector` at `:386` is a **host** halo exchange on a host-fresh array, correctly followed by
   `update to`. Compare `do_group_pass(..., omp_offload=.true.)`, which needs no transfer at all.

## Status

Reported 2026-07-14, found by running this skill's procedure while writing it. Source-only analysis
— not yet confirmed by a run. Confirming it needs a `CHECK_DIFFUSIVE_CFL=True` +
`MAX_TR_DIFFUSION_CFL=-1` + VarMix case; the cheap tell is a nondeterministic `num_itts` /
`CFL` diagnostic on GPU vs CPU.
