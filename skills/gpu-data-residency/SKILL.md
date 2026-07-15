---
name: gpu-data-residency
description: Decide where `!$omp target enter data map(...)` belongs for a MOM6 array, choose `map(to:)` vs `map(alloc:)`, and find every place a device-resident array must be copied back to the host (`target update from`) or refreshed on device (`target update to`). Use when porting a module to GPU, adding a device-resident array, auditing a port for missing transfers or unbalanced enter/exit data, or debugging a checksum mismatch, answers-differ-by-GPU-count, or stale-host symptom. Triggers on "where do I map this", "enter data", "exit data", "update from", "copy back", "needed on the host", "residency", "stale host", "map(to) vs map(alloc)", "missing transfer", "map balance".
---

# GPU data residency: where to map, and where to copy back

Answers two questions for a given array (or every array in a module):

1. **Where does its `enter data`/`exit data` go, and with which map kind?**
2. **Where must it be copied back** (`update from`) **or refreshed** (`update to`) **because a host-only consumer touches it?**

Read `knowledge/KNOWLEDGE.md` §3 Step 5 (mapping lifecycle) and §3 Step 8 (transfer discipline) if not already in context.
Depth: `knowledge/gpu-knowledge/03-openmp-mapping.md`, `12-diagnostics-io.md`, `02-pointer-usage.md`.

## The engine: every mapped array is a two-copy shadow state

Track **coherence between the host copy and the device copy** as you walk the code in execution
order. Almost every mapping bug in this tree is a state-machine violation.

| State | Host copy | Device copy |
|---|---|---|
| `UNMAPPED` | authoritative | does not exist |
| `SYNCED` | valid | valid |
| `HOST_FRESH` | authoritative | **stale / garbage** |
| `DEV_FRESH` | **stale / garbage** | authoritative |

| Event | Resulting state | Note |
|---|---|---|
| `enter data map(to: x)` | `SYNCED` | **Only if not already present.** On an already-present object this is a refcount bump and **copies nothing** — state unchanged (§8b finding C). |
| `enter data map(alloc: x)` | `HOST_FRESH` | Device side is garbage. Legal only if the next device touch is a *write*. |
| host write | `HOST_FRESH` | includes `!$OMP parallel do` loops — those are **host** CPU threads |
| device write | `DEV_FRESH` | |
| **host read while `DEV_FRESH`** | **BUG** | insert `!$omp target update from(x)` before it |
| **device read while `HOST_FRESH`** | **BUG** | insert `!$omp target update to(x)` before it |
| `update from(x)` | `SYNCED` | device → host |
| `update to(x)` | `SYNCED` | host → device |
| `exit data map(from: x)` | `UNMAPPED`, host valid | copies back |
| `exit data map(delete:/release: x)` | `UNMAPPED`, host **as it was** | **neither kind copies back** |

The machine tracks *coherence*, not *initialization*: `HOST_FRESH` on a freshly-`map(alloc:)`'d
local automatic array means "host is authoritative and holds garbage". Both reads are still wrong;
flag them separately.

Two rules the mechanical walk will not derive on its own:

- **`delete` vs `release`** — `delete` forces the refcount to **zero**, destroying any *outer*
  persistent mapping of the same object. Use `release` for scoped/per-call teardown; `delete` only
  in the `*_end` that mirrors the owning `enter data`. Never `map(delete:)` an object your scope
  does not own (`MOM_vert_friction.F90:1105` silently kills `MOM.F90:3190`'s map — §8b finding B).
- **Re-mapping never refreshes.** If host scalars/descriptors changed after the first map, the only
  refresh is `target update to(...)`. Never "re-map to refresh"; never re-`enter data` a parent
  struct after its members are attached (`c82e1254a`).

## Procedure

### Step 1 — Scope and inventory

Pick the array(s). For each, get every touch site interleaved with every region marker, in line
order:

```bash
scripts/residency-scan.sh <file.F90> <array-name>
```

It tags each line `MAP` / `XFER` / `DEV-REGION` / `DEV-HALO` / `HOST-THREADS` / `HOST-SINK` /
`TOUCH`. It is a *reading aid*, not an oracle — it tags the lines that open regions, and you still
have to read the code to see which touches fall inside them.

Also establish the array's identity, which fixes where the map goes:

| Kind | `enter data` site | `exit data` site |
|---|---|---|
| CS member (`ALLOCABLE_`) | in `<mod>_init`, next to `ALLOC_`, after `... = 0.0` | in `<mod>_end`, next to `DEALLOC_`, mirrored member-by-member, `delete` |
| Subroutine-scope scratch | at routine entry, `map(alloc:)` | at return, `release` (early-release once last use passes is fine) |
| Dummy argument | **not here** — the caller owns it; verify residency at every call site | — |
| Pointer member | `map(to:)` **never `alloc`**, guarded `if (associated(x))` | mirrored, same guard |

### Step 2 — Classify every touch HOST or DEVICE

This is where the analysis is won or lost. Reason about the **GPU build**
(`__NVCOMPILER_OPENMP_GPU`).

| Marker | Verdict |
|---|---|
| `do concurrent (...)` | **DEVICE** — the default compute idiom |
| `!$omp target teams` / `!$omp target ... loop` | **DEVICE** |
| `!$OMP parallel do` / `!$omp parallel` | **HOST** — CPU threads. *The single most common misread.* |
| plain `do` loop | **HOST** |
| plain `do` loop **inside** a `target teams loop` / DC | **DEVICE** (serial-k columns) |
| call to a `pure`/`elemental`/`declare target` helper from a device region | **DEVICE** |
| any other `call` | **HOST** unless proven otherwise |

Then check the touch against the **host-boundary catalogue** — the calls that are host-only and
therefore force a copy-back. See `references/host-boundaries.md` for the full list with its
verification commands. The load-bearing ones:

- `post_data` and the whole diag mediator; `hchksum`/`uvchksum`/... ; `save_restart`
- `pass_var`/`pass_vector` (no `omp_offload` argument exists); `start_group_pass`/`complete_group_pass`
- `do_group_pass(..., omp_offload=.true.)` is **DEVICE** — no transfer. Without the flag: HOST.
- any call into an untouched module (diabatic stack, ALE remap/regrid, restart — `knowledge/KNOWLEDGE.md` §2.4)

### Step 3 — Walk the ledger

In execution order, one row per event. **Branches matter more than anything else here**: a transfer
inside `if (cond)` does not dominate a read outside it. When a device write and a host read sit in
sibling branches, write down the predicate that reaches the read without the transfer — that
conjunction *is* the bug report.

| # | Line | Event | Host/Dev | State after | Verdict |
|---|---|---|---|---|---|
| 1 | `:209` | `map(alloc: khdt_x)` | — | `HOST_FRESH` (garbage) | ok |
| 2 | `:291` | write in `do concurrent` | DEV | `DEV_FRESH` | ok |
| 3 | `:394` | read in plain `do` | HOST | — | **BUG: needs `update from`** |

### Step 4 — Emit directives

Map kind, from the *first device touch* and who else reads the contents:

- first device touch is a **read**, or any host-set scalar/pointer descriptor is read on device
  → **`map(to:)`**. This includes every struct whose `associated()` state feeds device control flow
  (`map(alloc: Reg, Reg%Tr(:))` was the multi-GPU answer-change bug, `a774eb331`).
- first device touch is a **write**, pure workspace → **`map(alloc:)`**.
- CS shells → `map(alloc:)`, mapped **once**, before the child `_init`.

Transfer placement:

- Put the transfer at the **producer**, immediately upstream of the consumer. A transfer of a
  *different* array does not cover yours (`b29b27150`).
- **Decouple transfer from post**: one `update from` covering all consumers, guarded
  `if (CS%debug .or. CS%id_a>0 .or. CS%id_b>0)`, then the individual `if (id>0) call post_data(...)`
  (`MOM_diagnostics.F90:1825-1827`).
- Bracket an unavoidable host-only detour both ways: `from(...)` before, `to(...)` after
  (ALE at `MOM.F90:1036/1038`).
- At coarse sync points a blanket transfer is the codebase default (`MOM.F90:1091`) — match it
  unless profiling shows a stall.
- Guard with the matching intrinsic: `if (associated(x))` for pointers, `if (allocated(x))` for
  allocatables — never mixed. Never `map(...) if (present(optional))` inside a callee
  (`2108e0eba`).

### Step 5 — Lifecycle and balance checks

Run these over the routine/module regardless of what the walk found:

1. **Every `enter data` has a mirrored `exit data`** in the same scope (`15ca2a25f` leaked
   `b_denom_1`).
2. **No `delete` on an object this scope does not own** (§8b finding B).
3. **No copy-back expected from `delete`/`release`** — if the host needs the value, an `update from`
   or `map(from:)` must precede it.
4. **Parent mapped exactly once**, members attached after, refreshed with `update to(parent)`.
5. **Restart-registered, device-mutated fields** have a dominating `update from` before
   `save_restart` (currently latent-only — §8a item 17; do not let your port break it).
6. **Arrays-of-structs are not mapped element-by-element** on a hot path (`1865612de`).

### Step 6 — Report

Lead with the verdict. For each finding give: array, the two sites (device write → host read), the
**predicate that reaches the bad read**, the symptom it would produce, and the one-line fix. Then
the proposed directive set. Separate **confirmed** (you read both sites and the branch structure)
from **suspected** (needs a run).

## Symptoms this analysis explains

| Symptom | Likely residency cause |
|---|---|
| Checksum "mismatch" that isn't reproducible arithmetic | host-only checksum read a stale host copy — missing `update from` (`b29b27150`) |
| Answers differ **by GPU count** / run-to-run, control flow | `map(alloc:)` on a struct whose host-set scalars or `associated()` are read on device (`a774eb331`) |
| Device "addressing error" after init; `associated()` misbehaves | parent re-`enter data`'d after members attached (`c82e1254a`) |
| Correct on 1 GPU, wrong on N | missing `reduce`, or `alloc`-vs-`to` — latent until multi-device |
| Silent garbage in a host diagnostic only in some configs | transfer guarded by a *different* predicate than the read (the `khdt_x` shape — see `references/worked-example.md`) |
| GPU time dominated by attach/detach | array-of-structs mapped per element (`1865612de`) |
| Unexplained per-statement traffic | array touched in a device region with no explicit map at all (`bc05a6a89`) |

## Verification gate

A residency fix is a correctness change: it is subject to the same gate as any port
(`knowledge/KNOWLEDGE.md` §3 Step 9) — bit-identical `MOM_checksums` field checksums plus EFP
`write_energy`, and a **≥2-GPU run**, because the `alloc`-vs-`to` class of bug is latent on one
device. Never accept a nonzero diff as rounding.
