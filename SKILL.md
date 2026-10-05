---
name: porting-mom6-skill
description: Use whenever porting MOM6 (Modular Ocean Model 6) Fortran code to NVIDIA GPUs with nvfortran, or reviewing or debugging GPU-offloaded MOM6 code. Covers planning a port, converting loops to `do concurrent` or OpenMP `!$omp target` constructs, calling procedures from kernels, `!$omp target enter/exit data` and `target update` placement for `*_CS` fields and derived types such as `tv`, `G` and `GV`, polymorphic MOM_EOS and ALE calls, blocking to keep CPU performance, verifying bitwise answers, and known nvfortran bugs. Trigger for tasks like "port this subroutine to GPU", "add OpenMP offload to this loop", "why do GPU answers differ from CPU", "where should I map this array", or "review this GPU diff", even if the user does not name the skill or the directives.
---

# Porting MOM6 to GPUs

MOM6's GPU port is a work in progress, validated on the `double_gyre`, `benchmark`, and `benchmark_ALE`
configurations. Compute runs on the GPU in `do concurrent` or OpenMP `target` loops, data
movement is managed explicitly, and only the branches the target configurations exercise are
ported. Treat this as current best practice, not a finished spec: when you meet a pattern it does
not cover, make a decision consistent with it, flag it to the user, and suggest adding it here
once confirmed.

Findings are for nvfortran 26.3 unless marked, and tagged `[run-verified]`, `[source-only]` or
`[unverified]` in the references. Reproducers live in the uwagura/nvfortran-mres GitHub repository. The
`scripts/` directory is left empty for users to add their own build and verification scripts.

## Workflow for porting a routine

1. **Orient.** Check for existing work before designing: ask the user, and look for branches
   (`git branch -a`) that already touch the routine. Check prerequisites too: an EOS form the
   routine needs on the device that has no device-safe implementation is the real first task
   (`references/polymorphism.md`).
2. **Triage the calls inside the loops.** Calls outside the loop nest are host orchestration
   and cost nothing. For each call or function reference inside a loop you will offload:
   - **host-only** (`MOM_error`, `post_data`, checksums, halo passes, I/O): hoist it out of the
     loop, or return a status flag and act on it on the host afterwards;
   - **polymorphic** (`class(...)` bindings, MOM_EOS, ALE `Recon1d`): lift the call out of the
     loop or port a non-polymorphic path (`references/polymorphism.md`);
   - **external package** (CVMix, FMS): size the work before committing to the port;
   - **in-file, not `pure`**: make it `pure` and `!$omp declare target`
     (`references/device-calls.md`).
   Read the loop bodies themselves: function references look like array indexing.
3. **Decide: port in place, or refactor first.** Refactor first when a host-only call sits
   inside the loop nest, the loop writes module state, polymorphic dispatch happens inside it, or
   one loop body is too long for a reviewer to hold with its bitwise argument. A long routine is
   not a refusal by itself: read its call list first.
4. **Refactor, if needed, by verbatim extraction only.** Move the innermost side-effect-free
   arithmetic into a `pure`/`elemental` procedure, copying expressions character for character
   (parentheses included); return side effects as `intent(out)` flags; keep the old routine as a
   thin wrapper if callers need it.
5. **Port**, using the sections below and their references.
6. **Verify answers**, then **restore CPU performance**, usually by blocking
   (`references/verification.md`, `references/blocking.md`).
7. **Review before a pull request** (`references/verification.md` section 7).

If a step needs a build or a run you have not done, say so: write "unverified" and give the
exact experiment (configuration, build, what to compare). Never call something verified,
bitwise identical or passing without having run it.

**Stop and ask the user** when existing work already covers the target, a prerequisite is
unported, the refactor would be large, a polymorphic call cannot simply be lifted out of a loop,
a kernel in a target configuration needs an intrinsic without a reproducible version, or you
find a bug in existing code (report it with the condition that reaches it; fix it in its own
change, not inside the port).

## Porting conventions

- Promote column work arrays to 3-D `(i,j,k)` so kernels are plain `do concurrent`, rather than
  privatizing arrays per thread.
- Do not wrap the live code path in a big conditional that keeps the original loop as a host
  fallback. Lift genuinely unportable calls out one at a time (for example into a short host
  loop after the kernel).
- Port unexercised code only when leaving it would complicate the port: an unused `if` block
  inside a kernel goes along with it; a dead branch that would need its own transfers or
  duplicate work arrays on the host gets ported. Routines the target configurations never call
  stay unported.
- Do not introduce new structural patterns (such as per-configuration copies of a routine) just
  to make a port easier.
- Make control structures device-resident in their `_init` (released in `_end`) where possible,
  and map a derived type where it is owned, not where it is used; consumers only
  `target update` it.
- Put a condition on a single directive with its own `if (...)` clause, not a Fortran
  `if ... then` block around it.

## Comments and commit messages

- Comment sparingly and briefly. Say what the code cannot: a divergence from an existing
  pattern, a design choice a reviewer may question, a compiler workaround. Do not explain what
  the code does, and do not comment routine data transfers.
- Keep existing comments unless the port makes them wrong. Rewording or moving them creates
  merge conflicts with upstream.
- Keep commit messages short: a summary line and a few lines of body. Write more only for a
  large or confusing change. State the impact on answers, as MOM6 requires.

## Reviewing or debugging a port

Check your own diff first: a missing map or transfer, an unbalanced `enter`/`exit data`, or a
misplaced accumulation is far more common than a compiler bug. Audit each array's host and
device state through the routine (`references/data-mapping.md` section 7). Then check
`references/compiler-issues.md`. Report findings as confirmed (both sites read, branches
checked) or suspected (needs a run), each with the sites involved, the condition that reaches
the bad path, its symptom, and the fix.

## Choosing a loop construct

Full rules, with evidence: `references/loop-constructs.md`. In short:

1. **`do concurrent`** by default, with `k` first in a single header and every locality
   clause wrapped in `DO_LOCALITY(...)` (`#include "do_concurrent_compat.h"`).
2. **`!$omp target teams distribute parallel do collapse(n) private(...)`** when the body
   calls a procedure or needs per-iteration array scratch.
3. **`!$omp target teams loop collapse(n)`** only when the body makes no calls.

Hard rules:
- **Never put an automatic array in `local()`/`local_init()`.** It crashes or works by
  accident. Use a 3-D `(i,j,k)` workspace or form 2 with `private(w)`.
- **Never `reduce(+:)` over reals.** Reduction variables must be plain local variables, used
  only as `v = v op expr`.
- **For loops that run serially in `k`**, prefer `do concurrent (j)` → serial `do k` →
  `do concurrent (i)` where it is practical.
- **Put a gating mask in the header** (`do concurrent (j=..., i=..., mask(i,j) > 0.)`)
  rather than an `if` around the body.

## Calling a procedure from inside a kernel

Full rules, with evidence: `references/device-calls.md`. Hard rules:

- A callee in a `do concurrent` must be `pure`. Mark every device-called routine
  `!$omp declare target`; a cross-file callee without it fails to link.
- **Never pass an `(i,j,:)` section from inside a kernel.** It is repacked on the device heap
  and crashes at scale. Pass the whole array plus `i, j`.
- Declare array dummies explicit-shape and input scalars `VALUE`: worth ~30% and ~20%.
- Prefer caller-supplied workspace over automatics in the callee (+30%).
- `!DIR$ ATTRIBUTES FORCEINLINE` does nothing in nvfortran; inlining needs
  `-Minline=name:<routine>[,reshape]`. Check `-Minfo=inline`, and recheck answers after
  inlining.
- An elemental called on whole arrays outside a `do concurrent` silently runs on the host.

## Data mapping

Full rules, with evidence: `references/data-mapping.md`. Hard rules:

- **Grep for existing maps before adding one.** Resident data may only be refreshed with
  `target update`; a second `map(to:)` copies nothing.
- **`map(alloc:)` does not zero the device copy.** Prefer `map(alloc:)` plus initialization on
  the device; use `map(to:)` only when the device reads host-computed values, or host-set
  scalars, pointer descriptors, or `associated()` state of a struct.
- **Map every array a kernel references**, even in a branch your configuration never takes.
  Otherwise nvfortran copies it at every launch.
- **Derived types:** parent first, then members. Never `update to/from` or `map(from:)` the
  whole parent once a member is attached. A pointer component needs
  `map(to: parent, parent%ptr)`.
- **Every branch of an if/elseif chain needs its own transfer.** A missing `update from` can
  leave answers identical and break only control flow.
- **Put `enter data` below every early `return`.** Use `delete` only in the scope that owns the
  mapping.

## Preserving CPU and GPU performance together

A port must not change answers, either against the CPU build or against `dev/gpu`. Never
reorder floating-point arithmetic to make a loop parallel (`references/code-style-guide.md`).

**Order of work.** Early passes can concentrate on getting the computation onto the device and
reproducing answers. But a port is not ready for a pull request until CPU performance is back:
a CPU slowdown beyond about 1-2% will almost certainly not be accepted. Plan for that from the
start, even if the user's initial plan does not mention it.

**Blocking is the usual final step for restoring CPU performance**: block sizes are runtime
parameters whose defaults reproduce the old CPU loop structure, while GPU builds use one block
per domain. Write the GPU version so blocking can be added later without restructuring again:
whole-domain work arrays indexed so they can become block-sized, and loops in the forms of
`references/loop-constructs.md`. Layout, naming and parameters: `references/blocking.md`.
Measure the CPU clock before and after (`references/verification.md` section 5).

If keeping CPU and GPU performance together looks like it needs a substantial refactor rather
than a loop-construct swap, explain why, sketch one or two approaches, and ask the user before
proceeding.

## Verifying a port

Full checklist, with evidence: `references/verification.md`. In short:

- `ocean.stats` must match bitwise, against `dev/gpu` and between the CPU and GPU builds.
   Build with `-Mnofma`.
- `exp`, `log`, trig and `x**(1./n)` round differently on the GPU. Use `exp_repro`,
  `cuberoot` and `nth_root` in ported kernels. If a kernel in a target configuration needs
  another such intrinsic (`log`, `sin`, `tanh`, ...), warn the user and ask before porting it.
- Compare the CPU clock before and after the port; more than 1-2% slower blocks the PR.
- Confirm the kernel actually runs on the device (`-Minfo=accel`, `NVCOMPILER_ACC_NOTIFY=1`)
  before reading a GPU timing.
- A failure in some runs but not others often means uninitialized memory. Measure the rate.

## Reference files

Read the one for the task at hand; each starts with a contents line.

- `references/loop-constructs.md`: choosing a construct, `do concurrent` conventions, locality
  and automatics, reductions, column-loop structure, reading the schedule.
- `references/device-calls.md`: making a procedure device-callable, inlining, passing arrays,
  local arrays in callees.
- `references/data-mapping.md`: what is resident, where each kind of array is mapped, map kinds,
  derived types, host-only consumers and halo exchanges, hazards, auditing a routine.
- `references/polymorphism.md`: `class` types in kernels, the MOM_EOS `_loc` pattern, ALE
  remapping.
- `references/blocking.md`: blocking for CPU performance: parameters, wrapper, block loop, work
  arrays, answers.
- `references/verification.md`: build flags, answer checks, arithmetic order, host/device
  intrinsic differences, performance, intermittent failures, review before a pull request.
- `references/compiler-issues.md`: nvfortran problems that are live, fixed, stale or unverified,
  with workarounds and reproducers.
- `references/code-style-guide.md`: MOM6 Fortran style (full source:
  https://github.com/mom-ocean/MOM6/wiki/Code-style-guide).
