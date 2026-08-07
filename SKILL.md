---
name: mom6-gpu-loop-porting
description: Use this skill whenever porting MOM6 (Modular Ocean Model 6) Fortran code to run on GPU, or reviewing/debugging existing GPU-offloaded MOM6 code. This covers converting loops to `do concurrent` or OpenMP `!$omp target` offload directives, deciding where `!$omp target enter/exit data` and `target update` directives belong, handling MOM6 control-structure (`*_CS`) fields and derived types like `tv`/`G`/`US`/`GV` on the device, working around the MOM_EOS polymorphic-type GPU limitation, and preserving CPU performance while adding GPU support (e.g. loop tiling / "blocks"). Trigger this for tasks like "port this subroutine to GPU", "add OpenMP offload to this loop", "why is this GPU kernel giving different answers than the CPU", "how should I map this CS field to the device", or any work touching `!$omp target`, `do concurrent`, or GPU offload anywhere in the MOM6 codebase — even if the user doesn't use the word "skill" or spell out these directive names explicitly.
---

# MOM6 GPU Loop Porting

## Status and scope

MOM6's GPU port is a work in progress. The conventions here have been
validated on the `benchmark` and `benchmark_ALE` configurations, but the
codebase is large and only the code paths those configurations exercise have
been ported so far. Treat everything below as the current best practice, not
a finished spec — when you hit a pattern this document doesn't cover, make a
reasonable decision consistent with the philosophy here, flag it to the user,
and suggest adding it to this skill once it's confirmed to work.

The overall approach is: compute on the GPU with `do concurrent` or OpenMP
`target` loop constructs, manage data movement explicitly (don't rely on the
compiler to infer transfers), and only port the branches that are actually
exercised by the target configurations.

## Before porting any loop: check for MOM_EOS calls

GPUs can't execute through a polymorphic Fortran type's deferred procedures
from inside a kernel — `MOM_EOS` subroutines are typically dispatched this
way (an abstract `EOS` type selecting the equation-of-state routine at
runtime). This doesn't mean MOM_EOS-adjacent code is unportable: a kernel can
still be launched within an EOS subroutine on data that's already on the device, as
long as *that subroutine's own kernels* never touch the polymorphic `EOS`
type or its deferred procedures internally. The constraint is specifically
about the polymorphic dispatch happening inside a kernel, not about EOS logic
being off-limits.

So, before converting a loop:

1. Check whether the loop body calls a MOM_EOS subroutine.
2. If it does, check whether that call can be cleanly separated from the rest
   of the loop (e.g. hoisted into its own loop, called ahead of time into a
   temporary array) without a substantial restructuring of the surrounding
   code.
3. If separation is easy — do it, and port the surrounding loop normally.
4. If separation is *not* easy — a real refactor would be needed — **stop**.
   Don't attempt the port. Explain to the user why the EOS call blocks a
   straightforward port, sketch 1-2 plausible refactor approaches, and ask
   how they'd like to proceed before writing any offload code. Guessing here
   risks silently producing a kernel that won't compile or will crash on
   device, or a big unwanted refactor the user didn't ask for.

## Choosing a loop construct

Try these in order. Each fallback exists because of a specific, concrete
problem with the option above it — not because of general preference — so
don't skip straight to a fallback unless the described trigger actually
applies.

### 1. `do concurrent` (default choice)

`do concurrent` is preferred by default: it's standard Fortran (so the code
stays portable and readable even off-GPU), and it currently gets the best
code generation from `nvfortran`, the compiler this port targets.

```fortran
do concurrent( j=js:je, i=is:ie ) local(tmp)
  tmp = some_expression(i, j)
  output(i,j) = tmp * other_array(i,j)
enddo
```

Notes on `do concurrent` syntax, since it differs from an ordinary `do` loop
in easy-to-miss ways:

- Loop bounds are separated by a **colon**, not a comma:
  `do concurrent( i = is:ie, j = js:je )` — not `do i=is,ie; do j=js,je`.
- Collapse multi-dimensional loops into a single `do concurrent` statement
  rather than nesting separate `do concurrent` loops for each dimension.
  Nesting a `do concurrent` inside another `do concurrent` is fine (e.g. an
  outer 2-D horizontal loop containing an inner k-loop that itself needs to
  be a `do concurrent`) — it's collapsing what should be one multi-index
  statement into several separate ones that should be avoided.
- Declare loop-private scalar temporaries with `local(tmp)` so the GPU
  parallelization doesn't race on them.
- Remove any pre-existing `!$OMP parallel do` directive when converting a
  loop to `do concurrent` — the two shouldn't coexist on the same loop.
- `do concurrent` supports a mask condition, which is a natural fit for the
  common MOM6 pattern of `do i; do j; if (mask(i,j) > 0) ...`:

  ```fortran
  do concurrent( j=js:je, i=is:ie, mask(i,j) > 0 )
    output(i,j) = some_expression(i,j)
  enddo
  ```

  Prefer this masked form over keeping an `if` as the first statement inside
  the loop body when the condition is simply gating whether the iteration
  runs at all.

**Automatic arrays — read this carefully, the common summary of it is wrong.**
The problem is *not* "`do concurrent` cannot privatize an automatic array."
An automatic array declared in the **directive's own routine** and listed in
`private(...)` works fine, and is merged and checksum-gated
(`MOM_vert_friction.F90:575,737`). The construct that actually fails is an
automatic array **local to a callee** invoked from a device region and sized
from something that is *not* a dummy argument (e.g. `SZK_(GV)` →`GV%ke`, a
derived-type component), because that would need a device-side dynamic stack.
Swapping loop constructs does **not** help — the original failing case was
`!$omp target loop`, not `do concurrent`. Fix it at the callee instead: size
the automatic from a dummy argument, or better, have the caller pass the
workspace in (see `knowledge/gpu-knowledge/nvfortran-automatic-array-bug.md`
for the full table and the ranked workarounds). Note this bug only bites
**dynamic-memory** builds; static memory makes `SZK_` a compile-time constant.

**The real reasons to leave `do concurrent`** are the two below.

### 2. `!$omp target teams distribute parallel do collapse(n)` (when the kernel body calls a procedure)

`do concurrent`'s collapse is **inferred**, and nvfortran gives it up when the
body makes a call it cannot see through (see "Calling a procedure from inside
a kernel" below). An OpenMP `collapse(n)` is *asserted*, so it survives. This
is the form used by the merged device-call kernels — `MOM_vert_friction.F90:1443`
and the kappa-shear port's `kappa_shear_column` call site:

```fortran
!$omp target teams distribute parallel do collapse(2) private(f2, i, j)
do j=js,je ; do i=is,ie
  ! loop body, including a call to a !$omp declare target routine
enddo ; enddo
```

### 3. `!$omp target teams loop collapse(n)` (only when the body makes no calls)

Equivalent performance to the above when it works (measured within ~5%), but
on nvfortran 26.3 **`target teams loop` containing a call to a `declare target`
routine is a hard internal compiler error**:

```
nvnvvmd: error: parse invalid cast opcode for cast from 'float' to 'i8*'
NVFORTRAN-F-0155-Compiler failed to translate accelerator region
```

reproduced with and without `-stdpar`, and with both assumed-shape and
explicit-shape dummies. So prefer form 2 whenever the body calls anything.

## Calling a procedure from inside a kernel

Correctness rules (`declare target` vs force-inline, polymorphic `this`, etc.)
live in `knowledge/gpu-knowledge/08-cross-module-inlining.md`. Two additional
traps cost performance or crash without any compile-time warning:

**1. Never pass a non-contiguous array section.** Passing `Igu(i,j,:)` to an
assumed-shape `dimension(:)` dummy makes nvfortran repack the section into a
**per-thread temporary on the device heap**. The default heap (~8 MB) runs out
once there are enough columns, and the failure is a bare
`CUDA_ERROR_ILLEGAL_ADDRESS` with no message telling you to raise
`NV_ACC_CUDA_HEAPSIZE` (that friendly message only appears for *user-declared*
automatics, which the compiler null-checks; a compiler-generated repack is not).
`compute-sanitizer --tool memcheck` shows the real cause as
`Malloc/Free Warning encountered : Device-side malloc failed`. Pass the whole
3-D array plus the `i,j` indices instead.

**2. Declare those dummies explicit-shape, and give input scalars `VALUE`.**
Both are required, or the loop silently loses its collapse:

| dummies | input scalars | collapse? |
|---|---|---|
| assumed-shape `(isl:,jsl:,:)` | by reference | no |
| assumed-shape | `VALUE` | no |
| explicit-shape `(isl:iel,jsl:jel,nk)` | by reference | no |
| **explicit-shape** | **`VALUE`** | **yes** |

An assumed-shape dummy passes a descriptor by reference and a scalar without
`VALUE` passes an address; either makes `-Minfo` say `Reference argument
passing prevents parallelization: <name>` and drop to parallelizing the outer
loop only. `intent(out)` scalars may stay by reference. Getting both right is
worth **3-4x** and needs no build flags — prefer it over `-Minline`, which
hides the requirement in the build system where a user who rebuilds without
the flag silently gets the slow version. `kappa_shear_column` already uses
explicit-shape dummies (`dimension(SZI_(G),SZJ_(G),SZK_(GV)+1)`).

**Check it, don't assume it.** `-Minfo=all` should say `auto-collapsed` /
`collapse(2)`. To see the actual launch geometry, run with
`NVCOMPILER_ACC_NOTIFY=1`: a collapsed 360x180 loop reports `grid=507
block=128` (one thread per column), an un-collapsed one `grid=180 block=128`
(grid capped at `nj`). Also note `local(i,j)` naming the loop's own indices is
rejected outright (`NVFORTRAN-S-1045`), on CPU and GPU alike.

## Data mapping: what's already resident, and what you must map yourself

Two large derived types worth knowing about up front:

- **`G` (grid), `US` (unit scaling), `GV` (vertical grid)** and their
  constituent arrays can be assumed to already be resident on the device
  everywhere in the model, unless a user tells you otherwise for a specific
  case. You generally don't need to add mapping directives for these.
- **`tv` (thermodynamic variables)**, by contrast, is *not* yet persistently
  resident. Any subroutine that touches `tv` or its constituent arrays (e.g.
  `tv%S`, `tv%T`) needs those explicitly mapped to the device, from outside
  the subroutine that uses them.

When mapping a derived type like `tv`, order matters the same way it does for
`*_CS` structs (see `references/data-mapping-conventions.md`, Rule 2): map the
parent type first, then its constituent arrays.

**The gotcha that has caused segfaults in practice**: once you've mapped a
constituent array back with its own `target update from(...)` (because a
subroutine modified it), do **not** subsequently re-map the *parent type*
`tv` itself. Doing so after the arrays have already been updated can clobber
the array contents on the host, or leave them pointing at a device address
that no longer exists, which shows up as a segfault or an accelerator
failure — not usually as a silent wrong-answer bug, so it's easy to miss
until a run crashes. If only the arrays changed, map only the arrays back.

For the four rules governing *where in the code* a mapping directive belongs
(subroutine-local temporaries, static vs. dynamic `*_CS` fields, and
subroutine argument arrays), see `references/data-mapping-conventions.md`.

**Check transfer parity across every branch of an if/elseif chain.** When you
port one branch of a chain to the device and leave its siblings on the host,
the transfer that belongs to the ported branch is easy to miss, because the
sibling branches already have one that *looks* like it covers the case. This
shipped as a real bug: in `tracer_hordiff`, the `use_variable_mixing` branch
computes `khdt_x`/`khdt_y`/`Kh_u`/`Kh_v` in `do concurrent` on the device,
while the two branches below it compute the same arrays on the host and each
end with `!$omp target update to(...)`. The device branch needed an `update
from` and had none, so downstream **host** code read arrays that were never
written (fixed in `979be73e6`). When you finish a branch, ask: which side
computed these values, which side reads them next, and does *this* branch have
its own directive — not just the one belonging to the branch below it.

Note the direction of the two failure modes. A missing `update to` usually
shows up as wrong answers. A missing `update from` may not: if the stale host
values feed something that only influences control flow — an iteration count,
a limiter, a diagnostic threshold — answers can stay bitwise identical while
the run misbehaves in some other way entirely. See "When a run appears to
hang" below.

## Preserving CPU and GPU performance together

A port is only useful if it (a) doesn't change answers relative to the CPU
path, (b) doesn't change answers relative to `dev/gpu`, our development
branch, and (c) doesn't slow the CPU path down. All three matter equally —
a numerically-correct port that regresses CPU performance is not acceptable,
and neither is a fast GPU kernel that quietly reorders a summation (see the
arithmetic-reproducibility notes in `references/code-style-guide.md` — this
is a common way for (a)/(b) to fail silently).

**Loop tiling ("blocks") is the main tool for satisfying (c) without
sacrificing GPU throughput.** The idea: keep the same tile/block size for the
loop on both platforms, but choose different sizes per platform —
cache-friendly on CPU (often the default block size in `i`, and size 1 in `j`
and `k`), and large on GPU (often the whole default-sized 3-D array as one
block, to maximize parallel work per kernel launch). `MOM_continuity_PPM.F90`
and `MOM_CoriolisAdv.F90` have already been ported with this pattern
successfully and are good references for how the tiling is structured.

**If a subroutine or module looks like it'll need a substantial refactor to
get CPU and GPU performance to coexist** (not just a loop-construct swap),
don't push through it solo. Explain to the user why a refactor looks
necessary, sketch out a couple of ways it could be approached, and ask how
they'd like to proceed — the same escalation pattern as the MOM_EOS case
above.

## Verifying a port

Before considering a port done:

- **Correctness (a and b above)**: compare the `ocean.stats` file from a run
  with the port against `ocean.stats` from a run without it (or from
  `dev/gpu`). They should match. If they don't, suspect a reordered
  reduction, a mapping-order bug (see the `tv` gotcha above), or a race from
  a missing `local(...)` clause before looking elsewhere.
- **Performance (c above)**: check this with the CPU clocks already built
  into `MOM_cpu_clock` — see `references/performance-timing.md` for how to
  find an existing clock for the component you're porting, or add one if it
  doesn't exist yet. Separately, there's also a scripts directory intended
  for testing the target configurations against our target compiler settings
  for performance regressions more broadly. It may not be populated yet —
  check whether it exists and has usable scripts before assuming it's
  available, and ask the user if you can't find it.

**Beware intermittent failures — measure the rate, don't reason about one
run.** GPU kernels are deterministic, so a failure that appears in some runs
and not others almost always means *uninitialized memory* is being read
(`map(alloc:)` leaves the device copy unset; a never-written host array is
whatever was on the heap). Two traps when quantifying this:

- Separate CPU runs from GPU runs before computing any rate. Mixing them made
  a 4-run clean streak look like 8 and wrongly implicated a code change.
- Run rate tests **one at a time**. Two GPU jobs co-scheduled on a node slow
  each other ~3x, which a stall-detector reads as a hang.

Compare candidate against control by **alternating them in a single job** on
one device, so both see the same machine state, and report the count (e.g.
"11/24 vs 0/8, Fisher p = 0.019") rather than an impression.

## When a run appears to hang

An apparently hung GPU run is often not hung. Work outward:

1. `nvidia-smi --query-compute-apps=pid,used_memory` — is the GPU still busy?
2. `gdb -p <pid> -batch -ex "thread apply all bt"` on the **MOM6 rank**, not
   `mpirun` (`pgrep -x MOM6`; matching on the executable path also matches
   `mpirun`'s command line).
3. `NVCOMPILER_ACC_NOTIFY=15` traces every kernel launch, data action, region
   and wait. If the trace keeps growing, the process is executing, not stuck.
   Counting the actions tells you *what* is looping. Two warnings: the trace
   is enormous (13 GB in one case here — watch your quota), and the slowdown
   is large enough that a stall-detector will call a healthy run hung, so
   don't classify runs while it's on.

The case that motivated this: `benchmark_ALE` on GPU appeared to hang ~44% of
runs. The trace showed 5.9M kernel launches in `tracer_hordiff` before day 1
against ~300 expected. It was executing normally — an uninitialized host
`khdt_x` (the transfer-parity bug above) produced a garbage diffusive CFL, and
`num_itts = max(1, ceiling(max_CFL - ...))` is **unbounded**, so one timestep
became 89,159 iterations. Instrumenting the two suspect quantities with a
`write(0,...)` per timestep found it immediately after three wrong hypotheses
had been chased on inference alone. A stack sample landing in the OpenMP
runtime's lock was a red herring: with millions of data regions being entered
per second, that is simply where a sample is likely to land.

## Reference files

- `references/data-mapping-conventions.md` — the detailed rules for where
  `enter/exit data` and `target update` directives go for subroutine locals,
  static vs. dynamic `*_CS` fields, and subroutine argument arrays, plus a
  couple of `nvfortran`-specific mapping quirks (cheap zero-init `alloc` vs.
  `to`, and why conditionally-unused arrays still need at least an `alloc`).
- `references/performance-timing.md` — how to find or add a `MOM_cpu_clock`
  timer to check that a port hasn't regressed CPU performance.
- `references/code-style-guide.md` — a summary of MOM6's general Fortran
  style conventions (indentation, naming, loop-index letter conventions,
  arithmetic reproducibility) worth following in any code you write or
  restructure while porting. Full source:
  https://github.com/mom-ocean/MOM6/wiki/Code-style-guide
