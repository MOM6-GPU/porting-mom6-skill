# Loop constructs for GPU kernels

Rules for writing the parallel loop itself. Findings are for nvfortran 26.3 unless marked.
Tags: `[run-verified]` measured on a GPU, `[source-only]` read from code or commits, not re-run,
`[unverified]` reported but never reproduced.

Contents: 1 choosing a construct · 2 `do concurrent` conventions · 3 scalars in `local()` ·
4 arrays in `local()` · 5 reductions · 6 column loops · 7 early `exit` · 8 checking the schedule

## 1. Choosing a construct

Take the first that fits:

1. **`do concurrent`** by default. It is standard Fortran and gets nvfortran's best code.
2. **`!$omp target teams distribute parallel do collapse(n) private(...)`** when the body
   **calls a procedure**, or needs array scratch (section 4). The collapse is asserted, not
   inferred, and the explicit thread mapping avoids the compiler-chosen mappings that break
   below. Precedent: the column kernels in `vertvisc_coef` (`MOM_vert_friction.F90`).
3. **`!$omp target teams loop collapse(n)`** only when the body makes **no** calls.

Why calls push you to form 2 `[run-verified]`: when nvfortran picks the thread mapping itself
(`do concurrent` under `-stdpar=gpu`, or `target teams loop`) and puts an inner loop on
threads, a call to a non-inlined routine in the region can hit a code-generation bug. At
`-O0`/`-O1` it fails to compile with `parse invalid cast opcode for cast from 'double' to 'i8*'`
(`NVFORTRAN-F/S-0155`, reported at the wrong line). At `-O2` it compiles and either gives wrong
numbers or `CUDA_ERROR_ILLEGAL_ADDRESS`/`MISALIGNED_ADDRESS`. `distribute parallel do` avoids
it, and so does inlining the callee (`-Minline=name:<routine>`, plus `reshape` for array
arguments). Reproducers: nvfortran-mres repo, `make mre_badcast*` and
`docs/nvfortran-shared-slot-holder-bug-REPORT.md`.

A plain `do concurrent` that calls a `pure` routine usually works and keeps its collapse; just
check the build at `-O0` too, and see `device-calls.md` for how to declare the callee.

## 2. `do concurrent` conventions

- One header for all parallel indices, `k` first: `do concurrent (k=1:nz, j=js:je, i=is:ie)`.
  Separate headers are only for a deliberate nest (section 6).
- Wrap every locality clause in `DO_LOCALITY(...)` from `#include "do_concurrent_compat.h"`:
  `DO_LOCALITY(local(tmp))`, `DO_LOCALITY(reduce(max: x))`. It expands to nothing on
  compilers without F2018 locality support. Bare clauses build with nvfortran but break CI.
- Remove the `!$OMP parallel do` from a loop you convert.
- `local()` must not name the loop's own indices (`NVFORTRAN-S-1045`, CPU and GPU). Indices
  derived in the body, such as `i = isb + ii - 1` in a blocked loop, may be listed.
- Gating mask: put it in the header, `do concurrent (j=js:je, i=is:ie, G%mask2dT(i,j) > 0.)`,
  instead of an `if` wrapping the whole body `[run-verified]`. It keeps the collapse and is core
  F2008, so it needs no `DO_LOCALITY`. Before hoisting, check for statements outside the `if`:
  they ran at land points, and a later host loop may read what they wrote. When a loop does work
  over land points, it may make more sense to place the mask in the loop body instead of the
  `do concurrent` header:
  ```fortran
  do concurrent (j=js:je, i=is:ie)
    if (G%mask2dT(i,j) > 0.)
      ! work on ocean points
    else
      ! work on land points
  ```

## 3. Scalars in `local()`

**List every per-iteration scratch scalar in `local()`**, including inner serial loop
indices. nvfortran happens to privatize a scalar that each iteration writes before reading
even when it is unlisted (`-Minfo` says `Generating implicit private(...)`) `[run-verified]`,
but another compiler may not, and the clause tells readers the variable must be private.

`!$omp target` regions are different: OpenMP's default is shared, so **every** scalar written
in the body must be in `private(...)`, or the loop races and gives nondeterministic answers.

## 4. Arrays in `local()`

**Never put an automatic array in `local(...)` / `local_init(...)`.** This holds for any rank,
and whether the extent comes from a dummy (`nz`), a type component (`SZK_(GV)`), or bounds
like `SZI_(G)` `[run-verified]`. The outcome depends on the schedule nvfortran picks, shown by
`-Minfo=accel` on the loop that carries `local(w)`:

| `-Minfo` for the `local` loop | Automatic `w` | Fixed-size `w` |
|---|---|---|
| `CUDA thread blocks, CUDA threads(128) blockidx%x threadidx%x` (typical for `do concurrent (j,i)` with serial `k` loops inside) | **crash**, `CUDA_ERROR_ILLEGAL_ADDRESS` | passed (the compiler chose blocks-only) |
| `CUDA thread blocks ! blockidx%x`, inner loops on threads (typical for `do concurrent (j)` over `w(SZI_,SZK_)`) | passes **only if the host used `w` before the loop** (an assignment, or a `target enter data map`); otherwise undefined | passes |
| `CUDA thread blocks, CUDA threads(4) blockidx%x threadidx%y` | not seen | **silently wrong**: 4 iterations share one `w` |

With no host use, nvfortran drops `w`'s host allocation (no `pgf90_auto_alloc04_i8` in `-S`),
yet the kernel sizes its private copies from bounds that were never set. The result is
repeatable for a given binary but flips with unrelated edits: a crash, an
`Out of memory allocating 17648557553772920832 bytes`, or a pass. `local_init` behaves like
`local`. The schedule is not visible in the source, hence "never".

For per-iteration array scratch, use, best first:
- a 3-D workspace indexed `(i,j,k)`, CS-resident or passed in by the caller;
- form 2 of section 1 with `private(w)` and `w` automatic, which passed in every variant
  tested, including those where `local(w)` crashed.

Small fixed-size arrays in a column loop (`e(efp_digits)` in `MOM_coms.F90`) are fine, but after
putting any array in `local()`, check the schedule against the table. In static-memory builds
`SZK_(GV)` is a constant, so there the fixed-size column applies.

Reproducers: nvfortran-mres repo, `dc_local_automatics/` and `mre2_do_concurrent_local.F90`.

## 5. Reductions

- Wrap in `DO_LOCALITY(reduce(op: v))`. `v` must be a plain variable: `reduce(+: CS%n)` and
  `reduce(max: a(j))` are rejected. Reduce into a local scalar and assign it afterwards. For a
  per-row result, put `local(tmp)` on the outer loop and `reduce(max: tmp)` on an inner
  `do concurrent`, as in `tracer_epipycnal_ML_diff` (`MOM_tracer_hor_diff.F90`).
- In the body, `v` may appear only as `v = v op expr`, `v = expr op v`, or `v = f(..., v, ...)`
  (F2023 11.1.7.5). `flag = mask(i,j)` under `reduce(.or.: flag)` can lose a `.true.`; write
  `flag = flag .or. mask(i,j)`. A test like `if (x < v) v = x` is also non-conforming; write
  `v = min(v, x)`, behind an answer-date flag if the rounding could change.
- Never use `reduce(+:)` on reals: the order differs between CPU and GPU and between runs.
  Use `reproducing_sum` (`MOM_coms.F90`) or keep the sum serial. Integer `+`, `max`, `min`,
  `.or.` and `.and.` are safe.
- A `.or.`/`max` flag replaces a "search until found" loop: every iteration sets the flag,
  and the host branches on it after the loop (the `domore_*` flags in `MOM_tracer_advect.F90`).

## 6. Column loops

**Recommended form for loops that run serially in `k`**, such as sums over layers: parallel
over `j`, serial in `k`, parallel over `i`. On the GPU each `j` row gets a block and `i` runs on
its threads; on the CPU it keeps the stride-1 inner `i` loop. From `btstep`
(`MOM_barotropic.F90`):

```fortran
do concurrent (j=js:je)
  do k=1,nz
    do concurrent (I=is-1:ie)
      ubt_Cor(I,j) = ubt_Cor(I,j) + wt_u(I,j,k) * U_Cor(I,j,k)
    enddo
  enddo
enddo
```

It can be awkward when per-column scalars must carry across `k` (they become `(i)` arrays
indexed in the inner loop), or when a column needs private array scratch (section 4). Then use
one header over `(j,i)` with a serial `k` loop inside, or an OpenMP form from section 1 with
`private(...)` scratch for recurrences like the tridiagonal solves in `vertvisc`. Avoid a serial host `do k` wrapping a full 2-D
`do concurrent`: that is one kernel launch per level.

When nothing depends across `k`, put `k` in the header instead
(`do concurrent (k=1:nz, j=..., i=...)`). Merging a per-level launch loop this way gave about
20% on the `vertvisc` clock (`b95083139`) `[source-only]`.

## 7. Early `exit`

**An `exit` from a serial loop inside a `do concurrent` is fine; use it where it is natural.**
On 26.3 the insertion sort from `tracer_epipycnal_ML_diff` and an early-exit thickness search
both matched the host at `-O0` and `-O2` `[run-verified]` (nvfortran-mres repo,
`dc_early_exit/`).

An older rule banned `exit` in device loops because of commit `e23d6a7b1`, which says NVHPC
25.11 gave wrong answers. That report is `[unverified]`: it was never reproduced, 25.11 was
not available to test, and that loop also left its scratch scalars out of `local()`. Do not
rewrite an `exit` into an if-guard on the strength of it.

## 8. Checking the schedule

- Read `-Minfo=accel` on the `do concurrent` line itself:
  `Loop parallelized across CUDA thread blocks, CUDA threads(128) collapse(2) ... auto-collapsed`.
- `Reference argument passing prevents parallelization: <name>` is reported at a **call** and
  refers to the loop that contains the call, usually a serial inner loop. It says nothing about
  the enclosing `do concurrent`.
- `NVCOMPILER_ACC_NOTIFY=1` prints each launch's geometry. A collapsed 360x180 loop shows
  `grid=507 block=128`; one parallelized over `j` only shows `grid=180`.
- If an explicit `target teams` region launches far fewer teams than it has work, set
  `num_teams(ceiling(real(iterations)/128.))` by hand, as in `zonal_mass_flux`
  (`MOM_continuity_PPM.F90`) `[source-only]`.
