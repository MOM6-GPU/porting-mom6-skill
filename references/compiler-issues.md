# nvfortran issues: live, stale, and unverified

Compiler and runtime problems met while porting, with their status on nvfortran 26.3. Each
"Issue N" is a numbered section, with a reproducer, in the uwagura/nvfortran-mres repo's `README.MD`;
other reproducers are named by folder. When a problem here looks version-dependent, retest it on
the compiler you are using before working around it. When you find a new one, write a minimal
reproducer in nvfortran-mres and add a row.

## Live on 26.3

| Symptom | Trigger | Workaround | Evidence |
|---|---|---|---|
| `parse invalid cast opcode for cast from 'double' to 'i8*'` (`-O0`/`-O1`), or a clean build that gives **wrong answers** or `ILLEGAL_ADDRESS`/`MISALIGNED_ADDRESS` (`-O2`) | a call to a non-inlined routine in a region whose thread mapping the compiler chooses (`do concurrent`, `target teams loop`), with an inner loop on threads | `!$omp target teams distribute parallel do`, or `-Minline=name:<routine>` (`loop-constructs.md` section 1) | Issue 4, `make mre_badcast*` |
| Silent wrong answer: a sum short by about the inner trip count | an inner loop on `threadidx%x` that accumulates a loop-invariant addend; typically appears after `-Minline` | rewrite so the addend varies with the loop, or keep the loop serial; recheck answers after inlining | Issue 5 |
| `CUDA_ERROR_ILLEGAL_ADDRESS`, a garbage-size `Out of memory`, or a pass that flips with unrelated edits | an automatic array in `do concurrent` `local()` | 3-D workspace or OpenMP `private(w)` (`loop-constructs.md` section 4) | Issue 7, `dc_local_automatics/` |
| Loop silently runs on the host (`Accelerator restriction: Indirect function/procedure calls are not supported`), or `Failed to find device function` at run time, or "partially present" at `-O2` | a call through a polymorphic binding, or `this` passed to a procedure, inside a kernel | dispatch on the host, call free `_loc` procedures in the loop (`polymorphism.md`) | `polymorphism/` |
| No message, call not inlined | a dummy whose type contains a `pointer` to a type from another module (`G` has `Domain`) | none known; do not rely on inlining routines that take `G` | Issue 3 |
| `subprogram not inlined -- array reshaping not enabled` | `-Minline` of a routine with explicit-lower-bound dummies or a different-rank section | add `reshape`: `-Minline=reshape,name:<routine>` | Issue 2 |
| `!DIR$ ATTRIBUTES FORCEINLINE` has no effect | | `-Minline=name:<routine>` | `device_calls/` |
| Link fails: `multiple definition of '_NNN_<truncated path>'` | GPU objects built from a long source path; nvfortran names fat-binary segments by a truncated path | build from a short path | seen 2026-09 |
| Non-conforming code compiles | for example a local variable named `merge` while the `merge` intrinsic is used in a declaration | none; nvfortran's acceptance proves nothing, as ifx (the CPU compiler) will reject it | seen 2026-09 |
| `use of undefined value '%g'` from `llc`/`opt` at compile time (**fixed in 26.9**) | members of an unmapped derived type (`G`) used in a `do concurrent`, with `G` also implicitly copied by a `!$OMP parallel do` around inner `do concurrent` loops | on 26.3: map the type explicitly, and do not wrap `do concurrent` in `!$OMP parallel do` | `make mre1`: fails on 26.3, compiles on 26.9 |
| Different last bits from `exp`, `log`, `sin`, `tanh`, `x**(1./n)` on device | math library differences, even with `-Mnofma` | `exp_repro`, `cuberoot`, `nth_root`; ask the user about others (`verification.md` section 4) | `device_calls/intrinsic_bits.F90` |

## nvfortran 26.9 only

| Symptom | Trigger | Workaround | Evidence |
|---|---|---|---|
| Region silently runs on the host; `-Minfo`: `Accelerator restriction: datatype not supported: _in_NNN` | a module-scope `use omp_lib` (even `only:`) in an inlined callee's module or a module it uses | put `use omp_lib` inside the procedure that needs it; after a compiler upgrade, `grep -c 'datatype not supported'` the build log | Issue 6 |

## Not reproduced on 26.3

Older reports that did not reproduce on 26.3. Do not avoid these patterns because of them.

| Old report | Source | 26.3 result |
|---|---|---|
| Early `exit` from a loop under `do concurrent` gives wrong answers (25.11) | `e23d6a7b1` | correct at `-O0`/`-O2` (`dc_early_exit/`) |
| `map(to: x) if (present(x))` in a callee crashes (25.5, A100) | `2108e0eba` | works (`data_mapping/` v5) |
| `modulo()` not available on the device | `9aea28954` | works (`device_calls/`) |
| A callee without `declare target` or forced inlining gives silently wrong answers | `3cb184edd` (OpenACC-era) | same-file callee works; cross-file callee fails to link (`device_calls/`) |
| An automatic in a device-called routine cannot be sized from `SZK_(GV)` (25.x) | `05c74b56b` | works (`dc_local_automatics/sweep/`) |
| A null pointer or unallocated component named in a `do concurrent` crashes | earlier probe on a node without a GPU | works, 16 variants (`null_component/`) |
| Routines with derived-type arguments do not inline (< 26.3) | Issue 1 | fixed in 26.3 |

## Unverified

| Report | Where | Status |
|---|---|---|
| A `do concurrent` on the meridional tracer-flux loop gave wrong results on NVHPC 25.9 | comment in `MOM_tracer_hor_diff.F90` (`tracer_hordiff`), which keeps that loop as `!$omp target teams loop` | not retested on 26.3 |
| The OpenMP runtime launched far fewer teams than the work needed in `zonal_mass_flux` | `5b5f6b2b1`; the hand-set `num_teams` in `MOM_continuity_PPM.F90` | not retested; a performance issue only |
