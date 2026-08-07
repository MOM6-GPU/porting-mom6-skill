# Device calls, lost collapse, and transfer parity

> Findings from the `wave_speed` port (`src/diagnostics/MOM_wave_speed.F90`, branch
> `port/MOM_lateral_mixing_coefs`, August 2026, nvfortran **26.3**, `-O0 -mp=gpu -stdpar=gpu
> -gpu=mem:separate,cc80,sm_80 -Mnofma`). Complements doc 08, which covers the *correctness* rules
> for calling a procedure from a device region (`declare target` vs force-inline, polymorphic
> `this`). This doc covers the *performance and crash* traps that produce no compile-time warning,
> plus one host/device transfer bug that this port surfaced in already-merged code.
>
> Everything below was measured, not inferred. Reproducers live in
> `/scratch/cimes/uw4770/ws_experiments/` (`collapse_test.F90`, `collapse_time.F90`, `slice_test.F90`)
> and `/scratch/cimes/uw4770/repro_test/collapse/repro_collapse.F90`.
>
> **§1 was rewritten on 2026-08-04.** Its first version claimed that a call in a `do concurrent`
> defeats the inferred collapse unless the callee has explicit-shape dummies *and* `VALUE` scalars.
> A controlled reproducer disproved the `VALUE` half and most of the rest. What survives is a
> performance ranking, not a collapse switch. The corrected version is below; the retracted claim is
> kept visible in §1.3 because it was acted on in merged code.

---

## 1. `do concurrent`'s collapse is inferred; argument passing changes its cost

Doc 08 §7.3 says a bare `do concurrent` "generally handles `pure`/`elemental` calls without a
mandatory `declare target` — the definition just has to be visible." That is true for correctness.
For performance, how the callee declares its dummies is worth about 30%, and how it takes its input
scalars about another 20%.

nvfortran *infers* the collapse for `do concurrent` (unlike an OpenMP `collapse(n)`, which asserts
it), so it is worth checking. Read it off the `-Minfo` line for the `do concurrent` statement itself:

```
    348, Loop parallelized across CUDA thread blocks, CUDA threads(128) collapse(2) ! blockidx%x threadidx%x
           ! blockidx%x threadidx%x auto-collapsed
```

### 1.1 What "Reference argument passing prevents parallelization" actually means

```
    683, Reference argument passing prevents parallelization: c2_scale
```

This message is anchored to the **line of the call**, and it refers to the loop *containing* that
call — in a Newton iteration that is the serial `do itt=1,max_itt`, which was never going to be
parallelized and should not be. **It says nothing about the enclosing `do concurrent`.** In
`MOM_wave_speed.F90` the message appears for eleven scalars at the `tridiag_det_3d` call site while
the `do concurrent` two lines above reports `auto-collapsed`.

Do not use this message as a collapse indicator. Check the `do concurrent` line for `collapse(2)`.

### 1.2 Measured cost of the argument-passing choices

`repro_collapse.F90` — one thread per column, ten Newton iterations per column, each calling a
`pure` `declare target` routine. Compile-time knobs for dummy shape, `VALUE`, caller array shape,
and callee module. nvfortran 26.3, `-O0 -mp=gpu -stdpar=gpu -gpu=mem:separate,cc80,sm_80 -Mnofma`,
200 reps, checksums identical across all variants:

| callee dummies | input scalars | 360x180x22 | 120x60x75 | collapse? |
|---|---|---|---|---|
| **explicit-shape `(isl:iel,jsl:jel,nk)`** | **`VALUE`** | **0.070** | **0.118** | yes |
| explicit-shape | by reference | 0.090 | 0.139 | yes |
| assumed-shape `(:,:,:)` | `VALUE` | 0.106 | 0.150 | yes |
| assumed-shape | by reference | 0.125 | 0.169 | yes |
| 1-D dummy, caller passes `(i,j,:)` | either | **crash** | 55.9 | yes |

Every variant collapsed. Neither knob is a switch; both are ~20-30% each, and they compose.

The last row is the one that matters most, and it is a different failure entirely: a `(i,j,:)`
section into a 1-D assumed-shape dummy is ~400x slower at ALE size and dies with
`CUDA_ERROR_ILLEGAL_ADDRESS` at benchmark size. That is the device-heap repack of §2, not a collapse
question.

Two knobs that changed **nothing**: whether the caller's own arrays are explicit- or assumed-shape,
and whether the callee lives in the caller's module or another one.

### 1.3 What the retracted claim got wrong, and what is still unexplained

The original table asserted "no collapse" for three of the four dummy/scalar combinations. It came
from `collapse_time.F90`, where `v1` (`do concurrent` + call) genuinely does lose the collapse —
`-Minfo` gives it plain `blockidx%x` / `threadidx%x` with no `collapse(2)`. That much reproduces
today. The error was generalising from one harness to a rule, and reading the "Reference argument
passing" message as the loss indicator (§1.1) when it is not.

`repro_collapse.F90` failed to reproduce the loss under any combination of the four knobs above. The
one structural difference not yet tested is `collapse_time.F90`'s callee declaration
`real, dimension(isl:,jsl:,:)` — assumed-shape whose *lower bounds come from other dummy arguments*,
which my assumed-shape variant (`(:,:,:)`) does not exercise. That is the first thing to try if this
is picked up again.

Practical consequence: the collapse is not as fragile as this doc used to say, but explicit-shape
dummies are still the right default, and they are what makes it natural to pass the whole array plus
`i,j` instead of a section.

### 1.4 Prefer declarations over `-Minline`

`-Minline=reshape,name:<routine>` also inlines the callee (`reshape` is required whenever a dummy has
explicit lower bounds, otherwise nvfortran reports `subprogram not inlined -- array reshaping not
enabled`), which removes the argument-passing question altogether.

Prefer the declaration route anyway: `-Minline` puts a performance requirement in the build system,
where anyone who rebuilds without the flag silently gets the slow version and has no way to know.
Declarations encode it where the reader is already looking. `kappa_shear_column`
(`MOM_kappa_shear.F90`, `port/kappa-shear`) already declares its dummies
`dimension(SZI_(G),SZJ_(G),SZK_(GV)+1)`, arrived at independently.

### 1.5 In-model numbers

`wave_speed` on `benchmark_ALE`, 48 calls: **2.454 s** before the port, **0.072-0.092 s** after.

That spread is one binary measured six times, so treat the `(Ocean wave_speed)` clock as having
roughly ±25% run-to-run scatter at this problem size. Differences smaller than that cannot be read
off one run of each variant — an early comparison here did exactly that and drew the wrong
conclusion.

**Build gotcha if you do use flags:** `ocean_only/<build>/Makefile` line 18 assigns `FCFLAGS` with
`=` (mkmf-generated), overriding `config.mk`. Editing `config.mk` alone does not change an existing
build directory, and make does not track flag changes, so `rm` the affected `.o` as well.

---

## 2. Never pass a non-contiguous array section into a device callee

Passing `Igu(i,j,:)` to an assumed-shape `dimension(:)` dummy makes nvfortran repack the section into
a contiguous temporary **on the device heap, once per thread**. The default heap (~8 MB) is exhausted
once there are enough columns; the failure is a bare `CUDA_ERROR_ILLEGAL_ADDRESS`.

Confirmed with `slice_test.F90` (three variants, identical answers): slices crash at 64800 columns,
slices with `NV_ACC_CUDA_HEAPSIZE=2147483648` pass, whole-array-plus-`(i,j)` passes at the default
heap. `compute-sanitizer --tool memcheck` prints the direct evidence — `Malloc/Free Warning
encountered : Device-side malloc failed ... Address 0x0` — where the plain runtime prints nothing.

**Why there is no "raise the heap size" message.** The familiar `NV_ACC_CUDA_HEAPSIZE` diagnostic (as
in `kappa_shear_column`) comes from *user-declared automatics* in a device routine, which nvfortran
null-checks. A compiler-generated section repack is not null-checked, so the callee's first store goes
through a null pointer instead. Size dependence follows exactly: `nz*8` bytes/thread, ~11 MB for
benchmark against the ~8 MB default, only ~4.3 MB for ALE — which is why ALE never crashed.

Fix: pass the whole 3-D array plus `i,j`. This also lets the caller supply what would otherwise be
automatics in the callee (workaround 0 in `nvfortran-automatic-array-bug.md`).

---

## 3. nvfortran 26.3 construct/call compatibility

| construct | body contains a device call | result |
|---|---|---|
| `do concurrent` | yes | works, and stays collapsed; explicit-shape dummies and `VALUE` scalars are each worth ~20-30% (§1.2) |
| `!$omp target teams distribute parallel do collapse(2)` | yes, explicit-shape dummies | **works** — the merged precedent (`MOM_vert_friction.F90:1443`, kappa-shear's `kappa_shear_column` call site) |
| `!$omp target teams distribute parallel do collapse(2)` | yes, **assumed-shape** dummies | compiles, then dies at runtime with `CUDA_ERROR_MISALIGNED_ADDRESS`; a larger device heap does not help |
| `!$omp target teams loop collapse(2)` | yes | **hard ICE**: `nvnvvmd: error: parse invalid cast opcode for cast from 'float' to 'i8*'` → `NVFORTRAN-F-0155`. With and without `-stdpar`; with either dummy style |
| `!$omp target teams loop collapse(2)` | no | works, fastest of the OpenMP forms |

This is why `SKILL.md`'s fallback ordering was rewritten: it used to recommend `target teams loop` as
the preferred OpenMP fallback, which is unusable for any kernel containing a call on this toolchain.

Also: `local(i,j)` naming the loop's own indices is rejected unconditionally
(`NVFORTRAN-S-1045-DO CONCURRENT index name i may not appear in a locality spec`) — on CPU and GPU,
with and without inlining flags.

---

## 4. Candidate nvfortran bug reports

Four items, in descending order of how clear-cut they are:

1. **`target teams loop` + `declare target` call = ICE.** Unambiguous. Reproducer: `v4` in
   `collapse_test.F90`.
2. **`target teams distribute parallel do` + a call with assumed-shape dummies = misaligned
   address at runtime.** Reproducer: `v7`, same file.
3. **The collapse inference**, as seen in `collapse_time.F90` `v1` only — see §1.3, which withdraws
   the general form of this claim. Arguable as a missed optimization rather than a conformance bug:
   F2018 `do concurrent` *asserts* iteration independence, and the scalars named in the message were
   in the loop's own `local(...)` clause and passed to a `pure` procedure. Not worth filing until
   the trigger is actually isolated.
4. **The `nvompAcquireLock` / `__pgi_uacc_dataonb` crossover.** Under `-mp=gpu -stdpar=gpu` the
   OpenACC/stdpar runtime entry acquires an *OpenMP* runtime lock, i.e. the two runtimes share one
   present-table lock. Not shown to be a bug on its own — see §5, where it turned out to be a red
   herring — but worth knowing when reading stacks.

---

## 5. Transfer parity across if/elseif branches — a real bug this port surfaced

Fixed in **`979be73e6`**, `src/tracer/MOM_tracer_hor_diff.F90`. Pre-existing in merged code; the
`wave_speed` port only raised how often it bit.

`tracer_hordiff` picks a diffusivity three ways:

```
if (use_VarMix) then          ! computes khdt_x/khdt_y/Kh_u/Kh_v on the DEVICE (do concurrent)
                              !   -> had NO transfer at all
elseif (Resoln_scaled) then   ! computes them on the HOST, ends with target update to
else                          ! computes them on the HOST, ends with target update to
endif
if (CS%max_diff_CFL > 0.0) then
  !$omp target update from(khdt_x, khdt_y, Kh_u, Kh_v)   ! the ONLY device->host copy
```

With `USE_VARIABLE_MIXING = True` and `MAX_TR_DIFFUSION_CFL <= 0` (i.e. `benchmark_ALE`), the device
branch runs and the `update from` is skipped, so the **host** diffusive-CFL loop below reads memory
that was never written.

**Why it hid for so long.** The garbage feeds only `num_itts` and an inactive limiter, so runs that
complete are bitwise identical — `ocean.stats` matched the reference every time. And
`num_itts = max(1, ceiling(max_CFL - 4.0*EPSILON(max_CFL)))` has **no upper bound**: usually the
garbage was small and `num_itts` came out 1, but a `max_CFL` of `8.915808E+04` gave
**`num_itts = 89159`**, and the run looked hung. Correct `max_CFL` for that configuration is exactly
`0.0`; with the fix it prints `0.000000E+00` every step.

Measured, alternating builds within one job on one GPU: **11 of 24 runs stalled without the fix, 0 of
8 with it** (Fisher exact, one-tailed p = 0.019).

### 5.1 The generalisable rules

- When you port **one branch** of an if/elseif chain to the device, that branch needs its **own**
  transfer. The sibling branches' directives look like they cover the case and do not.
- A missing `update to` tends to give wrong answers. A missing `update from` may give **right answers
  and wrong behaviour**, if the stale values only reach control flow — an iteration count, a limiter,
  a diagnostic threshold. Bitwise-identical `ocean.stats` does not clear a suspect transfer.
- Unbounded iteration counts derived from field data are a robustness hazard independent of any GPU
  bug. `num_itts` here still has no cap.

### 5.2 How it was actually found, after three wrong hypotheses

Recorded because the *inference* path failed repeatedly and the *measurement* path worked at once.

Wrong turns: (a) blamed the port's ~2200 per-run `map(alloc:)`/`map(delete:)` present-table
mutations, on the strength of a stack sample sitting in `nvompAcquireLock`; (b) concluded the port was
exonerated because the backtrace was in unmodified code — *where* a failure manifests is not *what*
causes it; (c) blamed `cg1`, `wave_speed`'s own output, which instrumentation showed was healthy
(`max = 6.325`, zero non-finite) in the very timestep that exploded.

What worked, in order:

1. `nvidia-smi` — GPU still busy, so not a deadlock.
2. `gdb -p $(pgrep -x MOM6)` — note `-x`; matching the executable *path* also matches `mpirun`'s
   command line and you get a useless bash/mpirun stack.
3. `NVCOMPILER_ACC_NOTIFY=15` — 5.9M kernel launches in `tracer_hordiff` before day 1 against ~300
   expected per call. Proof it was executing, not stuck, and a pointer at the loop. (Caveats: 13 GB of
   trace, and the slowdown is big enough that a stall-detector will misclassify healthy runs — do not
   classify while tracing.)
4. `write(0,...)` of the two suspect scalars per timestep. Found it immediately.

Statistical hygiene for intermittent bugs: separate CPU from GPU runs before computing a rate (mixing
them made a 4-run clean streak look like 8 and wrongly implicated a code change); never co-schedule
two GPU jobs on one node while rate-testing (~3x mutual slowdown reads as a hang); and A/B by
alternating candidate and control **inside one job**.

---

## 6. Cross-references

- `08-cross-module-inlining.md` — correctness rules for device calls. §7.3's "bare `do concurrent` is
  more forgiving" is correct for answers, incomplete for performance; §1 above is the missing half.
- `nvfortran-automatic-array-bug.md` — workaround 0 (caller-supplied 3-D workspace) comes from this
  port; also records that nvfortran heap-allocates automatics.
- `SKILL.md` — "Choosing a loop construct", "Calling a procedure from inside a kernel",
  "When a run appears to hang".
- `references/data-mapping-conventions.md` — the four placement rules; §5 above is the branch-parity
  case they do not currently cover.
