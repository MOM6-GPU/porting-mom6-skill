# Compiler Workarounds — the `dev/gpu` nvfortran/NVHPC Bug Catalogue

> **Purpose.** This is the grep-friendly master index of every nvfortran/NVHPC compiler bug,
> workaround, and directive hit while porting MOM6 to NVIDIA GPUs on `dev/gpu`. Every row below is
> anchored to a `file:line` or a commit hash so it can be verified with `git show <hash>` or
> `sed -n '<line>p' <file>`. Read `00-architecture.md` §7.5 first. Deeper narrative treatments of
> some of these already exist in `04-do-concurrent-patterns.md` §3–4, `06-eos-layer.md`, and
> `08-cross-module-inlining.md` §5 — this document is the one-stop index that also adds items those
> docs don't cover (the A100/25.5 crash, struct-of-arrays flattening, in-flight private-clause bugs,
> the OpenACC→OpenMP migration history, and the `__NVCOMPILER_OPENMP_GPU` macro).
>
> **Method note.** No build or run was performed to produce this document; every entry is derived
> from source comments, directives, and `git log`/`git show` on `dev-gfdl..dev/gpu` plus the
> `origin/*` side branches. "Still open?" reflects what the comment/commit says, not independent
> verification.

---

## 1. Quick-index table

| # | Symptom / bug | Anchor | Workaround | Category | Still open? |
|---|---|---|---|---|---|
| 1 | Polymorphic `this` dispatch causes **runtime errors on GPU** in `do concurrent` | `MOM_EOS_Roquet_rho.F90:261,334,449,465,557,780,852,891`; `MOM_EOS_Wright.F90:107-109,230`; commits `7c7af5572`, `52a1b3954` | Duplicate each `elemental` type-bound proc as a free `_loc` function taking no `this`; array wrappers call `_loc` inside `do concurrent` | Genuine compiler bug (v-table/device dispatch) | **Fixed** for Roquet_rho & buggy_Wright; **all other EOS forms** (linear, UNESCO, Jackett06, TEOS10, Wright_full/red, Roquet_SpV) still use polymorphic elemental dispatch and are **not** GPU-safe |
| 2 | Residual **implicit device copy of `this`**. Verbatim comment (all 5 sites): *"NOTE: There is an implicit copy of \`this\` which cannot yet be prevented."* (only `Wright:1008,1048` add: *"Possibly because Nvidia cannot associate \`this\` with \`EOS%type\`."*) | `MOM_EOS_Wright.F90:1008,1048,1114,1147`; `MOM_EOS_Roquet_rho.F90:817` | None — noted, not fixed | Genuine compiler bug (suspected) | **Open / unresolved** |
| 3 | `do concurrent` **cannot reduce into an array element** (`itmp(i,j)`-style reduction target) | `MOM_tracer_hor_diff.F90:962` | Reduce into a scalar (`itmp`) with `DO_LOCALITY(reduce(max:itmp))`, then assign scalar into the array after the loop | Genuine compiler bug / OpenMP-standard restriction (reduction var must be a named scalar) | Worked around; also documented in `04-do-concurrent-patterns.md` §4.4 |
| 4 | `modulo()` **not implemented on all systems** (incl. NVIDIA GPU device runtime) | `MOM_intrinsic_functions.F90:232`; upstream fix commit `9aea28954` (branch `cuberoot-no-modulo`, merged) | Replace `modulo(e,3)` with explicit `e_a - e_r*3` arithmetic in `rescale_cbrt` | Genuine compiler/library gap | **Fixed**, merged into current source |
| 5 | `x**(1.0/n)` lowers to `exp((1/n)*log(x))` — **imprecise / bit-different** vs. host libm | `MOM_intrinsic_functions.F90:120-131` (`nth_root`, doc comment) | Fixed-iteration Newton's method on `y^n - x = 0` using only `*,+,/`; used by `MOM_barotropic.F90`'s `bt_rem = av_rem**Instep` | Numerical-portability workaround (not a bug, a bit-repro requirement) | Resolved by design |
| 6 | `do concurrent` on the **meridional tracer-flux face loop gives the wrong result** | `MOM_tracer_hor_diff.F90:1464` — `! this gives wrong result when using do concurrent on NVHPC 25.9` | Kept as `!$omp target teams loop collapse(2) private(...)` instead of `do concurrent` | **Genuine compiler bug, version-specific (NVHPC 25.9)** | Open (workaround in place; not re-tested on later NVHPC) |
| 7 | **Early `exit` inside a `do concurrent`-nested loop gives wrong answers** | Commit `e23d6a7b1` — *"NVHPC 25.11 didn't like the early exit and would give wrong answers."* `MOM_tracer_hor_diff.F90` insertion-sort loop | Replace `do k2=k,2,-1 ; if (cond) exit` with an `if`-guarded loop body (`if (cond) then ... endif ; enddo`, no `exit`) | **Genuine compiler bug, version-specific (NVHPC 25.11)** | Fixed by rewrite; pattern (avoid `exit`/`return`/`cycle` inside device-offloaded loop bodies) should be treated as a hard rule |
| 8 | OpenMP runtime **under-launches teams** (17 vs. ~238 expected) for a target-teams region | Commit `5b5f6b2b1` — *"For some reason omp runtime was only starting a kernel with 17 blocks when the openacc version would start it with 238 or something like that."* `MOM_continuity_PPM.F90:701,707` (`nteams = ceiling(...) / 128.`) | Manually compute `num_teams(nteams)` from the tile's iteration count instead of relying on the runtime's default team count | Genuine compiler/runtime scheduling deficiency | **Open / permanent workaround** — the manual `num_teams` clause is load-bearing, not decorative |
| 9 | Mandatory function inlining or **wrong answers** | Commit `3cb184edd` — *"IMPORTANT: However for OpenMP, inlining of ratio_max and flux_elem is MANDATORY. do so with `-Minline=name:ratio_max,name:flux_elem`. Otherwise results are incorrect."* | `-Minline=name:...` compiler flag (later superseded, see #10) | Genuine compiler bug (cross-procedure device call miscompiled without inlining) | Superseded, not re-verified as fixed |
| 10 | Avoiding the fragile `-Minline=name:...` flag list | Commit `4e3f1b758` — *"add !NVF\$ INLINE to ratio_max flux_elem … instead of compiling with -Minline=name:flux_elem,name:flux_elem_OBC,name:ratio_max, can compile with -Minline=pragma instead."* | Source-level `!NVF$ INLINE` directive + `-Minline=pragma` build flag | Build-portability improvement | Superseded again by #11 |
| 11 | `!NVF$ INLINE` replaced with a portable directive; **large -O2 perf win** | Commit `93dbbd36e` — *"remove nvf inline and replace with intel forceinline … Significantly improves performance of blocked zonal/meridional_mass_flux at -O2"* | `!DIR$ ATTRIBUTES FORCEINLINE :: flux_elem` / `flux_elem_OBC` (`MOM_continuity_PPM.F90:1086,1149`); `ratio_max` (`:3086`) now has **no** inline directive at all (relies on `pure function` + compiler default) | Performance-driven directive swap | Current state; **`-Minline`/forceinline is still effectively load-bearing for correctness per #9**, so this is a correctness-adjacent perf change, not a pure perf change |
| 12 | `thread_limit(128)` removed from the same `target teams` region that got `num_teams` | Commit `4e3f1b758` diff: `-!$omp target teams num_teams(nteams) thread_limit(128)` → `+!$omp target teams num_teams(nteams)` (`MOM_continuity_PPM.F90:707`) | Dropped `thread_limit` clause; `num_teams` alone retained | Perf/behavior tuning | Not explained in commit message — flagged for follow-up |
| 13 | `omp target teams loop` region **crashed / segfaulted** | Commit `5274c3a8e` "fix segfault" — removed `!$omp target` / `!$omp parallel loop collapse(2)` wrapper around a k-recurrence loop in `PressureForce_FV_Bouss`, replaced with `do concurrent` | `do concurrent (j=js:je, I=Isq:Ieq)` in place of `!$omp target … !$omp parallel loop collapse(2) … !$omp end target` | Genuine compiler/runtime bug (crash) | Fixed by rewrite |
| 14 | General instability of `!$omp target` regions on some compilers | Commit `0f05b360f` — *"PGF: Convert omp target region to do concurrents. Some compilers do not handle the omp target syntax very well. Switching to do concurrent seems to minimize these issues."* (Note: "PGF" here = the **Pressure Gradient Force** module prefix used throughout `MOM_PressureForce_*.F90`/`MOM.F90` commit messages, **not** the PGI/PGF90 compiler.) | Prefer `do concurrent` over `!$omp target`/`!$omp parallel loop` where both are equally expressive | General reliability preference, feeds guiding principle §0.3 of `00-architecture.md` | Ongoing house rule |
| 15 | Reversion: `omp target teams loop` → `do concurrent` | Commit `e8b0ecfbf` "omp target teams loop -> do concurrent" (`MOM_continuity_PPM.F90`, −85/+54 lines) | Same direction as #14 | Reliability/perf preference | Merged |
| 16 | **A100 (Stellar) crash with nvfortran 25.5** | Commit `2108e0eba` — *"Remove eta_bt transfer from find_eta_2d that was causing crashes on stellar A100s with nvfortran 25.5"* — removed `!$omp target enter data map(to: eta_bt) if (present(eta_bt))` from `find_eta_2d` (`MOM_interface_heights.F90`) | Delete the conditional `map(to:)` of an **optional** dummy argument guarded by `if (present(...))`; rely on `eta_bt` being resident via its caller's own mapping | **Genuine compiler bug, version+arch specific (NVHPC 25.5, A100)** | Fixed by removal; the underlying pattern (`map(...) if (present(optional_arg))`) is suspect in general — see #17 |
| 17 | Conditional `!$omp target enter data if(...) map(to:...)` lines commented out during the OpenACC→OpenMP migration (**see corrected history below — the original "four `present()` lines, never enabled" claim was wrong**) | Commit `f74525ae8` "Transition OpenACC to OpenMP" — *"We lose present() but overall it seems to work."* added **four** `!!!$omp target enter data if(...) &` commented blocks to `MOM_PressureForce_FV.F90`, of which **only one** uses the Fortran `present()` intrinsic (`if(present(pbce))`); the other three are `if(use_EOS)`, `if(use_p_atm)`, `if(.not. use_p_atm)` | The `if(present(pbce))` block was **un-commented and enabled the same day** by `d4ba8d69d` "OpenMP: PBCE on GPU" (2024-12-06), then **refactored out** by `9bfe7d358` "PF: Move pbce and eta management out of fn" (2025-01-14). **None of the four lines exist in current source.** | The "We lose present()" quote refers to the Fortran **`present(optional_arg)` intrinsic**: the diff carries no OpenACC `present()` data clauses at all, only commented-out `if(present(pbce)) map(to: pbce)` conditional maps | **Not open in the way originally stated.** The `present(pbce)+map` pattern *was* enabled and used for ~5 weeks, then removed by a **refactor**, not because of a compiler crash. Its link to bug #16 (an actual A100/25.5 crash on `map(to:eta_bt) if(present(eta_bt))`) is a foreshadowing of the identical construct, not causal evidence — see §2.1 #17 |
| 18 | Struct-of-arrays member arrays are expensive to **attach/detach** on device | Commit `1865612de` "Convert structs of arrays to flat arrays" — *"Flattening these arrays halves time when compiling for GPU. Lots of time was being spent 'attaching' and 'detaching' the member arrays to/from each struct on the GPU."* `MOM_tracer_hor_diff.F90` (−155/+131 lines) | Flatten arrays-of-derived-type-members into plain flat arrays indexed by a combined index, at the cost of ~20% more memory (700k→830k elements/array in benchmark) | Performance workaround for deep-copy/attach overhead (not a bug) | Merged; documented also in `00-architecture.md` §2.3 and `01-memory-control-structures.md` |
| 19 | `do concurrent` locality specifiers (`local`, `reduce`) **not supported on all compilers in active use** | Commits `d2a72eddd` "DO_LOCALITY compatibility macro", `cd178dd52` "DO_LOCALITY() bugfix" | `DO_LOCALITY(X)` macro in `src/framework/do_concurrent_compat.h`: expands to `X` if `HAVE_FC_DO_CONCURRENT_LOCAL` else to `;` (a no-op that avoids a dangling line-continuation `&` parse error). Feature-detected by `ac/m4/mom6_fc_do_concurrent_local.m4` → `ac/configure.ac:172` → `HAVE_FC_DO_CONCURRENT_LOCAL` | Portability/feature-detection (not nvfortran-specific; guards **against compilers that don't have it**, e.g. older gfortran) | Resolved by design; permanent infra |
| 20 | `do concurrent` formatting broke some compilers' parsers | Commit `e5444b4e5` "expand and indent do concurrents for gcc" | Reformat compact `do concurrent (...) ; stmt ; enddo` one-liners into expanded/indented multi-line form | Cross-compiler portability (gcc/gfortran, not nvfortran) | Merged |
| 21 | `makedep` (the in-house Fortran dependency scanner) **couldn't parse empty macro functions** or valueless `-D` flags, needed for `HAVE_FC_DO_CONCURRENT_LOCAL`-style defines | Commit `6474597b1` "Makedep: Support empty macro functions" | Parser patch: supports `#define foo(x)` with no body, and command-line `-DMACRO` without `=value` | Build-tooling fix, prerequisite for #19's infra | Merged |
| 22 | `__NVCOMPILER_OPENMP_GPU` macro used to change **default block sizes and disable CPU-only early-exit optimizations** | `MOM_continuity_PPM.F90:1406,1434,1442,2416,2443,2450,3120`; `MOM_CoriolisAdv.F90:2094`; commit `93dbbd36e` body: *"if `__NVCOMPILER_OPENMP_GPU` macro is defined (to be replaced at a later time), set default n?block to 0"* | `#ifdef __NVCOMPILER_OPENMP_GPU` / `#ifndef` guards: (a) default `niblock/njblock/nkblock = 0` (whole-domain, GPU) vs. `32/4/1` (CPU); (b) disable the `domore`/`if (.not.domore) exit` early-exit convergence check entirely under GPU builds — GPU kernels always run the fixed iteration count instead of testing for early convergence | Both a compile-time tuning knob (block sizes) **and** an early-exit avoidance identical in spirit to bug #7 (early exit from a converging loop is unsafe/meaningless once the loop body is spread across GPU threads) | Permanent, by design; commit flags the macro itself as *"to be replaced at a later time"* — i.e. considered a stopgap |
| 23 | Missing variables in `private()`/`DO_LOCALITY(local(...))` clauses causing suspected **data races** in the BBL viscosity kernel | Unmerged branch `origin/merge-omp-debug`, commits `7a51e5fb3` "cdrag locality fixes?", `fc4068582` "Private D_vel_[pm]wq" — question mark in the commit subject signals this was exploratory/unconfirmed debugging | Add `cdrag`, `cdrag_sqrt`, `cdrag_sqrt_H`, `cdrag_sqrt_H_RL`, `D_vel_p`, `D_vel_m` to `private()`/`DO_LOCALITY(local(...))`/`DO_LOCALITY(local_init(...))` clauses of the `target teams loop collapse(2) thread_limit(128)` region in `MOM_set_viscosity.F90:803-812` | **Own-code porting bug class** (incomplete privatization), not a compiler bug — but a recurring hazard: any scalar written inside a `target teams`/`do concurrent` body must be explicitly privatized or it silently races | **Unmerged / in-flight** as of this writing — not yet landed on `dev/gpu` |
| 24 | Data-upload / mapping bugs introduced by refactors, mistaken for compiler bugs | Commits `6f3a42d53` "Horvisc: Grid bugfix on GPU (Leith)" (*"The Leith params introduced new conditional loops for some of the horizontal viscosity metric arrays. This mangled some of the uploads to the GPU."*), `b404caae2` "Horvisc: biharm bugfix" (*"Forgot to copy cs%biharm_const2_xx"*), `799836a54` "dev/gfdl merge: CS%ntrunc vertvisc_limit_vel bug" (*"Accidentally added to CS%ntrunc inside the loop, rather than outside."*) | Fix the missing `map`/copy or the misplaced accumulation | **Own-code bug, not a compiler bug** | Fixed; included here specifically as a *negative example* — don't misattribute application bugs to nvfortran |

---

## 1a. Symptom-signature → decision-rule index (for a debugging agent)

Match an observed failure against a **symptom signature** in the middle column, then apply the
**decision rule**. `SILENT` = wrong answer with **no crash and no diagnostic** (the dangerous class).

| # | Symptom signature (what you observe) | Decision rule (what to do) |
|---|---|---|
| 1 | Runtime error / illegal-address / device-dispatch fault the moment an `elemental` **type-bound** EOS proc (`this%…`) is invoked inside a `do concurrent`/`target` region | Call a free `_loc` function that takes **no `this`**; keep the type-bound proc as a thin host wrapper |
| 2 | No crash; only the `! implicit copy of `this`` comment — a `class(...)` actual still appears in a device loop | Port the remaining `this`-taking call (incl. `density_anomaly_elem_buggy_Wright`) to a `_loc` variant so `this` leaves the loop; else accept the copy |
| 3 | Build error or wrong reduction value when the `reduce`/`!$omp` target is an **array element** `itmp(i,j)` | Reduce into a **named scalar**, assign the scalar into the array element after the loop |
| 4 | Link/runtime "unsupported intrinsic" **or** wrong value from `modulo()` on the device | Replace `modulo(e,3)` with arithmetic `e_a - e_r*3` |
| 5 | `SILENT` bit-level divergence vs host in `x**(1.0/n)` (lowered to `exp((1/n)*log x)`) | Use fixed-iteration Newton `nth_root` (only `* + /`) |
| 6 | `SILENT` wrong numeric result, **only on NVHPC 25.9**, on the meridional tracer-flux face loop, when that loop is a `do concurrent` | **Do not** convert to `do concurrent`; keep `!$omp target teams loop collapse(2) private(...)` |
| 7 | `SILENT` wrong answer in the tracer insertion-sort under **NVHPC 25.11**, when the inner loop uses an early `exit` | Replace `exit` with an `if`-guarded body; **never** `exit`/`return`/`cycle` inside a device-offloaded loop |
| 8 | Kernel launches far fewer teams/blocks than the OpenACC equivalent (e.g. **17 vs ~238**); large perf loss, correct answer | Add manual `num_teams(ceiling(real(tile_iters)/128.))` |
| 9 | Wrong answers (**not** merely slow) when `ratio_max`/`flux_elem` are **not inlined** into a device region | Force inlining (`!DIR$ ATTRIBUTES FORCEINLINE`, or historically `-Minline=name:…`) |
| 10/11 | Build fragility from a hand-maintained `-Minline=name:…` list; or poor `-O2` perf of blocked mass-flux | Source-level `!DIR$ ATTRIBUTES FORCEINLINE :: name` (current) |
| 12 | *(no observable signature — undocumented `thread_limit(128)` removal)* | None; flagged for follow-up |
| 13 | Segfault in a `!$omp target`/`parallel loop` region wrapping a **k-recurrence** in `PressureForce_FV_Bouss` | Rewrite as `do concurrent (j=…, I=…)` |
| 14/15 | Assorted crashes/miscompiles in `!$omp target` regions that are equally expressible as `do concurrent` | Prefer `do concurrent` |
| 16 | **Crash** on Stellar **A100 + NVHPC 25.5** from `map(to: eta_bt) if (present(eta_bt))` (optional dummy) in `find_eta_2d` | Delete the conditional `map` of the optional arg; rely on the caller's mapping |
| 18 | ~2× GPU wall time dominated by "attaching/detaching" struct-of-arrays member arrays | Flatten arrays-of-derived-type-members into flat indexed arrays |
| 19 | Build error: compiler rejects `do concurrent (...) local(...)`/`reduce(...)` | Wrap specifiers in `DO_LOCALITY(...)`; feature-detect via `HAVE_FC_DO_CONCURRENT_LOCAL` |
| 20 | gcc/gfortran parse error on compact one-line `do concurrent (...) ; stmt ; enddo` | Expand/indent into multi-line form |
| 21 | `makedep` fails on `#define foo(x)` (empty body) or a valueless `-DMACRO` | Apply the makedep parser patch (`6474597b1`) |
| 22 | *(compile-time knob, not a failure)* GPU build wants whole-domain blocks / must skip CPU-only early-exit convergence tests | `#ifdef __NVCOMPILER_OPENMP_GPU` → block size 0 and drop `if(.not.domore) exit` |
| 23 | Suspected **data race** / nondeterministic wrong answers in the BBL viscosity `target teams loop` | Add **every** scalar written in the body (`cdrag*`, `D_vel_p/m`, …) to `private()`/`DO_LOCALITY(local[_init])` |
| 24 | Wrong answers on GPU appearing right after a refactor, resembling a compiler bug | **First** check for a missing `map`/copy or a misplaced accumulation in your own diff — *before* blaming nvfortran |

---

## 2. What's a genuine nvfortran/NVHPC compiler bug vs. OpenMP semantics/perf tuning

Sweeping the whole set above, three clusters emerge. Keep them separate — conflating them is the
single biggest risk of this catalogue being misread.

### 2.1 Genuine compiler bugs (miscompilation/crash, version-identifiable)

- **#1 / #2** — polymorphic `class(...) :: this` dispatch inside `do concurrent`/target regions:
  runtime errors (fixed via `_loc` free functions) and a residual implicit copy (unfixed).
  **Verified nuance on #2 ("`this` unused" is only partly true).** Reading the five loop bodies:
  - `MOM_EOS_Wright.F90:1114,1147` and `MOM_EOS_Roquet_rho.F90:817` (the `calculate_density_derivs_*`
    routines) call the `..._loc` free function inside the `do concurrent` and **never reference `this`**
    — the pure "implicit copy even though unused" case.
  - `MOM_EOS_Wright.F90:1008,1048` (`calculate_density_array_2d/3d_buggy_Wright`) are different: the
    `if (present(rho_ref))` branch still calls `density_anomaly_elem_buggy_Wright(this, T, S, p, rho_ref)`
    — i.e. `this` **is** syntactically passed into the device loop, because **no `_loc` variant of the
    anomaly function exists** (only `density_elem_buggy_Wright_loc`, used in the `else` branch, drops it).
    The anomaly function itself never reads `this` (it uses module-level coefficients `a0,a1,b0,c0,…`), so
    the argument is *dead*, but the compiler still materialises the copy because a `class(...)` actual
    appears in the call. So the fix here is not "the copy is spurious"; it is "port the anomaly path to a
    `_loc` free function too, then `this` disappears from the loop." Whether that removes the copy is
    unverified (no build was run).
- **#3** — `do concurrent` reduction into an array element instead of a named scalar.
- **#6** — NVHPC 25.9: `do concurrent` gives a **silently wrong numerical result** (no crash) on the
  meridional tracer-flux loop in `MOM_tracer_hor_diff.F90`. This is the scariest class of bug because
  it doesn't crash — it silently corrupts answers.
- **#7** — NVHPC 25.11: early `exit` from a loop nested inside `do concurrent` gives wrong answers.
- **#8** — OpenMP runtime under-launching teams (17 instead of ~238) relative to the equivalent
  OpenACC code — a scheduling/heuristic bug in the runtime, not the compiler proper, but still a
  genuine defect requiring a manual `num_teams` override.
- **#9** — (historical) without explicit inlining of `ratio_max`/`flux_elem`, OpenMP-compiled results
  were *incorrect*, not just slow — points at a miscompilation of the un-inlined cross-procedure call
  inside a device region.
- **#13** — segfault in a `!$omp target` region wrapping a recurrence loop.
- **#16** — A100 + NVHPC 25.5 crash from mapping an **optional** dummy argument
  (`map(to: eta_bt) if (present(eta_bt))` in `find_eta_2d`, `MOM_interface_heights.F90`); this one is a
  **confirmed crash**, fixed by removing the transfer in `2108e0eba`.
- **#17** — *(downgraded — see the corrected table row 17).* Originally presented as "the same
  `if(present())+map` pattern flagged suspect a release earlier and never re-enabled." That framing does
  **not** survive verification: in `f74525ae8` only one of the four commented `if(...)` blocks used
  `present()`, and that one (`if(present(pbce))`) was **enabled the same day** (`d4ba8d69d`) and later
  removed by a plain **refactor** (`9bfe7d358`), not because of a crash. So #17 is **not independent
  evidence** that `present()+map` is a compiler bug — bug #16 remains the only *demonstrated* instance.
  The construct is identical, however, which is why the foreshadowing link below still stands.
  > **Resolved (2026-07-14):** *"We lose present()"* means the Fortran `present(optional_arg)`
  > **intrinsic**, not the OpenACC `present()` data clause. The `f74525ae8` diff contains **no**
  > OpenACC `present()` data clauses anywhere; what it does contain are commented-out conditional
  > OpenMP maps of the form `!!!$omp target enter data if(present(pbce)) map(to: pbce)` — the author
  > tried intrinsic-`present()` conditional maps and disabled them. Both meanings of `present()` are
  > therefore the same meaning here, and the #17-↔-#16 foreshadowing link **stands**: `f74525ae8`
  > disabled the very construct (`map(to: X) if(present(X))`) that later crashed an A100 under
  > NVHPC 25.5 and was removed outright by `2108e0eba`.

### 2.2 OpenMP/`do concurrent` semantics constraints (not bugs — the standard genuinely requires this)

- **#3** (also listed above) is arguably standards-conformant: `do concurrent` reductions must name a
  scalar reduction variable; MOM6's fix (reduce into a scalar, then store) is the *correct* idiom, not
  a workaround for broken behavior.
- **#22**'s early-exit removal under `__NVCOMPILER_OPENMP_GPU`: an `exit`/early-termination
  convergence test is meaningless once loop iterations are spread across GPU threads that don't share
  loop-carried state — this is an inherent semantic mismatch between "iterate until converged" serial
  algorithms and "all iterations execute independently" parallel loops, not a compiler defect.
- **#19/#20** are cross-compiler portability guards (`DO_LOCALITY`, GCC formatting), not nvfortran bugs
  at all — several other compilers are the ones lacking the feature.

### 2.3 Performance-only tuning (correct either way, chosen for speed)

- **#8**'s `num_teams` computation (once the under-launch defect is worked around, the specific
  formula `ceiling(tile_size/128.)` is a tuning choice).
- **#11**'s switch from `!NVF$ INLINE` to `!DIR$ ATTRIBUTES FORCEINLINE`, explicitly for an "-O2 perf"
  win (though see the correctness caveat cross-referenced from #9).
- **#12**'s removal of `thread_limit(128)` (undocumented reason — flagged, not resolved).
- **#18**'s struct-of-arrays flattening (halved GPU time; a memory/attach-cost optimization).
- **#14/#15**'s general preference for `do concurrent` over `!$omp target` (partly reliability, partly
  simplicity — commit messages don't quantify a perf delta).

---

## 3. nvfortran/NVHPC versions referenced in source or commit history

| Version | Where mentioned | What broke |
|---|---|---|
| **NVHPC 25.5** | Commit `2108e0eba` | Crash on **Stellar A100** GPUs from `map(to: eta_bt) if (present(eta_bt))` in `find_eta_2d` |
| **NVHPC 25.9** | `MOM_tracer_hor_diff.F90:1464` | `do concurrent` gives a wrong numerical result on the meridional tracer-flux face loop; kept as `!$omp target teams loop collapse(2)` instead |
| **NVHPC 25.11** | Commit `e23d6a7b1` | Early `exit` inside a loop nested in `do concurrent` produces wrong answers in the tracer insertion-sort |

No other nvfortran point-release is named in-tree. The `__NVCOMPILER_OPENMP_GPU` predefined macro
(used at `MOM_continuity_PPM.F90:1406` et al. and `MOM_CoriolisAdv.F90:2094`) is version-agnostic — it
detects "compiling for NVIDIA GPU OpenMP offload" in general, not a specific release, and its commit
(`93dbbd36e`) explicitly calls it a stopgap: *"to be replaced at a later time."*

The **ifort** bit-reproducibility issue in commit `5f413739b` ("Barotropic: frhat[uv] HYBRID repro
fix") is **not** an nvfortran bug — included here only as a contrast case: *"it seems possible that
Intel has added a reduction-like optimization, even at -O0"* in a loop-fission refactor. Cross-compiler
bit-repro bugs are a distinct risk category from the NVHPC-specific ones above.

---

## 4. Directive / build-flag reference

| Directive / flag | Purpose | Where used | Anchor |
|---|---|---|---|
| `!$omp declare target` | Marks a `pure`/`elemental` helper as device-callable so it can be invoked from inside a `do concurrent`/`target` region without a host round-trip | 21 occurrences across `MOM_coms.F90`, `MOM_intrinsic_functions.F90`, `MOM_set_viscosity.F90`, `MOM_vert_friction.F90` | e.g. `MOM_intrinsic_functions.F90:51` (`cuberoot`), `:133` (`nth_root`), `:181` (`rescale_cbrt`), `:246` (`descale`); `MOM_vert_friction.F90:437,2101,2611` |
| `!$omp target teams num_teams(N)` | Manually sets the team count because the OpenMP runtime under-launched (bug #8) | `MOM_continuity_PPM.F90:707,1811` | `nteams = ceiling(real((j_end-j_start+1)*(i_end-i_start+1))/128.)` computed just above each site (`:701`, `:1806`) |
| `thread_limit(N)` | Caps threads/team; used once in the current tree, was removed elsewhere (bug #12) | `MOM_set_viscosity.F90:803` (`thread_limit(128)`); removed from `MOM_continuity_PPM.F90` by `4e3f1b758` | — |
| `collapse(2)` | Flattens two loop levels into one iteration space for a `target teams loop`/`parallel loop` | Pervasive — `MOM_vert_friction.F90:737,938,1223,1255`; `MOM_set_viscosity.F90:803`; `MOM_tracer_hor_diff.F90:1465` | — |
| `!DIR$ ATTRIBUTES FORCEINLINE :: name` | Forces inlining of a cross-procedure call inside a device loop; supersedes `!NVF$ INLINE`/`-Minline=name:...` (bugs #9-#11) | `MOM_continuity_PPM.F90:1086` (`flux_elem`), `:1149` (`flux_elem_OBC`) | Introduced by `93dbbd36e` |
| `-Minline=name:ratio_max,name:flux_elem` (build flag, **not** in-tree) | Historical mandatory-inline requirement before the source-level directive existed | Commit `3cb184edd` message | Superseded — no longer needed once `!DIR$ ATTRIBUTES FORCEINLINE` is in source |
| `DO_LOCALITY(X)` macro | Conditionally applies `do concurrent` locality specifiers (`local`, `local_init`, `reduce`) only if the compiler supports them | `src/framework/do_concurrent_compat.h`; used in `MOM_continuity_PPM.F90` (11×), `MOM_CoriolisAdv.F90` (42×), `MOM_barotropic.F90` (5×), `MOM_tracer_hor_diff.F90` (9×), `MOM_tracer_advect.F90` (8×), `MOM_coms.F90` (3×), `MOM_vert_friction.F90` (6×), `MOM_set_viscosity.F90` (6×), `MOM_sum_output.F90` (4×) | Defined by `HAVE_FC_DO_CONCURRENT_LOCAL` (below) |
| `HAVE_FC_DO_CONCURRENT_LOCAL` | Autoconf-detected macro: does this Fortran compiler support `do concurrent (...) local(...)`? | `ac/m4/mom6_fc_do_concurrent_local.m4`, invoked from `ac/configure.ac:172` | Feeds `DO_LOCALITY(X)` — see `d2a72eddd`, `cd178dd52` |
| `__NVCOMPILER_OPENMP_GPU` (predefined macro, not MOM6-defined) | Detects "compiling for NVIDIA GPU via OpenMP offload" to switch default block sizes (`0` vs `32/4/1`) and to disable CPU-only early-exit convergence checks | `MOM_continuity_PPM.F90:1406,1434,1442,2416,2443,2450,3120`; `MOM_CoriolisAdv.F90:2094` | Introduced by `93dbbd36e`; explicitly called a stopgap in that commit's message |
| `omp_offload` (optional Fortran argument, not a directive) | Forwarded to `mpp_do_group_update` so halo exchanges use device-resident (GPU-aware MPI) buffers | `config_src/infra/FMS2/MOM_domain_infra.F90:1143`, passed `.true.` at ~25 call sites | See `03-openmp-mapping.md`, `11-halos-domains.md` (per `00-architecture.md` §7.3) |

---

## 5. Chronology (oldest → newest, by commit)

1. `f74525ae8` (2024-12-06) "Transition OpenACC to OpenMP" — *"We lose present() but overall it
   seems to work."* Added four commented `!!!$omp target enter data if(...) &` blocks to
   `MOM_PressureForce_FV.F90`; **only one used `present()`** (`if(present(pbce))`). That one was
   **enabled the same day** by `d4ba8d69d` "OpenMP: PBCE on GPU" and later removed by the refactor
   `9bfe7d358` (2025-01-14). "We lose present()" refers to the Fortran **`present()` intrinsic**, not
   the OpenACC data clause — the diff has no OpenACC `present()` clauses, only the commented-out
   `if(present(pbce))` conditional maps (see table row 17). Corrected from the original "commented
   out, never enabled" claim.
2. `5274c3a8e` (2025-10-23) "fix segfault" — `omp target`+`parallel loop` around a k-recurrence
   crashed; replaced with `do concurrent`.
3. `1865612de` (2025-11-27) "Convert structs of arrays to flat arrays" — attach/detach cost halved
   GPU time in `MOM_tracer_hor_diff`.
4. `9aea28954` (upstream `dev-gfdl`, merged) "cuberoot: Replace modulo() with arithmetic ops" —
   `modulo()` not implemented on all platforms including NVIDIA GPU.
5. `0f05b360f` (2025-08-07) "PGF: Convert omp target region to do concurrents" — *"Some compilers do
   not handle the omp target syntax very well."*
6. `3cb184edd` (2026-03-17) "use openmp instead of openacc" — mandatory `-Minline` requirement first
   documented.
7. `5b5f6b2b1` (2026-03-17) "add teams spec to problematic target region" — manual `num_teams` to fix
   OpenMP runtime under-launching teams (17 vs. ~238).
8. `4e3f1b758` (2026-05-06) "add !NVF\$ INLINE to ratio_max flux_elem" — source-directive alternative
   to `-Minline=name:...`; also drops `thread_limit(128)` from the `num_teams` region.
9. `e8b0ecfbf` (2026-05-01) "omp target teams loop -> do concurrent".
10. `2108e0eba` (2026-04-30) "Remove eta_bt transfer from find_eta_2d that was causing crashes on
    stellar A100s with nvfortran 25.5".
11. `7c7af5572` / `52a1b3954` — EOS `_loc` free-function pattern for Roquet_rho/Wright polymorphic
    `this` runtime errors.
12. `93dbbd36e` (Kblock continuity reconstruction) — `!NVF$ INLINE` → `!DIR$ ATTRIBUTES FORCEINLINE`;
    introduces `__NVCOMPILER_OPENMP_GPU` macro for block-size defaults and early-exit removal.
13. `e23d6a7b1` (2026-03-31) "swap early exit to if guard in insert sort" — NVHPC 25.11 wrong answers
    from early `exit`.
14. `MOM_tracer_hor_diff.F90:1464` NVHPC 25.9 `do concurrent` wrong-result comment (commit not
    independently identified by hash in the sweep; comment is in the current merged source).
15. *(unmerged, in-flight)* `origin/merge-omp-debug`: `7a51e5fb3`/`fc4068582` — suspected missing
    `private()`/`DO_LOCALITY(local(...))` variables in the BBL viscosity `target teams loop` region.

---

## 6. Explicitly NOT a compiler bug (contrast cases, so they aren't miscatalogued later)

- **`buggy_Wright_EOS` / "buggy" naming** (`MOM_EOS_Wright.F90:5,43` etc.) — this is an **upstream
  MOM6 legacy EOS variant name**, predating the GPU port, deliberately retaining old science-level
  arithmetic bugs (*"a poor implementation (missing parenthesis and bugs)"*) for backward answer
  reproducibility. It is unrelated to nvfortran; don't confuse "buggy" in the type name with an
  nvfortran defect. (It happens to be one of only two EOS forms ported to `_loc` free-function form —
  see item #1 — which is why it appears throughout this catalogue.)
- **`5f413739b`** ifort bit-repro regression — a different compiler (Intel), included in §3 only as a
  contrast case.
- **`6f3a42d53`, `b404caae2`, `799836a54`** (item #24) — missed `map`/copy and misplaced accumulation
  bugs introduced by the GPU-porting authors themselves, not by the compiler.
- **`e5444b4e5`, `d2a72eddd`, `cd178dd52`** (`DO_LOCALITY`, GCC formatting) — these guard **against
  compilers that lack a feature nvfortran has**, not against an nvfortran defect.

---

## 7. Cross-references

- `00-architecture.md` §7.1 (EOS polymorphism), §7.5 (this catalogue's origin note), §5 (k-blocking,
  `__NVCOMPILER_OPENMP_GPU` block-size defaults).
- `04-do-concurrent-patterns.md` §3 (`num_teams` category A/B), §4.4 (array-element reduction bug),
  §5.1-5.2 (early exit, `modulo()`, polymorphic dispatch narrative).
- `06-eos-layer.md` — full detail on the `_loc` free-function pattern and which EOS forms remain
  polymorphic/unported.
- `08-cross-module-inlining.md` §3 (the exact `-Minline` requirement), §5 (nvfortran-refused device
  constructs).
- `01-memory-control-structures.md` — struct-of-arrays flattening (item #18) in the CS-member context.
- `03-openmp-mapping.md` §1.4 ("Balance bugs and their fixes") for the broader map/unmap bug history
  that items #16, #17, #24 are drawn from.

---

## Verification notes

Independent Opus verification against source + `git` on `dev/gpu` (no build/run performed).

**Confirmed verbatim / hash-accurate (spot-checked all 24 quick-index rows and the §4 directive table):**

- **Comments, verbatim:** `MOM_tracer_hor_diff.F90:1464` `! this gives wrong result when using do
  concurrent on NVHPC 25.9`; `:962` `! nvfortran do concurrent cannot reduce array elements`; all five
  `! NOTE: There is an implicit copy of `this` which cannot yet be prevented.` sites
  (`MOM_EOS_Wright.F90:1008,1048,1114,1147`, `MOM_EOS_Roquet_rho.F90:817`); `buggy_Wright`
  "poor implementation (missing parenthesis and bugs)" at `MOM_EOS_Wright.F90:5`, type at `:43`.
- **Commit messages, verbatim:** `e23d6a7b1` ("NVHPC 25.11 didn't like the early exit and would give
  wrong answers"), `5b5f6b2b1` (17-vs-238 teams), `3cb184edd` (MANDATORY `-Minline`), `4e3f1b758`
  (`-Minline=pragma` alternative **and** the `-…thread_limit(128)` → `+…` drop, confirmed in the diff),
  `93dbbd36e` ("remove nvf inline and replace with intel forceinline … at -O2" and the
  `__NVCOMPILER_OPENMP_GPU` block-size note), `2108e0eba` (Stellar A100 / nvfortran 25.5),
  `5f413739b` ("Intel has added a reduction-like optimization, even at `-O0`"), and every hash in
  §6/§24 (`6f3a42d53`, `b404caae2`, `799836a54`, `1865612de`, `5274c3a8e`, `0f05b360f`, `e8b0ecfbf`,
  `9aea28954`, `d2a72eddd`, `cd178dd52`, `e5444b4e5`, `6474597b1`, `52a1b3954`).
- **Anchors, exact:** `FORCEINLINE` at `MOM_continuity_PPM.F90:1086,1149`; `ratio_max` `pure function`
  at `:3086` with **no** inline directive; `num_teams` at `:707,1811` with `nteams` at `:701,1806`;
  `thread_limit(128)` present at `MOM_set_viscosity.F90:803`; `!$omp declare target` at
  `MOM_intrinsic_functions.F90:51,133,181,246` and `MOM_vert_friction.F90:437,2101,2611` (21 total
  across the four named files); `modulo()` comment at `MOM_intrinsic_functions.F90:232`; all eight
  `__NVCOMPILER_OPENMP_GPU` guards (`MOM_continuity_PPM.F90:1406,1434,1442,2416,2443,2450,3120`;
  `MOM_CoriolisAdv.F90:2094`); `DO_LOCALITY` counts (continuity 11, CoriolisAdv 42, tracer_hor_diff 9);
  `omp_offload` dummy at `MOM_domain_infra.F90:1143`. `e23d6a7b1`'s `exit`→`if`-guard rewrite and
  `5274c3a8e`'s `!$omp target`→`do concurrent` rewrite both match the described workaround.
- **Branch `origin/merge-omp-debug`:** `7a51e5fb3` "cdrag locality fixes?" (question mark present) and
  `fc4068582` "Private D_vel_[pm]wq", both by Marshall Ward, both touch `MOM_set_viscosity.F90`;
  the added `local`/`local_init`/`private` variables (`cdrag_sqrt`, `cdrag_sqrt_H`, `cdrag_sqrt_H_RL`,
  `cdrag`, `D_vel_p`, `D_vel_m`) match. Still unmerged on `dev/gpu`.

**Corrected:**

1. **Row 2 / §2.1** — the blanket "`this` is unused inside the device loop" is only true for
   `Wright:1114,1147` and `Roquet_rho:817` (which call `_loc`). At `Wright:1008,1048` the
   `present(rho_ref)` branch **passes `this`** to `density_anomaly_elem_buggy_Wright` (no `_loc` variant
   exists); the argument is dead inside that function but is still syntactically present in the loop.
   Fix framing added.
2. **Row 17 / §2.1 #16-#17 / §5 item 1** — the original "**four** `if(present(...)) map(to:...)` lines
   left commented out, **never enabled**, foreshadowing bug #16" was substantially wrong. In
   `f74525ae8` only **one** of the four commented blocks uses `present()` (`if(present(pbce))`; the
   others are `if(use_EOS)`, `if(use_p_atm)`, `if(.not. use_p_atm)`). That one block was **un-commented
   and enabled the same day** by `d4ba8d69d` "OpenMP: PBCE on GPU", then removed ~5 weeks later by the
   **refactor** `9bfe7d358` "PF: Move pbce and eta management out of fn" — not by a crash. No such line
   exists in current source. "We lose present()" does refer to the Fortran `present()` intrinsic — the
   diff holds no OpenACC `present()` data clauses, only the commented-out `if(present(pbce))`
   conditional maps — so the construct `f74525ae8` disabled is the same one that later crashed in bug
   #16.

**Independent completeness sweep:** grepped `src/` and `config_src/` for nvfortran/NVHPC/nvidia/
"wrong answer|result"/"didn't like"/"cannot yet"/segfault/crash comments. No **nvfortran/GPU** bug
comment is missing from the catalogue. Two non-nvfortran comments were deliberately excluded as
out of scope: `MOM_diagnostics.F90:305` ("some compiler options can force at least one iteration…" —
a legacy ANSI-F77 loop-trip-count workaround, not GPU) and `mom_cap.F90:85` ("Model does not compile
with `use ESMF, only:`" — an ESMF module-use quirk, not nvfortran).

**Confidence:** High for every verbatim comment, commit-message quote, hash, and file:line anchor
(all directly checked). High for the Row 2 and Row 17 corrections (checked the loop bodies and the
`f74525ae8`→`d4ba8d69d`→`9bfe7d358` chain directly). The meaning of "We lose present()" is settled:
it is the Fortran intrinsic, not the OpenACC data clause. The catalogue's "Still open?"
statuses remain as-reported (no build/run was performed to re-test them), per the document's own method
note.
