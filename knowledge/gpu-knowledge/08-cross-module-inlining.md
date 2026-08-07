# Cross-module calls and inlining on `dev/gpu`

> Drills into architecture doc §0.4, §5, §7.1, §7.5. Scope: every place a helper subroutine/function
> is called from inside a device compute region (`!$omp target`/`!$omp loop`/`do concurrent`), the
> directives and refactors that make that legal on nvfortran, the exact `-Minline` requirement, and
> the cases where duplication (not sharing) was used to route around the problem.

---

## 1. The core rule

**A subroutine/function called from inside an OpenMP `target` compute region must either (a) carry
`!$omp declare target` so the device compiler emits device code for it, or (b) be force-inlined at
the call site.** Getting this wrong does not reliably fail to compile — commit `3cb184edd` is
explicit that skipping it produces **silently wrong numerical results**, not a compile error or
crash. This is the single most expensive class of bug in the port because it doesn't show up until a
checksum diverges.

> **Force-inline directive changed at HEAD — read this before trusting §2/§3 below.** The inline
> mechanism went through *three* forms, and the current tree (`dev/gpu` HEAD, after the merged
> k-blocking commit `93dbbd36e`) is on the **third**:
> 1. `3cb184edd` — build flag only: `-Minline=name:ratio_max,name:flux_elem` (MANDATORY).
> 2. `4e3f1b758` — source pragma `!NVF$ INLINE` above each of `flux_elem`/`flux_elem_OBC`/`ratio_max`
>    (compile with `-Minline=pragma`); also dropped `thread_limit(128)`.
> 3. `93dbbd36e` — replaced `!NVF$ INLINE` with the Intel-style **`!DIR$ ATTRIBUTES FORCEINLINE :: <name>`**
>    on `flux_elem` (`MOM_continuity_PPM.F90:1086`) and `flux_elem_OBC` (`:1149`), and **removed the
>    directive from `ratio_max` entirely** — its commit body: *"remove nvf inline and replace with intel
>    forceinline / Significantly improves performance of blocked zonal/meridional_mass_flux at -O2."*
>
> So **there are currently zero `!NVF$ INLINE` directives in `src/`** (`grep -rF 'NVF$ INLINE' src/`
> returns nothing) and `ratio_max` now carries **no** inline directive at all despite still being
> called from `!$omp target`/`!$omp loop` regions (see §3). The prose below that says
> "`!NVF$ INLINE`" describes the historical `4e3f1b758` state; the catalogue rows have been corrected
> to HEAD.

A second, distinct rule governs `class(*)`/polymorphic dispatch: passing a polymorphic `this` into a
device region causes **runtime errors** on nvfortran (§4), not silent wrongness — so the two failure
modes must be told apart when debugging.

`do concurrent` regions are more forgiving **for correctness**: nvfortran's `-stdpar=gpu` lowering
appears to handle plain `elemental`/`pure` procedure calls without a mandatory `declare target` —
only the definition matters if it's *also* reached from an `!$omp target` region. They are *not* more
forgiving for performance: an out-of-line call costs the inferred collapse unless the dummies are
explicit-shape and the input scalars carry `VALUE`
(`15-device-calls-collapse-and-transfer-parity.md` §1). The declare-target/inline requirement
bites specifically on the `!$omp target teams` / `!$omp target ... !$omp loop` idiom used for the
hand-tuned team-count kernels (§5 of `00-architecture.md`). Note that in the current tree the
`flux_elem`/`ratio_max` call sites in continuity are all inside that `!$omp target teams` + `!$omp
loop collapse(2)` idiom (e.g. `flux_elem` at `MOM_continuity_PPM.F90:715-724`, `ratio_max` at
`:768-769`), **not** a bare `do concurrent` — which is exactly why they need the FORCEINLINE
treatment (the earlier draft's "`ratio_max` inside a bare `do concurrent` at `:1622-1630`" example was
inaccurate: `:1622-1630` are `flux_elem` calls inside an `!$omp loop`).

---

## 2. Catalogue of device-callable helpers

All 21 `!$omp declare target` occurrences in `src/` (this total *includes* the one `!$omp declare
target(pr, I_pr)` for module-level constant arrays — it is one of the 21, not extra), plus the
force-inlined helpers in continuity. **At HEAD the inline directive is `!DIR$ ATTRIBUTES FORCEINLINE`
on `flux_elem`/`flux_elem_OBC` and none on `ratio_max`** (see the banner in §1); the rows below have
been corrected accordingly. "Caller directive" is the enclosing compute-region form that requires the
callee to be device-resident.

| Helper | file:line (def) | Directive (at HEAD) | Caller / enclosing region | Reason |
|---|---|---|---|---|
| `flux_elem` (elemental subroutine) | `src/core/MOM_continuity_PPM.F90:1087` | `!DIR$ ATTRIBUTES FORCEINLINE :: flux_elem` (`:1086`) | `!$omp target teams` / `!$omp loop` in `zonal_mass_flux`/`meridional_mass_flux` (calls at `:719,1060,1457,1622,1626,1630,1823,2162,2464,2629,2632,2635`; note `:724,1065,1462,1828,2167,2469` are `flux_elem_OBC`) | Mandatory inline — commit `3cb184edd`: "Otherwise results are incorrect." Was `!NVF$ INLINE` (`4e3f1b758`); changed to Intel FORCEINLINE by `93dbbd36e`. |
| `flux_elem_OBC` (elemental subroutine) | `:1150` | `!DIR$ ATTRIBUTES FORCEINLINE :: flux_elem_OBC` (`:1149`) | OBC call sites listed above | Same as `flux_elem`; was also listed explicitly in the `-Minline=name:...` flag |
| `ratio_max` (pure function) | `:3086` | **none at HEAD** (was `!NVF$ INLINE` at `:3085`, removed by `93dbbd36e`) | `!$omp target`/`!$omp loop` in `zonal_mass_flux` (`:768,769,792,793,813,814,833,834,851,852`) and `meridional_mass_flux` (`:1872,1873,1897,1898,1917,1918,1938,1939,1955,1956`) | Originally mandatory-inline (`3cb184edd`); directive since removed — see §3 |
| `efp_decompose` (pure subroutine) | `src/framework/MOM_coms.F90:778` (directive `:779`) | `!$omp declare target` | `do concurrent` reduction loop in `increment_block_ints` (`:721-728`) | Per-point EFP bin decomposition inside the reproducing-sum block reduction; must be device-resident since the reduction body is compiled for target |
| module constants `pr`, `I_pr` | `src/framework/MOM_coms.F90:69` | `!$omp declare target(pr, I_pr)` | referenced from inside `efp_decompose`/reduction body | Data (not code), but same device-residency requirement extends to module-level constant arrays consumed by device code |
| `cuberoot` (elemental function) | `src/framework/MOM_intrinsic_functions.F90:51` (directive), `:50` (def) | `!$omp declare target` | called from `MOM_barotropic.F90`, EOS files, etc. inside `do concurrent`/`omp target` | Replaces `x**(1/3)` (transcendental `exp/log` lowering) with a deterministic bit-exact iterative kernel; must be device-resident everywhere it's used |
| `nth_root` (elemental function) | `:133` (directive), `:132` (def) | `!$omp declare target` | `MOM_barotropic.F90`'s `bt_rem = av_rem**Instep` | Replaces `x**(1/n)` for general integer `n`; same transcendental-lowering problem as `cuberoot` |
| `rescale_cbrt` (pure subroutine) | `:181` (directive), `:180` (def) | `!$omp declare target` | called from `cuberoot` | Helper to `cuberoot`; must be device-resident since its caller is |
| `descale` (pure function) | `:246` (directive), `:245` (def) | `!$omp declare target` | called from `cuberoot` | Same as `rescale_cbrt` |
| `find_coupling_coef_gl90` | `src/parameterizations/vertical/MOM_vert_friction.F90:437` (directive), `:436` (def) | `!$omp declare target` | `!$omp target teams distribute parallel do collapse(2)` in `vertvisc_coef` (called `:1600,1904`) | Per-column GL90 vertical-viscosity coupling coefficient, called once per (I,j) inside the teams-distribute loop at `:1443` |
| `find_coupling_coef_k` (pure subroutine) | `:2101` (directive), `:2099` (def) | `!$omp declare target` | same teams-distribute loop, called `:1594,1898` | BBL-aware coupling coefficient |
| `find_coupling_coef` (non-pure subroutine) | `:2611` (directive), `:2609` (def) | `!$omp declare target` | same loop, called `:1679,1983` (ice-shelf branch) | Shelf-drag coupling coefficient variant |
| `find_L_open_uniform_slope` (pure subroutine) | `src/parameterizations/vertical/MOM_set_viscosity.F90:1251,1258` (directive, ×2) `:1241` (def) | `!$omp declare target` | called from BBL open-area column kernels inside device loops in `set_viscous_ML`/related | Column-local open-fraction geometry, one of 4 `find_L_open_*` variants selected per bottom-shape case |
| `find_L_open_concave_trigonometric` | `:1294,1316` (dir, ×2), `:1283` (def) | `!$omp declare target` | same family | ditto (uses `atan`-derived constant `C2pi_3`, still device-safe since it's a compile-time parameter) |
| `find_L_open_concave_iterative` | `:1389,1438` (dir, ×2), `:1378` (def) | `!$omp declare target` | same family | Newton-iteration fallback when the closed-form concave case is ill-conditioned |
| `test_L_open_concave` | `:1718,1742` (dir, ×2), `:1705` (def) | `!$omp declare target` | same family (diagnostic/verification path) | Consistency check on `find_L_open_concave_*` outputs, itself device-resident |
| `find_L_open_convex` | `:1802,1836` (dir, ×2), `:1788` (def) | `!$omp declare target` | same family | Convex bottom-shape case; iterates with `maxitt` |
| `set_v_at_u` (pure function) | `:1966` (dir, ×1), `:1951` (def) | `!$omp declare target` | thickness-weighted interpolation used inside device column loops | Cross-staggering interpolation (v at u-points) |
| `set_u_at_v` (pure function) | `:2012` (dir, ×1), `:1997` (def) | `!$omp declare target` | ditto | u at v-points |

**Note on the double directive:** in `MOM_set_viscosity.F90`, each of the 5 `find_L_open_*`/
`test_L_open_concave` routines carries `!$omp declare target` **twice** — once immediately after the
dummy-argument declarations, once again after the local-variable declarations, both before any
executable statement (e.g. lines 1251 and 1258 both sit inside `find_L_open_uniform_slope`, which
starts at 1241 and has no other procedure boundary between them). `set_v_at_u`/`set_u_at_v` carry it
only once. This accounts for 12 total `declare target` occurrences across 7 subroutines/functions in
this file (5×2 + 2×1 = 12) and is presumably harmless (a repeated directive on the same scope), but is
catalogued here since it's an idiosyncrasy specific to this file — the pattern is not used anywhere
else in the 21-occurrence inventory.

Total: 21 `!$omp declare target` directives (matches the count in `00-architecture.md` §6.1) across
4 files (`MOM_coms.F90` (2: module constants `declare target(pr, I_pr)` + `efp_decompose`),
`MOM_intrinsic_functions.F90` (4), `MOM_vert_friction.F90` (3), `MOM_set_viscosity.F90` (12, including
the double-directive idiosyncrasy noted below)) — verified by `grep -rn "declare target" src/ | wc -l`
= 21. **Force-inline in `MOM_continuity_PPM.F90` at HEAD: 2 `!DIR$ ATTRIBUTES FORCEINLINE` directives
(`flux_elem` `:1086`, `flux_elem_OBC` `:1149`), and zero on `ratio_max`.** (The "3 `!NVF$ INLINE`
sites" phrasing in `00-architecture.md` §7.5 and §6.1 predates `93dbbd36e` and is now stale — there
are 0 `!NVF$ INLINE` directives in the tree.)

**Note on "cross-module":** in the strict sense (helper defined in a *different module* than the
caller) only `cuberoot`/`nth_root` (`MOM_intrinsic_functions` called from `MOM_barotropic`/EOS) and
the EOS `_loc` functions (defined in `MOM_EOS_Wright`/`MOM_EOS_Roquet_rho`, dispatched from
`MOM_EOS`) qualify. `flux_elem`/`ratio_max`, the `find_coupling_coef*` family, and the `find_L_open_*`
family are all same-module (private module procedures called from a sibling subroutine in the same
file) — but they hit the *identical* compiler problem, because the OpenMP device-code generation
model treats "outside the lexical scope of the enclosing `!$omp target` construct" as the boundary
that requires `declare target`, not the Fortran module boundary. The knowledge-base title tracks the
pain point ("procedure call inside a device loop"), which is broader than literal cross-`module`
calls.

---

## 3. The exact `-Minline` requirement

Commit `3cb184edd` ("use openmp instead of openacc", `src/core/MOM_continuity_PPM.F90`, +169/−198)
is the OpenACC→OpenMP translation of `MOM_continuity_PPM.F90` and states the rule verbatim:

> IMPORTANT: However for OpenMP, inlining of ratio_max and flux_elem is MANDATORY. do so with
> `-Minline=name:ratio_max,name:flux_elem`. Otherwise results are incorrect.

That commit's translation table (from the same message) is itself useful background for the OpenACC→
OpenMP mapping used throughout the port:

> outer parallel region followed by multiple inner acc loops is equivalent to an outer omp target
> followed by multiple inner omp loops. oacc parallel loop seems to be equivalent to omp target loop.

Follow-up commit `4e3f1b758` ("add !NVF\$ INLINE to ratio_max flux_elem") replaces the build-flag
requirement with a source-level directive, so the build system no longer has to enumerate function
names:

> instead of compiling with `-Minline=name:flux_elem,name:flux_elem_OBC,name:ratio_max`, can compile
> with `-Minline=pragma` instead.

Diff (`4e3f1b758`) adds `!NVF$ INLINE` directly above each of the three definitions:
```fortran
!> Evaluates the zonal mass or volume fluxes in an element.
!NVF$ INLINE
elemental subroutine flux_elem(u, h, h_p1, h_L, h_L_p1, h_R, h_R_p1, uh, duhdu, visc_rem, &
                               G_dy_Cu, G_IareaT, G_IareaT_p1, G_IdxT, G_IdxT_p1, dt, &
                               vol_CFL, por_face_area)
```
```fortran
!> Return the maximum ratio of a/b or maxrat.
!NVF$ INLINE
pure function ratio_max(a, b, maxrat) result(ratio)
```
The same commit also **removes** a `thread_limit(128)` clause from the `!$omp target teams
num_teams(nteams)` construct that wraps these calls (kept as `num_teams(nteams)` only, verified at
`MOM_continuity_PPM.F90:707,1811`) — inlining changed the register/resource footprint enough that the
previous thread-limit tuning was dropped.

**Third form (current HEAD) — commit `93dbbd36e`.** The k-blocking-continuity merge ("Use blocking in
k dimension for continuity reconstruction (#165)") replaced the `!NVF$ INLINE` pragmas with the
Intel-style `!DIR$ ATTRIBUTES FORCEINLINE :: <name>` directive on `flux_elem` and `flux_elem_OBC`,
and **deleted the directive from `ratio_max` outright.** Its commit body states the motivation:
*"remove nvf inline and replace with intel forceinline / Significantly improves performance of blocked
zonal/meridional_mass_flux at -O2."* Diff evidence (`git show 93dbbd36e -- src/core/MOM_continuity_PPM.F90`):
```diff
-!NVF$ INLINE
+!DIR$ ATTRIBUTES FORCEINLINE :: flux_elem
 elemental subroutine flux_elem(...)
-!NVF$ INLINE
+!DIR$ ATTRIBUTES FORCEINLINE :: flux_elem_OBC
 elemental subroutine flux_elem_OBC(...)
-!NVF$ INLINE
 pure function ratio_max(a, b, maxrat) result(ratio)
```
So at HEAD the `-Minline=pragma` recipe no longer targets `!NVF$ INLINE` (there are none); inlining is
now driven by the `!DIR$ ATTRIBUTES FORCEINLINE` directive (which nvfortran honours) on the two
`flux_elem` routines.

> **Resolved (2026-07-14):** The gap is a deliberate removal resting on implicit device codegen.
> `93dbbd36e` removed `!NVF$ INLINE` from `ratio_max` without replacement while giving
> `flux_elem`/`flux_elem_OBC` FORCEINLINE, and **no `-Minline` exists in any in-repo or
> mkmf-template build config** — so no flag is quietly standing in for the directive. `ratio_max` is
> still called from `!$omp target`/`loop` regions in `MOM_continuity_PPM.F90`. Correctness at HEAD
> therefore rests on nvfortran implicitly compiling/inlining a small same-file `pure` function for
> the device: empirically fine on the tested toolchain (the commit is merged and checksum-gated), but
> fragile. **Recommendation:** add `!DIR$ ATTRIBUTES FORCEINLINE :: ratio_max` for parity with
> `flux_elem`. Never imitate the gap in new code.

**Without either the flag or the pragma:** per `3cb184edd`'s commit message, the OpenMP target
version of `zonal_mass_flux`/`meridional_mass_flux` produces **incorrect results** (not a build
failure) if `flux_elem`/`ratio_max` are compiled as genuine out-of-line device subroutine calls
instead of being inlined. The exact failure mode (register spill, missing device symbol resolution at
link time, or a silent no-op) is not spelled out in the commit — only the empirical fact that results
are wrong — so this is catalogued as an observed nvfortran limitation, not a fully diagnosed root
cause.

---

## 4. The pure/elemental refactoring pattern (the blessed fix)

The house style for "this needs restructuring for the device, but must not change one bit of
arithmetic" is: **extract the innermost, side-effect-free arithmetic into a `pure`/`elemental`
procedure, and never touch the expressions themselves.** Four concrete instances:

### 4.1 `efp_decompose` — reproducing-sum decomposition (`MOM_coms.F90:779`)

```fortran
!> Decompose one real into its 6 signed EFP bin contributions.  NaNs and
!! overflows are reported by flags, rather than the module-level error
!! logicals, so that the routine is free of side effects.
pure subroutine efp_decompose(r, e, rmag, is_nan, is_ovf)
  !$omp declare target
```
The doc comment is explicit about *why* it's `pure`: side-effect-freedom (no touching the module-level
`overflow_error`/`NaN_error` flags) is what makes it safe to call from inside a `do concurrent`
reduction body (`increment_block_ints`, `:721-728`) without breaking the compiler's ability to reason
about the loop. Errors are returned as `is_nan`/`is_ovf` output arguments and only folded into the
module flags by the (non-pure, host-side) caller after the reduction completes.

### 4.2 EOS `_loc` free functions — the `this`-copy fix (§7.1 of `00-architecture.md`)

Origin commit `52a1b3954` ("Added local versions of density_elem and density_derivs without "this"
argument", `MOM_EOS_Wright.F90`) states the problem directly in a doc comment it introduces:

```fortran
!> Wrapper for density_elem_buggy_Wright_loc created to preserve API while calling
!! density_elem_buggy_Wright without "this" variable that causes runtime errors on
!! gpu runs with nvfortran.
real elemental function density_elem_buggy_Wright(this, T, S, pressure)
  class(buggy_Wright_EOS), intent(in) :: this !< This EOS
  ...
  density_elem_buggy_Wright = density_elem_buggy_Wright_loc(T, S, pressure)
end function density_elem_buggy_Wright
```
The original type-bound elemental (`class(this)` dummy) is kept **only as a thin wrapper** for API
compatibility with code that still dispatches through it; the arithmetic itself was moved verbatim
into a new free function with the identical body and an `_loc` suffix, dropping the `this` argument
entirely. Later commit `7c7af5572` ("EOS: 2D and 3D density implementations of methods (#185)")
extends the same split to `calculate_density_derivs_3d`/`calculate_stanley_density_2d`/
`calculate_density_second_derivs_2d` in both `MOM_EOS_Wright.F90` and `MOM_EOS_Roquet_rho.F90`.

The `_loc` functions are then called from `do concurrent (k,j,i)` device loops directly, e.g.
`MOM_EOS_Wright.F90:1058`:
```fortran
do concurrent (k=ks:ke, j=js:je, i=is:ie)
  rho(i,j,k) = density_elem_buggy_Wright_loc( T(i,j,k), S(i,j,k), pressure(i,j,k))
enddo
```
— no `this`, no polymorphic dispatch inside the device loop at all.

### 4.3 Intrinsic replacements — `cuberoot`/`nth_root` (`MOM_intrinsic_functions.F90`)

`cuberoot` predates the GPU port (added by the upstream `dev-gfdl` history); the dev/gpu-specific
changes are (a) adding `!$omp declare target` to `cuberoot`, `rescale_cbrt`, and `descale`
(`dev-gfdl...dev/gpu` diff for this file — no separate commit message survives on `dev/gpu`'s
first-parent history, folded into `546907fee`/`7b706ecbc`), and (b) adding a **new** function,
`nth_root`, generalizing the pattern to arbitrary integer roots for `MOM_barotropic.F90`'s
`bt_rem = av_rem**Instep`. `nth_root`'s doc comment states the motivating problem precisely:

```fortran
!> Bit-stable n-th root of x for x in (0, +inf) and integer n >= 1, suitable
!! for evaluation inside `!$omp target` / `do concurrent` offloaded regions.
!!
!! Lowering `x**(1.0/n)` via the compiler produces `exp((1.0/n)*log(x))` — two
!! transcendentals whose last-bit rounding differs between host libm and CUDA
!! libdevice. This routine avoids that path entirely: it uses fixed-iteration
!! Newton on y^n - x = 0, with y^(n-1) evaluated as repeated multiplication,
!! and one bit-precision-polishing iteration at the end.
elemental function nth_root(x, n) result(root)
  !$omp declare target
```
This is the pattern in miniature: identify an intrinsic (`x**(1/n)`) whose device lowering is not
bit-reproducible with the host, and replace it with a deterministic, fixed-iteration-count,
multiply/add-only kernel that is `elemental` (so it vectorizes/parallelizes trivially) and carries
`!$omp declare target` (so it's legal inside `!$omp target` regions too).

### 4.4 `flux_elem`/`ratio_max` — already `elemental`/`pure`, just needed force-inlining (§3 above)

Unlike 4.1–4.3, these were **already** `elemental`/`pure` before the GPU port (they long predate
`dev/gpu`); the fix here was not a restructuring of the arithmetic at all, just a force-inline
directive (or the build-flag equivalent) — which itself churned three times: `-Minline` flag
(`3cb184edd`) → `!NVF$ INLINE` (`4e3f1b758`) → `!DIR$ ATTRIBUTES FORCEINLINE` on the two `flux_elem`
routines, none on `ratio_max` (`93dbbd36e`; see §1 banner and §3). This is the cheapest end of the
spectrum: when the helper is already side-effect-free, the only remaining problem is *compiler code
generation* (inlining), not source structure.

**Common thread across all four:** none of the fixes reorder floating-point operations or change
which numbers get added to which. `efp_decompose` moves code, not arithmetic. The EOS `_loc` split is
a literal copy-paste of the RHS with the `this` dummy dropped. `nth_root`/`cuberoot` replace a
compiler-lowered intrinsic with an equivalent-precision hand-written kernel (arithmetically different
from `x**(1/n)` at the bit level, but *designed* to match to round-off, and unit-tested via
`Test_cuberoot`). `flux_elem`/`ratio_max` are untouched. This is consistent with guiding principle #2
in `00-architecture.md`: bitwise reproducibility is mandatory, and `pure`/`elemental` extraction is
the preferred restructuring tool precisely because it cannot silently reorder a reduction.

---

## 5. Constructs nvfortran refused inside device regions

### 5.1 Early `exit`/`return` inside a loop under `do concurrent` — commit `e23d6a7b1`

```
swap early exit to if guard in insert sort

NVHPC 25.11 didn't like the early exit and would
give wrong answers.
```
Diff, `src/tracer/MOM_tracer_hor_diff.F90:942-946` (insertion-sort inner loop, itself nested inside an
outer `do concurrent (j=js-1:je+1, i=is-1:ie+1)`):
```fortran
  do concurrent (j=js-1:je+1, i=is-1:ie+1)
    do k=2,num_srt(i,j) ; if (rho_srt(i,k,j) < rho_srt(i,k-1,j)) then
      ! The last segment needs to be shuffled earlier in the list.
-     do k2 = k,2,-1 ; if (rho_srt(i,k2,j) >= rho_srt(i,k2-1,j)) exit
+     do k2 = k,2,-1 ; if (rho_srt(i,k2,j) < rho_srt(i,k2-1,j)) then
        itmp = k0_srt(i,k2-1,j) ; k0_srt(i,k2-1,j) = k0_srt(i,k2,j) ; k0_srt(i,k2,j) = itmp
        tmp = rho_srt(i,k2-1,j) ; rho_srt(i,k2-1,j) = rho_srt(i,k2,j) ; rho_srt(i,k2,j) = tmp
        tmp = h_srt(i,k2-1,j) ; h_srt(i,k2-1,j) = h_srt(i,k2,j) ; h_srt(i,k2,j) = tmp
-     enddo
+     endif ; enddo
    endif ; enddo
  enddo
```
The transformation negates the exit condition (`>=` → `<`) and turns it into a body `if`-guard that
still executes every iteration of the `k2` loop (doing nothing once the sorted position is found)
rather than jumping out of it early. Behaviorally identical on a CPU; the NVHPC 25.11 device code
generator for `do concurrent` mis-compiled the `exit`-from-inner-loop form and produced wrong values
— this is catalogued as a **compiler bug worked around**, not a language limitation (`exit` from
`do concurrent` bodies is otherwise permitted by the standard here since it doesn't cross the
`do concurrent` loop itself, only the ordinary nested `do`).

### 5.2 `modulo()` — "not implemented on all systems"

`MOM_intrinsic_functions.F90`'s `rescale_cbrt` avoids `modulo()` entirely, with the reasoning
recorded in-line:
```fortran
  ! modulo() is not implemented on all systems, so compute the remainder as
  ! r = n - 3*q.

  e_x = e_a - e_r * 3
```
This particular replacement (`850504a98`, "cuberoot: Replace modulo() with arithmetic ops") actually
originates upstream on `dev-gfdl` itself (authored by Marshall Ward, NOAA) — i.e. GFDL had already
hit non-`modulo()`-supporting platforms (explicitly named as "e.g. NVIDIA GPUs" in that commit
message) before `dev/gpu` forked, and `dev/gpu` inherited the fix along with the merge, then layered
`!$omp declare target` on top of the resulting `rescale_cbrt`/`descale`/`cuberoot`. The floor-division
identity used instead, `⌊e/3⌋ = (e + sign(1,e) - 1) / 3`, replaces `modulo(e,3)` with `sign()` +
integer truncating division, both of which are safe on device.

### 5.3 Polymorphic dispatch / `select type` — the unresolved case

Documented directly in-source (`MOM_EOS_Wright.F90:1008`, `:1048`, `:1114`, `:1147`;
`MOM_EOS_Roquet_rho.F90:817`) as an **open, unresolved** problem, distinct from the fixed cases above:
```fortran
  ! NOTE: There is an implicit copy of `this` which cannot yet be prevented.
  !   Possibly because Nvidia cannot associate `this` with `EOS%type`.

  if (present(rho_ref)) then
    do concurrent (k=ks:ke, j=js:je, i=is:ie)
      rho(i,j,k) = density_anomaly_elem_buggy_Wright(this, T(i,j,k), S(i,j,k), &
          pressure(i,j,k), rho_ref)
    enddo
  else
    do concurrent (k=ks:ke, j=js:je, i=is:ie)
      rho(i,j,k) = density_elem_buggy_Wright_loc( T(i,j,k), S(i,j,k), pressure(i,j,k))
    enddo
  endif
```
Note the asymmetry within a single subroutine (`calculate_density_array_3d_buggy_Wright`,
`MOM_EOS_Wright.F90:1024`): the common (`rho_ref` absent) branch was converted to the `_loc` form and
avoids `this` entirely; the `rho_ref`-present branch **still passes `this`** into
`density_anomaly_elem_buggy_Wright`, which has not (yet) been given a `_loc` sibling, so it still
carries the "implicit copy of `this`" comment as an acknowledged, uncorrected cost. This is exactly
the case flagged unresolved in `00-architecture.md` §7.5. The root architectural problem — `EOS_type`
wraps `class(EOS_base), allocatable :: type` (`MOM_EOS.F90:162`) and dispatches via
`EOS%type%calculate_density_...` (`MOM_EOS.F90:344` et seq., chosen via `select type
(t => EOS%type)` at init, `MOM_EOS.F90:2143`) — remains in place for every EOS variant except
buggy-Wright and Roquet_rho, which is why the doc calls the EOS layer "being rewritten" rather than
"rewritten."

### 5.4 `select type` at the dispatch boundary (not itself inside a device loop)

Worth noting for completeness: the `select type (t => EOS%type)` construct in `MOM_EOS.F90:2143`
(used only during `EOS_init` to call type-specific parameter setters) is host-only and not itself a
device-region problem — it is the *runtime v-table dispatch through `class(EOS_base)`* at
per-gridpoint call sites (`EOS%type%density_elem(...)`, `EOS%type%calculate_density_array_3d(...)`)
that nvfortran cannot resolve on device, which is why the fix is at the call site (duplicate as a
free function) rather than at the type declaration.

---

## 6. Where code is duplicated (not shared) to avoid a cross-module/cross-procedure device call

| Duplication | Files | What's duplicated | Maintenance cost |
|---|---|---|---|
| `density_elem_buggy_Wright` / `density_elem_buggy_Wright_loc` | `MOM_EOS_Wright.F90:90-117` | Full arithmetic body duplicated verbatim between the type-bound wrapper (kept for API compatibility with polymorphic callers) and the free `_loc` function | Any bugfix or unit change to the Wright density formula must be applied in the `_loc` body; the wrapper is a 1-line pass-through so it can't drift on its own, but a future edit to one and not the other would silently split behavior between the polymorphic-dispatch call path and the device call path |
| `calculate_density_derivs_elem_buggy_Wright` / `..._loc` | `MOM_EOS_Wright.F90:199-241` | Same pattern for density-derivative arithmetic | Same risk, doubled: this is now the *third* copy of closely related Wright-EOS arithmetic (density, density anomaly (not yet split, still polymorphic-only), derivatives) living in the file, at different stages of the `_loc` migration |
| Roquet_rho density/derivs `_loc` siblings | `MOM_EOS_Roquet_rho.F90` (mirrors `MOM_EOS_Wright.F90` structure) | Same wrapper/`_loc` split, independently re-implemented per EOS formulation | Every additional EOS type that gets GPU-ported (currently only 2 of 9 `form_of_EOS` cases: buggy-Wright, Roquet_rho) needs its own from-scratch `_loc` split; there is no shared free-function template, so the split is copy-pasted per formulation rather than factored once |
| `find_L_open_uniform_slope` / `_concave_trigonometric` / `_concave_iterative` / `_convex` | `MOM_set_viscosity.F90:1241-1802` | Four independent bottom-shape-specific column kernels, each `!$omp declare target`, rather than one dispatching helper | This is a pre-existing physics branch structure (different closed-form/iterative solutions per bottom curvature case), not new duplication caused by porting — but the device-residency requirement means *all four* must independently carry the directive and be kept in sync if the calling convention changes, since there is no shared dispatch layer that could paper over an inconsistency |
| `find_coupling_coef` / `find_coupling_coef_k` / `find_coupling_coef_gl90` | `MOM_vert_friction.F90:436,2099,2609` | Three separate per-column coupling-coefficient kernels (BBL-drag, GL90, ice-shelf-drag variants) each independently marked `!$omp declare target` | Same shape as above: physics-driven branching pre-dates the port, but porting forces each branch to be independently verified device-safe; a future change to the shared thickness/coupling logic must be replicated three times unless a common inner helper is factored out (none currently exists) |

**Overall maintenance cost pattern:** the `_loc`-suffix duplication (EOS) is the clearest *port-induced*
duplication — it exists solely because polymorphic `this` cannot cross into device code, and doubles
the number of functions that must be kept in sync per EOS formulation as more formulations are ported.
The `find_L_open_*`/`find_coupling_coef*` families are pre-existing physics-driven branch structures
that the port did not duplicate further, but whose per-branch `!$omp declare target` marking means
device-safety must be independently re-verified for each variant rather than centrally.

---

## 7. Prescriptive rules — the device-call checklist

Distilled from the cases above. **When a device compute region (`!$omp target`/`!$omp loop`/
`do concurrent`) must call a helper, walk this checklist in order:**

1. **Is the helper side-effect-free?** It must not touch module-level state, do I/O, or mutate
   `save`d/`intent(inout)` module data. If not, refactor: extract the innermost arithmetic into a new
   `pure` (or `elemental` for scalar-per-element) procedure and return errors/flags as arguments —
   the `efp_decompose` model (`MOM_coms.F90:778`, side-effect-freedom is *why* it can sit in a
   `do concurrent` reduction body). Never reorder the floating-point arithmetic while doing this
   (guiding principle #2, `00-architecture.md`).
2. **Does it carry no `class(*)`/polymorphic dummy?** A `class(this)` argument forces an implicit copy
   nvfortran mishandles on device (`MOM_EOS_Wright.F90:1048`, "implicit copy of `this` which cannot yet
   be prevented"). If it has one, duplicate it as a free `_loc` function with the `this` dummy dropped
   and the body copy-pasted verbatim (`density_elem_buggy_Wright_loc`, commit `52a1b3954`), and call
   the `_loc` form from the loop.
3. **Which device-region idiom is the call inside?**
   - *Bare `do concurrent`*: nvfortran's `-stdpar=gpu` lowering generally handles `pure`/`elemental`
     calls without a mandatory `declare target` — the definition just has to be visible.
     **But correct answers are not the whole story here:** `do concurrent`'s collapse is *inferred*,
     and an out-of-line call gives it up unless the callee's array dummies are **explicit-shape** and
     its input scalars carry **`VALUE`**. Worth 3-4x, with no compile-time warning beyond an
     `-Minfo` line reading `Reference argument passing prevents parallelization`. See
     `15-device-calls-collapse-and-transfer-parity.md` §1.
   - *`!$omp target teams` / `!$omp target … !$omp loop`*: the callee **must** be device-resident —
     either `!$omp declare target` on the definition, or force-inlined at compile (`!DIR$ ATTRIBUTES
     FORCEINLINE :: <name>` at HEAD; historically `!NVF$ INLINE` + `-Minline=pragma`). Getting this
     wrong is **silently wrong numbers, not a build error** (`3cb184edd`).
4. **Declare-target vs force-inline — which?** Column/point kernels that are large or reused widely get
   `!$omp declare target` (all 21 catalogued helpers). Tiny leaf `elemental`/`pure` helpers that the
   compiler *can* inline get force-inlined instead — this measurably outperformed out-of-line device
   calls (`93dbbd36e`: FORCEINLINE "significantly improves performance … at -O2"). If you add `!$omp
   declare target`, it must appear after all declarations and before the first executable statement
   (the `MOM_set_viscosity.F90` routines even repeat it twice, `:1251`/`:1258` — harmless idiosyncrasy).
5. **Same-module or cross-module?** Irrelevant to the compiler: the boundary that triggers the
   requirement is "outside the lexical scope of the enclosing `!$omp target` construct", not the
   Fortran `module` boundary (§2 note). A private same-module helper called from a sibling subroutine
   hits the identical problem as a genuine cross-module call. Duplicate-per-module only when forced
   (the `_loc` split is re-implemented per EOS formulation — no shared template, §6).
6. **Verify.** There is no compile-time signal for the silent-wrongness failure mode. The only proof is
   a **`MOM_checksums` hchksum/uchksum + reproducing-sum energy comparison, CPU build vs GPU build,
   bit-for-bit** (`00-architecture.md` §7.2, `MOM_checksums.F90:2680`). A port that compiles and runs
   but whose helper silently failed to inline will diverge only here.

### The "never do inside a device region" list

| Never | Why | Evidence |
|---|---|---|
| Pass a `class(*)`/polymorphic `this` into the loop | nvfortran emits an implicit device copy it mishandles → runtime errors | `MOM_EOS_Wright.F90:1008,1048,1114,1147`, `MOM_EOS_Roquet_rho.F90:817`; fix `52a1b3954`/`7c7af5572` |
| Dispatch through a v-table (`EOS%type%method(...)`) | nvfortran cannot resolve the `class(EOS_base)` v-table on device | `MOM_EOS.F90:162,2143`; §5.3/§5.4 |
| Early `exit`/`return` out of an inner loop under `do concurrent` | NVHPC 25.11 mis-compiles it → **wrong answers**; rewrite as a negated `if`-guard that still iterates | `e23d6a7b1`, `MOM_tracer_hor_diff.F90:942-946` |
| `modulo()` | not implemented on all device targets (NVIDIA GPUs); use `sign()` + truncating integer division | upstream `850504a98` (Marshall Ward/NOAA), `MOM_intrinsic_functions.F90:232-235` |
| `x**(1/n)` / `x**(1./3.)` where bit-reproducibility matters | lowered to `exp((1/n)*log(x))` — two transcendentals whose last bit differs between host libm and CUDA libdevice | `nth_root`/`cuberoot`, `MOM_intrinsic_functions.F90:120-132,50` |
| Call a helper into an `!$omp target teams`/`loop` without a *guaranteed* inline or `declare target` | silent wrong numbers, no build error | `3cb184edd` ("MANDATORY … otherwise results are incorrect") |
| Allocate, do I/O, or post a diagnostic inside the loop | forces host round-trips / is illegal on device | `00-architecture.md` §9 |

---

## 8. Cross-references

- `docs/gpu-knowledge/00-architecture.md` §0.4 (guiding principle), §5 (k-blocking hybrid kernel using
  `flux_elem`/`ratio_max`), §7.1 (EOS polymorphism), §7.5 (compiler workarounds index).
- Planned/pending docs referenced but not yet written at time of writing: `06-eos.md` (full EOS
  architecture), `13-compiler-workarounds.md` (broader nvfortran-bug catalogue including
  `num_teams`/`thread_limit` tuning (`5b5f6b2b1`), `omp target teams loop` → `do concurrent`
  reversions (`e8b0ecfbf`), and the A100/nvfortran-25.5 crash workaround (`2108e0eba`) — these are
  compiler-workaround entries adjacent to but outside the cross-module-inlining scope of this
  document.

---

## Verification notes

Verified against source at `dev/gpu` HEAD and git history (source + git only; no build/run).

**Confirmed (verbatim / exact):**
- The **21 `!$omp declare target`** count and every catalogued def line: `MOM_intrinsic_functions.F90`
  cuberoot `:50`/nth_root `:132`/rescale_cbrt `:180`/descale `:245`; `MOM_coms.F90` `declare
  target(pr,I_pr)` `:69` and `efp_decompose` def `:778`/directive `:779`; `MOM_vert_friction.F90`
  find_coupling_coef_gl90 `:436`, _k `:2099`, plain `:2609`; `MOM_set_viscosity.F90` 12 occurrences.
- The **double-directive idiosyncrasy** in `MOM_set_viscosity.F90` (e.g. `find_L_open_uniform_slope`
  starts `:1241`, directives at `:1251` after dummy decls and `:1258` after local decls, both before
  the first executable at `:1260`). 5×2 + 2×1 = 12 confirmed.
- Commit `3cb184edd` message ("inlining of ratio_max and flux_elem is MANDATORY … Otherwise results
  are incorrect") — verbatim. Commit `4e3f1b758` (added `!NVF$ INLINE` to all 3, dropped
  `thread_limit(128)`) — verbatim + diff. Commit `52a1b3954` message and the wrapper doc-comment
  (`MOM_EOS_Wright.F90:107-109`). Commit `e23d6a7b1` message ("NVHPC 25.11 didn't like the early exit
  and would give wrong answers") + diff (`MOM_tracer_hor_diff.F90:942-946`, `>=`→`<`, `exit`→guard).
- `modulo()` origin `850504a98` — confirmed upstream, author **Marshall Ward (NOAA)**, message names
  "NVIDIA GPUs" and the `sign()` simplification; the in-source comment `:232-235`.
- The **5 "implicit copy of `this`"** occurrences: `MOM_EOS_Wright.F90:1008,1048,1114,1147` +
  `MOM_EOS_Roquet_rho.F90:817`. The **asymmetric `_loc` conversion** in
  `calculate_density_array_3d_buggy_Wright` (`:1024`): `rho_ref`-present branch still passes `this`
  (`:1053`), absent branch uses `_loc` (`:1058`) — exact.
- The `nth_root` doc-comment (`exp((1/n)*log(x))` transcendental-lowering rationale, Newton iteration)
  — verbatim (`:120-132`).
- **NVHPC version numbers are correct as differentiated:** `e23d6a7b1` = **25.11**; `2108e0eba` =
  **25.5** (a *different* commit — "crashes on stellar A100s with nvfortran 25.5"). No conflict; both
  right per their own commit.

**Corrected (material):**
- **`!NVF$ INLINE` is stale.** At HEAD (after merged commit `93dbbd36e`, "remove nvf inline and
  replace with intel forceinline") there are **zero** `!NVF$ INLINE` directives in `src/`. `flux_elem`
  (`:1086`) and `flux_elem_OBC` (`:1149`) now carry `!DIR$ ATTRIBUTES FORCEINLINE :: <name>`, and
  **`ratio_max` carries no inline directive at all**. Rewrote §1 banner, §2 catalogue rows + total,
  §3 (added the third-form subsection), and §4.4.
- **§1's `do concurrent` example was wrong:** it claimed "`ratio_max` inside a bare `do concurrent` at
  `:1622-1630`". Those lines are `flux_elem` calls inside an `!$omp loop`; `ratio_max` is called at
  `:768-769` etc. inside `!$omp target`/`!$omp loop`. Corrected.
- Minor: `efp_decompose` def is `:778` (directive `:779`), not `:779`; catalogue's `flux_elem`
  call-site list mixed in `flux_elem_OBC` sites (`:724,1065,…`) — split out.

**Confidence:** High. Every quoted commit message, doc comment, and line number was re-derived from
the tree. The former open item — how `ratio_max` stays correct with no inline directive and no
`declare target` while called from a device region — is resolved in §3: no `-Minline` flag is
covering for it, so correctness rests on nvfortran's implicit device codegen for a small same-file
`pure` function, which works on the tested toolchain but should be pinned with an explicit
FORCEINLINE.
