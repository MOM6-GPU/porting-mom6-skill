# The EOS layer: runtime polymorphism vs. nvfortran (dev/gpu)

> Drills into architecture doc §7.1. Scope: `src/equation_of_state/`. This is the canonical
> case study for "`class(*)`/runtime polymorphism is a disaster on device" — read this before
> touching any polymorphic dispatch elsewhere in the tree.

---

## 1. The old dispatch: `EOS_type` → `class(EOS_base)` → deferred elemental procedures

### 1.1 The wrapper type

`EOS_type` (`MOM_EOS.F90:117-164`) is a plain (non-polymorphic) derived type holding scalar
parameters (unit-scaling factors, freezing-point coefficients) plus one polymorphic component:

```fortran
!> A control structure for the equation of state
type, public :: EOS_type ; private
  integer :: form_of_EOS = 0
  ...
  !> The instance of the actual equation of state
  class(EOS_base), allocatable :: type
end type EOS_type
```
(`MOM_EOS.F90:117`, component at `:162`)

### 1.2 The abstract base with deferred elemental bindings

`EOS_base` (`MOM_EOS_base_type.F90:13-74`) is `abstract` and declares nine `deferred` type-bound
procedures, all `elemental`, all taking the polymorphic `this` as their first dummy:

```fortran
type, abstract :: EOS_base
contains
  procedure(i_density_elem), deferred :: density_elem
  procedure(i_density_anomaly_elem), deferred :: density_anomaly_elem
  procedure(i_spec_vol_elem), deferred :: spec_vol_elem
  procedure(i_spec_vol_anomaly_elem), deferred :: spec_vol_anomaly_elem
  procedure(i_calculate_density_derivs_elem), deferred :: calculate_density_derivs_elem
  procedure(i_calculate_density_second_derivs_elem), deferred :: calculate_density_second_derivs_elem
  procedure(i_calculate_specvol_derivs_elem), deferred :: calculate_specvol_derivs_elem
  procedure(i_calculate_compress_elem), deferred :: calculate_compress_elem
  procedure(i_EOS_fit_range), deferred :: EOS_fit_range
  ! shared, non-deferred fallbacks provided by the base module:
  procedure :: calculate_density_array_2d => a_calculate_density_array_2d
  procedure :: calculate_density_array_3d => a_calculate_density_array_3d
  procedure :: calculate_density_derivs_2d => a_calculate_density_derivs_2d
  procedure :: calculate_density_derivs_3d => a_calculate_density_derivs_3d
  procedure :: calculate_density_second_derivs_2d => a_calculate_density_second_derivs_2d
  ...
end type EOS_base
```
(`MOM_EOS_base_type.F90:13-74`)

Every deferred interface is `elemental` and carries `class(EOS_base), intent(in) :: this` as the
first argument (e.g. `i_density_elem`, `:81-88`). The **base fallbacks** (`a_calculate_density_array_2d`
at `:268`, `a_calculate_density_array_3d` at `:298`, `a_calculate_density_derivs_2d/3d` at `:428/456`,
`a_calculate_density_second_derivs_2d` at `:542`) all have the same shape: take a whole array section,
and apply the elemental deferred procedure to it *through `this`*:

```fortran
! a_calculate_density_array_3d, MOM_EOS_base_type.F90:298-326
subroutine a_calculate_density_array_3d(this, T, S, pressure, rho, dom, rho_ref)
  class(EOS_base), intent(in) :: this
  ...
  if (present(rho_ref)) then
    rho(is:ie, js:je, ks:ke) = this%density_anomaly_elem(T(is:ie, js:je, ks:ke), &
        S(is:ie, js:je, ks:ke), pressure(is:ie, js:je, ks:ke), rho_ref)
  else
    rho(is:ie, js:je, ks:ke) = this%density_elem(T(is:ie, js:je, ks:ke), &
        S(is:ie, js:je, ks:ke), pressure(is:ie, js:je, ks:ke))
  endif
end subroutine a_calculate_density_array_3d
```

This is a **whole-array assignment invoking an elemental type-bound procedure through a polymorphic
`this`** — the compiler must resolve, per call, which concrete override of `density_elem`/
`density_anomaly_elem` to invoke (v-table dispatch), and then apply it across an array section. This
is the fallback path used by **every EOS form that does not supply its own array/2d/3d override**
(see §3): `linear`, `UNESCO`, `Jackett06`, `TEOS10`, `Wright_full`, `Wright_red`, `Roquet_SpV`.

### 1.3 Concrete class selection: `select case` / `allocate` at init

The concrete class is chosen once, at `EOS_init`/`EOS_manual_init` time, with an ordinary
`select case` driving `allocate(<concrete-type> :: EOS%type)`:

```fortran
! MOM_EOS.F90:2122-2148
if (allocated(EOS%type)) deallocate(EOS%type) ! Needed during testing which re-initializes
select case (EOS%form_of_EOS)
  case (EOS_LINEAR)
    allocate(linear_EOS :: EOS%type)
  case (EOS_UNESCO)
    allocate(UNESCO_EOS :: EOS%type)
  case (EOS_WRIGHT)
    allocate(buggy_Wright_EOS :: EOS%type)
  case (EOS_WRIGHT_FULL)
    allocate(Wright_full_EOS :: EOS%type)
  case (EOS_WRIGHT_REDUCED)
    allocate(Wright_red_EOS :: EOS%type)
  case (EOS_JACKETT06)
    allocate(Jackett06_EOS :: EOS%type)
  case (EOS_TEOS10)
    allocate(TEOS10_EOS :: EOS%type)
  case (EOS_ROQUET_RHO)
    allocate(Roquet_rho_EOS :: EOS%type)
  case (EOS_ROQUET_SPV)
    allocate(Roquet_SpV_EOS :: EOS%type)
end select
select type (t => EOS%type)
  type is (linear_EOS)
    call t%set_params_linear(Rho_T0_S0, dRho_dT, dRho_dS, dRho_dp)
  type is (buggy_Wright_EOS)
    call t%set_params_buggy_Wright(use_Wright_2nd_deriv_bug)
end select
```

Every call site downstream (e.g. `calculate_density_3d`, `MOM_EOS.F90:427`) then dispatches through
`EOS%type%calculate_density_array_3d(...)` — ordinary Fortran type-bound-procedure dispatch on a
polymorphic `allocatable` component.

### 1.4 Why this is fatal on device

Two distinct, stacked problems:

1. **The outer v-table dispatch itself.** `EOS%type%calculate_density_array_3d(...)` is a runtime
   (dynamic) dispatch — the compiler must consult a type descriptor attached to the allocatable
   polymorphic component to find the correct concrete procedure. This dispatch happens once per host
   call (not per grid point), so it is not the primary GPU blocker by itself — but it means the
   *body* of whatever gets called still carries `class(EOS_base)/class(<concrete>), intent(in) :: this`
   as a live dummy argument in its interface.
2. **The polymorphic `this` inside device loops.** Any elemental procedure bound through `this`
   (`this%density_elem(...)`, `this%density_anomaly_elem(...)`, or even a same-module wrapper that
   takes `this` as its first dummy) that gets called *inside* a `do concurrent`/OpenMP-target region
   forces nvfortran to attempt to pass/copy the polymorphic descriptor into the device region per
   iteration. In practice this either (a) silently produces an **implicit copy of `this` that cannot
   be prevented** (documented in-source, see §2.3), which is a correctness/perf hazard, or (b) throws
   an outright **runtime error on GPU with nvfortran** for the forms that hadn't yet been rewritten
   (the exact phrase used repeatedly in-source, e.g. `MOM_EOS_Wright.F90:108-109`,
   `MOM_EOS_Roquet_rho.F90:260-261`). Passing `class(*)`/polymorphic `this` into a `do concurrent` or
   `target` region is exactly the failure mode called out in `00-architecture.md` guiding principle 4
   and quick-reference item in §9 ("**Never** pass `class(*)`/polymorphic `this`").

Net effect: the seven EOS forms that still rely purely on the `EOS_base` fallbacks (§1.2) cannot be
called from inside a device array loop at all without hitting this. The two forms that needed
GPU-resident 2D/3D density evaluation (Wright and Roquet_rho, used by the ported
`MOM_PressureForce_FV.F90`/`MOM_density_integrals.F90` pressure-gradient path) had to be rewritten.

---

## 2. The new strategy: free `_loc` kernels + explicit `do concurrent` overrides

### 2.1 Pattern

For each rewritten form, every elemental kernel that used to be a type-bound procedure
`foo_elem_XXX(this, T, S, p, ...)` is **duplicated** as a free (non-type-bound) elemental function
`foo_elem_XXX_loc(T, S, p, ...)` with **no `this` parameter at all**, and the original
`this`-taking procedure becomes a **thin one-line wrapper** that just calls the `_loc` version (kept
only so the deferred-interface contract in `EOS_base` is still satisfied for scalar/1-D call sites
that go through the ordinary polymorphic path). New **direct** `calculate_density_array_2d/3d`,
`calculate_density_derivs_2d/3d`, and (for Roquet_rho) `calculate_density_second_derivs_2d`
overrides are added on the concrete type, each containing an explicit `do concurrent (k,j,i)` /
`do concurrent (j,i)` loop that calls the `_loc` kernel directly — never through `this`.

### 2.2 Quoted example — Wright `_loc`/wrapper pair (`MOM_EOS_Wright.F90:90-117`)

```fortran
real elemental function density_elem_buggy_Wright_loc(T, S, pressure)
  real, intent(in) :: T        !< potential temperature relative to the surface [degC].
  real, intent(in) :: S        !< salinity [PSU].
  real, intent(in) :: pressure !< pressure [Pa].
  ! Local variables
  real :: al0, p0, lambda
  al0 = (a0 + a1*T) +a2*S
  p0 = (b0 + b4*S) + T * (b1 + T*(b2 + b3*T) + b5*S)
  lambda = (c0 +c4*S) + T * (c1 + T*(c2 + c3*T) + c5*S)
  density_elem_buggy_Wright_loc = (pressure + p0) / (lambda + al0*(pressure + p0))
end function density_elem_buggy_Wright_loc

!> Wrapper for density_elem_buggy_Wright_loc created to preserve API while calling
!! density_elem_buggy_Wright without "this" variable that causes runtime errors on
!! gpu runs with nvfortran.
real elemental function density_elem_buggy_Wright(this, T, S, pressure)
  class(buggy_Wright_EOS), intent(in) :: this !< This EOS
  real, intent(in) :: T, S, pressure
  density_elem_buggy_Wright = density_elem_buggy_Wright_loc(T, S, pressure)
end function density_elem_buggy_Wright
```

This is the literal pattern introduced by commit `52a1b3954` ("Added local versions of
density_elem and density_derivs without 'this' argument"). Its diff shows the **before** state was
exactly the disaster case described in §1.4: the 2D override itself used to call the `this`-taking
form *inside* the `do concurrent`:

```diff
   else
     do concurrent (j=js:je, i=is:ie)
-      rho(i,j) = density_elem_buggy_Wright(this, T(i,j), S(i,j), pressure(i,j))
+      rho(i,j) = density_elem_buggy_Wright_loc( T(i,j), S(i,j), pressure(i,j))
     enddo
```
(commit `52a1b3954`, `MOM_EOS_Wright.F90`)

### 2.3 Quoted example — 3D override calling the `_loc` kernel (`MOM_EOS_Wright.F90:1024-1061`)

```fortran
subroutine calculate_density_array_3d_buggy_Wright(this, T, S, pressure, rho, dom, rho_ref)
  class(buggy_Wright_EOS), intent(in) :: this
  ...
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
end subroutine calculate_density_array_3d_buggy_Wright
```

Two things worth flagging for the knowledge base:

- The **plain-density branch** (`rho_ref` absent) calls the `_loc` free function — clean, no `this`.
- The **anomaly branch** (`rho_ref` present) **still calls `density_anomaly_elem_buggy_Wright(this, ...)`**
  (2D at `:1013`, 3D at `:1053`, both verified) — i.e. in `MOM_EOS_Wright.F90` the `_loc` treatment
  was only applied to `density_elem` and `calculate_density_derivs_elem`. There is **no**
  `density_anomaly_elem_buggy_Wright_loc` free function in the file at all (grep-confirmed: the only
  `_loc` kernels in Wright are `density_elem_buggy_Wright_loc` at `:90` and
  `calculate_density_derivs_elem_buggy_Wright_loc` at `:199`).

There are therefore **two structurally different residuals**, and it matters that the knowledge base
keep them apart:

  1. **A genuinely `this`-dereferencing device loop (unfinished work).** The Wright anomaly branch
     (`:1013, :1053`) calls `density_anomaly_elem_buggy_Wright(this, ...)` *inside* the `do concurrent`
     body. That is not a compiler limitation — `MOM_EOS_Roquet_rho.F90` proves the fix is available:
     it *does* have `density_anomaly_elem_Roquet_rho_loc` (`:274`) and its array overrides call it in
     the anomaly branch (`:744, :783`). Wright simply never got a `density_anomaly` `_loc` kernel
     written. This is a mechanical, finishable gap, not an nvfortran wall.
  2. **A residual `this` that appears only in the *signature*, never in the loop body (compiler
     limitation).** The Wright and Roquet derivs overrides call *only* the `_loc` kernel inside the
     loop (Wright `:1117, :1150`; Roquet `:820, :854`) yet still carry the
     "**implicit copy of `this` which cannot yet be prevented**" comment. Here `this` is dereferenced
     nowhere in the region — it survives only because the enclosing subroutine must declare
     `class(<concrete>_EOS), intent(in) :: this` to satisfy the `EOS_base` deferred-binding contract,
     and nvfortran materializes/copies that descriptor for the device region regardless. This is the
     genuinely open compiler issue.

  **Correction to an earlier draft of this doc:** it is *not* the case that Roquet's
  `calculate_density_derivs_2d` "was never converted to call a `_loc` kernel." Line `:820` does call
  `calculate_density_derivs_elem_Roquet_rho_loc`; the loop body never touches `this`. The retained
  "implicit copy…cannot yet be prevented" note at Roquet `:817` is therefore case (2) — a genuine
  compiler-limitation annotation on a fully-converted path — and its wording is simply the *older*
  phrasing, left inconsistent with the newer "…called via their free-function (`_loc`) form rather
  than through the polymorphic `this` binding, which causes runtime errors in `do concurrent` regions
  offloaded to the GPU with nvfortran" comment used on the sibling 3D-density, 3D-derivs, and
  2D-second-derivs overrides (`:778-780, :850-852, :889-891`). Both comments describe the *same*
  case-(2) residual; only the wording differs. The `MOM_EOS_Wright.F90` comment sites are
  `:1008, :1048, :1114, :1147`; the sole Roquet "implicit copy" site is `:817`.

  > **FABLE-CHECK (reviewed 2026-07-14 — resolution or current status in KNOWLEDGE.md §8a/§8b):** The Wright anomaly branch (`:1013, :1053`) is the one place a device `do concurrent`
  > body still literally passes `this`. Confirm whether this is deliberate (a form the ported
  > PressureForce path never exercises with `rho_ref` present, so it was left) or an oversight. Roquet
  > (`:274, :744, :783`) shows the `_loc` fix is trivially available, so if any GPU code path reaches
  > `calculate_density_array_2d/3d_buggy_Wright` *with* `rho_ref`, this is a live correctness bug, not a
  > tolerated limitation. Look at callers of `calculate_density(..., rho_ref=...)` for `EOS_WRIGHT` in
  > `MOM_density_integrals.F90` / `MOM_PressureForce_FV.F90`.

### 2.4 `int_density_dz_wright`: whole-routine offload, not just the elemental kernel

`int_density_dz_wright` (`MOM_EOS_Wright.F90:426-706`) is not called through `EOS_base` dispatch at
all (it's a free module procedure used directly by `MOM_PressureForce_FV.F90`/
`MOM_density_integrals.F90`), so it has no `this`/polymorphism problem — but it is architecturally
part of the same EOS-layer port and was rewritten in commit `692abbc67` ("port int_density_dz_wright")
from plain nested `do`-loops to:
- `!$omp target enter data map(alloc: z0pres, al0_2d, p0_2d, lambda_2d, intz)` / matching
  `exit data map(release:...)` bracketing the whole routine (`:548, :704`),
- `do concurrent (j=..., i=...)` for the pointwise vertical-integral computation (`:550-605`),
- `!$omp target teams loop collapse(2) private(...)` (long private list: `hWght, hL, hR, iDenom,
  hWt_LL, hWt_LR, hWt_RR, hWt_RL, m, wt_L, wt_R, wtT_L, wtT_R, al0, p0, lambda, dz, p_ave, I_al0,
  I_Lzz, eps, eps2, intz`) for the horizontal (Boole's-rule) integrals in x and y (`:608-654,
  657-703`), replacing the old collapsed `do j=... ; do I=...` form. `target teams loop` is used
  here (not `do concurrent`) because the horizontal loops carry a serial inner `do m=2,4` and a
  sizeable private scalar/array list per (i,j) — the same pattern vertical friction uses for its
  tridiagonal column solve (see `00-architecture.md` §4.3).

---

## 3. Per-form port status

Objective evidence: count of `do concurrent` occurrences per EOS source file (a form with array/2d/3d
overrides has them; a form still on the `EOS_base` fallback path has zero, because its only elemental
kernels are plain scalar `elemental function`s with no device directives of their own):

| File | `do concurrent` count | Has `_loc` kernels? | Overrides `array_2d/3d`, `derivs_2d/3d`? | Overrides `second_derivs_2d`? | Status |
|---|---|---|---|---|---|
| `MOM_EOS_Wright.F90` (`buggy_Wright_EOS`) | 9 | yes (partial: `density_elem`, `derivs_elem` only) | yes | **no** | **Ported** (density + 1st derivs 2D/3D); anomaly branch still passes `this` |
| `MOM_EOS_Roquet_rho.F90` (`Roquet_rho_EOS`) | 11 | yes (density, anomaly, derivs, 2nd-derivs) | yes | **yes** | **Ported**, most complete conversion |
| `MOM_EOS_linear.F90` (`linear_EOS`) | 0 | no | no | no | Polymorphic fallback only |
| `MOM_EOS_UNESCO.F90` (`UNESCO_EOS`) | 0 | no | no | no | Polymorphic fallback only |
| `MOM_EOS_Jackett06.F90` (`Jackett06_EOS`) | 0 | no | no | no | Polymorphic fallback only |
| `MOM_EOS_TEOS10.F90` (`TEOS10_EOS`) | 0 | no | no | no | Polymorphic fallback only |
| `MOM_EOS_Wright_full.F90` (`Wright_full_EOS`) | 0 | no | no | no | Polymorphic fallback only |
| `MOM_EOS_Wright_red.F90` (`Wright_red_EOS`) | 0 | no | no | no | Polymorphic fallback only |
| `MOM_EOS_Roquet_SpV.F90` (`Roquet_SpV_EOS`) | 0 | no | no | no | Polymorphic fallback only |

Confirmed by `grep -c 'type, extends\|procedure ::.*array_2d\|...' `: each of the seven unported
files' `type, extends (EOS_base) :: <Name>_EOS ... end type` block contains **no** override of
`calculate_density_array_2d/3d`, `calculate_density_derivs_2d/3d`, or
`calculate_density_second_derivs_2d` — every one of them relies purely on the `EOS_base` fallbacks
in §1.2, i.e. on `this%density_elem(...)` applied elementally across whole array sections. Any code
path that ends up calling `calculate_density_3d`/`calculate_density_derivs_3d` for `EOS_LINEAR`,
`EOS_UNESCO`, `EOS_JACKETT06`, `EOS_TEOS10`, `EOS_WRIGHT_FULL`, `EOS_WRIGHT_REDUCED`, or
`EOS_ROQUET_SPV` still routes through the polymorphic `this%..._elem` dispatch and is the disaster
case from §1.4 if it is ever invoked from a device loop. `EOS_DEFAULT` in `MOM_EOS.F90:192` is
`EOS_WRIGHT_FULL_STRING` — i.e. **the model's default EOS is still on the unported path**; only
configurations that explicitly select `WRIGHT` (buggy) or `ROQUET_RHO` get the GPU-safe direct
kernels.

---

## 4. `calculate_density_*` generic wrappers: fast path + rescale path (`MOM_EOS.F90`)

`MOM_EOS.F90` is the generic front door (`interface calculate_density`, `:70-78`;
`calculate_density_derivs`, `:87-92`; `calculate_density_second_derivs`, `:101-104`). Every 2D/3D
generic wrapper (`calculate_density_2d`, `calculate_density_3d` at `:427`,
`calculate_density_derivs_2d`, `calculate_density_derivs_3d` at `:1055`,
`calculate_density_second_derivs_2d` at `:1236`) has the identical shape: test whether the EOS's
unit-rescaling factors are all exactly `1.0` and, if so, skip the rescale copies and call
`EOS%type%calculate_..._2d/3d` directly on the caller's arrays; otherwise rescale `T`/`S`/`pressure`
into temporaries first. E.g. `calculate_density_3d` (`MOM_EOS.F90:427-485`):

```fortran
if ((EOS%RL2_T2_to_Pa == 1.0) .and. (EOS%R_to_kg_m3 == 1.0) .and. &
    (EOS%C_to_degC == 1.0) .and. (EOS%S_to_ppt == 1.0)) then
  call EOS%type%calculate_density_array_3d(T, S, pressure, rho, domain, rho_ref=rho_ref)
else ! This is the same as above, but with some extra work to rescale variables.
  pres(is:ie, js:je, ks:ke) = EOS%RL2_T2_to_Pa * pressure(is:ie, js:je, ks:ke)
  Ta(is:ie, js:je, ks:ke)   = EOS%C_to_degC   * T(is:ie, js:je, ks:ke)
  Sa(is:ie, js:je, ks:ke)   = EOS%S_to_ppt    * S(is:ie, js:je, ks:ke)
  if (present(rho_ref)) then
    call EOS%type%calculate_density_array_3d(Ta, Sa, pres, rho, domain, rho_ref=EOS%R_to_kg_m3*rho_ref)
  else
    call EOS%type%calculate_density_array_3d(Ta, Sa, pres, rho, domain)
  endif
endif
```

This fast path exists purely for the common non-dimensional-testing configuration (scale factors
== 1) and avoids allocating/filling three full-size rescale temporaries (`pres`, `Ta`, `Sa`) per
call — a performance optimization orthogonal to the polymorphism problem, but relevant because it
determines whether the call into `EOS%type%calculate_density_array_3d` receives the caller's own
arrays directly (fast path) or freshly-computed local temporaries (rescale path); either way the
call itself still goes through the same `EOS%type%...` dynamic dispatch described in §1.3.

The dispatch-level guard everywhere upstream of the fast path is `if (.not. allocated(EOS%type))
call MOM_error(FATAL, ...)` (e.g. `:1034, :1096, :1184`) — a defensive check that the concrete class
was actually allocated by `EOS_init`/`EOS_manual_init` before any dispatch is attempted.

---

## 5. `int_density_dz_*` / `int_spec_vol_dp_*` pressure-integral routines

These are free module procedures (not `EOS_base` type-bound), called directly from
`MOM_PressureForce_FV.F90` for the two EOS-specific fast paths (Wright, linear) and from
`src/core/MOM_density_integrals.F90`'s **generic** PCM/PLM/PPM routines
(`int_density_dz_generic_pcm/plm/ppm`, `int_spec_vol_dp_generic_pcm/plm`) for every other EOS form
— the generic routines call back into the polymorphic `calculate_density`/`calculate_density_derivs`
generic interface, so they inherit whichever port status the underlying EOS form has (§3).

Port status by direct grep for `do concurrent` in each EOS-specific integral file:

| Routine | File | Ported? |
|---|---|---|
| `int_density_dz_wright` | `MOM_EOS_Wright.F90:426` | **Yes** — `do concurrent` (pointwise) + `!$omp target teams loop collapse(2)` (Boole's-rule horizontal integrals), commit `692abbc67` |
| `int_spec_vol_dp_wright` | `MOM_EOS_Wright.F90:713` (same file, later section) | **No** — plain nested loops, zero directives in this region |
| `int_density_dz_linear` / `int_spec_vol_dp_linear` | `MOM_EOS_linear.F90:277,483` | No |
| `int_density_dz_wright_full` / `int_spec_vol_dp_wright_full` | `MOM_EOS_Wright_full.F90:397,669` | No |
| `int_density_dz_wright_red` / `int_spec_vol_dp_wright_red` | `MOM_EOS_Wright_red.F90:399,671` | No |
| Roquet_rho | — | Has **no** `int_density_dz`/`int_spec_vol_dp` of its own; always routes through `int_density_dz_generic_*` in `MOM_density_integrals.F90`, which calls the ported `calculate_density_3d`/`calculate_density_derivs_3d` array entry points for Roquet_rho specifically |

So within `MOM_EOS_Wright.F90` itself the port is **half-done**: the density-anomaly integral
(`int_density_dz_wright`) is GPU-resident, but the companion specific-volume integral
(`int_spec_vol_dp_wright`, used for the non-Boussinesq path) is not.

### `port/pressureforce-benchmark_ALE` — k-blocking `int_density_dz_generic_plm`

This branch (commits `8c6881e6c` "move kblock inside int_density_dz_generic_plm", `1038a4921`,
`a3e889601`, on top of merged commit `c82e1254a` "vertvisc: Fix CS memory management") k-blocks the
**generic** ALE/PLM pressure-integral path in `src/core/MOM_density_integrals.F90` and its submodule
implementation `MOM_density_integrals_s.F90` — a different track from the EOS-layer `_loc` rewrite.
Signature change: `int_density_dz_generic_plm(k, ...)` → `int_density_dz_generic_plm(kstart, kend,
...)`, with `dpa`/`intz_dpa`/`intx_dpa`/`inty_dpa` gaining a `SZK_(GV)` third dimension so a whole
k-range can be processed per call. Inside `generic_plm_update_dpa`, the k-loop is folded into the
existing tiled `do concurrent`:

```fortran
! before (single k, passed in from caller)
do concurrent (j=jstart:jend, i=istart:iend)
  ...
enddo
! after (k-blocked)
do concurrent (k=kstart:kend, j=jstart:jend, i=istart:iend)
  ii = i-istart+1 ; jj = j-jstart+1
  dz = e(i,j,K) - e(i,j,K+1)
  ...
enddo
```

but the **EOS evaluation itself inside that k-loop still calls the generic `calculate_density`
interface** (`call calculate_density(T5, S5, p5, T25, TS5, S25, r5, EOS, EOSdom_h5, rho_ref=rho_ref)`,
wrapped in its own `do k=kstart,kend` loop, separate from the k-blocked `do concurrent`) — i.e. this
branch k-blocks the *driver* loop structure but does not touch or depend on which EOS form is
active; it inherits the polymorphic-dispatch status of whatever `EOS%type` is allocated, same as the
unmodified generic PCM/PPM routines. This confirms the k-blocking effort (§5 of
`00-architecture.md`) and the EOS `_loc`-rewrite effort are two independent, not-yet-unified tracks.

Branches `remotes/origin/eos-3d`, `remotes/origin/efficient_density_integrals_new_api`,
`efficient_density_integrals_rebase`, `efficient_density_integrals_stanley` are earlier/alternative
staging points for the same `calculate_density_3d`/`_derivs_3d`/`_second_derivs_2d` API additions
that commit `7c7af5572` ultimately merged (its own commit message: "These were extracted from a
larger pull request supporting pressure density integrals (#156)" — i.e. `#185`/`7c7af5572` is a
narrower cherry-pick of that larger, still-unmerged effort).

---

## 6. What remains, boilerplate cost, and the endgame

### 6.1 What remains

- **Seven of nine EOS forms** (`linear`, `UNESCO`, `Jackett06`, `TEOS10`, `Wright_full`,
  `Wright_red`, `Roquet_SpV`) have **no** GPU-safe 2D/3D density/derivs path at all — including
  `Wright_full`, which is `EOS_DEFAULT` (`MOM_EOS.F90:192`). Any GPU run using the default EOS (or
  any of the other six) that reaches a 3D array density/derivs call falls straight back onto
  `EOS_base`'s `this%..._elem` array-section fallback (§1.2), the exact pattern documented as fatal.
- Even within the two "ported" forms, the conversion is **incomplete**: `buggy_Wright_EOS` has no
  `calculate_density_second_derivs_2d` override (falls back to `EOS_base`'s), and its
  `density_anomaly_elem` path still passes `this` inside `do concurrent` (§2.3). `int_spec_vol_dp_wright`
  (companion to the ported `int_density_dz_wright`) is unported.
- The "implicit copy of `this` which cannot yet be prevented" comment is left **unresolved** in
  five places (`MOM_EOS_Wright.F90:1008,1048,1114,1147`; `MOM_EOS_Roquet_rho.F90:817`). As split in
  §2.3, these are **not all the same thing**:
    - `MOM_EOS_Wright.F90:1114,1147` (derivs 2D/3D) and `MOM_EOS_Roquet_rho.F90:817` (derivs 2D) sit
      above `do concurrent` bodies that call **only** the `_loc` kernel and never touch `this` — these
      are the pure **compiler-limitation** case: `this` survives only in the formal parameter list
      (mandated by the `EOS_base` deferred-binding contract) and nvfortran's do-concurrent/target
      lowering materializes/copies that descriptor regardless. No workaround for this specific residual
      is recorded in-source yet.
    - `MOM_EOS_Wright.F90:1008,1048` head `calculate_density_array_2d/3d_buggy_Wright`, whose **anomaly
      branch** (`:1013,:1053`) genuinely dereferences `this` inside the loop. That part is
      **unfinished work** (no `density_anomaly_elem_buggy_Wright_loc` exists), finishable by copying
      Roquet's approach — see the FABLE-CHECK in §2.3.

### 6.2 Boilerplate cost of the `_loc` duplication, per form

For a form fully converted like `Roquet_rho`, the pattern requires, per elemental kernel that needs
device access:
1. A new free `_loc` elemental function/subroutine with the same body but no `this` (full
  duplication of the arithmetic — not a refactor, a copy).
2. The original `this`-taking type-bound procedure reduced to a one-line wrapper calling the `_loc`
  version (kept only to satisfy `EOS_base`'s deferred interface for scalar/1-D generic callers).
3. A new `calculate_density_array_2d`/`_3d`, `calculate_density_derivs_2d`/`_3d`, and (optionally)
  `calculate_density_second_derivs_2d` override on the concrete type, each hand-written with its own
  `do concurrent` loop and `dom(...)` index bookkeeping — duplicating the loop-and-index-slicing
  logic that `EOS_base`'s fallback already provides generically for free.

Concretely: `MOM_EOS_Roquet_rho.F90` grew from a file with only elemental kernels to one with **11
distinct `do concurrent` loops**, each hand-duplicating the domain-slicing arithmetic already present
once in `MOM_EOS_base_type.F90`'s generic fallbacks. Commit `7c7af5572` alone added +174 lines to
`MOM_EOS_Roquet_rho.F90` and +77 to `MOM_EOS_Wright.F90` purely for the 2D/3D array overrides (plus
+92 to `MOM_EOS_base_type.F90` for the generic fallback siblings that non-ported forms still use).
Scaling this same treatment to the remaining seven forms would mean **7× more duplicated kernels**
(each EOS form already has ~7-9 elemental kernels: density, anomaly, spec_vol, spec_vol anomaly,
derivs, second-derivs, specvol-derivs, compress) plus 7×3-5 hand-written array-loop overrides — a
large, mechanical, error-prone amount of copy-paste, and every future bugfix to an elemental kernel's
math must now be applied in two places (the `this`-taking original and the `_loc` copy) unless the
original is reduced to a pure pass-through (as done for the fully-converted kernels).

### 6.3 Prescriptive recipe: port one more EOS form, bit-for-bit

This is the exact, mechanical procedure to give any remaining form (`linear`, `UNESCO`, `Jackett06`,
`TEOS10`, `Wright_full`, `Wright_red`, `Roquet_SpV`) a GPU-safe 2D/3D path, matching the merged
`buggy_Wright`/`Roquet_rho` pattern. It touches **exactly one file**, `MOM_EOS_<Form>.F90` — no
change to `MOM_EOS.F90` is needed, because the generic front-door wrappers there already dispatch to
`EOS%type%calculate_density_array_2d/3d`, `..._derivs_2d/3d`, `..._second_derivs_2d` (§4); overriding
those bindings on the concrete type is automatically picked up. Use `MOM_EOS_Roquet_rho.F90` as the
reference implementation (the most complete conversion).

**Step 0 — scope.** Decide which of the five array entry points the GPU path actually needs. The full
set is `calculate_density_array_2d`, `calculate_density_array_3d`, `calculate_density_derivs_2d`,
`calculate_density_derivs_3d`, `calculate_density_second_derivs_2d`. Roquet_rho overrides all five;
buggy_Wright overrides the first four (no `second_derivs_2d`). Each override you add shadows the
`EOS_base` `a_*` fallback **for that form only**; forms you don't touch keep using the fallback.

**Step 1 — `_loc` free kernels (one per elemental kernel the overrides call).** For each of
`density_elem`, `density_anomaly_elem`, `calculate_density_derivs_elem`, and (only if doing
second_derivs) `calculate_density_second_derivs_elem`, add a free (non-type-bound) elemental
procedure `<kernel>_<form>_loc(T, S, pressure[, ref][, out args])` whose body is **copied verbatim**
from the existing `<kernel>_<form>` with the `this` dummy deleted. Do **not** refactor or re-parenthesize
the arithmetic — bitwise reproducibility (`00-architecture.md` principle 2) requires identical
floating-point operation order. Reference bodies: Roquet `density_elem_Roquet_rho_loc` (`:204`),
`density_anomaly_elem_Roquet_rho_loc` (`:274`), `calculate_density_derivs_elem_Roquet_rho_loc`
(`:375`), `calculate_density_second_derivs_elem_Roquet_rho_loc` (`:466`). **Do not skip
`density_anomaly` — that omission is exactly the unfinished-work gap in Wright (§2.3).**

**Step 2 — reduce each original kernel to a one-line wrapper.** Keep the
`class(<form>_EOS), intent(in) :: this` dummy and the `!>` doc comment (the deferred `EOS_base`
interface still requires the binding for scalar/1-D generic callers), replace the body with a single
call to the `_loc` version. Reference: Roquet `density_elem_Roquet_rho` (`:262-268`),
`density_anomaly_elem_Roquet_rho` (`:335-342`).

**Step 3 — declare the array overrides in the type block.** In `type, extends(EOS_base) :: <form>_EOS
… contains`, add (mirroring Roquet `:185-195`):
```fortran
procedure :: calculate_density_array_2d       => calculate_density_array_2d_<form>
procedure :: calculate_density_array_3d       => calculate_density_array_3d_<form>
procedure :: calculate_density_derivs_2d      => calculate_density_derivs_2d_<form>
procedure :: calculate_density_derivs_3d      => calculate_density_derivs_3d_<form>
procedure :: calculate_density_second_derivs_2d => calculate_density_second_derivs_2d_<form> ! optional
```

**Step 4 — implement each override.** Copy the **signature** of the matching `EOS_base` fallback
(`MOM_EOS_base_type.F90`: `a_calculate_density_array_2d` `:268`, `_array_3d` `:298`,
`a_calculate_density_derivs_2d` `:428`, `_derivs_3d` `:456`, `a_calculate_density_second_derivs_2d`
`:542`). Body: unpack `dom(rank,2)` into `is/ie, js/je[, ks/ke]`, then write an explicit
`do concurrent (…, j=js:je, i=is:ie)` that calls the `_loc` kernel. For the density arrays,
write **both** branches — `present(rho_ref)` → `density_anomaly_elem_<form>_loc(…, rho_ref)`, else
`density_elem_<form>_loc(…)`. Reference: Roquet `calculate_density_array_3d_Roquet_rho` (`:782-789`),
`calculate_density_derivs_3d_Roquet_rho` (`:853-855`), `calculate_density_second_derivs_2d_Roquet_rho`
(`:892-893`).

**Step 5 — annotate.** Add the NOTE comment above the loop; prefer the newer, accurate wording
(Roquet `:778-780`) over the legacy "implicit copy" phrasing. Leave `this` in the signature
(unavoidable — see §2.3 case 2).

**Step 6 — self-check (no build).** `grep -c 'do concurrent' MOM_EOS_<Form>.F90` should rise by the
number of loops added (Wright=9, Roquet=11 for reference). Confirm **no** `<form>_EOS` array override
calls `<kernel>_<form>(this, …)` inside a `do concurrent` — every device-region call must target a
`_loc` kernel. The build+numerical check (`MOM_checksums` hchksum CPU vs GPU, `00-architecture.md`
§9) is the acceptance gate but is out of scope for a source-only agent.

**Out of scope for this recipe:** the `int_density_dz_<form>` / `int_spec_vol_dp_<form>` pressure
integrals (§2.4, §5) are a *separate* whole-routine offload (`enter data` + `do concurrent` +
`target teams loop`), independent of the `_loc` dispatch rewrite. Only `int_density_dz_wright` is
done; `int_spec_vol_dp_wright` and every other form's integrals are not.

### 6.4 Proposal (not existing code): a polymorphism-free end-state

> **Everything in this subsection is a *proposal* sketched from the evidence, not code present in the
> tree.** It is offered so a future agent has a target architecture; verify feasibility before acting.

The `_loc` recipe (§6.3) removes `this` from the *loop body* but cannot remove it from the override
*signature* — the residual descriptor copy (§2.3 case 2) persists because `EOS_base`'s deferred
bindings mandate the `this` dummy. A structural fix would drop the `class(EOS_base), allocatable ::
type` component (`MOM_EOS.F90:162`) and the abstract type entirely, replacing runtime v-table
dispatch with a compile-time `select case (EOS%form_of_EOS)` at the generic front door:

- The elemental `_loc` kernels are **already `this`-free**, so they carry over unchanged and become
  plain public module procedures of each `MOM_EOS_<Form>` module.
- The generic wrappers in `MOM_EOS.F90` (`calculate_density_3d` etc., §4) replace
  `call EOS%type%calculate_density_array_3d(…)` with `select case (EOS%form_of_EOS) ; case
  (EOS_WRIGHT) ; call calculate_density_array_3d_buggy_Wright(…) ; case (EOS_ROQUET_RHO) ; … ; end
  select`, calling the form's array routine as a module procedure with **no `this`**. This eliminates
  both the v-table dispatch (§1.4 problem 1) and the descriptor-copy residual (§1.4 problem 2)
  structurally, in one place, rather than per kernel.
- Instance parameters move onto `EOS_type` directly (a small non-polymorphic surface: `buggy_Wright`
  needs only `three` + `use_Wright_2nd_deriv_bug`; `linear` needs `Rho_T0_S0, dRho_dT, dRho_dS,
  dRho_dp`; most forms carry only module `parameter`s already, e.g. Wright's `a0…c5`). `set_params`
  fills the relevant fields; the existing `allocate(<concrete> :: EOS%type)` + `select type`
  (`:2123`) is deleted.
- Trade-off: the `select case` must enumerate every form at each of the ~5 array entry points (a
  fixed, bounded amount of code), versus today's open-ended per-form `_loc` duplication. It also
  couples `MOM_EOS.F90` to every form module (already effectively true via `use`). The upstream
  `eos-3d` / `efficient_density_integrals_*` branches and PR `#156` are the place to check whether a
  variant of this is already in progress before re-deriving it.

### 6.5 Is there a plan to eliminate polymorphism entirely?

**No integration/removal plan is recorded in-source.** What exists today is a per-form, per-kernel,
opt-in escape hatch (`_loc` + explicit override) applied to exactly the two forms actually exercised
by the ported pressure-force path (`buggy_Wright_EOS`, `Roquet_rho_EOS`), while `EOS_base` itself
(the abstract type, its `deferred`/polymorphic interface, and its `this`-based fallbacks) is
untouched and still the only implementation for seven forms including the default. There is no
in-repo comment, TODO, or branch proposing to replace `EOS_type`'s `class(EOS_base), allocatable`
component with a non-polymorphic tagged-union / `select case (form_of_EOS)` dispatch at the call
site (which would sidestep the v-table/`this`-descriptor problem structurally rather than by
per-kernel duplication) — the `_loc` pattern as merged is a **local workaround**, not the
architectural fix `00-architecture.md` alludes to when it says "the EOS layer is being rewritten to
avoid it." Based on the commit history (`7c7af5572`/`#185` explicitly extracted from a larger,
still-unmerged PR `#156`, plus the parallel `eos-3d`/`efficient_density_integrals_*` staging
branches), the wider rewrite is in progress upstream but not yet visible as a completed design in
this tree; a porting agent picking this up next should treat "convert the remaining 7 forms with the
same `_loc` boilerplate" as the known-mechanical stopgap, and treat "replace `class(EOS_base)`
dispatch with a non-polymorphic form" as the still-open architectural question.

---

## Verification notes

Opus verification pass (source + git only; no build/run). Checked every factual claim against
`MOM_EOS.F90`, `MOM_EOS_base_type.F90`, `MOM_EOS_Wright.F90`, `MOM_EOS_Roquet_rho.F90`, and commits
`7c7af5572`, `52a1b3954`, `692abbc67`, plus the named branches.

**Confirmed (verified against code/git):**
- `do concurrent` per-form table (§3): Wright 9, Roquet_rho 11, all seven other forms 0 — exact.
- Wright anomaly branch still passes `this` inside `do concurrent`: 2D `:1013`, 3D `:1053` — verified.
- `Wright_full` is `EOS_DEFAULT` (`MOM_EOS.F90:186,192`) and has zero `do concurrent` — verified.
- `int_spec_vol_dp_wright` (`:713-957`) has **no** device directives → unported; `int_density_dz_wright`
  (`:426-706`) has `enter data` (`:548`) / `exit data` (`:704`) / `do concurrent` / `target teams loop`
  → ported (`692abbc67`) — both verified.
- `EOS_type` `:117`, `class(EOS_base), allocatable :: type` `:162`; `select case`/`allocate` at
  `:2123` (doc said `:2122`, off by one — left as-is, immaterial); base fallbacks at
  `:268/298/428/456/542`; deferred bindings and generic wrappers (`calculate_density_3d :427`,
  `_derivs_3d :1055`, `_second_derivs_2d :1236`, `.not. allocated` guards) — all verified.
- Commit stats: `7c7af5572` +174 Roquet / +77 Wright / +92 base / +241 `MOM_EOS.F90`; `52a1b3954`
  +35/−8 Wright; `692abbc67` — verified. `#156` extraction message — verified.
- Branches `eos-3d`, `efficient_density_integrals_{new_api,rebase,stanley}`,
  `port/pressureforce-benchmark_ALE` (commit `8c6881e6c`, `int_density_dz_generic_plm(kstart,kend,…)`
  signature) — all exist and match — verified.
- Roquet is the more complete conversion: `_loc` kernels for density/anomaly/derivs/2nd-derivs
  (`:204/274/375/466`); array overrides call `_loc` in both branches (`:744/783/788`, derivs
  `:820/854`, 2nd-derivs `:893`) — verified.

**Corrected:**
- §2.3 previously said Roquet's `calculate_density_derivs_2d` (`:817`) "was never converted to call a
  `_loc` kernel." **False** — line `:820` calls `calculate_density_derivs_elem_Roquet_rho_loc`; the
  loop never touches `this`. Rewrote §2.3/§6.1 to split the two residual types: (1) a genuinely
  `this`-dereferencing device loop = **unfinished work** (Wright anomaly branch, fixable because
  Roquet already has `density_anomaly_elem_Roquet_rho_loc :274`); (2) a `this` that appears only in
  the signature = **genuine nvfortran limitation** (Wright derivs `:1114/1147`, Roquet `:817`). The
  `:817` "implicit copy" note is stale wording for case (2), not evidence of an unconverted path.

**Enhanced:**
- Added §6.3 (fully prescriptive 6-step port recipe: exact procedures, files, signatures, self-check).
- Added §6.4 (labeled *proposal*: `select case (form_of_EOS)` polymorphism-free end-state).
- Renumbered old §6.3 → §6.5 (unchanged content).

**FABLE-CHECK markers:** 1 (§2.3 — whether the Wright anomaly `this`-passing branch is a tolerated
dead path or a live GPU correctness bug; needs a caller trace in `MOM_density_integrals.F90` /
`MOM_PressureForce_FV.F90`).

**Confidence:** High. All quantitative claims (counts, line numbers, commit stats, branch existence)
independently reproduced from source and git. The one substantive error was in causal reasoning, not
in the underlying line references, and has been corrected. Residual uncertainty is confined to the
single flagged FABLE-CHECK, which requires a build/run or deeper caller trace to close definitively.
