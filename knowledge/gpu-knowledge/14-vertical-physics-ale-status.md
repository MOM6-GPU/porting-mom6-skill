# Vertical physics (diabatic) and ALE remap/regrid: porting status (dev/gpu)

> Drills into architecture doc §1, §4.3, §6.3. Scope: `src/parameterizations/vertical/`,
> `src/parameterizations/lateral/MOM_mixed_layer_restrat.F90` /
> `MOM_thickness_diffuse.F90`, and `src/ALE/`. This is the **largest remaining porting
> surface** on `dev/gpu`: the entire vertical-mixing (diabatic) stack and the ALE
> vertical-Lagrangian remap/regrid machinery are essentially untouched on mainline, with
> a scatter of in-flight, non-converged branches attacking pieces of it. Read this before
> starting any work in these directories.

---

## 1. Mainline status: confirmed host-only (`git diff --numstat dev-gfdl...dev/gpu`)

| File | + | − | Status |
|---|---|---|---|
| `src/parameterizations/vertical/MOM_set_diffusivity.F90` | 0 | 0 | **untouched** |
| `src/parameterizations/vertical/MOM_CVMix_KPP.F90` | 0 | 0 | **untouched** |
| `src/parameterizations/vertical/MOM_energetic_PBL.F90` | 0 | 0 | **untouched** (mainline only) |
| `src/parameterizations/lateral/MOM_mixed_layer_restrat.F90` | 0 | 0 | **untouched** |
| `src/ALE/MOM_regridding.F90` | 0 | 0 | **untouched** |
| `src/ALE/MOM_remapping.F90` | 0 | 0 | **untouched** |
| `src/parameterizations/vertical/MOM_diabatic_driver.F90` | 3 | 0 | cosmetic only (device buffer sync around one diagnostic) |
| `src/ALE/MOM_ALE.F90` | 3 | 0 | cosmetic only (same pattern) |
| `src/parameterizations/lateral/MOM_thickness_diffuse.F90` | 3 | 0 | cosmetic only (same pattern) |
| `src/parameterizations/vertical/MOM_kappa_shear.F90` | 2 | 1 | trivial: CPU `!$OMP parallel do` `shared()` clause fix (data-race bugfix, not a GPU port) |
| `src/parameterizations/vertical/MOM_diabatic_aux.F90` | 13 | 12 | **partial**: one subroutine (`find_uv_at_h`) ported |
| `src/parameterizations/vertical/MOM_vert_friction.F90` | 775 | 184 | ported (merged, reference pattern — see architecture doc §4.3) |
| `src/parameterizations/vertical/MOM_set_viscosity.F90` | 501 | 384 | ported (merged, reference pattern) |

The three "3-line" hits (`MOM_diabatic_driver.F90`, `MOM_ALE.F90`, `MOM_thickness_diffuse.F90`) are
all the *identical* pattern — wrapping a `call find_eta(...)` with manual
`!$omp target update to(h)` / `!$omp target enter data map(alloc: eta)` /
`!$omp target exit data map(from: eta)` so a host-side diagnostic call can read a
device-resident `h`. E.g. in `MOM_diabatic_driver.F90`:

```fortran
  if (CS%id_e_predia > 0) then
    !$omp target update to(h)
    !$omp target enter data map(alloc: eta)
    call find_eta(h, tv, G, GV, US, eta, dZref=G%Z_ref)
    !$omp target exit data map(from: eta)
    call post_data(CS%id_e_predia, eta, CS%diag)
  endif
```

This is plumbing around one specific diagnostic hook (`find_eta` itself was ported separately,
see `src/core/MOM_interface_heights.F90` and branch `find-eta-gpu`/`find-eta-merge` below) — it is
**not** evidence of any diabatic/ALE porting. `MOM_set_diffusivity.F90`, `MOM_CVMix_KPP.F90`,
`MOM_energetic_PBL.F90`, `MOM_mixed_layer_restrat.F90`, `MOM_regridding.F90`, `MOM_remapping.F90`
have **zero** diff against `dev-gfdl` — verified by empty `git diff --numstat` output (not merely
absent from a grep) for each file individually.

The one real (if narrow) exception is `MOM_diabatic_aux.F90::find_uv_at_h` — see §2.4.

---

## 2. The diabatic driver: structure, dispatch, and the heavy column kernels

### 2.1 Top-level dispatch (`MOM_diabatic_driver.F90`)

- `diabatic` (`:279`, `end` `:530`) — top-level entry called from `MOM.F90` (`step_MOM_thermo`).
  Chooses between an ALE-coordinate path and a legacy layered path.
- `diabatic_ALE_legacy` (`:535`–`:1242`) — older ALE-mode implementation, still present.
- `diabatic_ALE` (`:1247`–`:1873`) — current ALE-mode implementation (~630 lines).
- `layered_diabatic` (`:1877`–`:2868`) — non-ALE (isopycnal/layered) implementation (~1000 lines).
- `adiabatic` (`:2905`) — no-mixing pass-through path.

All three dispatch routines (`diabatic_ALE_legacy`, `diabatic_ALE`, `layered_diabatic`) call, in
sequence, the same set of column-physics kernels (confirmed by grep across the file — the three
`set_diffusivity` call-site pairs below live one per routine: 694/697 in `diabatic_ALE_legacy`,
1409/1412 in `diabatic_ALE`, 2113/2116 in `layered_diabatic`):

| Kernel | Call sites (line #s) | Module |
|---|---|---|
| `set_diffusivity` | 694, 697, 1409, 1412, 2113, 2116 | `MOM_set_diffusivity.F90` |
| `KPP_compute_BLD` / `KPP_calculate` | 757/760, 763/766, 1473/1476, 1479/1482, 2172/2175, 2178/2181 | `MOM_CVMix_KPP.F90` |
| `differential_diffuse_T_S` | 826, 2253 | `MOM_diabatic_aux.F90` |
| `energetic_PBL` / `energetic_PBL_get_MLD` | 912/915, 1563/1566 | `MOM_energetic_PBL.F90` |
| `bulkmixedlayer` | 2051, 2056, 2460 | `MOM_bulk_mixed_layer.F90` |
| `tracer_vertdiff_Eulerian` / `triDiagTS_Eulerian` | 1016/1017/1019, 1664/1665 | `MOM_diabatic_aux.F90` |
| `tracer_vertdiff` / `triDiagTS` | 2419/2420/2422, 2509/2510/2512 | `MOM_diabatic_aux.F90` |
| `regularize_layers` | 2539 | `MOM_regularize_layers.F90` |

(`calculate_kappa_shear`/`MOM_kappa_shear.F90` is invoked one level up, from
`MOM_dynamics_split_RK2.F90`/`set_viscous_ML`, feeding `Kd_shear`/`Kv_shear` into `visc`, which
`set_diffusivity` and `energetic_PBL` then consume — so it is on the same critical path even
though it isn't a direct call from `diabatic_driver.F90`.)

### 2.2 Which kernels are inherently serial-in-k

All of these are **column-physics kernels**: for a fixed `(i,j)` they operate on the full
`k=1..nz` water column and cannot be vectorized/parallelized over `k` because they are recurrence
relations up or down the column:

- **`tracer_vertdiff` / `triDiagTS`** — literally solve a tridiagonal system per column (Thomas
  algorithm: forward elimination down `k`, back-substitution up `k`). This is exactly the pattern
  already ported in `MOM_vert_friction.F90` (`!$omp target teams loop collapse(2)` over `(i,j)`
  with a serial inner `k` loop, 3 `declare target` column kernels) — the architecture doc's
  "vertvisc-style teams-loop treatment" is the template these must reuse.
- **`set_diffusivity` → `find_N2`, `find_TKE_to_Kd`** — build interface buoyancy/N² profiles top-down
  and then propagate a `maxEnt`/`kb`-indexed recurrence up and down the column (see §3 excerpts:
  `maxEnt(i,K) = ds_dsp1(i,K)*(maxEnt(i,K-1) + htot(i))`, `htot` accumulates across `k`).
- **`energetic_PBL`** — an iterative TKE-budget integration that walks down the column mixing
  layers into an evolving mixed layer (mech_TKE/conv_PErel accumulate across `k`); see §5 for the
  most advanced in-flight attempt.
- **`KPP_calculate`/`KPP_compute_BLD`** — searches down the column for the boundary-layer depth
  (an OBL search + shape-function evaluation), then computes a diffusivity profile — column-local
  but with a top-down search recurrence.
- **`kappa_shear`** — solves an implicit/iterative shear-instability closure per column
  (`Calc_kappa_shear_vertex`), similarly recurrence-based.
- **`differential_diffuse_T_S`** — solves a diffusion-like tridiagonal in `k` for T/S separately
  from the main mixing.
- **`bulkmixedlayer`** — mixed-layer entrainment/detrainment logic, inherently sequential
  top-down through the mixed-layer slab.

None of these can be flattened to `do concurrent (k,j,i)` the way continuity/CoriolisAdv were;
they need the two-level pattern already proven in `MOM_vert_friction.F90`: parallelize over
`(i,j)` (whole-column-per-thread), keep `k` as a private serial loop inside a `declare target`
column kernel.

### 2.3 `layered_diabatic` vs `diabatic_ALE`

Both drivers call the *same* physics kernels but differ in bookkeeping: `layered_diabatic` mixes
directly into an isopycnal grid and calls `regularize_layers` at the end to repair degenerate
layers; `diabatic_ALE` defers grid regeneration to the subsequent ALE remap step
(`ALE_regridding_and_remapping`, `MOM.F90:916`/`:1037`) and has no `regularize_layers` call. This
means **porting the column kernels once (in `MOM_set_diffusivity.F90`/`MOM_CVMix_KPP.F90`/
`MOM_energetic_PBL.F90`) benefits all three dispatch paths** (`diabatic_ALE_legacy` included) — there
is no need to port them per-driver.

### 2.4 The one real precedent: `find_uv_at_h` in `MOM_diabatic_aux.F90`

This is the only genuine (if narrow) GPU port merged anywhere in the diabatic stack. Before:
plain nested `do j / do i` with `!$OMP parallel do`. After:

```fortran
  !$omp target enter data map(alloc: a_w,a_e,a_s,a_n,b1,d1,c1)
  !$omp target teams loop private(sum_area,Idenom,a_w,a_e,a_s,a_n,b_denom_1,b1,d1,c1) &
  !$omp   map(to: ea, eb, h) map(from: u_h, v_h)
  do j=js,je
    do concurrent (i=is:ie)
      ...
    enddo
    if (mix_vertically) then
      do concurrent (i=is:ie)
        ...
      enddo
      do k=2,nz ; do concurrent (i=is:ie)      ! forward elimination — k stays a serial do-loop
        c1(i,k) = eb(i,j,k-1) * b1(i)
        ...
      enddo ; enddo
      do k=nz-1,1,-1 ; do concurrent (i=is:ie)  ! back substitution — k stays a serial do-loop
        u_h(i,j,k) = u_h(i,j,k) + c1(i,k+1)*u_h(i,j,k+1)
        ...
      enddo ; enddo
    ...
  enddo
  !$omp target exit data map(release: a_w,a_e,a_s,a_n,b1,d1,c1)
```

This is exactly the target pattern for the rest of the diabatic stack: `!$omp target teams loop`
over `j` (or `(i,j)` collapsed), `do concurrent (i=...)` for the parallel dimension, and explicit
serial `do k=...` for the tridiagonal recurrence — nothing else in `MOM_diabatic_aux.F90` or the
rest of the diabatic stack has received this treatment yet.

---

## 3. In-flight: `remotes/edoyango/port-set_diffusivity` (and siblings `port/set_diffusivity`,
`set_diffusivity-kjiarrs`)

`git log --oneline dev/gpu..remotes/edoyango/port-set_diffusivity`:

```
fc8ca6c7d more kji in find_tke_to_kd
ba65ee4bf kji arrays set_density_ratios
57b119b1f ijk arrays in find_n2
efe136545 set_diffusivity: move remaining OMP j-loop blocks out into own j-loops
fdac8e44d set_diffusivity: block j loops in add_drag_diffusivity
5d5a1d3e7 set_diffusivity: move ML_radiation, tidal_mixing, int_tides out of OMP loop
e7d7fb983 set_diffusivity: consolidate thickness_to_dz into single dz array
7978a3236 set_diffusivity: block j loops in find_TKE_to_Kd and set_density_ratios
3670d6c68 set_diffusivity: block j loops in calculate_bkgnd_mixing and find_TKE_to_Kd
02974f28d set_diffusivity: block j loops in find_N2 and find_rho_bottom
4a4290548 add j dimension to find_N2 related arrs
```

`git diff --numstat dev/gpu...remotes/edoyango/port-set_diffusivity`:

```
204   2   src/core/MOM_interface_heights.F90
130 110   src/parameterizations/vertical/MOM_bkgnd_mixing.F90
693 584   src/parameterizations/vertical/MOM_set_diffusivity.F90
```

The superset branch `remotes/edoyango/port/set_diffusivity` (10 commits ahead of this one, adds
`add_LOTW_BBL_diffusivity` tile-index support and explicit `! TODO` markers for porting/blocking)
touches an even wider footprint — it also pulls in `MOM_isopycnal_slopes.F90` (+400/−359),
`MOM_lateral_mixing_coeffs.F90`, `MOM_MEKE.F90`, `MOM_stoch_eos.F90` — showing that
`set_diffusivity` cannot be ported in isolation; its buoyancy/N² inputs are shared with the
lateral-mixing-coefficient and MEKE machinery.

### 3.1 What transformation is actually applied

**This is not the blessed k-blocking template.** There is **zero** `omp target`, `do concurrent`,
or `DO_LOCALITY` anywhere in this branch's diff (`grep -c` on the diff returns 0). It is a
*preparatory* refactor only, in two parts:

1. **j-blocking (promote scalar-`j` column routines to `nj`-wide row-blocks).** Every helper
   (`find_N2`, `find_TKE_to_Kd`, `add_drag_diffusivity`, `set_density_ratios`, ...) is rewritten
   from `real, dimension(SZI_(G),SZK_(GV))` (a single j-row, called once per `j` inside an
   `!$OMP parallel do` over `j`) to `real, dimension(SZI_(G),SZK_(GV),nj)` (a block of `nj` rows,
   dummy args promoted to `jstart`/`jend`/`nj`), and the call site is hoisted out of the per-`j`
   OMP loop so it processes a whole row-block at once. Excerpt (`fdac8e44d`, `add_drag_diffusivity`):
   ```fortran
   -subroutine add_drag_diffusivity(h, u, v, tv, fluxes, visc, j, TKE_to_Kd, maxTKE, &
   +subroutine add_drag_diffusivity(h, u, v, tv, fluxes, visc, jstart, jend, nj, TKE_to_Kd, maxTKE, &
                                    kb, rho_bot, G, GV, US, CS, Kd_lay, Kd_int, Kd_BBL)
   -  integer,                          intent(in)    :: j    !< j-index of row to work on
   -  real, dimension(SZI_(G),SZK_(GV)), intent(in)   :: TKE_to_Kd
   +  integer,                          intent(in)    :: jstart, jend, nj
   +  real, dimension(SZI_(G),SZK_(GV),nj), intent(in) :: TKE_to_Kd
   ```
   This is the row-block analogue of the `niblock/njblock/nkblock` idea from §5 of the
   architecture doc, but applied by hand at the *array-dimension* level rather than through the
   CS-parameter block-size machinery used in `MOM_continuity_PPM.F90`.

2. **Array dimension reordering, `(i,k,j)` → `(i,j,k)`.** Once row-blocked, local work arrays like
   `dRho_int`, `rho_0`, `dsp1_ds`, `maxEnt` are declared `(SZI_(G),nj,SZK_(GV))` instead of
   `(SZI_(G),SZK_(GV),nj)`, i.e. `k` is pushed to the *last* (slowest-varying) dimension and `j`
   becomes the middle dimension. Excerpt (`fc8ca6c7d`, `find_TKE_to_Kd`):
   ```fortran
   -  real, dimension(SZI_(G),SZK_(GV),nj) :: &
   +  real, dimension(SZI_(G),nj,SZK_(GV)) :: &
        ds_dsp1, dsp1_ds, maxEnt, rho_0, ...
   ...
   -  do j=jstart,jend ; jj = j - jstart + 1 ; do k=2,nz-1 ; do i=is,ie
   -    dsp1_ds(i,k,jj) = 1.0 / ds_dsp1(i,k,jj)
   +  do k=2,nz-1 ; do j=jstart,jend ; jj = j - jstart + 1 ; do i=is,ie
   +    dsp1_ds(i,jj,k) = 1.0 / ds_dsp1(i,jj,k)
      enddo ; enddo ; enddo
   ```
   and loop nests are correspondingly re-ordered so `k` is the **outer** loop and `(j,i)` the
   inner ones — i.e. the branch is deliberately exposing `(i,j)` as the parallel dimension pair
   and isolating the serial recurrence (`maxEnt`/`kb`-indexed) to the outer `k` loop, in
   preparation for eventually replacing the inner `(j,i)` nest with `do concurrent` and keeping
   `k` a private serial loop — the same shape as the `find_uv_at_h` precedent in §2.4, just not
   yet wired up with any device directives.

**Verdict:** genuine, disciplined groundwork toward the vertvisc-style pattern, but pre-device —
no `omp target`/`do concurrent`/`DO_LOCALITY` has landed yet on any of the three sibling branches.

---

## 4. In-flight: `remotes/edoyango/port/thickness_diffuse`

`git diff --numstat dev/gpu...remotes/edoyango/port/thickness_diffuse`:

```
  1    1   pkg/CVMix-src
  1    1   src/ALE/MOM_ALE.F90
  3    3   src/ALE/PLM_functions.F90
 69   31   src/core/MOM_PressureForce_FV.F90
 30   23   src/core/MOM_PressureForce_Montgomery.F90
 62  456   src/core/MOM_density_integrals.F90
589    0   src/core/MOM_density_integrals_s.F90      (new submodule file)
400  359   src/core/MOM_isopycnal_slopes.F90
  4    0   src/core/MOM_stoch_eos.F90
241    0   src/equation_of_state/MOM_EOS.F90
158   16   src/equation_of_state/MOM_EOS_Roquet_rho.F90
 77    0   src/equation_of_state/MOM_EOS_Wright.F90
 92    0   src/equation_of_state/MOM_EOS_base_type.F90
 12    1   src/parameterizations/lateral/MOM_MEKE.F90
 47    3   src/parameterizations/lateral/MOM_lateral_mixing_coeffs.F90
338  347   src/parameterizations/lateral/MOM_thickness_diffuse.F90
  4    0   src/parameterizations/vertical/MOM_internal_tide_input.F90
  4    0   src/parameterizations/vertical/MOM_set_diffusivity.F90
```

This is **much further along** than `set_diffusivity` — it is the one branch in this whole survey
that has actual device directives in the vertical/lateral-mixing space (via `git log --oneline`:
`add DO_LOCALITY macro`, `port remaining loops in thickness_diffuse`, `port thickness diffuse
full`, `submodulify`/`desubmodulify` — module split into `MOM_thickness_diffuse_s.F90` submodule).
Excerpt from `add DO_LOCALITY macro` (`6d8b47efc`), showing real `do concurrent` + `DO_LOCALITY`
already in place, being tidied to use the macro instead of hand-written `local()`/`local_init()`
clauses:

```fortran
-      do concurrent (j=jstart:jend, i=is-1:ie) &
-          local_init(drdiA, drdiB, drdkL, drdkR) &
-          local(drdz, hg2L, hg2R, haL, haR, dzaL, dzaR, wtL, wtR, ...)
+      do concurrent (j=jstart:jend, i=is-1:ie) DO_LOCALITY(local_init(drdiA, drdiB, drdkL, drdkR))
```

**EOS/mixing-coefficient dependencies pulled in:** `thickness_diffuse_full` (isopycnal/GM
height-diffusion) needs horizontal density gradients and the Fukumori-type internal wave speed
`cg1`, so this branch is forced to also port:
- `MOM_density_integrals.F90`/new `MOM_density_integrals_s.F90` — the `int_density_dz_generic_plm`
  PLM pressure/density integral used for slope estimates (heavy k-blocking work visible in the log:
  `int_density_dz_generic_plm: tile ... in intx_dpa/inty_dpa update`, "use 2d calc dens", "move k
  loops inside").
- `MOM_isopycnal_slopes.F90` (`calc_isoneutral_slopes`) — feeds `Slope`/`N2` to the streamfunction
  limiter.
- `MOM_EOS.F90` / `MOM_EOS_base_type.F90` / `MOM_EOS_Wright.F90` / `MOM_EOS_Roquet_rho.F90` — the
  same EOS `_loc`-function device pattern documented in `06-eos-layer.md`, extended/duplicated
  here rather than reused, since the merged EOS port only covers Wright/Roquet.
- `MOM_MEKE.F90` / `MOM_lateral_mixing_coeffs.F90` — `cg1`/`Rd`/eddy-length inputs are shared
  state consumed by `thickness_diffuse`.
- `MOM_PressureForce_FV.F90` / `MOM_PressureForce_Montgomery.F90` — touched for the same PLM
  density-integral change, since both pressure-force and thickness-diffusion reuse
  `int_density_dz_generic_plm`.

So thickness_diffuse is not an isolated kernel — porting it drags in essentially the whole
density/pressure-integral subsystem. This branch shows the *cost* of that in practice: 17 files
touched to port one lateral-mixing routine.

---

## 5. EPBL (`MOM_energetic_PBL.F90`): untouched on mainline, but a striking naive-port
experiment exists off-tree (`remotes/origin/epbl-3d` and siblings)

This is a bigger and more informative in-flight effort than the branches enumerated in the
original task brief, and directly relevant to the "heavy column kernels" question in §2.2, so it
is documented here in full.

`git log --oneline dev/gpu..remotes/origin/epbl-3d`:
```
c954189f9 ePBL: do concurrent
c627c0d09 ePBL: Overly aggressive inlining
05c74b56b ePBL: Replace auto array size (nk=75)
99fd0e4f3 ePBL: 3D test: remove 2d/1d copies and directives
8dd5fe2d8 Test 3d version of epbl_column
5166b07aa ePBL: Submodule for MOM_wave_interface
bf394e40d ePBL debug: minor cleanups
a433f1eb4 ePBL: Remove redundant module loads in submodule
```
`git diff --numstat dev/gpu...remotes/origin/epbl-3d`:
```
   4     3   src/core/MOM_interface_heights.F90
 415  3674   src/parameterizations/vertical/MOM_energetic_PBL.F90
5031     0   src/parameterizations/vertical/MOM_energetic_PBL_smod.F90   (new submodule)
  19     0   src/parameterizations/vertical/smod.mk                      (submodule makefile)
  46   432   src/user/MOM_wave_interface.F90
 475     0   src/user/MOM_wave_interface_smod.F90                        (new submodule)
```
Note the branch also submodulifies `MOM_wave_interface` (commit `5166b07aa`) — because
`get_Langmuir_Number` lives there and must be reachable/inlinable from the device kernel — and touches
`MOM_interface_heights.F90`; the port is not confined to the two EPBL files. (`git log --oneline
dev/gpu..remotes/origin/epbl-3d` is 13 commits; the 8 shown above are the most recent, ending the
lineage at `c954189f9`.) Sibling branches `epbl-debug`, `epbl-debug-3d`, `epbl-debug-submod`,
`epbl-debug-submod-github` are earlier checkpoints of the same lineage, converging on `epbl-3d`.

**The approach taken here is deliberately the opposite of the k-blocking template**: instead of
tiling and hoisting communication, the entire per-column TKE-budget subroutine
(`ePBL_column`/`ePBL_column_3d`, thousands of lines) is marked `pure`, and the outer `(i,j)` loop
is wrapped directly in `do concurrent`, calling the whole column kernel as one opaque body. From
the final commit (`c954189f9`, message: *"The magic of do concurrent has sped this up 100x to
~2ms/step"*):

```fortran
-  !$omp target loop private(SpV_dt)
-  do j=js,je
+  do concurrent (j=js:je, i=is:ie)
+    if (G%mask2dT(i,j) > 0.) then
     ...
```

```fortran
-subroutine ePBL_column_3d(h, dz, u, v, ..., G, i, j, TKE_gen_stoch, TKE_diss_stoch, tmpval)
+pure subroutine ePBL_column_3d(h, dz, u, v, ..., G, i, j, TKE_gen_stoch, TKE_diss_stoch, tmpval)
```

Two nvfortran-workaround commits accompany this ("naive whole-kernel `do concurrent`" is not free):

- **`05c74b56b` "Replace auto array size (nk=75)"** — local work arrays inside the column kernel
  were declared `real, dimension(SZK_(GV)+1) :: ...` (a size derived from a derived-type member,
  i.e. an automatic array with a runtime-known but non-dummy-argument extent). This was replaced
  with a **hardcoded literal `75`** to get past a device-compilation limitation on automatic arrays
  sized from non-argument expressions inside a `do concurrent`/`pure` procedure — a real
  nvfortran-workaround worth cataloguing (see `13-compiler-workarounds.md`), and a portability
  landmine (breaks silently if `GV%ke /= 75`).
- **`c627c0d09` "Overly aggressive inlining"** (commit message: *"This is too hideous to describe
  in words. But I am trying to inline as much as possible."*) — manually inlines calls like
  `get_Langmuir_Number`/`find_mstar` because cross-module calls from inside the `pure`/device
  region don't resolve cleanly, echoing architecture-doc principle 4 ("cross-module calls inside
  device loops are painful... must be inlined").

**Assessment:** this is real evidence that (a) EPBL *can* be parallelized over `(i,j)` as a single
opaque `pure` column kernel without restructuring its internal `k` recurrence at all (the "naive
port" contrast case referenced in the architecture doc for `bodner-naive-port`), and (b) doing so
still requires nontrivial workarounds (automatic-array sizing, manual inlining) and is not yet a
CPU-preserving k-blocked port — it is a single-target (GPU-only, `do concurrent` whole-kernel)
experiment, not obviously mergeable as-is (mainline principle 1: "preserve CPU performance").
It has **not** touched `dev/gpu` and `MOM_energetic_PBL.F90` remains at 0/0 there.

---

## 6. `MOM_set_viscosity.F90` / `MOM_vert_friction.F90` tiling branches (adjacent, in-flight)

Three further edoyango branches continue tiling work on the *already-merged* viscosity/friction
modules (not diabatic/ALE, but immediately adjacent and using the same idiom, useful precedent):

- `remotes/edoyango/port-set_viscous_BBL-tile` — tiles temporaries in `set_viscous_BBL`
  (`TILE_SIZE_X`×`TILE_SIZE_Y`, tile-local indices), `MOM_set_viscosity.F90` +238/−176.
- `remotes/edoyango/port-set_viscous_ML-tile` — tiles `set_viscous_ML` (block-size inputs, 2D
  promotion of 1D temporaries, global array transfers), +678/−476.
- `remotes/edoyango/tile-vertvisc-coef` — promotes `vertvisc_coef`/`vertvisc`/`vertvisc_remnant`
  locals to 3D arrays and adds `bind(parallel,teams)` to a `do concurrent`, `MOM_vert_friction.F90`
  +457/−351 (one commit literally named `claude tile vertvisc_coef` — LLM-assisted).

These confirm the "j/i-tiling of column-local temporaries" idiom is the actively-used
transformation across all the vertical-physics branches right now (as opposed to the
niblock/njblock/nkblock CS-parameter block machinery used in `MOM_continuity_PPM.F90`) — expect
this to be the template that eventually lands for `set_diffusivity`/KPP/EPBL too.

---

## 7. ALE remap/regrid: why per-column reconstruction is architecturally hard to offload

### 7.1 Call structure

`ALE_regridding_and_remapping` (`MOM.F90:1900`–`:2073`) is called from `step_MOM_thermo`
(`MOM.F90:916` and `:1037`, once per branch of an ALE/non-ALE conditional). Internally
(`MOM_ALE.F90`):

- `regridding_main(CS%remapCS, CS%regridCS, ...)` generates the new target grid — dispatches via
  a plain integer `select case (CS%regridding_scheme)` (`MOM_regridding.F90:1281`) over
  `build_grid_HyCOM1`/`build_grid_adaptive`/z-star/sigma/rho builders. This part is *not*
  polymorphic and would k-block reasonably (it's the same shape of problem as other dispatch code
  already ported elsewhere).
- The remap step is a **plain host loop calling a per-column subroutine**, not any kind of
  `do concurrent`:
  ```fortran
  do j = G%jsc-1,G%jec+1 ; do i = G%isc-1,G%iec+1
    call remapping_core_h(CS%remapCS, nz, h_orig(i,j,:), tv%S(i,j,:), nz, h(i,j,:), &
                          tv_local%S(i,j,:))
    call remapping_core_h(CS%remapCS, nz, h_orig(i,j,:), tv%T(i,j,:), nz, h(i,j,:), &
                          tv_local%T(i,j,:))
  enddo ; enddo
  ```
  (`MOM_ALE.F90:745-748`; the identical pattern recurs at `:836/838` for tracers, `:1192`/`:1267`
  for velocity remap, and inside `coord_rho.F90:277/279`, `coord_hycom.F90:240/241` for
  coordinate-generator-internal remaps.)

### 7.2 Why `remapping_core_h`/`Recon1d_*` resist the k-blocking template

1. **Ragged/variable column length.** `remapping_core_h(CS, n0, h0, u0, n1, h1, u1, ...)` takes
   `n0`/`n1` as *arguments*, not fixed to `GV%ke` — columns can have a different active layer count
   (thin/vanished layers), and the internal sub-cell intersection (`intersect_src_tgt_grids`)
   builds per-column index arrays (`isrc_start`, `isrc_end`, `isub_src`, ...) whose *sizes and
   control flow depend on the column's own data*. This is fundamentally a per-column
   variable-length merge/list-intersection algorithm — the opposite of the uniform, statically-
   shaped stencil work `do concurrent`/SIMD lanes want.

2. **Polymorphic dispatch (`class(Recon1d)`), the same v-table problem already catalogued for
   EOS.** `Recon1d` (`src/ALE/Recon1d_type.F90:16`) is `abstract` with **eight `deferred`
   type-bound procedures** (`init`, `reconstruct`, `average`, `f`, `dfdx`, `check_reconstruction`,
   `unit_tests`, `destroy`, plus `init_parent`/`reconstruct_parent`), and `remapping_CS` carries
   `class(Recon1d), pointer :: reconstruction` (`MOM_remapping.F90:83`). When
   `CS%remapping_scheme == REMAPPING_VIA_CLASS`, the core routine dispatches through this pointer:
   ```fortran
   if (CS%remapping_scheme == REMAPPING_VIA_CLASS) then
     call CS%reconstruction%reconstruct(h0, u0)
     call CS%reconstruction%remap_to_sub_grid(h0, u0, n1, h_sub, ...)
   else ! Uses the OM4-era integer-keyed select-case reconstruction functions instead
   ```
   (`MOM_remapping.F90:273-300`). This is architecturally identical to the pre-port
   `EOS_type`/`class(EOS_base)` situation documented in `06-eos-layer.md` §7.1: nvfortran cannot
   resolve v-table dispatch on device. **The escape hatch is the OM4-era `else` path**: it is
   already non-polymorphic (integer-keyed `select case`, no `class(Recon1d)` deref), so it needs no
   `_loc`-style rewrite at all — a device port can simply set `remapping_scheme /= REMAPPING_VIA_CLASS`
   and target the `remap_src_to_sub_grid_om4`/`remap_sub_to_tgt_grid_om4`/`build_reconstructions_1d`
   functions directly. **This is exactly what `remotes/origin/jorge/diagnostics_port` does** (see §7.3):
   it marks that whole OM4 call chain `!$omp declare target` and fixes the ragged sizing with a
   `NK_GPU_MAX` parameter — so the earlier draft's "no branch is attempting to port `Recon1d_*`" is
   **incorrect**; a preparatory device-enablement attempt exists, it just avoids `Recon1d` rather
   than de-polymorphizing it.
   Note: even in the *non*-polymorphic (`else`) branch, dispatch is still per-scheme via
   `select case` over ~9 reconstruction kinds (PCM/PLM/PLM_hybgen/PPM_CW/PPM_H4/PPM_IH4/
   PPM_hybgen/PQM/...) inside a routine called once per column — so columns using different
   schemes (rare, but the scheme is a runtime CS setting, uniform per-run in practice) would
   otherwise diverge; in practice this is not the blocking issue, the deferred-procedure dispatch
   and ragged sizing are.

3. **Deep, non-inlined call chains inside the per-column kernel.** `remapping_core_h` alone calls
   `intersect_src_tgt_grids`, `build_reconstructions_1d`, `remap_src_to_sub_grid[_om4]` (or the
   polymorphic `CS%reconstruction%remap_to_sub_grid`), and `remap_sub_to_tgt_grid[_om4]` — roughly a
   half-dozen cross-procedure calls per column (`adjust_h_sub` is present in source but currently
   commented out at `:280`/`:809`), each of which would need `!$omp declare target` + guaranteed
   inlining (architecture-doc principle 4) to avoid an interpreted/indirect call inside a device
   loop. This is precisely the set `jorge/diagnostics_port` blanket-tags `!$omp declare target`.

4. **Optional arguments and automatic-size locals.** `remapping_core_h` has optional dummy args
   (`net_err`, `PCM_cell`) and internal automatic arrays sized by the dummy `n0`/`CS%degree+1`
   (`ppoly_r_coefs(n0,CS%degree+1)`) — both patterns are awkward inside `do concurrent`/`pure`
   procedures on current nvfortran (the EPBL "hardcode nk=75" workaround in §5 is exactly this
   class of problem, just for a fixed-size case; ALE's is worse because the size is *not* even
   fixed at `GV%ke`, it's a runtime sub-column count).

### 7.3 Branches touching ALE files — one preparatory remap attempt, the rest incidental

Full survey of every branch/remote with a nonzero diff against `MOM_remapping.F90` /
`MOM_regridding.F90` / `MOM_ALE.F90` / `Recon1d_type.F90`:

| Branch | Files touched | Nature |
|---|---|---|
| most branches (`bodner-naive-port`, `port/pressureforce-benchmark_ALE`, `find-eta-gpu`, `find-eta-merge`, `find-eta-with-CS`, `feat/new-diag-manager`, `port/thickness_diffuse`) | `MOM_ALE.F90` only, 1–3 lines | The `find_eta` device-buffer-sync pattern from §1, or a one-line diag-manager hook — **not** a remap port |
| `remotes/origin/cmake/amd-flang` | `MOM_remapping.F90` 36/34 | AMD/flang build-portability fixes, not NVIDIA/nvfortran GPU work |
| **`remotes/origin/jorge/diagnostics_port`** | **`MOM_remapping.F90` 82/62** | **The one real remap-offload attempt.** Adds `!$omp declare target` to the entire OM4 per-column call chain (`remapping_core_h`, `build_reconstructions_1d`, `intersect_src_tgt_grids`, `remap_src_to_sub_grid[_om4]`, `remap_sub_to_tgt_grid_om4`, `interpolate_column`, `average_value_ppoly` — all tagged `! GPU PORT DIAGNOSTICS`) and introduces `integer, parameter, public :: NK_GPU_MAX = 500` (`:47`), rewriting the ragged automatic locals `frac_pos`/`k_src` to `NK_GPU_MAX+1` fixed size (`:1275-1277`) — the ALE analogue of the EPBL `nk=75` workaround (§5). It is a **single messy WIP commit** (`c5810df0b` "diagnostics kinda clean"); it does **not** yet wrap the `MOM_ALE.F90:745` host loop in a device region (no `omp target`/`do concurrent` added there), so the routines are device-*callable* but not yet device-*driven*. Still: this is genuine preparatory device-enablement, and it validates the "target the OM4 path, sidestep `Recon1d`" strategy (§7.2). |
| `remotes/JorgeG94/jorge/use_dp_as_real` | `MOM_ALE.F90` 243/241, `MOM_regridding.F90` 386/384, `MOM_remapping.F90` 521/519, `Recon1d_type.F90` 52/50 | Precision refactor: replace `-r8` compiler flag with explicit `real(real64)` kinds everywhere (`801689d87`, "use real64 everywhere instead of -r8") — a plausible *prerequisite* for device work (explicit kinds matter for device code per architecture-doc conventions) but **not itself an offload** |
| `remotes/edoyango/submod-conversion` | `MOM_ALE.F90` 39/1244, `MOM_regridding.F90` 45/2372, `MOM_remapping.F90` 30/2445, `Recon1d_type.F90` 6/187 | Mechanical, codebase-wide module→module+submodule split via a script (`convert_to_submodules.py`, ~290 new `_s.F90` files) for **incremental compile-speed**, not GPU-related; same submodule idiom later reused by hand in the `thickness_diffuse`/`epbl-3d` branches |
| ~all branches | `Recon1d_type.F90` 4/2 | Ubiquitous license-header swap (SPDX identifier), **not** code — ignore |

**Conclusion for Q5 (corrected): exactly one branch — `remotes/origin/jorge/diagnostics_port` —
is attempting to make the `remapping_core_h`/OM4 reconstruction chain device-executable**
(`declare target` + `NK_GPU_MAX` fixed sizing), but it stops at making the routines *callable* on
device; **no branch yet drives them from a device loop**, and none touches the polymorphic
`Recon1d_*` path. So remapping remains the least mature corner of the effort, but it is **not** true
that "zero work, preparatory or otherwise" exists — the de-polymorphization question is effectively
answered (use the OM4 path) and the ragged-sizing question has a candidate answer (`NK_GPU_MAX`);
what's missing is the driving device loop + bitwise validation.

> **FABLE-CHECK (reviewed 2026-07-14 — resolution or current status in KNOWLEDGE.md §8a/§8b):** Is the `diagnostics_port` strategy (OM4 select-case path + blanket `declare target`
> + `NK_GPU_MAX=500` fixed sizing) the right long-term direction, or a dead end? Two concerns a
> porting agent should resolve before building on it: (1) `NK_GPU_MAX=500` over-allocates every
> column to 500 layers of private stack per thread — check whether that blows the device stack /
> register budget for realistic `GV%ke` (~75), vs. sizing at `GV%ke`; (2) the OM4 path still
> `select case`s over ~9 reconstruction kinds per column — confirm nvfortran handles that branch
> divergence acceptably inside a `target teams loop`. Look at `MOM_remapping.F90:47,273-300,1275`
> on `remotes/origin/jorge/diagnostics_port` and compare against the merged EOS `_loc` approach.

---

## 8. Prioritized remaining-work assessment

Ranked by how directly the subsystem sits on the per-timestep critical path (called every
dynamic/thermo step, unconditionally, vs. only under specific runtime configs) and by current
in-flight momentum:

### Tier 1 — on the critical path every thermodynamic step, must eventually port, work started

Recommended dependency ordering: **EOS `_loc` coverage (see below) → `find_N2`/density inputs →
`set_diffusivity` → KPP/EPBL → `kappa_shear`**. The tridiagonal solves (#1) are independent and can
go first as a warm-up.

1. **`MOM_diabatic_aux.F90` tridiagonal solves** (`tracer_vertdiff`/`triDiagTS`, `triDiagTS_Eulerian`).
   - *Approach:* copy the in-file `find_uv_at_h` template verbatim (§2.4): `!$omp target teams loop`
     over `j`, `do concurrent (i=is:ie)` for the parallel dimension, explicit serial `do k` for the
     forward-elimination / back-substitution recurrence, with the tridiagonal work arrays
     (`b1`,`c1`,`d1`) `map(alloc:)`/`map(release:)` around the region. Equivalently the merged
     `MOM_vert_friction.F90` teams-loop-collapse(2) form (`:737`,`:938`,`:1223`,`:1255`, 3
     `!$omp declare target` column kernels at `:437`,`:2101`,`:2611`).
   - *Hazards:* these routines take `tv%T`/`tv%S` **pointer** members of `thermo_var_ptrs` (arch-doc
     §2.3) — map the target arrays, not the container; guard any diagnostic post behind `id_* > 0`
     device-update (arch-doc §7.4). No restart registration on these temporaries.
   - *Why first:* lowest remaining effort — the precedent already lives in the same file.
2. **`MOM_set_diffusivity.F90`** — called unconditionally from all three diabatic drivers (§2.1),
   several call sites per step.
   - *Approach:* finish the existing `port-set_diffusivity` lineage — its j-blocking +
     `(i,j,k)`-reorder groundwork already exposes `(i,j)` as the parallel pair with `k` isolated as
     the serial recurrence (`maxEnt`/`kb`), i.e. the `find_uv_at_h`/vertvisc shape minus the device
     directives. Add `DO_LOCALITY`-annotated `do concurrent (j,i)` + serial-`k`, or teams-loop.
   - *Dependency (must land first):* its buoyancy/N² inputs (`find_N2`, `calc_isoneutral_slopes`,
     density integrals) are shared with `lateral_mixing_coeffs`/MEKE and call EOS — so the EOS `_loc`
     chain and `MOM_isopycnal_slopes.F90` need device coverage first (this is exactly why the
     `port/set_diffusivity` superset drags those in; §3). Prefer merging the EOS/density-integral
     work from `port/thickness_diffuse` (§4) before wiring device directives here.
   - *Hazards:* `set_diffusivity` writes into `visc%Kd_*` which include restart-target **pointer**
     fields of `vertvisc_type` (arch-doc §2.3, §7.4) — keep those device-resident and only
     `target update from` on diag/restart boundaries.
3. **`MOM_energetic_PBL.F90`** — the default boundary-layer scheme in most configs, every thermo step.
   - *Approach:* the `epbl-3d` branch proves `(i,j)`-parallel `do concurrent` over a `pure`
     whole-column kernel works, but as a **GPU-only, non-CPU-preserving** variant (§5). Two viable
     paths: (a) accept the naive `pure`-column `do concurrent` if the "preserve CPU performance"
     principle can be waived here (pending the `bodner-naive-port` precedent discussion), or (b) redo
     in blessed teams-loop style. Either way reuse `epbl-3d`'s two hard-won workarounds: manual
     inlining of `get_Langmuir_Number`/`find_mstar` (cross-module calls, arch-doc principle 4) and a
     fixed-size replacement for the `SZK_(GV)+1` automatic column arrays.
   - *Hazard:* the hardcoded `nk=75` in `epbl-3d` is a portability landmine — parameterize to `GV%ke`
     or a validated `NK_MAX` before reuse. Depends on `MOM_wave_interface` being device-reachable
     (submodulified on `epbl-3d`).
4. **`MOM_CVMix_KPP.F90`** — alternative/complementary boundary-layer scheme, same call frequency as
   EPBL when enabled. **Zero in-flight work found** — the least-started "always runs" kernel.
   - *Approach:* same two-level column pattern (OBL-depth search + shape function is a top-down
     column recurrence, §2.2). Hazard: KPP calls into the external CVMix package (`pkg/CVMix-src`);
     those cross-library calls must be `declare target` + inlinable or duplicated, a bigger inlining
     surface than the other Tier-1 items.
5. **`MOM_kappa_shear.F90`** — feeds `Kd_shear`/`Kv_shear` consumed by `set_diffusivity`/EPBL; only
   change on `dev/gpu` is an unrelated CPU OpenMP data-race fix (`shared()` clause). No in-flight GPU
   work. *Approach:* per-column iterative closure (`Calc_kappa_shear_vertex`) → same teams-loop shape;
   port after `set_diffusivity` since its output is that routine's input. Note it is invoked from the
   **dynamics** side (`set_viscous_ML`), so its device data must be live across the dycore→diabatic
   boundary.

### Tier 2 — on the critical path but conditional on runtime config, or ALE-specific
6. **ALE remap (`MOM_remapping.F90`/`Recon1d_*`)** — runs every timestep when ALE mode is active
   (the default coordinate mode in most modern MOM6 configs), so functionally Tier-1 in practice,
   but ranked here because it is **architecturally the hardest problem in this whole survey**
   (ragged per-column sizing + polymorphic `Recon1d` dispatch + deep call chains, §7). **Not zero
   in-flight work** (corrected): `remotes/origin/jorge/diagnostics_port` has already tagged the OM4
   per-column call chain `!$omp declare target` and replaced the ragged locals with an `NK_GPU_MAX`
   fixed size (§7.3) — so the two hard design questions have candidate answers (de-polymorphize by
   *using the OM4 select-case path* rather than rewriting `Recon1d`; fix ragged sizing with a max-`nk`
   pad). *Recommended approach:* validate/adopt that branch's OM4-path direction (resolve the
   FABLE-CHECK in §7.3 on `NK_GPU_MAX` stack cost first), then add the missing driving device loop at
   `MOM_ALE.F90:745` (`do concurrent (j,i)` over columns), then bitwise-validate against CPU with
   `MOM_checksums`. This is still the highest-risk item and should be treated as a research problem,
   but it is no longer a blank slate.
7. **`MOM_regridding.F90`** (grid generation) — simpler than remapping (integer `select case`
   dispatch, no polymorphism found), likely portable with the standard template once someone
   starts; currently zero in-flight work.
8. **`MOM_thickness_diffuse.F90`** (lateral, GM/isopycnal height diffusion) — technically lateral
   not vertical, but bundled here because it's explicitly grouped with this porting frontier in
   the architecture doc. Furthest along of any item in this document (real `do concurrent` +
   `DO_LOCALITY` in-tree on `port/thickness_diffuse`), but drags in the density-integral/EOS
   subsystem (§4) — expect a wide-footprint PR when it lands.
9. **`MOM_mixed_layer_restrat.F90`** (Bodner MLE restratification) — zero on mainline; one
   contrasting **naive** in-flight port (`bodner-naive-port`) exists per the architecture doc,
   useful as a second data point alongside `epbl-3d` for the "naive vs. k-blocked" trade-off
   discussion, but not surveyed in depth here (out of this document's primary file list).

### Tier 3 — peripheral / infrequent / not on the hot loop
10. **`MOM_diag_mediator.F90`** and diagnostic-only device-buffer-sync shims (the `find_eta`
    pattern in §1) — real but low-value work; already tracked in `12-diagnostics-io.md`.
11. Regridding coordinate generators (`coord_rho.F90`, `coord_hycom.F90`, etc.) — called once per
    remap step, but simple compared to the reconstruction machinery itself; will likely port
    "for free" once `MOM_regridding.F90`/`MOM_remapping.F90` are handled.

**Bottom line:** the diabatic column-physics kernels (Tier 1) are individually tractable — each
one needs the same `!$omp target teams loop` + `do concurrent(i)` + serial-`do k` shape already
proven twice (`MOM_vert_friction.F90` merged, `MOM_diabatic_aux.F90::find_uv_at_h` merged) and
partially rehearsed a third time (`epbl-3d`, naive variant). The real open problem is **ALE
remapping**: its two design questions (polymorphic per-column dispatch + ragged column sizes) are the
hardest in this survey, but — correcting the earlier draft — they are **not untouched**:
`jorge/diagnostics_port` has a preparatory device-enablement pass (OM4 path + `declare target` +
`NK_GPU_MAX`) that answers both in candidate form (§7.3). What remains is the driving device loop over
columns and bitwise validation, not a from-scratch redesign.

---

## 9. Branches referenced in this document (for follow-up)

| Branch | Role |
|---|---|
| `remotes/edoyango/port-set_diffusivity` | set_diffusivity j-blocking + array reordering (pre-device) |
| `remotes/edoyango/port/set_diffusivity` | superset of above, +BBL tile indices, TODO markers |
| `remotes/edoyango/set_diffusivity-kjiarrs` | earlier checkpoint of the same lineage |
| `remotes/edoyango/port/thickness_diffuse` | furthest-along lateral/vertical-mixing port, real `do concurrent`+`DO_LOCALITY` |
| `remotes/origin/epbl-3d` (+ `epbl-debug*`) | naive whole-column `do concurrent` port of EPBL, "100x" claim, workaround catalogue |
| `remotes/edoyango/port-set_viscous_BBL-tile`, `port-set_viscous_ML-tile`, `tile-vertvisc-coef` | adjacent tiling work on already-merged viscosity/friction modules, same idiom |
| `remotes/origin/find-eta-gpu`, `find-eta-merge`, `find-eta-with-CS` | ported `find_eta`/`MOM_interface_heights.F90`, consumed by the 3-line `MOM_ALE.F90`/`MOM_diabatic_driver.F90`/`MOM_thickness_diffuse.F90` diffs on mainline |
| `remotes/origin/jorge/diagnostics_port` | **the one preparatory remap-offload attempt**: `declare target` on the OM4 `remapping_core_h` chain + `NK_GPU_MAX` fixed sizing (§7.3); not yet device-driven |
| `remotes/JorgeG94/jorge/use_dp_as_real` | `-r8` → explicit `real64` kinds across ALE, plausible device-code prerequisite |
| `remotes/edoyango/submod-conversion` | mechanical, whole-codebase module→submodule split (compile-speed, not GPU) |
| `remotes/origin/bodner-naive-port` | naive-port contrast case for `MOM_mixed_layer_restrat.F90` + density integrals (architecture doc §6.2) |

No file outside `docs/gpu-knowledge/14-vertical-physics-ale-status.md` was created; no build or
run was performed for this survey.

---

## Verification notes

Independently verified against source + git (branch `dev/gpu`, baseline `dev-gfdl`); no code built or run.

**Confirmed exactly:**
- §1 numstat table — every row reproduced via per-file `git diff --numstat dev-gfdl...dev/gpu`
  (0/0 for the six untouched modules; 3/0 for `MOM_diabatic_driver`/`MOM_ALE`/`MOM_thickness_diffuse`;
  2/1 `kappa_shear`; 13/12 `MOM_diabatic_aux`; 775/184 vertvisc; 501/384 set_viscosity).
- The three "3-line" diffs are byte-for-byte the `find_eta` device-buffer-sync pattern quoted;
  `kappa_shear` is a CPU `!$OMP ... shared()` data-race fix (not a port).
- `find_uv_at_h` port pattern (§2.4): `!$omp target teams loop` over `j` + `do concurrent (i)` +
  serial `do k` recurrence, `map(alloc/release)` on tridiagonal temporaries — exact.
- Diabatic dispatch structure and all kernel call-site line numbers (§2.1 table) — exact.
- `port-set_diffusivity`: numstat (204/2, 130/110, 693/584), commit log, and **zero** added
  `omp target`/`do concurrent`/`DO_LOCALITY` lines (preparatory-only) — confirmed.
- `port/thickness_diffuse`: 18-file footprint (incl. `pkg/CVMix-src` submodule bump → "17 source
  files") and real device directives (135 added `do concurrent`/`DO_LOCALITY` lines) — confirmed.
- `epbl-3d`: `pure ePBL_column[_3d]`, `do concurrent (j,i)`, and all three commit-message claims
  ("100x…~2ms/step", "Replace auto array size (nk=75)" → literal `75`, "Overly aggressive
  inlining") — confirmed verbatim.
- ALE: `MOM_ALE.F90:745-748` host loop; `Recon1d` abstract type + 10 deferred procedures
  (`Recon1d_type.F90:16`); `class(Recon1d), pointer` (`:83`) + `REMAPPING_VIA_CLASS` dispatch
  (`:273-300`); `MOM_regridding.F90:1281` integer `select case`, no polymorphism — all confirmed.
- Tier prioritization is consistent with the arch-doc §4 call tree (diabatic column kernels run
  every thermo step; ALE remap every step in ALE mode; thickness_diffuse/MLE are config-conditional).

**Corrected:**
- §5 `epbl-3d` numstat block **understated the footprint** — the branch touches 6 files
  (`MOM_interface_heights.F90`, `smod.mk`, `MOM_wave_interface.F90` + `_smod`), not just the two
  EPBL files. Fixed, with note that `MOM_wave_interface` is submodulified for device reachability.
- §7.3 / §8 **the central "no branch attempts remap offload / zero in-flight work, preparatory or
  otherwise" claim was wrong.** `remotes/origin/jorge/diagnostics_port` (`MOM_remapping.F90` 82/62,
  not 78/61) marks the entire OM4 per-column chain `!$omp declare target` and adds
  `NK_GPU_MAX = 500` fixed sizing (`:47`,`:1275-1277`) — a genuine preparatory device-enablement
  attempt that sidesteps `Recon1d` polymorphism via the OM4 select-case path. It stops short of a
  driving device loop. Rewrote the §7.3 row, the §7.2 point-2/3 claims, the §7.3 conclusion, the §8
  Tier-2 item 6, and the bottom line accordingly; added it to §9.
- §7.2 point 3 listed `adjust_h_sub` as a live per-column call — it is **commented out**
  (`MOM_remapping.F90:280`,`:809`,`:843`). Corrected; call list updated to the real OM4 chain.
- §2.1/§2.3: the kernels are dispatched from **three** routines (`diabatic_ALE_legacy` included),
  not two — clarified (set_diffusivity sites 694/697 live in the legacy path).
- Minor numstat drift on `cmake/amd-flang` `MOM_remapping.F90` (36/34, doc had 32/33) — updated.

**Enhancements:** Tier-1 targets now carry per-item recommended approach (merged precedent:
`find_uv_at_h`/vertvisc teams-loop for tridiagonals, `epbl-3d` for whole-column, EOS `_loc` +
`isopycnal_slopes` prerequisite chain for `set_diffusivity`), known hazards (pointer `tv%T`/`tv%S`
and restart-target `visc%Kd_*` members, diag `id_*>0` guards), CVMix external-library inlining risk,
and an explicit dependency ordering (EOS `_loc` → N²/density → set_diffusivity → KPP/EPBL →
kappa_shear). ALE remap now points at the concrete `diagnostics_port` groundwork.

**FABLE-CHECK markers:** 1 (the `NK_GPU_MAX`/OM4-path strategy question in §7.3).

**Confidence:** High. Every numstat, line number, commit hash, and code excerpt was checked directly
against the tree; the one material correction (diagnostics_port remap groundwork) was verified by
reading the actual `declare target`/`NK_GPU_MAX` hunks on that remote branch.
