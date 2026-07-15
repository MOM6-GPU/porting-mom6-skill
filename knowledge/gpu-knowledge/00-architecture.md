# MOM6 Architecture for GPU Porting (dev/gpu)

> **Note (2026-07-14):** this anchor document predates the verification pass over docs 01–14.
> Known-stale spots have been corrected inline; where this file and a numbered doc disagree,
> **the numbered doc and `KNOWLEDGE.md` win.**

> **Purpose.** This is the anchor document for the MOM6 GPU-porting knowledge base. It describes the
> code architecture a porting agent must understand *before* touching anything: directory layout,
> the control-structure (CS) pattern, the time-stepping call tree, memory conventions, and a
> quantified inventory of GPU work already merged on `dev/gpu`. Every other document in
> `docs/gpu-knowledge/` drills into one subsystem; read this one first.
>
> **Repo state at time of writing.** Branch `dev/gpu`, upstream baseline `dev-gfdl`. Diffing
> `dev-gfdl...dev/gpu` = 320 commits, 52 files, +8310/−4615 lines = the totality of merged GPU work.
> The port targets NVIDIA GPUs with **NVHPC / nvfortran** using **OpenMP target offload** +
> **Fortran `do concurrent`**. Do not build or run the code; study source + git only.

---

## 0. Guiding principles of the port (context that shapes every decision)

1. **Preserve CPU performance.** The blessed strategy is **k-blocking / tiling**: loops are
   restructured into blocks (`niblock`/`njblock`/`nkblock`) so one source form runs well on both
   CPU (cache blocking, block sizes `32/4/1`) and GPU (whole-array, block sizes `0/0/0`). See
   merged commits `b8c471cfa` (Kblock coradcalc), `93dbbd36e` (k-block continuity), and local branch
   `kblock-hor-visc`.
2. **Bitwise reproducibility is mandatory.** Refactors must not reorder floating-point operations.
   Extracting code into `pure`/`elemental` subroutines is the preferred restructuring tool.
   Reproducing sums use exact fixed-point integer arithmetic (`MOM_coms`) so results are independent
   of thread/PE order.
3. **`do concurrent` is the default parallel idiom.** OpenMP `target teams` directives are used only
   for reductions or where `do concurrent` misbehaves/underperforms (documented per case).
4. **Cross-module calls inside device loops are painful.** They must be force-inlined or duplicated
   as `!$omp declare target` helpers. (CORRECTION: at HEAD the live directive is
   `!DIR$ ATTRIBUTES FORCEINLINE`; `!NVF$ INLINE` and `-Minline=name:` are historical —
   `93dbbd36e`; see doc 08.) **`class(*)`/runtime polymorphism is a disaster on device** — the EOS
   layer is being rewritten to avoid it.
5. **Document nvfortran bugs.** This is frontier work; genuine compiler bugs are hit and worked
   around with comments and directives that must be catalogued.

---

## 1. Directory and module layout

Source lives under `src/` (~244 F90 files) with the FMS coupling/infra under `config_src/`.

| Directory | Role | Key modules |
|---|---|---|
| `src/core/` | Dynamical core (momentum + continuity), prognostic state, grids | `MOM.F90`, `MOM_dynamics_split_RK2.F90`, `MOM_continuity_PPM.F90`, `MOM_CoriolisAdv.F90`, `MOM_PressureForce_FV.F90`, `MOM_barotropic.F90`, `MOM_variables.F90`, `MOM_grid.F90`, `MOM_verticalGrid.F90`, `MOM_open_boundary.F90` |
| `src/parameterizations/lateral/` | Lateral (horizontal) subgrid physics | `MOM_hor_visc.F90`, `MOM_thickness_diffuse.F90`, `MOM_mixed_layer_restrat.F90`, `MOM_MEKE.F90`, `MOM_lateral_mixing_coeffs.F90` |
| `src/parameterizations/vertical/` | Vertical physics / mixing | `MOM_vert_friction.F90`, `MOM_set_viscosity.F90`, `MOM_diabatic_driver.F90`, `MOM_set_diffusivity.F90`, `MOM_CVMix_KPP.F90`, `MOM_energetic_PBL.F90`, `MOM_kappa_shear.F90` |
| `src/ALE/` | Vertical Lagrangian remap (ALE), regridding, 1-D reconstructions | `MOM_ALE.F90`, `MOM_regridding.F90`, `MOM_remapping.F90`, `PPM/PLM/PQM_functions.F90`, `Recon1d_*.F90`, `coord_*.F90` |
| `src/equation_of_state/` | Equation of state (density) | `MOM_EOS.F90`, `MOM_EOS_base_type.F90`, `MOM_EOS_Wright.F90`, `MOM_EOS_Roquet_rho.F90`, others |
| `src/tracer/` | Tracer registry, advection, diffusion | `MOM_tracer_registry.F90`, `MOM_tracer_types.F90`, `MOM_tracer_advect.F90`, `MOM_tracer_hor_diff.F90` |
| `src/diagnostics/` | Runtime diagnostics, energy/mass integrals | `MOM_diagnostics.F90`, `MOM_sum_output.F90` |
| `src/framework/` | Infrastructure: domains, comms, IO, params, diag mediator, checksums | `MOM_domains.F90`, `MOM_coms.F90`, `MOM_diag_mediator.F90`, `MOM_restart.F90`, `MOM_file_parser.F90`, `MOM_hor_index.F90`, `MOM_checksums.F90`, `MOM_intrinsic_functions.F90`, `do_concurrent_compat.h`, `MOM_memory_macros.h` |
| `src/initialization/` | State/grid initialization | `MOM_state_initialization.F90` |
| `config_src/infra/{FMS1,FMS2}/` | Thin wrappers over GFDL FMS (mpp domains, IO, diag manager) | `MOM_domain_infra.F90`, `MOM_diag_manager_infra.F90`, `MOM_coms_infra.F90` |
| `config_src/memory/` | Compile-time memory model selection | `dynamic_symmetric/MOM_memory.h`, `dynamic_nonsymmetric/MOM_memory.h` |
| `config_src/drivers/` | Top-level drivers (solo, coupled) | `solo_driver/MOM_driver.F90` |

**Physics/infra split to keep in mind:** the *dynamical core* (`src/core`) is where the bulk of the
GPU port has happened; the *vertical physics* (`diabatic`, `set_diffusivity`, KPP, EPBL) and *ALE
remapping/regridding* are largely **untouched on `dev/gpu`** and in-flight on side branches (see §6).

---

## 2. The control-structure (CS) pattern

Every module owns exactly one derived type `<name>_CS` (declared `type, public :: X_CS ; private` —
public name, private members) that holds **all persistent per-module state**: runtime parameters,
diagnostic IDs (init to −1), work arrays, halo group-pass handles, and pointers to child CSs. It is
populated by a `<name>_init` routine and torn down by `<name>_end`.

### 2.1 Member styles

Members come in three flavours; which one is used has direct GPU-mapping consequences:

- **Macro-allocatable arrays** via `MOM_memory_macros.h`, e.g. in `MOM_dynamics_split_RK2.F90:90`:
  `real ALLOCABLE_, dimension(NIMEMB_PTR_,NJMEM_,NKMEM_) :: CAu, PFu, diffu`. `ALLOCABLE_` expands to
  `,allocatable`; `ALLOC_(x)` to `allocate(x)`. This is the dominant, GPU-friendly form.
- **Bare `pointer` arrays**, e.g. `real, pointer, dimension(:,:) :: taux_bot => NULL()`
  (`MOM_dynamics_split_RK2.F90:151`). Pointers are used mainly so an array can be a **target in the
  restart registry** (`MOM_variables.F90:294` comment) or aliased across modules. Pointers complicate
  device mapping and aliasing analysis.
- **Nested child CSs**, in two idioms: *by value/embedded* (`type(hor_visc_CS) :: hor_visc`,
  `type(continuity_CS) :: continuity_CSp`, `MOM_dynamics_split_RK2.F90:244,246`) or *by pointer*
  (`type(vertvisc_CS), pointer :: vertvisc_CSp => NULL()`, line 252).

### 2.2 The nesting tree

`MOM_control_struct` (`MOM.F90:204`) is the root, allocated once. It holds the prognostic state as
macro-allocatables (`h, T, S` at `MOM.F90:205`; `u, uh, uhtr`; `v, vh, vhtr`), the grid/vertical-grid
pointers (`G`, `GV`, `US`), shared containers (`tv`, `visc`, `ADp`, `CDp`), and the child module CSs.
The dynamics hub is `dyn_split_RK2_CSp` (pointer, `MOM.F90:407`), which in turn embeds
`hor_visc`, `continuity_CSp`, `CoriolisAdv`, `barotropic_CSp` (by value) and points to
`vertvisc_CSp`, `set_visc_CSp`, `ALE_CSp`.

```
MOM_control_struct (MOM.F90:204)              ← allocated once, mapped alloc at MOM.F90:3xxx
├── prognostic state: h,T,S,u,v,uh,vh,uhtr,vhtr (macro-allocatable, mapped `to`)
├── G / GV / US                                (grid; G%* metrics uploaded in bulk)
├── tv  (thermo_var_ptrs, allocatable)         ← T,S pointers + eqn_of_state
├── visc (vertvisc_type, allocatable)
├── ADp / CDp (accel_/cont_diag_ptrs, pointer aliases)
└── dyn_split_RK2_CSp (pointer, MOM.F90:407)   ← the dycore hub CS
    ├── CAu,PFu,diffu,eta,u_av,... (macro-allocatable 3D)   ← enter/exit data by member
    ├── hor_visc          (continuity_CS, by value)
    ├── continuity_CSp    (continuity_CS, by value; params only, block sizes)
    ├── CoriolisAdv       (CoriolisAdv_CS, by value)
    ├── barotropic_CSp    (barotropic_CS, by value; large frhatu/frhatv/... arrays)
    ├── vertvisc_CSp      (vertvisc_CS, pointer)
    └── set_visc_CSp      (set_visc_CS, pointer)
```

### 2.3 Shared "bag of state" containers (`MOM_variables.F90`)

These are threaded through many CSs so several modules alias the same fields:

- `thermo_var_ptrs` (`:79`) — pointer `T`, `S`, `p_surf`, `frazil`, allocatable `SpV_avg`, and
  `type(EOS_type), pointer :: eqn_of_state`.
- `ocean_internal_state` (`:138`) — all-pointer aliases to `T,S,u,v,h,uh,vh` and accelerations.
- `accel_diag_ptrs` (`:167`) / `cont_diag_ptrs` (`:241`) — pointer diagnostic aliases (`CS%ADp`,`CS%CDp`).
- `vertvisc_type` (`:258`) — **hybrid**: allocatable drag fields (`bbl_thick_u`, `kv_bbl_u`, `Ray_u`)
  plus pointer fields (`MLD`, `Kd_shear`, `Kv_shear`) that must be restart-registry targets.
- `BT_cont_type` (`:317`) — all-allocatable barotropic face-area coupling arrays.

**GPU implication:** the offload unit is the *whole CS object with its allocatable members*. Deep-copy
("attach/detach") of derived-type member arrays on device is expensive; commit `1865612de` flattened
arrays-of-structs to flat arrays because time was wasted attaching member arrays per struct
(halved GPU time in `MOM_tracer_hor_diff`).

---

## 3. Memory conventions

### 3.1 Compile-time memory model — `config_src/memory/`

Two dynamic configs on the include path differ by exactly one `#define`:
`dynamic_symmetric/MOM_memory.h` defines `SYMMETRIC_MEMORY_`, `dynamic_nonsymmetric` undefs it. Both
undef `STATIC_MEMORY_` (so `dev/gpu` uses dynamic allocation). Halo width `NIHALO_ = NJHALO_ = 2`.
Static memory would substitute real `NIGLOBAL_`/`NK_` at build time.

### 3.2 Macros — `src/framework/MOM_memory_macros.h`

- Attribute macros: `ALLOCABLE_`→`,allocatable`, `PTR_`→`,pointer`, `ALLOC_(x)`→`allocate(x)`,
  `DEALLOC_(x)`, `TO_NULL_`→`=>NULL()` (all no-ops in static mode).
- Heap-shape macros (dynamic): `NIMEM_`,`NJMEM_`→`:`; the **B (velocity/corner) forms depend on
  symmetric memory** — `NIMEMB_`/`NJMEMB_`→`0:` if symmetric else `:`; `NIMEMB_SYM_`→`0:` always.
- Dummy-argument shape macros (the `SZ*` family, appear in nearly every subroutine signature):
  `SZI_(G)`→`G%isd:G%ied`, `SZJ_(G)`→`G%jsd:G%jed`, `SZK_(G)`→`G%ke`, `SZK0_(G)`→`0:G%ke`,
  `SZIB_(G)`→`G%IsdB:G%IedB`, `SZJB_(G)`→`G%JsdB:G%JedB`.

### 3.3 Index conventions — `src/framework/MOM_hor_index.F90` / `MOM_grid.F90`

`hor_index_type` (`MOM_hor_index.F90:18`) scalar integers, replicated into `ocean_grid_type`
(`MOM_grid.F90:28`):

- Cell-center (h/tracer point): `isc/iec`, `jsc/jec` (computational); `isd/ied`, `jsd/jed` (data =
  computational + halos); `isg/ieg` (global).
- Cell-vertex / velocity B-grid (capital-I/J "B"): `IscB/IecB`, `JscB/JecB`; `IsdB/IedB`, `JsdB/JedB`.
- In **symmetric** mode the B indices start one lower: `IsdB = isd-1` (`MOM_hor_index.F90:92-96`),
  matching `NIMEMB_`→`0:`. Upper bounds always equal center upper bounds.

Canonical array shapes (doc block `MOM_hor_index.F90:178`): `h(isd:ied,jsd:jed)`,
`q(IsdB:IedB,JsdB:JedB)`, `u(IsdB:IedB,jsd:jed)`, `v(isd:ied,JsdB:JedB)`. Grid metrics are named by
stagger: `T`=tracer/h, `Cu`=C-grid u, `Cv`=C-grid v, `Bu`=B-grid corner; reciprocals prefixed `I`
(`IareaT`). **Code loop ranges should always be written for symmetric memory** — non-symmetric then
also works (with a less efficient halo pattern).

### 3.4 Where the state lives

Prognostic arrays `h,T,S` (h/tracer points), `u,uh,uhtr` (u points), `v,vh,vhtr` (v points) are
macro-allocatables inside `MOM_control_struct` (`MOM.F90:205-216`). They are passed by argument down
the call tree (as `u,v,h`), aliased into `tv%T`, `tv%S`, and into the `ocean_internal_state` pointer
container for diagnostics. The vertical grid (`GV%ke` layers, `GV%sInterface`, unit factors) is a
`pointer` carried everywhere.

---

## 4. Time-stepping call tree (the GPU-critical path)

### 4.1 Top level — `MOM.F90` `step_MOM`

`step_MOM` (`MOM.F90:522`) drives a coupling timestep. Within it (order depends on ALE/thermo
splitting flags):

- `step_MOM_thermo` (`MOM.F90:911, 1032`; def `:1731`) → `diabatic(...)` (`:1828`) → ALE remap.
- `step_MOM_dynamics` (`MOM.F90:985`; def `:1226`) → `step_MOM_dyn_split_RK2(...)` (`:1388`) [or the
  `RK2b`/unsplit variants].
- `step_MOM_tracer_dyn` (`MOM.F90:999`; def `:1598`) → `advect_tracer(...)` (`:1650`) →
  `tracer_hordiff(...)` (`:1653`).
- Diagnostics posted throughout via `post_data` (`MOM_diag_mediator.F90`).

Group halo update of the coupled state uses the **GPU-aware** path:
`call do_group_pass(pass_uv_T_S_h, G%Domain, ..., omp_offload=.true.)` (`MOM.F90:2112`).

### 4.2 The split RK2 dynamical core — `MOM_dynamics_split_RK2.F90`

`step_MOM_dyn_split_RK2` (def `:302`) is a **predictor–corrector** scheme separating fast barotropic
and slow baroclinic modes. Actual internal call sequence (absolute line numbers in
`MOM_dynamics_split_RK2.F90`; group passes created at `:506-517`):

**Predictor stage**
1. `PressureForce(h,tv,...)` → `CS%PFu,CS%PFv,CS%pbce,CS%eta_PF` (`:527`)
2. `CorAdCalc(u_av,v_av,h_av,...)` → `CS%CAu_pred,CS%CAv_pred` (`:589`) — Coriolis + momentum advection
3. `set_viscous_ML(...)` (`:640`), `vertvisc_coef(up,vp,...)` (`:650`), `vertvisc_remnant(...)` (`:651`)
4. `btcalc(h)` (`:671`), `bt_mass_source` (`:673`)
5. `continuity(u_inst,v_inst,h,hp,...,BT_cont)` → provisional `hp` (`:695`); `set_dtbt` (`:715/:719`)
6. **`btstep(...)`** — barotropic sub-cycling (`:726`), returns `u_accel_bt,v_accel_bt,eta_pred`
7. `vertvisc_coef` (`:801`) / `vertvisc(up,vp,...,AD_pred)` (`:817`) / `vertvisc_remnant` (`:834`)
8. `continuity(up,vp,h,hp,...,u_cor=u_av,v_cor=v_av)` (`:853`); `radiation_open_bdry_conds` (`:868`)

**Corrector stage**
9. `bt_mass_source(hp,...)` (`:894`), `PressureForce(hp,...)` (`:909`, if `begw/=0`)
10. `horizontal_viscosity(u_av,v_av,h_av,...)` → `CS%diffu,CS%diffv` (`:962`)
11. `CorAdCalc(...)` → `CS%CAu,CS%CAv` (`:972`)
12. `btstep(...)` again (`:1023`)
13. `vertvisc_coef(u_inst,v_inst)` (`:1099`) / `vertvisc(...,ADp)` (`:1108`) / `vertvisc_remnant` (`:1121`)
14. `continuity(u_inst,v_inst,h_tmp,h,...)` → final `h` (`:1148`); `radiation_open_bdry_conds` (`:1172`)
15. Optional stored `CorAdCalc(...,CAu_pred,CAv_pred)` for next predictor (`:1206`)

Group-pass execution points: `pass_eta` (`:580/:658`), `pass_visc_rem` (`:661-685`), `pass_uvp`
(`:829-847`), `pass_hp_uv` (`:860`), `pass_vector(u_av,v_av)` OBC (`:877`), `pass_uv` (`:1118-1137`),
`pass_h` (`:1153`), `pass_av_uvh` (`:1163-1181`) — all `omp_offload=.true.`. **The pure compute
kernels (continuity_PPM, CorAdCalc, PressureForce_FV, hor_visc) contain no halo updates and no
reproducing sums by design** — communication is hoisted into this driver and `btstep`.

So the per-timestep dycore hot loop is: **PressureForce → CorAdCalc → vert_visc(coef/remnant) →
btcalc → continuity → btstep (inner barotropic loop) → hor_visc → continuity**, with ~10 group halo
updates, all on the `omp_offload=.true.` path.

### 4.3 Key sub-solvers

- **`continuity` / `MOM_continuity_PPM.F90`** — PPM finite-volume mass transport. Public `continuity`
  → `zonal_mass_flux` / `meridional_mass_flux` → PPM reconstruction. **Fully k-blocked** with
  `niblock/njblock/nkblock` (CS members `:76-78`), hybrid `do concurrent` + `!$omp target teams` (see §5).
- **`CorAdCalc` / `MOM_CoriolisAdv.F90`** — Coriolis + advection of momentum (KE gradient, vorticity
  flux). Heavily restructured (56 `do concurrent`, 61 `omp target`); k-blocking merged (`b8c471cfa`).
- **`PressureForce` / `MOM_PressureForce_FV.F90`** — finite-volume pressure gradient. Two entry points
  `PressureForce_FV_nonBouss` (`:122`) and `PressureForce_FV_Bouss` (`:947`); calls into
  `MOM_density_integrals.F90` (`int_density_dz_*`) which call EOS. 30 `do concurrent`, 34 `omp target`.
- **`btstep` / `MOM_barotropic.F90`** — barotropic solver (`:480`, 6868-line module). Sub-cycles many
  small barotropic timesteps in `btstep_timeloop` (`:2376`); `set_dtbt` (`:3797`) sets the sub-step.
  Most heavily ported module by raw directive count (242 `do concurrent`, 106 `omp target`).
- **`horizontal_viscosity` / `MOM_hor_visc.F90`** — lateral friction (Laplacian + biharmonic,
  Smagorinsky/Leith). 62 `do concurrent`, 144 `omp target`. k-blocking in-flight on `kblock-hor-visc`.
- **`vertvisc` / `MOM_vert_friction.F90`** — implicit vertical friction (tridiagonal solve per column).
  `vertvisc_coef`/`vertvisc`/`vertvisc_remnant`. Uses `!$omp target teams loop collapse(2)` with a
  serial inner tridiagonal k-loop (see §5). 3 `!$omp declare target` column kernels.
- **`diabatic` / `MOM_diabatic_driver.F90`** — vertical mixing dispatcher (`diabatic` `:279` →
  `diabatic_ALE`/`layered_diabatic`). **Essentially unported on `dev/gpu`** (3-line change).

---

## 5. The k-blocking / tiling transformation (the blessed pattern)

The core CPU-and-GPU-preserving refactor. Loops are rewritten so a horizontal/vertical **block** is
processed at a time. Block sizes are CS parameters resolved at init: on NVHPC GPU builds they default
to `0` (meaning "whole domain / no cache blocking"), on CPU to e.g. `32/4/1`
(`MOM_continuity_PPM.F90:3120-3129`). When `0`, the block is set to the full loop extent
(`if (niblock == 0) niblock = ...`, `MOM_continuity_PPM.F90:188`).

Representative hybrid kernel from continuity (`MOM_continuity_PPM.F90:696-736`): an outer host loop
strides over blocks, block-local indices `ii = i-i_start+1`, `jj = j-j_start+1` reuse small work
arrays, and the compute region is an explicit `!$omp target teams num_teams(nteams)` (team count
computed by hand — `nteams = ceiling(real((j_end-j_start+1)*(i_end-i_start+1))/128.)`) wrapping
`!$omp loop collapse(2)` over the tile, calling `!$omp declare target` helpers (`flux_elem`,
`ratio_max`). The manual team count exists because nvfortran's OpenMP runtime under-launched teams
(commit `5b5f6b2b1`). Elsewhere the same file uses plain `do concurrent (k=1:nz, j=..., i=...)`
(`:430`) where the compiler schedules acceptably.

Bitwise preservation: k-blocking changes only *loop structure*, never the arithmetic order within a
column reconstruction, so results stay bit-identical — verified with `MOM_checksums`. Merged examples:
`93dbbd36e` (continuity reconstruction), `b8c471cfa` (CoriolisAdv). In-flight: `kblock-hor-visc`
(single commit rewrites `MOM_hor_visc.F90`, +1365/−1113), `kblock-coradcalc`.

---

## 6. Quantified inventory of GPU work

### 6.1 Merged on `dev/gpu` (`git diff --stat dev-gfdl...dev/gpu`, 320 commits, 52 files)

**Ported (heavy churn, GPU-resident):**

| Subsystem | File | +/− | Status |
|---|---|---|---|
| Continuity (PPM) | `MOM_continuity_PPM.F90` | +1575/−1171 | k-blocked, hybrid dc+omp |
| Barotropic solver | `MOM_barotropic.F90` | +1003/−804 | ported; `omp_offload` halos |
| Vertical friction | `MOM_vert_friction.F90` | +775/−184 | teams-loop tridiagonal |
| Coriolis/advection | `MOM_CoriolisAdv.F90` | +764/−565 | k-blocked (`b8c471cfa`) |
| Set viscosity (BBL/ML) | `MOM_set_viscosity.F90` | +501/−384 | ported, declare-target kernels |
| Horizontal viscosity | `MOM_hor_visc.F90` | +399/−195 | ported (full k-block on branch) |
| EOS dispatch | `MOM_EOS.F90` + base/Wright/Roquet | +361/+154/+290/+291 | Wright+Roquet 2D/3D direct |
| Tracer hor. diff | `MOM_tracer_hor_diff.F90` | +319/−268 | ported; flat-array refactor |
| Tracer advection | `MOM_tracer_advect.F90` | +319/−227 | ported (`#54`); multi-GPU fixes |
| Reproducing sums | `MOM_coms.F90` | +317/−177 | block-based EFP (`8593a732a`) |
| Pressure force FV | `MOM_PressureForce_FV.F90` | +215/−136 | ported (Wright integrals) |
| Dycore driver | `MOM_dynamics_split_RK2.F90` | +244/−105 | maps CS members, halo offload |
| Top driver | `MOM.F90` | +260/−40 | top-level CS/grid mapping |
| Intrinsics | `MOM_intrinsic_functions.F90` | +104/−23 | `cuberoot`/`nth_root` declare-target |

Infra added: `do_concurrent_compat.h` (`DO_LOCALITY` macro), `ac/m4/mom6_fc_do_concurrent_local.m4`
+ `ac/configure.ac:172` (feature detection → `HAVE_FC_DO_CONCURRENT_LOCAL`), `omp_offload` optional
arg on `do_group_pass` (`config_src/infra/FMS2/MOM_domain_infra.F90:1143`).

**Directive totals across `src/`:** 698 `do concurrent`, 829 `omp target`, 213 `enter data`,
168 `exit data`, 21 `declare target`. (Counts re-verified by doc 03's verification pass.)

### 6.2 In-flight on branches

| Branch | Scope | Pattern |
|---|---|---|
| `kblock-hor-visc` | `MOM_hor_visc.F90` k-blocking | blessed k-block |
| `kblock-coradcalc` | `MOM_CoriolisAdv.F90` k-block (subset merged) | blessed k-block |
| `bodner-naive-port` | `MOM_mixed_layer_restrat.F90` (Bodner MLE), density integrals | **naive** port (contrast case) |
| `port/pressureforce-benchmark_ALE` | `MOM_density_integrals.F90` int_density_dz PLM | ALE pressure integrals |
| `diag_map_mediator_port` | `MOM_diag_mediator.F90` (+365) | diagnostics offload |
| `fix/nan_repro_sum` | `MOM_coms.F90` NaN in large-domain repro sum | reproducibility bugfix |
| `remotes/edoyango/acc-btstep` | `MOM_barotropic.F90` OpenACC kernels + async | alternative btstep offload |
| `remotes/edoyango/bugfix-traceradvection-multigpu` | `MOM_tracer_advect.F90` `Reg%Tr(:)` mapping | multi-GPU correctness |
| `remotes/edoyango/port-set_diffusivity` | `MOM_set_diffusivity.F90` (+1277) | vertical-mixing port |
| `remotes/edoyango/port/thickness_diffuse` | `MOM_thickness_diffuse.F90` (+685) | lateral-mixing port |
| `remotes/edoyango/gpu-port-tracking` | `.testing/tools/track_gpu_port.py` + CI | **port coverage tooling** (noport/toport sentinels) |
| `remotes/edoyango/benchmark_ALE_nvtx_clocks` | nvtx markers on clocks | profiling practice |

### 6.3 Untouched on `dev/gpu` (host-only; major porting surface remaining)

`MOM_diabatic_driver.F90` (3 lines), `MOM_set_diffusivity.F90` (0), `MOM_CVMix_KPP.F90` (0),
`MOM_energetic_PBL.F90` (0), `MOM_mixed_layer_restrat.F90` (0), `MOM_ALE.F90` (3),
`MOM_regridding.F90` (0), `MOM_remapping.F90` (0), `MOM_diag_mediator.F90` (0, host-only IO path).
The **entire vertical-mixing (diabatic) stack and ALE remap/regrid remain to be ported.**

---

## 7. Cross-cutting subsystems a porting agent must know

### 7.1 EOS — runtime polymorphism is the enemy

`EOS_type` (`MOM_EOS.F90:117`) wraps a **polymorphic allocatable component**
`class(EOS_base), allocatable :: type` (`:162`). `EOS_base` (`MOM_EOS_base_type.F90:13`) is an abstract
type with **deferred elemental type-bound procedures** (`density_elem`, etc.). The concrete class is
chosen with `allocate(<concrete> :: EOS%type)` in a `select case`/`select type` (`MOM_EOS.F90:2122`),
and calls dispatch through `EOS%type%calculate_density_...`. **nvfortran cannot resolve this v-table
dispatch on device, and passing polymorphic `this` into a `do concurrent`/target region forces a
mishandled implicit copy.** The fix (merged, `7c7af5572`, `52a1b3954`): each elemental kernel is
duplicated as a free `_loc` function with no `this`, and new `calculate_density_array_3d` /
`_derivs_3d` / `_second_derivs_2d` implementations wrap `do concurrent (k,j,i)` calling the `_loc`
kernel. **Only buggy-Wright and Roquet_rho are ported;** linear, UNESCO, Jackett06, TEOS10,
Wright_full/red, Roquet_SpV still fall back to polymorphic elemental dispatch. See `06-eos.md`.

### 7.2 Bitwise reproducibility — `MOM_coms` EFP sums + `MOM_checksums`

Global integrals (`write_energy`, `MOM_sum_output.F90`) use **Extended Fixed Point (EFP)** reproducing
sums: each real is decomposed into a 6-word base-2⁴⁶ signed integer (`EFP_type`, `MOM_coms.F90:103`),
integers are summed exactly (associative regardless of order), and reconstructed. The GPU port
(`8593a732a`) partitions the domain into **blocks small enough that no block sum can overflow the 17
carry bits**, and each block is a `do concurrent` with `DO_LOCALITY(reduce(+: block_sum))` +
`reduce(max:...)` over exact integers (`increment_block_ints`, `MOM_coms.F90:618-772`), with
`efp_decompose` a `pure`/`!$omp declare target` helper (`:778`). Because the reduction is over exact
integers, thread/PE scheduling cannot change the bits. `MOM_checksums.F90` verifies ports via a
`popcnt`-based bitcount checksum (`:2680`) mod 10⁹ — two runs agree only if fields are bit-identical.
See `07-reproducibility.md`.

### 7.3 Halos, domains, and the `omp_offload` path

`MOM_domains.F90` re-exports `create_group_pass`/`do_group_pass`/`pass_var`/`pass_vector` over FMS mpp
(`config_src/infra/FMS2/MOM_domain_infra.F90`). The GPU port added an optional `omp_offload` argument
to `do_group_pass` (`:1143`) forwarded to `mpp_do_group_update`, so halo exchanges operate on
device-resident buffers (GPU-aware). It is passed `.true.` at 26 call sites (14 gated behind
`if (G%nonblocking_updates)`, 12 unconditional — doc 11 §4) across dynamics,
barotropic, and tracer modules. Halo width is 2 (`NIHALO_`). See `11-halos-domains.md`.

### 7.4 Diagnostics / IO — still host-only on `dev/gpu`

`MOM_diag_mediator.F90` (`post_data` generic, `:73`) is **unchanged on `dev/gpu`** — posting a
diagnostic implies a device→host transfer of the field. Directives guard transfers behind diag-ID
checks (`!$omp target update from(...) if (CS%id_... > 0)`, e.g. `MOM_tracer_hor_diff.F90:722`). The
offload of the mediator itself is on branch `diag_map_mediator_port`. See `12-diagnostics-io.md`.

### 7.5 Compiler workarounds

Catalogue-worthy so far: mandatory inlining of `ratio_max`/`flux_elem` (`3cb184edd`: "Otherwise
results are incorrect"; historically via `-Minline=name:`/`!NVF$ INLINE`, at HEAD via
`!DIR$ ATTRIBUTES FORCEINLINE` on flux_elem/flux_elem_OBC only — `93dbbd36e`; `ratio_max` currently
carries no directive, see KNOWLEDGE.md §8a item 11), `!$omp declare target` on all point/column
kernels, manual `num_teams` (`5b5f6b2b1`),
`omp target teams loop -> do concurrent` reversions (`e8b0ecfbf`), `modulo()` avoided in `cuberoot`
(not implemented on all targets), EOS `_loc` free functions to avoid `this` copies, "implicit copy of
`this` which cannot yet be prevented" (unresolved), and the `A100 nvfortran 25.5` crash avoided by
removing an `eta_bt` transfer (`2108e0eba`). See `13-compiler-workarounds.md`.

---

## 8. Port-coverage tooling (branch `edoyango/gpu-port-tracking`)

`.testing/tools/track_gpu_port.py` cross-references gcov execution coverage against auto-detected
ported regions (`do concurrent`, `!$omp target[ teams][ loop]` blocks). Manual overrides use in-source
sentinels: `!@start noport ... !@end noport` (never port — serial bookkeeping) and
`!@start toport ... !@end toport` (needs porting but not a structural loop). Every marker requires an
explicit matching `!@end`. Executed-but-unported lines are split into "portable" vs "not portable"
(allocate/IO/call/control-flow). This is the objective progress metric for the whole effort.

---

## 9. Quick reference — where to look first

- **Add a device kernel:** copy the `do concurrent (k,j,i) DO_LOCALITY(local(...))` pattern; for
  reductions use `DO_LOCALITY(reduce(+:...))`; for tridiagonal columns use
  `!$omp target teams loop collapse(2)` with explicit `private`.
- **Map a new CS array:** `ALLOC_(CS%x(...)); CS%x=0.0; !$omp target enter data map(to: CS%x)` and the
  mirrored `map(delete:)` next to `DEALLOC_` in `*_end`.
- **Call a helper from device:** add `!$omp declare target` and ensure it inlines.
- **Verify a port:** compare `MOM_checksums` hchksum/uchksum and reproducing-sum energy output CPU vs GPU.
- **Never** pass `class(*)`/polymorphic `this`, allocate inside a device loop, or reorder a
  floating-point reduction.
