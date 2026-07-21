# Memory Handling and Control-Structure Allocation (dev/gpu)

> Companion to `00-architecture.md` §2 (CS pattern) and §3 (memory conventions). Where
> `00-architecture.md` sketches the CS pattern and macro table, this document goes to the bottom of
> it: exact macro expansions, the concrete symmetric/non-symmetric offset mechanics, full type
> definitions for the five+ study-target CS types with allocation-site line numbers, and a
> line-by-line reconstruction of how the CS object graph is built and mapped onto the GPU at runtime.
> See `02-pointer-usage.md` for the deep dive on *why* individual fields are `pointer` (restart
> registry, cross-module aliasing) — this document only touches that where it bears on allocation.

---

## 1. The compile-time memory model

### 1.1 `config_src/memory/` — the one-`#define` difference

Two files select the memory layout at build time; both are otherwise byte-identical:

- `config_src/memory/dynamic_symmetric/MOM_memory.h:37`: `#define SYMMETRIC_MEMORY_`
- `config_src/memory/dynamic_nonsymmetric/MOM_memory.h:37`: `#undef  SYMMETRIC_MEMORY_`

Both files (`:41`) `#undef STATIC_MEMORY_` — `dev/gpu` always builds dynamic (heap-allocated,
runtime-shaped) arrays, never static (compile-time-shaped) arrays. Both set `NIHALO_ = NJHALO_ = 2`
(`:30,33`) and `NIGLOBAL_`/`NJGLOBAL_`/`NK_`/`NIPROC_`/`NJPROC_` to `NONSENSE_*` placeholders
(`:11-21`) that are never actually used in dynamic mode (they only matter for `STATIC_MEMORY_`
builds, where they're substituted by the build system with real numbers). Both `#include
<MOM_memory_macros.h>` (`:43`) which does the real work, branching on whether `STATIC_MEMORY_` is
defined.

### 1.2 `src/framework/MOM_memory_macros.h` — full macro catalogue (dynamic-mode expansions)

Attribute/action macros (`:101-110`, identical role in both modes, only dynamic shown):
| Macro | Expands to (dynamic) | Static-mode value |
|---|---|---|
| `ALLOCABLE_` | `,allocatable` | *(nothing)* |
| `PTR_` | `,pointer` | *(nothing)* |
| `ALLOC_(x)` | `allocate(x)` | *(nothing)* |
| `DEALLOC_(x)` | `deallocate(x)` | *(nothing)* |
| `TO_NULL_` | `=>NULL()` | *(nothing)* |

Heap-shape macros for declaring `ALLOCABLE_`/`PTR_` members inside a type (dynamic mode, `:114-161`):
| Macro | Dynamic expansion | Notes |
|---|---|---|
| `NIMEM_` / `NJMEM_` | `:` | h/tracer-point extent |
| `NIMEMB_PTR_` / `NJMEMB_PTR_` | `:` (**always**, regardless of symmetric) | see note below |
| `NIMEMB_` / `NJMEMB_` | `0:` if `SYMMETRIC_MEMORY_` else `:` (`:126-140`) | velocity/corner (B) extent |
| `NIMEMB_SYM_` / `NJMEMB_SYM_` | `0:` unconditionally | *always*-symmetric B arrays |
| `NKMEM_` | `:` | layer extent |
| `NKMEM0_` | `0:` | interface extent |
| `NK_INTERFACE_` | `:` | interface extent (heap-shape macro, `NK_+1` in static mode) |

Dummy-argument / stack-shape macros, the `SZ*` family used in nearly every subroutine signature
(`:167-182`):
```
SZI_(G)   -> G%isd:G%ied      SZJ_(G)  -> G%jsd:G%jed
SZK_(G)   -> G%ke             SZK0_(G) -> 0:G%ke
SZIB_(G)  -> G%IsdB:G%IedB    SZJB_(G) -> G%JsdB:G%JedB
SZIBS_(G) -> G%isd-1:G%ied    SZJBS_(G)-> G%jsd-1:G%jed   ! "always symmetric" dummy shape
```
Plus decomposition-invariant `SZDI_/SZDIB_/SZDJ_/SZDJB_` (`:188-195`) that are the same in both memory
models — used where a routine must always see the symmetric-shaped index range regardless of the
build's `SYMMETRIC_MEMORY_` setting.

**The `NIMEMB_PTR_` subtlety.** In *dynamic* mode `NIMEMB_PTR_`/`NJMEMB_PTR_` expand to plain `:`
(`MOM_memory_macros.h:122,125`) — **not** `0:` — even when `SYMMETRIC_MEMORY_` is set. This is
because in dynamic mode the array is declared with assumed/deferred shape (`:`); the actual lower
bound (`0` vs `1`) is fixed later at the `ALLOC_(...)` call site using explicit bounds computed from
`hor_index_type`/`ocean_grid_type` (`IsdB`, which is itself `isd-1` when symmetric — see §2). In
*static* mode, by contrast, `NIMEMB_PTR_` is literally `NIMEMB_` (`MOM_memory_macros.h:53,56`)
because the shape has to be baked into the declaration at compile time — there's no later `ALLOC_`
call to fix it up. So: **dynamic-mode symmetric offsets live in the runtime `ALLOC_` bounds, not in
the type declaration**; static-mode symmetric offsets live in the macro expansion itself.

### 1.3 Concrete example — `ocean_grid_type` in `MOM_grid.F90`

```fortran
! MOM_grid.F90:78-92 (h-point vs u-point metric declarations)
real ALLOCABLE_, dimension(NIMEM_,NJMEM_) :: &
  mask2dT, geoLatT, geoLonT, dxT, IdxT, dyT, IdyT, areaT, IareaT, sin_rot, cos_rot
real ALLOCABLE_, dimension(NIMEMB_PTR_,NJMEM_) :: &
  mask2dCu, OBCmaskCu, geoLatCu, geoLonCu, dxCu, IdxCu, IdxCu_OBCmask, dyCu, IdyCu, dy_Cu, IareaCu, areaCu
```
In dynamic mode both declarations reduce to `dimension(:,:)` — the difference between an h-point
array (`NIMEM_`) and a u-point array (`NIMEMB_PTR_`) disappears at declaration time and is entirely
determined by the bounds passed to `ALLOC_` in `allocate_metrics` (`MOM_grid.F90:536-617`):
```fortran
! MOM_grid.F90:547-548
ALLOC_(G%dxT(isd:ied,jsd:jed))       ; G%dxT(:,:) = 0.0
ALLOC_(G%dxCu(IsdB:IedB,jsd:jed))    ; G%dxCu(:,:) = 0.0
```
`IsdB` here is `G%IsdB`, computed in `MOM_grid_init` (`:303-313`, mirroring `hor_index_init`):
```fortran
! MOM_grid.F90:306-310
if (G%symmetric) then
  G%IscB = G%isc-1 ; G%JscB = G%jsc-1
  G%IsdB = G%isd-1 ; G%JsdB = G%jsd-1
  G%IsgB = G%isg-1 ; G%JsgB = G%jsg-1
endif
```
So **symmetric memory gives every B-staggered (u/v/corner) array one extra row/column on the west/
south side** (`IsdB = isd-1` instead of `isd`), so that `u(IsdB:IedB, jsd:jed)` includes the west
face of the westernmost h-cell — every h-cell's both faces are present. Non-symmetric memory keeps
`IsdB = isd` (velocity data domain same size as tracer data domain), which is smaller and requires a
different (more argument-passing-heavy, "get the extra column from a neighbor") halo-update pattern
that is legal but less efficient — hence the doc-comment "code should always be written for
symmetric memory" (`MOM_hor_index.F90:175-176`).

### 1.4 `src/framework/MOM_hor_index.F90` — where the offset is actually computed

`hor_index_type` (`:18-57`) is the single source of truth for these bounds; `hor_index_init`
(`:65-102`) sets the B (capital-letter) bounds from the h/tracer bounds:
```fortran
! MOM_hor_index.F90:89-99
HI%IscB = HI%isc ; HI%JscB = HI%jsc
HI%IsdB = HI%isd ; HI%JsdB = HI%jsd
HI%IsgB = HI%isg ; HI%JsgB = HI%jsg
if (HI%symmetric) then
  HI%IscB = HI%isc-1 ; HI%JscB = HI%jsc-1
  HI%IsdB = HI%isd-1 ; HI%JsdB = HI%jsd-1
  HI%IsgB = HI%isg-1 ; HI%JsgB = HI%jsg-1
endif
HI%IecB = HI%iec ; HI%JecB = HI%jec
HI%IedB = HI%ied ; HI%JedB = HI%jed
HI%IegB = HI%ieg ; HI%JegB = HI%jeg
```
Upper bounds (`IecB`/`IedB`/`IegB`) are **always** equal to the center upper bounds — only the lower
bound moves. `ocean_grid_type` (`MOM_grid.F90:28-216`) duplicates every one of these fields as its own
scalars (`G%isc`, `G%IsdB`, ...) rather than referencing `G%HI` directly in hot code, because `G%HI`
(a `hor_index_type` value, `MOM_grid.F90:31`) is copied wholesale in `MOM_grid_init` (`G%HI = HI`,
`:271`) and the flat scalars are what every `SZI_`/`SZIB_` macro and every loop bound actually reads.

Canonical shapes (doc block, `MOM_hor_index.F90:178-182`):
```
h(isd:ied, jsd:jed)          q(IsdB:IedB, JsdB:JedB)
u(IsdB:IedB, jsd:jed)        v(isd:ied, JsdB:JedB)
```

### 1.5 Static-mode expansions (for contrast; not used on `dev/gpu`)

If `STATIC_MEMORY_` were defined, `NIMEM_` → `(((NIGLOBAL_-1)/NIPROC_)+1+2*NIHALO_)`
(`MOM_memory_macros.h:31`) — a compile-time arithmetic expression substituting real
`NIGLOBAL_`/`NIPROC_`/`NIHALO_` values baked in by the build system (`NIHALO_=2` always) — and
`NKMEM_` → `NK_` (a literal layer count, `:65`), so every array gets a fixed shape at compile time.
`dev/gpu` never uses this path (`STATIC_MEMORY_` is `#undef`'d in both memory configs), but static
mode is why the `SZ*` macros exist at all: dummy-argument declarations must work identically whether
the actual heap array behind them is shaped by a runtime `G%isd` or a compile-time arithmetic
expression.

---

## 2. `verticalGrid_type` (`MOM_verticalGrid.F90:26-101`)

Unlike `ocean_grid_type`, the vertical grid has almost no macro-allocatable arrays — it is dominated
by scalars (unit-conversion factors `H_to_m`, `H_to_kg_m2`, `Angstrom_H`, ...) plus four small,
**plain** (non-macro) `allocatable` 1-D arrays sized by `nk` (number of layers), not by horizontal
extent:
```fortran
! MOM_verticalGrid.F90:41-46, 63-65
real, allocatable, dimension(:) :: sLayer      !< layer-center coordinate values
real, allocatable, dimension(:) :: sInterface  !< interface coordinate values
real, allocatable, dimension(:) :: g_prime, Rlay
```
`GV` itself is a `pointer` in every owner (`MOM_control_struct%GV`, `MOM.F90:240-241`), allocated
once by `verticalGridInit`:
```fortran
! MOM_verticalGrid.F90:122-124, 242-245
if (associated(GV)) call MOM_error(FATAL, 'verticalGridInit: called with an associated GV pointer.')
allocate(GV)
...
allocate( GV%sInterface(nk+1) )
allocate( GV%sLayer(nk) )
allocate( GV%g_prime(nk+1), source=0.0 )
allocate( GV%Rlay(nk), source=0.0 )
```
and torn down by `verticalGridEnd` (`:361-368`, `deallocate(GV%g_prime, GV%Rlay)` then
`deallocate(GV%sInterface, GV%sLayer)` then `deallocate(GV)`). Because the vertical grid is tiny
(`O(nk)`, not `O(ni*nj*nk)`), most of its data reaches device kernels by-value — scalar
unit-conversion factors passed through argument lists, or the whole small `GV` derived type passed as
an `intent(in)` dummy that nvfortran can firstprivate-copy.

**Correction (verified against source).** It is *not* true that `GV` never appears in an `!$omp target
enter data` list. `MOM.F90:3650` contains an **active** directive:
```fortran
! MOM.F90:3650
!$omp target enter data map(to: GV, GV%Rlay, GV%g_prime)
```
i.e. the whole `GV` object plus its `Rlay` and `g_prime` member arrays are explicitly mapped `to` the
device. Instructively, an earlier attempt to do the same map immediately after `verticalGridInit`
(`MOM.F90:3053-3054`) is **commented out** with the note explaining why it was moved:
```fortran
! MOM.F90:3056-3057
! This does not work.  GV%RLay changes sometime later.
!!!$omp target enter data map(to: GV, GV%Rlay, GV%g_prime)
```
So the live map was deliberately relocated to a point (`:3650`, after the vertical grid parameters are
rescaled) where `GV%Rlay`/`GV%g_prime` no longer change on the host. `GV%sLayer`/`GV%sInterface` are
*not* in any device map — only `GV`, `Rlay`, and `g_prime`.

> **Resolved (2026-07-14):** The `GV` device map is load-bearing, not vestigial. `GV%Rlay` is read
> inside a device `do concurrent` — the `Rml_max`-vs-`GV%Rlay` binary density search in
> `tracer_epipycnal_ML_diff` (`MOM_tracer_hor_diff.F90`) — so `initialize_MOM`'s
> `map(to: GV, GV%Rlay, GV%g_prime)` is genuinely consumed. Treat the relocated map as required, and
> keep the ordering constraint the disabled sibling records: it must stay after the host-side rescale.

---

## 3. The five-plus CS types: definitions and allocation sites

### 3.1 `MOM_control_struct` (`MOM.F90:204-477`) — the root

Not a pointer or allocatable itself — it is a **plain value type** owned by the driver
(`config_src/drivers/solo_driver/MOM_driver.F90:174`: `type(MOM_control_struct) :: MOM_CSp`) and
passed down by `intent(inout)`. This matters enormously for GPU mapping — see §5.

Prognostic state is macro-allocatable, declared directly in the type:
```fortran
! MOM.F90:205-216
real ALLOCABLE_, dimension(NIMEM_,NJMEM_,NKMEM_) :: h, T, S
real ALLOCABLE_, dimension(NIMEMB_PTR_,NJMEM_,NKMEM_) :: u, uh, uhtr
real ALLOCABLE_, dimension(NIMEM_,NJMEMB_PTR_,NKMEM_) :: v, vh, vhtr
```
allocated in `initialize_MOM` at `MOM.F90:3110-3115`:
```fortran
ALLOC_(CS%u(IsdB:IedB,jsd:jed,nz))   ; CS%u(:,:,:) = 0.0
ALLOC_(CS%v(isd:ied,JsdB:JedB,nz))   ; CS%v(:,:,:) = 0.0
ALLOC_(CS%h(isd:ied,jsd:jed,nz))     ; CS%h(:,:,:) = GV%Angstrom_H
ALLOC_(CS%uh(IsdB:IedB,jsd:jed,nz))  ; CS%uh(:,:,:) = 0.0
ALLOC_(CS%vh(isd:ied,JsdB:JedB,nz))  ; CS%vh(:,:,:) = 0.0
!$omp target enter data map(to: CS%u, CS%v, CS%h, CS%uh, CS%vh)
```
and deallocated in `MOM_end` (`:4778`: `DEALLOC_(CS%u) ; DEALLOC_(CS%v) ; DEALLOC_(CS%h)`; the
`MOM_end` subroutine itself begins at `:4698`, and calls `end_dyn_split_RK2(CS%dyn_split_RK2_CSp)` at
`:4733`).

Nested/embedded children (excerpt; the type has ~30 child-CS members):
```fortran
! MOM.F90:233-234, 244, 261, 340, 342, 403-460 (representative)
type(ocean_grid_type), allocatable  :: G_in     !< allocatable (was plain value; see 81680c15d, §6)
type(ocean_grid_type), pointer :: G => NULL()   !< pointer alias to the active grid
type(thermo_var_ptrs), allocatable :: tv
type(vertvisc_type),   allocatable :: visc      !< allocatable (was plain value; see 81680c15d/c82e1254a, §6)
type(accel_diag_ptrs), allocatable :: ADp
type(cont_diag_ptrs)               :: CDp       !< embedded value (not allocatable/pointer)
type(MOM_dyn_split_RK2_CS), pointer :: dyn_split_RK2_CSp => NULL()
type(set_visc_CS), allocatable :: set_visc_CSp  !< allocatable (was plain value; see c82e1254a, §6)
type(thickness_diffuse_CS)    :: thickness_diffuse_CSp   !< embedded value
type(MEKE_CS)                 :: MEKE_CSp               !< embedded value
```
Allocation sites for the notable ones: `allocate(CS%tv)` (`:2564`), `allocate(CS%G_in)` (`:2985`,
followed by `!$omp target enter data map(to: CS%G_in)` at `:3034`), `allocate(CS%ADp)` (`:3189`,
`!$omp target enter data map(alloc: CS%ADp)` `:3190`), `allocate(CS%visc)` (`:3277`, `map(alloc:
CS%visc)` `:3278`), `allocate(CS%dyn_split_RK2_CSp)` (`:3258`, `map(alloc: CS%dyn_split_RK2_CSp)`
`:3259`).

### 3.2 `MOM_dyn_split_RK2_CS` (`MOM_dynamics_split_RK2.F90:89-281`) — the dycore hub

Pointer member of the parent (`CS%dyn_split_RK2_CSp`, `MOM.F90:407`), allocated at `MOM.F90:3258` (see
above) — **the CS type itself has no `_init`-time `allocate(CS)` inside its own module**; the parent
allocates it because "this module does not have its own control structure, but shares the same
control structure with MOM.F90" (module doc, `MOM_dynamics_split_RK2.F90:2113-2116`).

Array members are macro-allocatable:
```fortran
! MOM_dynamics_split_RK2.F90:90-95
real ALLOCABLE_, dimension(NIMEMB_PTR_,NJMEM_,NKMEM_) :: &
  CAu, CAu_pred, PFu, PFu_Stokes, diffu
```
Allocated (with the GPU-map immediately following each pair) in `register_restarts_dyn_split_RK2`,
`:1350-1368`:
```fortran
! MOM_dynamics_split_RK2.F90:1350-1368
ALLOC_(CS%diffu(IsdB:IedB,jsd:jed,nz)) ; CS%diffu(:,:,:) = 0.0
ALLOC_(CS%diffv(isd:ied,JsdB:JedB,nz)) ; CS%diffv(:,:,:) = 0.0
!$omp target enter data map(to: CS%diffu, CS%diffv)
ALLOC_(CS%CAu(IsdB:IedB,jsd:jed,nz))   ; CS%CAu(:,:,:)   = 0.0
ALLOC_(CS%CAv(isd:ied,JsdB:JedB,nz))   ; CS%CAv(:,:,:)   = 0.0
!$omp target enter data map(to: CS%CAu, CS%CAv)
ALLOC_(CS%CAu_pred(IsdB:IedB,jsd:jed,nz)) ; CS%CAu_pred(:,:,:)   = 0.0
ALLOC_(CS%CAv_pred(isd:ied,JsdB:JedB,nz)) ; CS%CAv_pred(:,:,:)   = 0.0
!$omp target enter data map(to: CS%CAu_pred, CS%CAv_pred)
ALLOC_(CS%PFu(IsdB:IedB,jsd:jed,nz))   ; CS%PFu(:,:,:)   = 0.0
ALLOC_(CS%PFv(isd:ied,JsdB:JedB,nz))   ; CS%PFv(:,:,:)   = 0.0
!$omp target enter data map(to: CS%PFu, CS%PFv)
ALLOC_(CS%eta(isd:ied,jsd:jed))       ; CS%eta(:,:)    = 0.0
ALLOC_(CS%u_av(IsdB:IedB,jsd:jed,nz)) ; CS%u_av(:,:,:) = 0.0
ALLOC_(CS%v_av(isd:ied,JsdB:JedB,nz)) ; CS%v_av(:,:,:) = 0.0
ALLOC_(CS%h_av(isd:ied,jsd:jed,nz))   ; CS%h_av(:,:,:) = GV%Angstrom_H
!$omp target enter data map(to: CS%eta, CS%u_av, CS%v_av, CS%h_av)
```
A second batch (`uhbt`, `visc_rem_u/v`, `pbce`, `eta_PF`, `u_accel_bt/v_accel_bt`) is allocated in
`initialize_dyn_split_RK2` at `:1631-1648`, several with `map(alloc:)` instead of `map(to:)` — a
deliberate choice, flagged by an in-source `TODO`:
```fortran
! MOM_dynamics_split_RK2.F90:1350-1351
! TODO: Are these initializations necessary?  If not, then we can do
!   map(alloc:) rather than map(to:)
```
(i.e. arrays that are always written by a kernel before being read don't need the host-computed
initial zero copied over — `map(alloc:)` skips that copy; arrays read before first write, e.g. `eta`
which may seed itself from `h` on device, need `map(to:)`.)

Bare `pointer` members exist for restart-registry targeting: `real, pointer, dimension(:,:) ::
taux_bot => NULL()` (`:151`), `tauy_bot` (`:153`), and `type(BT_cont_type), pointer :: BT_cont =>
NULL()` (`:155`) (see `02-pointer-usage.md` for why).

Nested children — **mixed by-value and by-pointer in the same type**:
```fortran
! MOM_dynamics_split_RK2.F90:243-264
type(hor_visc_CS)        :: hor_visc            !< by value
type(continuity_CS)      :: continuity_CSp      !< by value (continuity_CS = continuity_PPM_CS, see §3.3)
type(CoriolisAdv_CS)     :: CoriolisAdv          !< by value
type(PressureForce_CS)   :: PressureForce_CSp    !< by value
type(vertvisc_CS), pointer :: vertvisc_CSp      => NULL()  !< pointer
type(set_visc_CS), pointer :: set_visc_CSp      => NULL()  !< pointer
type(barotropic_CS)      :: barotropic_CSp       !< by value
type(SAL_CS)             :: SAL_CSp              !< by value
type(tidal_forcing_CS)   :: tides_CSp            !< by value
type(harmonic_analysis_CS) :: HA_CSp             !< by value
type(ALE_CS), pointer    :: ALE_CSp             => NULL()  !< pointer
```
**Only the children that own device-resident arrays are separately entered** onto the device with
their own `map(alloc:)` immediately before their `_init` call, in `initialize_dyn_split_RK2`
(`:1689-1747`) — namely `continuity_CSp`, `PressureForce_CSp`, `hor_visc`, `barotropic_CSp` (all
by-value) and `vertvisc_CSp` (pointer). **Not every by-value child gets its own map:**
`CoriolisAdv` (`:1692`), `SAL_CSp` (`:1695`), `tides_CSp` (`:1696`) and `HA_CSp` (`:1698`) are
`_init`-ed with **no preceding `map(alloc:)`** (verified: `grep` finds no `map(...CS%CoriolisAdv...)`
etc. anywhere in the module) — they ride along inside the parent's whole-struct
`map(alloc: CS%dyn_split_RK2_CSp)` (`MOM.F90:3259`) or are effectively host-only parameter holders.
```fortran
! MOM_dynamics_split_RK2.F90:1689, 1704, 1708, 1711-1712, 1744 (the mapped children)
!$omp target enter data map(alloc: CS%continuity_CSp)
call continuity_init(Time, G, GV, US, param_file, diag, CS%continuity_CSp, CS%OBC)
call CoriolisAdv_init(...)          ! :1692  — NO map(alloc: CS%CoriolisAdv)
...
!$omp target enter data map(alloc: CS%PressureForce_CSp)
call PressureForce_init(...)
!$omp target enter data map(alloc: CS%hor_visc)
call hor_visc_init(Time, G, GV, US, param_file, diag, CS%hor_visc, ADp=CS%ADp)
allocate(CS%vertvisc_CSp)
!$omp target enter data map(alloc: CS%vertvisc_CSp)
call vertvisc_init(...)
...
!$omp target enter data map (alloc: CS%barotropic_CSp)
call barotropic_init(...)
```
This is the "attach a scalar struct first, let its own `_init` attach its array members second"
pattern discussed in §5 — it is the mechanism, not `ALLOCABLE_` vs plain `allocate`, that determines
whether a member ends up device-resident.

Teardown, `end_dyn_split_RK2` (`:2049-2089`), mirrors this exactly, member-by-member:
```fortran
! MOM_dynamics_split_RK2.F90:2052-2083
!$omp target exit data map(delete: CS%barotropic_CSp)
call barotropic_end(CS%barotropic_CSp)
call vertvisc_end(CS%vertvisc_CSp)
deallocate(CS%vertvisc_CSp)
call hor_visc_end(CS%hor_visc)
!$omp target exit data map(delete: CS%hor_visc)
...
DEALLOC_(CS%diffu) ; DEALLOC_(CS%diffv)
!$omp target exit data map(delete: CS%diffu, CS%diffv)
...
deallocate(CS)
```
(the last `deallocate(CS)` deallocates the pointer `MOM_dyn_split_RK2_CS` object itself, called from
`MOM.F90:4733`: `call end_dyn_split_RK2(CS%dyn_split_RK2_CSp)`).

### 3.3 `continuity_PPM_CS` (aliased `continuity_CS`) (`MOM_continuity_PPM.F90:41-80`) — params only

```fortran
! MOM_continuity_PPM.F90:41-80 (full type)
type, public :: continuity_PPM_CS ; private
  logical :: initialized = .false.
  type(diag_ctrl), pointer :: diag
  logical :: upwind_1st, monotonic, simple_2nd, aggress_adjust, vol_CFL, better_iter, &
             use_visc_rem_max, marginal_faces
  real :: tol_eta, tol_vel, CFL_limit_adjust, h_marg_min
  integer :: niblock         !< The i block size used in array calculations [nondim].
  integer :: njblock         !< The j block size used in array calculations [nondim].
  integer :: nkblock         !< The k block size used in reconstruction calculations [nondim].
end type continuity_PPM_CS
```
Zero array members — nothing to allocate or map. This is the "k-block/tile size" home: `niblock`,
`njblock`, `nkblock` are the CS parameters mentioned in `00-architecture.md` §5, resolved once at
init and read every timestep by `continuity_PPM`/`zonal_mass_flux`/`meridional_mass_flux` to decide
block extents (`if (niblock == 0) niblock = ...`, `:188`). `MOM_continuity.F90:11` aliases the name:
`use MOM_continuity_PPM, only : continuity_CS=>continuity_PPM_CS`. Embedded by value in the parent
(`MOM_dynamics_split_RK2.F90:246`); because it has no arrays it is never itself the target of an
`!$omp target enter data`/`map` directive anywhere in the tree — it's mapped only as part of its
parent's `map(alloc: CS%continuity_CSp)` (a zero-array struct maps almost for free — just its scalar
bytes).

### 3.4 `hor_visc_CS` (`MOM_hor_visc.F90:43-254`) — mixed macro-allocatable and plain-allocatable arrays

Roughly 30 logical/real scalar parameters, then two populations of array members:
```fortran
! MOM_hor_visc.F90:141-153 (macro-allocatable, "standard shape" arrays)
real ALLOCABLE_, dimension(NIMEM_,NJMEM_) :: Kh_bg_xx
real ALLOCABLE_, dimension(NIMEM_,NJMEM_) :: Ah_bg_xx
real ALLOCABLE_, dimension(NIMEM_,NJMEM_) :: reduction_xx
! MOM_hor_visc.F90:145,156-161 (plain, bare allocatable — conditionally-present diagnostics/options)
real, allocatable :: Kh_bg_2d(:,:)
real, allocatable :: Kh_Max_xx(:,:), Ah_Max_xx(:,:), Ah_Max_xx_KS(:,:)
real, allocatable :: n1n2_h(:,:), n1n1_m_n2n2_h(:,:)
```
Both populations are allocated in `hor_visc_init` (`:2867-2966`), e.g.
```fortran
! MOM_hor_visc.F90:2867-2876 (macro form, unconditional, mapped immediately)
ALLOC_(CS%dx2h(isd:ied,jsd:jed))        ; CS%dx2h(:,:)    = 0.0
...
!$omp target enter data map(alloc: CS%dx2h, CS%dy2h, CS%dx2q, CS%dy2q)
!$omp target enter data map(alloc: CS%dx_dyT, CS%dy_dxT, CS%dx_dyBu, CS%dy_dxBu)
! :2883-2884 (plain form, conditional on a runtime flag)
allocate(CS%Kh_Max_xx(Isd:Ied,Jsd:Jed), source=0.0)
allocate(CS%Kh_Max_xy(IsdB:IedB,JsdB:JedB), source=0.0)
```
**The macro-vs-plain choice does not itself decide device residency** — both populations get mapped
explicitly when a kernel needs them. `CS%Kh_Max_xx` (bare `allocatable`, never touched by
`ALLOCABLE_`) is entered onto device later in the same routine:
```fortran
! MOM_hor_visc.F90:3432-3433
!$omp target enter data map(to: CS%Kh_max_xx) if (CS%Laplacian)
!$omp target enter data map(to: CS%Kh_max_xy) &
```
The real distinction: macro-allocatable members are the *unconditional*, "always exists, always
canonical (isd:ied)-shaped" arrays declared inline in the type using the `SZ*`-family conventions;
plain-`allocatable` members are typically **conditionally allocated** (only if a particular
Smagorinsky/Leith/anisotropic/Zanna-Bolton option is on) so `allocated(CS%x)` doubles as both the
memory-presence flag and the physics on/off flag — using `ALLOCABLE_` (which is a no-op attribute
tweak in static mode, meaningless for a conditionally-present array) would not fit that dual-purpose
usage, so these are always genuine plain Fortran `allocatable`, checked with `if (allocated(...))`
at deallocation (`hor_visc_end`, `:3701-3729`: `if (allocated(CS%Kh_Max_xx)) deallocate(CS%Kh_Max_xx)`).

### 3.5 `vertvisc_CS` (`MOM_vert_friction.F90:56-196`) — mostly scalars plus interface-staggered arrays

```fortran
! MOM_vert_friction.F90:100-111
real ALLOCABLE_, dimension(NIMEMB_PTR_,NJMEM_,NK_INTERFACE_) :: a_u        !< u-drag coeff at interfaces
real ALLOCABLE_, dimension(NIMEMB_PTR_,NJMEM_,NK_INTERFACE_) :: a_u_gl90
real ALLOCABLE_, dimension(NIMEMB_PTR_,NJMEM_,NKMEM_)        :: h_u        !< effective layer thickness
real ALLOCABLE_, dimension(NIMEM_,NJMEMB_PTR_,NK_INTERFACE_) :: a_v, a_v_gl90
real ALLOCABLE_, dimension(NIMEM_,NJMEMB_PTR_,NKMEM_)        :: h_v
real, pointer, dimension(:,:) :: a1_shelf_u => NULL()  !< pointer: restart target for ice-shelf coupling
real, pointer, dimension(:,:) :: a1_shelf_v => NULL()
...
type(PointAccel_CS), pointer :: PointAccel_CSp => NULL()  !< child CS, pointer
```
`vertvisc_CS` is itself a **pointer** member of `MOM_dyn_split_RK2_CS` (`vertvisc_CSp`,
`MOM_dynamics_split_RK2.F90:252`), `allocate`d explicitly (`allocate(CS%vertvisc_CSp)`,
`:1711`) rather than being a plain embedded value like `hor_visc`/`continuity_CSp` — because
`vertvisc_init` needs to hand a stable address to `PointAccel`/restart registrations that outlive the
call, and (per `02-pointer-usage.md`) the module wanted an explicit-lifetime object rather than an
implicitly-copied value component.

### 3.6 Summary: allocatable/pointer choice per CS, parent relationship

| CS type | Member-in-parent kind | Parent | Array members | Alloc site (parent map) |
|---|---|---|---|---|
| `MOM_control_struct` | plain value (driver-owned) | `MOM_driver.F90:174` | macro (`h,T,S,u,v,...`) | `MOM_driver.F90:282` (`map(alloc: MOM_CSp)`) |
| `MOM_dyn_split_RK2_CS` | `pointer` | `MOM_control_struct` | macro (`CAu,PFu,diffu,...`) | `MOM.F90:3258-3259` |
| `continuity_PPM_CS` | embedded value | `MOM_dyn_split_RK2_CS` | none | `MOM_dynamics_split_RK2.F90:1689` |
| `hor_visc_CS` | embedded value | `MOM_dyn_split_RK2_CS` | macro + plain mix | `MOM_dynamics_split_RK2.F90:1708` |
| `barotropic_CS` | embedded value | `MOM_dyn_split_RK2_CS` | (large, not detailed here) | `MOM_dynamics_split_RK2.F90:1744` |
| `vertvisc_CS` | `pointer` | `MOM_dyn_split_RK2_CS` | macro (interface-staggered) | `MOM_dynamics_split_RK2.F90:1711-1712` |
| `set_visc_CS` | `allocatable` (was embedded value; changed by `81680c15d`/`c82e1254a`) | `MOM_control_struct` | mostly scalar/diag | `MOM.F90:3710` (after `c82e1254a`) |
| `vertvisc_type` (`visc`) | `allocatable` (was embedded value; changed by `81680c15d`) | `MOM_control_struct` | mixed allocatable/pointer, see §4 | `MOM.F90:3277-3278` |

---

## 4. `MOM_variables.F90` — the shared "bag of state" containers

### 4.1 `thermo_var_ptrs` (`:79-131`)

```fortran
! MOM_variables.F90:81-98 (excerpt)
real, pointer :: T(:,:,:) => NULL()        !< potential temperature — POINTER
real, pointer :: S(:,:,:) => NULL()        !< salinity — POINTER
real, pointer :: p_surf(:,:) => NULL()     !< POINTER (conditionally allocated)
type(EOS_type), pointer :: eqn_of_state => NULL()
...
real, allocatable, dimension(:,:,:) :: SpV_avg   !< genuinely ALLOCATABLE (no external alias needed)
```
`T`/`S` are pointers purely so `CS%tv%T => CS%T` (`MOM.F90:3119`) can alias the *same* physical memory
as the dycore's macro-allocatable `CS%T`; there is no separate allocation for `tv%T` — it is set once
`CS%T` exists. `p_surf`/`frazil`/`salt_deficit` are allocated conditionally with plain
`allocate(..., source=0.0)` (`MOM.F90:3165-3170`: `if (use_p_surf_in_EOS) allocate(CS%tv%p_surf(isd:ied,jsd:jed), source=0.0)`)
— pointer, not macro-allocatable, precisely because they're conditional and because
`register_restart_field` needs a stable `target`.

### 4.2 `vertvisc_type` (`:258-313`) — the textbook hybrid, with the exact cited comment

```fortran
! MOM_variables.F90:259-292 (allocatable drag/BBL fields)
real, allocatable, dimension(:,:) :: &
  bbl_thick_u, bbl_thick_v, kv_bbl_u, kv_bbl_v, ustar_BBL, &
  BBL_meanKE_loss, BBL_meanKE_loss_sqrtCd, taux_shelf, tauy_shelf
real, allocatable, dimension(:,:,:) :: Ray_u, Ray_v

! MOM_variables.F90:294-312 (pointer fields, WITH THE EXACT COMMENT)
! The following elements are pointers so they can be used as targets for pointers in the restart registry.
real, pointer, dimension(:,:)   :: MLD => NULL()
real, pointer, dimension(:,:)   :: h_ML => NULL()
real, pointer, dimension(:,:)   :: sfc_buoy_flx => NULL()
real, pointer, dimension(:,:,:) :: Kd_shear => NULL()
real, pointer, dimension(:,:,:) :: Kv_shear => NULL()
real, pointer, dimension(:,:,:) :: Kv_shear_Bu => NULL()
real, pointer, dimension(:,:,:) :: Kv_slow  => NULL()
real, pointer, dimension(:,:,:) :: TKE_turb => NULL()
```
Allocated via `safe_alloc_ptr` (`MOM_safe_alloc.F90:72-94`, a guarded `if (.not.associated(ptr))
allocate(ptr(...), source=0.0)` helper) in `set_visc_register_restarts`
(`MOM_set_viscosity.F90:2899-2901`: `call safe_alloc_ptr(visc%Kv_shear, isd, ied, jsd, jed, nz+1)`).
Device mapping of `Kv_shear`/`Kv_shear_Bu` is *not* colocated with the allocation — it happens later,
inside `set_visc_init` (after the surrounding `visc` struct itself has been `target update`d), per the
fix in `c82e1254a` (§6.2).

### 4.3 `accel_diag_ptrs` (`:167-238`) / `cont_diag_ptrs` (`:241-255`)

Both are **all-pointer** diagnostic-alias structs — no allocation of their own; every member is
pointed at an array owned elsewhere (`Accel_diag%diffu => CS%diffu`, `MOM_dynamics_split_RK2.F90:1664`
etc.). `CS%ADp` in `MOM_control_struct` is `allocatable` (`:340`, the struct itself is allocated,
`allocate(CS%ADp)`, `MOM.F90:3189`) but every field *inside* the allocated struct is a bare pointer
aliasing someone else's macro-allocatable array — the struct exists purely for the diagnostics layer
to hold one handle instead of a dozen.

### 4.4 `BT_cont_type` (`:317-352`) — all genuinely allocatable

```fortran
! MOM_variables.F90:318-347 (representative)
real, allocatable :: FA_u_EE(:,:), FA_u_E0(:,:), FA_u_W0(:,:), FA_u_WW(:,:)
real, allocatable :: uBT_WW(:,:), uBT_EE(:,:)
real, allocatable :: h_u(:,:,:), h_v(:,:,:)
type(group_pass_type) :: pass_polarity_BT, pass_FA_uv
```
No pointers at all — `BT_cont_type` is never a restart-registry target and never aliased by another
module, so there is no forcing reason for pointer semantics; `alloc_BT_cont_type`/
`dealloc_BT_cont_type` (`MOM_variables.F90:567-639`) do plain `allocate(BT_cont%FA_u_EE(...))`. The
*owning* member, however, is a pointer: `type(BT_cont_type), pointer :: BT_cont => NULL()`
(`MOM_dynamics_split_RK2.F90:155`) — because `BT_cont` is conditionally allocated (only if
`marginal_faces`/certain barotropic options are set) and is handed out to several call sites
(`horizontal_viscosity(..., hu_cont=CS%BT_cont%h_u, ...)`, `:1754`) that need `associated()` gating.

---

## 5. The CS object graph — nesting tree with allocation-site line numbers

```
MOM_CSp : type(MOM_control_struct)                     ! plain value, driver-owned
          MOM_driver.F90:174 (declared)
          MOM_driver.F90:282  !$omp target enter data map(alloc: MOM_CSp)   <- BEFORE initialize_MOM runs
│
├─ CS%G_in       : ocean_grid_type, allocatable         MOM.F90:233 (decl) / :2985 allocate / :3034 map(to:)
├─ CS%G          : ocean_grid_type, pointer  => G_in or rotated copy        MOM.F90:234
├─ CS%GV         : verticalGrid_type, pointer            MOM_verticalGrid.F90:124 allocate(GV)
├─ CS%tv         : thermo_var_ptrs, allocatable           MOM.F90:244 (decl) / :2564 allocate(CS%tv)
├─ CS%visc       : vertvisc_type, allocatable             MOM.F90:261 (decl) / :3277 allocate / :3278 map(alloc:)
├─ CS%ADp        : accel_diag_ptrs, allocatable           MOM.F90:340 (decl) / :3189 allocate / :3190 map(alloc:)
├─ CS%CDp        : cont_diag_ptrs, embedded value         MOM.F90:342
├─ CS%h,T,S,u,v,uh,vh,uhtr,vhtr : macro-allocatable        MOM.F90:205-216 (decl) / :3110-3185 ALLOC_+map(to:)
├─ CS%set_visc_CSp : set_visc_CS, allocatable              MOM.F90:420 (decl) / :3710ish allocate (post c82e1254a)
└─ CS%dyn_split_RK2_CSp : MOM_dyn_split_RK2_CS, pointer    MOM.F90:407 (decl)
   │                                                        MOM.F90:3258 allocate / :3259 map(alloc:)
   │
   ├─ CS%CAu,PFu,diffu,eta,u_av,h_av,... : macro-allocatable
   │      MOM_dynamics_split_RK2.F90:90-147 (decl)
   │      MOM_dynamics_split_RK2.F90:1352-1368, 1631-1648 (ALLOC_ + map(to:)/map(alloc:))
   │
   ├─ CS%hor_visc          : hor_visc_CS, embedded value
   │      MOM_dynamics_split_RK2.F90:244 (decl) / :1708 map(alloc:) / hor_visc_init allocates its own arrays
   │
   ├─ CS%continuity_CSp    : continuity_CS(=continuity_PPM_CS), embedded value
   │      MOM_dynamics_split_RK2.F90:246 (decl) / :1689 map(alloc:)  — params only, no arrays
   │
   ├─ CS%CoriolisAdv       : CoriolisAdv_CS, embedded value
   │      MOM_dynamics_split_RK2.F90:248 (decl)
   │
   ├─ CS%PressureForce_CSp : PressureForce_CS, embedded value
   │      MOM_dynamics_split_RK2.F90:250 (decl) / :1704 map(alloc:)
   │
   ├─ CS%barotropic_CSp    : barotropic_CS, embedded value
   │      MOM_dynamics_split_RK2.F90:256 (decl) / :1744 map(alloc:)
   │
   ├─ CS%vertvisc_CSp      : vertvisc_CS, pointer
   │      MOM_dynamics_split_RK2.F90:252 (decl) / :1711 allocate / :1712 map(alloc:)
   │
   └─ CS%set_visc_CSp      : set_visc_CS, pointer (this module's own alias, distinct object from CS%set_visc_CSp above)
          MOM_dynamics_split_RK2.F90:254 (decl) — set via `CS%set_visc_CSp => set_visc` (:1715), NOT separately allocated
```

Teardown mirrors this tree exactly in reverse, member-by-member (`end_dyn_split_RK2`,
`MOM_dynamics_split_RK2.F90:2049-2089`; `MOM_end`, `MOM.F90:4698-4802`), each child's own `_end` routine
called first, then its `!$omp target exit data map(delete: ...)`, then (where allocatable/pointer)
the Fortran `deallocate`.

---

## 6. Three commits that shaped this model

### 6.1 `81680c15d` — "Allocate MOM CS and both viscosity CS on GPU"

Root cause: nvfortran was raising ambiguous **"partial presence"** errors on fields of the top-level
`CS` in code paths downstream of `MOM.F90` — i.e. some member of `MOM_control_struct` was present on
device while a sibling member (needed by the same kernel/region) was not, and the compiler could not
reconcile the mixed presence state of a single struct. The fix: change **two** members of
`MOM_control_struct` from plain-embedded-value to `allocatable` (verified against the commit diff —
`git show 81680c15d` touches only these two type declarations, plus `MOM_driver.F90` and
`MOM_set_viscosity.F90`), decoupling their device presence from the parent struct's:
```fortran
! MOM.F90 diff (81680c15d)
-  type(ocean_grid_type) :: G_in                   !< Input grid metric
+  type(ocean_grid_type), allocatable  :: G_in     !< Input grid metric
...
-  type(vertvisc_type) :: visc !< structure containing vertical viscosities, ...
+  type(vertvisc_type), allocatable :: visc !< ...
```
(`set_visc_CSp` was **not** made `allocatable` here — that came one commit later in `c82e1254a`,
§6.2; this commit only *maps* `set_visc_CSp` onto the device, it does not change its declaration.)

> **Resolved (2026-07-14):** "Partial presence" is the literal NVIDIA runtime diagnostic, not a
> reconstruction. The NVHPC OpenMP/OpenACC runtime raises a FATAL "partially present" error when a
> mapping's address range partially overlaps an existing present-table entry — exactly the
> whole-struct-over-attached-member overlap described here. Rely on this mechanism as a general rule.
plus explicit `allocate(CS%G_in)` (before `G_in => CS%G_in`) and `allocate(CS%visc)` before their
respective `map(to:)`/`map(alloc:)` directives, and a whole-struct `!$omp target update to(CS)` right
after the grid upload (`MOM.F90:3104`, new in this commit) — which the commit message flags as breaking the project's
own derived-type handling rule ("we do an update(CS) after the grid has been uploaded... this needs
exploration") but was needed and apparently harmless once `G_in`/`visc` were decoupled as allocatables.
The commit further notes the `G_in`→allocatable change specifically fixed a **performance** problem
("excessive grid transfers... severely degrading performance"), distinct from the correctness
"partial presence" issue that motivated `visc`.

### 6.2 `c82e1254a` — "vertvisc: Fix CS memory management"

A regression from `81680c15d`: `CS%visc` was being **allocated twice** — once implicitly by giving it
the `allocatable` attribute plus an early `allocate(CS%visc)`, and (unclear from the single commit,
but per the message) a second time overwriting the pointers `visc%Kv_shear`/`visc%Kv_shear_Bu` that
`set_visc_register_restarts` had already set up, corrupting `associated()` state used for flow control
in device kernels — causing addressing errors in `double_gyre` runs. Fix, concretely:
- `CS%set_visc_CSp` changed from embedded value to `allocatable` (`MOM.F90` diff:
  `-  type(set_visc_CS)  :: set_visc_CSp` / `+  type(set_visc_CS), allocatable :: set_visc_CSp`),
  matching the same pattern used for `G_in`/`visc` in the prior commit ("This may not be needed, but
  it is consistent with other types").
- The single combined `!$omp target enter data map(to: CS%visc, CS%set_visc_CSp)` before
  `set_visc_init` was split: `CS%visc` is now updated with `!$omp target update to(visc)` **inside**
  `set_visc_init` (`MOM_set_viscosity.F90:3322-3324`), and `CS%set_visc_CSp` gets its own
  `allocate(CS%set_visc_CSp)` + `map(alloc:)` in `MOM.F90` immediately before the call.
- The device `map(alloc:)`/`map(to:)` for `visc%Kv_shear`/`visc%Kv_shear_Bu` was **moved out of**
  `set_visc_register_restarts` (removed the early `!$omp target enter data map(alloc: visc%Kv_shear)`
  right after `safe_alloc_ptr`) and **into** `set_visc_init`, applied only *after* the `target update
  to(visc)` scalar sync:
  ```fortran
  ! MOM_set_viscosity.F90:3322-3330 (post-fix)
  !$omp target update to(visc)
  !$omp target update to(CS)
  !$omp target enter data map(to: visc%Kv_shear)    if (associated(visc%Kv_shear))
  !$omp target enter data map(to: visc%Kv_shear_Bu) if (associated(visc%Kv_shear_Bu))
  ```
- In `MOM_vert_friction.F90`'s `vertvisc_coef`, the stale in-source comment explaining that
  `Kv_shear` is "persistently mapped... so map(to:) would not copy host updates" was removed along
  with the workaround it justified; `Kv_shear_Bu` now uses `!$omp target update to(visc%Kv_shear_Bu)`
  (was: a redundant `map(to:)` re-enter-data every call) matching how `Kv_shear` was already handled,
  and the matching `map(release: visc%Kv_shear_Bu)` at exit was deleted (no longer needed since it's
  not separately entered per-call).

This pair of commits establishes the working rule for this codebase (visible nowhere as a written
rule except in these two commit messages): **any CS-type field that participates in cross-call
`associated()`-gated flow control, or whose presence must be decoupled from a sibling field's
presence, should be declared `allocatable` (not an embedded value) in its parent**, so its device
lifetime can be managed independently with its own `map(alloc:)`/`allocate()` pair rather than being
swept up in whatever bulk-struct directive covers its parent.

### 6.3 `1865612de` — "Convert structs of arrays to flat arrays"

Scope: `src/tracer/MOM_tracer_hor_diff.F90` only (`tracer_epipycnal_ML_diff`, `:757` region). Before:
```fortran
! pre-commit (struct-of-arrays, one small array per j-index)
type(p2d), dimension(SZJ_(G)) :: deep_wt_Lu, deep_wt_Ru, hP_Lu, hP_Ru
...
do j=js,je
  k_size = max(2*max_srt(j),1)
  allocate(deep_wt_Lu(j)%p(IsdB:IedB,k_size))
  ...
  !$omp target enter data map(alloc: deep_wt_Lu(J)%p, deep_wt_Ru(J)%p, hP_Lu(J)%p, hP_Ru(J)%p, &
  !$omp   k0a_Lu(j)%p, k0a_Ru(j)%p, k0b_Lu(j)%p, k0b_Ru(j)%p)
enddo
```
After: a single flat `real, dimension(:,:,:), allocatable :: deep_wt_Lu, ...` sized once
(`k_size = max over all j of 2*max_srt(j)`, via a `do concurrent ... DO_LOCALITY(reduce(max:k_size))`
reduction) and allocated/entered **once**:
```fortran
allocate(k0a_Lu(IsdB:iedB,k_size,jsd:jed))
...
!$omp target enter data map(alloc: deep_wt_Lu, deep_wt_Ru, hP_Lu, hP_Ru, k0a_Lu, k0a_Ru, k0b_Lu, k0b_Ru)
```
The commit message is explicit about the mechanism: each `type(p2d)`/`type(p2di)` element was a
**separate derived-type instance with its own pointer-array component**, so mapping `SZJ_(G)`
(`O(njglobal)`) of them onto the device meant `O(njglobal)` individual "attach the array component to
this struct instance" operations — each one a distinct OpenMP mapping bookkeeping entry — rather than
one attach of one flat array. "Flattening these arrays halves time when compiling for GPU. Lots of
time was being spent 'attaching' and 'detaching' the member arrays to/from each struct on the GPU."
Tradeoff acknowledged in the message: ~20% more memory (padding every j-slice out to the same
`k_size`, versus each j's exact `2*max_srt(j)`), traded for the attach/detach overhead reduction. This
is the same underlying cost (`00-architecture.md` calls it "the offload unit is the whole CS object
with its allocatable members... deep-copy of derived-type member arrays is expensive") observed at a
finer grain — even without a big CS, an array *of small derived-type instances each holding an array*
pays the same per-instance attach tax, and the fix generalizes: prefer one flat array with an extra
dimension over an array of one-member-array structs whenever the outer array's extent is a loop-index
range rather than a true type/kind distinction.

---

## 7. Practical checklist derived from the above

1. Deciding embedded-value vs `allocatable` vs `pointer` for a new child CS field:
   - No arrays, always exists → embedded value is fine (`continuity_PPM_CS` pattern, §3.3).
   - Has arrays, always exists for the life of the parent, no `associated()` gating needed elsewhere
     → embedded value still works (`hor_visc`, `barotropic_CSp`), mapped via the parent's
     `map(alloc:)` + its own `_init`'s per-array `map(to:)`/`map(alloc:)`.
   - Conditionally allocated, or needs independent device-presence lifetime from its parent, or is a
     restart-registry target → `allocatable` (post-`81680c15d`/`c82e1254a` idiom) or `pointer` (older
     idiom, still used for `vertvisc_CSp`/`set_visc_CSp` *inside* `MOM_dyn_split_RK2_CS`, as opposed
     to the top-level `MOM_control_struct` copies which are now `allocatable`).
2. Every macro-allocatable array needs its own `ALLOC_`/zero-init/`!$omp target enter data
   map(to:|alloc:)` triplet at init and matching `DEALLOC_`/`map(delete:)` at `_end` — see
   `00-architecture.md` §9's "Map a new CS array" recipe; this document's §3 gives the concrete
   line-referenced exemplars to copy from.
3. Avoid arrays of small derived-type instances that each carry an array component (the `type(p2d),
   dimension(SZJ_(G))` anti-pattern) when the outer index is a loop range, not a real type
   distinction — flatten to one array with an extra dimension (`1865612de`).
4. Symmetric-memory offsets are a runtime `ALLOC_`-call-site concern in dynamic mode, not a macro
   concern — always compute B-point bounds from `hor_index_type`/`ocean_grid_type`'s `IsdB`/`JsdB`
   etc. (which already encode the `-1` symmetric offset), never hand-roll `isd-1`.

### 7.1 Prescriptive rules for adding GPU-resident CS state (grounded in cited code)

These are the load-bearing rules distilled from §§3–6; follow them when introducing a new CS member
or a new child CS on `dev/gpu`.

- **Declare persistent 3-D/2-D work arrays as macro-allocatable, not `pointer`.** Use
  `real ALLOCABLE_, dimension(SZ*-family macros) :: x` (e.g. `MOM_dynamics_split_RK2.F90:90`,
  `MOM.F90:205`). Reserve `pointer` **only** for arrays that must be a `target` in the restart
  registry or aliased across modules (the `MOM_variables.F90:294` comment — *"pointers so they can be
  used as targets for pointers in the restart registry"* — is the canonical justification; e.g.
  `visc%Kv_shear`, `taux_bot`). Pointers complicate device mapping and force `associated()`-gated
  paths; do not reach for them by default.
- **Declare a new *child CS* as `allocatable`, not embedded-value, unless it is trivially small and
  always-present.** Embedded-value is acceptable only for the zero-array / small-scalar case
  (`continuity_PPM_CS`, §3.3) or an always-present array-owning child whose presence never needs to be
  decoupled from a sibling's (`hor_visc`, `barotropic_CSp`). The moment a child is **conditionally
  allocated**, needs an **independent device-presence lifetime**, or participates in cross-call
  **`associated()`-gated flow control**, make it `allocatable` — this is the explicit lesson of
  `81680c15d` (`G_in`, `visc`) and `c82e1254a` (`set_visc_CSp`). A top-level embedded-value struct is
  an anti-pattern precisely because its device presence gets entangled with the parent's (the
  "partial presence" failure, §6.1).
- **Order at init: allocate/attach the parent *shell* before its members.** The parent CS is entered
  with `map(alloc:)` (scalars only) *before* its `_init` runs, and each array member is entered by its
  own `_init` afterwards. Concretely: `MOM_CSp` is `map(alloc:)`-ed in the driver
  (`MOM_driver.F90:282`) **before** `initialize_MOM` (`:286`); `CS%dyn_split_RK2_CSp` is
  `allocate`+`map(alloc:)`-ed (`MOM.F90:3258-3259`) **before** its arrays are allocated in
  `register_restarts_dyn_split_RK2`/`initialize_dyn_split_RK2`. Never map a member array before the
  struct that contains it exists on device.
- **Co-locate the map with the allocate, and mirror it in `_end`.** Each macro-allocatable array gets
  an `ALLOC_(...)` + zero-init + `!$omp target enter data map(to:|alloc:)` triplet in `*_init`
  (`MOM_dynamics_split_RK2.F90:1352-1368`) and a matching `DEALLOC_(...)` + `!$omp target exit data
  map(delete: ...)` in `*_end` (`:2065-2083`), member-by-member in reverse order. Choose `map(to:)`
  only when the host-computed initial value is read before the first device write; otherwise
  `map(alloc:)` skips the copy (the in-source `TODO` at `:1350-1351` documents exactly this choice).
  **Exception (the `c82e1254a` rule):** for a `pointer` array whose `associated()` state is set up in a
  *separate* registration routine (`safe_alloc_ptr`), do **not** map it at the allocation site — map it
  later, *after* a `!$omp target update to(<parent struct>)`, guarded by `if (associated(...))`
  (`MOM_set_viscosity.F90:3323-3330`). Mapping it early double-allocates and corrupts the pointer state.
- **Never build an array of small derived types that each own an array component when the outer index
  is a loop range** (`type(p2d), dimension(SZJ_(G))`, §6.3) — every instance pays a separate device
  attach/detach. Flatten to one `allocatable` array with an extra dimension (`1865612de`). This is the
  same "deep-copy of derived-type member arrays is expensive" cost the whole-CS model pays, seen at
  finer grain.
- **`GV` is a genuine exception to "map every array."** Only `GV`, `GV%Rlay`, `GV%g_prime` are mapped
  (`MOM.F90:3650`), and only after rescaling; the rest of the vertical grid reaches kernels by-value.
  Do not assume vertical-grid arrays are device-resident by default (see the §2 correction).

---

## Verification notes

Opus verification pass (source + git only; no build/run). Every line number and macro expansion in
this document was checked against the current `dev/gpu` tree unless flagged below.

**Confirmed against code/git:**
- All `MOM_memory_macros.h` expansions and line ranges: attribute macros (`:101-110`), heap-shape
  macros (`:114-161`), the `NIMEMB_PTR_`→`:` (dynamic, `:122,125`) vs `NIMEMB_PTR_`→`NIMEMB_` (static,
  `:53,56`) subtlety, `SZ*` family (`:167-182`), `SZD*` (`:189-195`). The document's central claim —
  that dynamic-mode symmetric offsets live in the runtime `ALLOC_` bounds (`IsdB`/`JsdB`), not the type
  declaration, while static mode bakes them into the macro — is correct.
- `config_src/memory/{dynamic_symmetric,dynamic_nonsymmetric}/MOM_memory.h`: byte-identical except the
  `SYMMETRIC_MEMORY_` `#define`/`#undef` at `:37`; both `#undef STATIC_MEMORY_` (`:41`), `NIHALO_=2`.
- `hor_index_init` B-bound logic (`MOM_hor_index.F90:89-99`), `MOM_grid_init` mirror (`:303-313`),
  `allocate_metrics` ALLOC_ sites (`:547-548`).
- `MOM_control_struct` layout and every allocation/map site in `MOM.F90`: `G_in` (`:233`/`:2985`/`:3034`),
  `visc` (`:261`/`:3277-3278`), `tv` (`:244`/`:2564`), `ADp` (`:340`/`:3189-3190`),
  `dyn_split_RK2_CSp` (`:407`/`:3258-3259`), `set_visc_CSp` (`:420`/`:3709-3711`), prognostic state
  (`:205-216`/`:3110-3115`), `tv%T => CS%T` (`:3119`), `p_surf` (`:3165`), `map(alloc: MOM_CSp)` before
  `initialize_MOM` (`MOM_driver.F90:282` before `:286`).
- `MOM_dyn_split_RK2_CS` full member list and allocation/teardown (`:89-281`, `:1352-1368`,
  `:1631-1648`, `:1689-1747`, `:2049-2089`, module doc `:2113-2116`).
- `continuity_PPM_CS` (`:41-80`, zero arrays), `vertvisc_CS` arrays (`:100-111`), `verticalGrid_type`
  arrays and `verticalGridInit`/`End` (`:106-247`/`:361-368`).
- `MOM_variables.F90`: `thermo_var_ptrs` (`:79`), `vertvisc_type` with the exact restart-registry
  comment at `:294`, `accel_diag_ptrs` (`:167`), `cont_diag_ptrs` (`:241`), `BT_cont_type` (`:317`),
  `safe_alloc_ptr` (`MOM_safe_alloc.F90`, `if (.not.associated(ptr))` guard), `set_visc` post-fix
  block (`MOM_set_viscosity.F90:3322-3330`).
- All three commit narratives (`81680c15d`, `c82e1254a`, `1865612de`) match their commit messages and
  diffs, including the "partial presence" motivation, the double-allocation/`associated()`-corruption
  regression fix, and the `~700k→~830k (~20%)` memory tradeoff. `1865612de` scope confirmed as
  `MOM_tracer_hor_diff.F90` only.

**Corrected:**
1. **§2 (major):** the claim that no `GV` member is ever mapped and "it never appears in an `!$omp
   target enter data` list" is **false**. `MOM.F90:3650` has an active `map(to: GV, GV%Rlay,
   GV%g_prime)`, with an instructive commented-out earlier attempt at `:3057`. Rewritten with the
   correct facts.
2. **§3.2:** the blanket claim that *"each by-value child is separately entered with its own
   `map(alloc:)`"* overstated the code — `CoriolisAdv`, `SAL_CSp`, `tides_CSp`, `HA_CSp` are by-value
   children with **no** preceding `map(alloc:)` (`grep`-verified). Only `continuity_CSp`,
   `PressureForce_CSp`, `hor_visc`, `barotropic_CSp` (+ pointer `vertvisc_CSp`) are individually mapped.
3. **§6.1:** "change **three** members … to `allocatable`" corrected to **two** (`G_in`, `visc`); the
   `git show 81680c15d` diff changes only those two declarations. `set_visc_CSp` became `allocatable`
   in `c82e1254a`, not here.
4. **Minor line numbers:** `MOM_end` subroutine starts at `:4698` (was cited as `:4680`); the three
   `DEALLOC_(CS%u/v/h)` are all on `:4778` (was `:4778-4779`); `NK_INTERFACE_` reclassified in the
   table from "dummy-arg form" to a heap-shape macro (`NK_+1` in static mode).

**Confidence:** High. Nearly all line numbers and macro/expansion details were verified exact against
the working tree; the few discrepancies were small offsets (now fixed) plus the two substantive
overstatements above (GV mapping, by-value-child mapping) and one miscount (two vs three members).
The subtle symmetric-memory / `ALLOC_`-site vs macro-expansion story — the document's core technical
thesis — is fully correct.
