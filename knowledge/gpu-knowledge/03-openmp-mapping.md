# OpenMP Target Data-Mapping Infrastructure (dev/gpu)

> **Purpose.** Inventory of the *proven-works* patterns for making MOM6 control-structure (CS)
> members and local scratch arrays device-resident under OpenMP target offload. This is a "what
> works" catalogue for extending the port to new modules — not a tutorial on OpenMP itself. Read
> `00-architecture.md` §2, §6, §7.3 first.

**Directive counts across `src/` + `config_src/`** (reproduced by
`grep -rn "omp target enter data\|omp target exit data\|omp target update\|omp declare target\|map(" src/ config_src/`):

| Directive | Count |
|---|---|
| `!$omp target enter data` | 213 |
| `!$omp target exit data` | 168 |
| `!$omp target update` | 398 |
| `!$omp declare target` | 21 |

---

## 1. The canonical CS-member lifecycle

The dominant, hand-written idiom co-locates the mapping directive directly next to the
`ALLOC_`/`DEALLOC_` macro call (`MOM_memory_macros.h`) that creates/destroys the host array. There
is **no wrapper macro or subroutine** for this — see §4.

### 1.1 Allocate → enter data (init routine)

`src/core/MOM_dynamics_split_RK2.F90:1350-1368` (`register_restarts_dyn_split_RK2`):

```fortran
! TODO: Are these initializations necessary?  If not, then we can do
!   map(alloc:) rather than map(to:)
ALLOC_(CS%diffu(IsdB:IedB,jsd:jed,nz)) ; CS%diffu(:,:,:) = 0.0
ALLOC_(CS%diffv(isd:ied,JsdB:JedB,nz)) ; CS%diffv(:,:,:) = 0.0
!$omp target enter data map(to: CS%diffu, CS%diffv)
ALLOC_(CS%CAu(IsdB:IedB,jsd:jed,nz))   ; CS%CAu(:,:,:)   = 0.0
ALLOC_(CS%CAv(isd:ied,JsdB:JedB,nz))   ; CS%CAv(:,:,:)   = 0.0
!$omp target enter data map(to: CS%CAu, CS%CAv)
...
ALLOC_(CS%eta(isd:ied,jsd:jed))       ; CS%eta(:,:)    = 0.0
ALLOC_(CS%u_av(IsdB:IedB,jsd:jed,nz)) ; CS%u_av(:,:,:) = 0.0
ALLOC_(CS%v_av(isd:ied,JsdB:JedB,nz)) ; CS%v_av(:,:,:) = 0.0
ALLOC_(CS%h_av(isd:ied,jsd:jed,nz))   ; CS%h_av(:,:,:) = GV%Angstrom_H
!$omp target enter data map(to: CS%eta, CS%u_av, CS%v_av, CS%h_av)
```

Note the host-side zero-init (`CS%diffu(:,:,:) = 0.0`) *before* the map — `map(to:)` copies that
initialized value up. The in-source `TODO` at line 1350 flags that this is sometimes wasted work
where `map(alloc:)` (no copy) would do — see §3 kind-selection rules.

### 1.2 Matching exit data → deallocate (`*_end` routine)

`src/core/MOM_dynamics_split_RK2.F90:2065-2083` (`end_dyn_split_RK2`) — directives appear in
**reverse order** of the enter-data calls, immediately paired with `DEALLOC_`:

```fortran
DEALLOC_(CS%diffu) ; DEALLOC_(CS%diffv)
!$omp target exit data map(delete: CS%diffu, CS%diffv)
DEALLOC_(CS%CAu)   ; DEALLOC_(CS%CAv)
!$omp target exit data map(delete: CS%CAu, CS%CAv)
DEALLOC_(CS%CAu_pred) ; DEALLOC_(CS%CAv_pred)
!$omp target exit data map(delete: CS%CAu_pred, CS%CAv_pred)
DEALLOC_(CS%PFu)   ; DEALLOC_(CS%PFv)
!$omp target exit data map(delete: CS%PFu, CS%PFv)
...
DEALLOC_(CS%eta) ; DEALLOC_(CS%eta_PF) ; DEALLOC_(CS%pbce)
!$omp target exit data map(delete: CS%eta, CS%eta_PF, CS%pbce)
DEALLOC_(CS%h_av) ; DEALLOC_(CS%u_av) ; DEALLOC_(CS%v_av)
!$omp target exit data map(delete: CS%u_av, CS%v_av, CS%h_av)
```

This enter/exit pair (register-restarts ↔ end) is the module-lifetime idiom. A second, shorter-lived
idiom wraps a **single subroutine call**: allocate/enter-data at entry, exit-data/deallocate at
return, for pure scratch (non-CS) arrays — see §1.3.

### 1.3 Local-scratch (non-CS, subroutine-scoped) lifecycle

Two flavours:

- **Stack-declared automatic arrays**, entered/exited around their live range inside one routine —
  `src/core/MOM_continuity_PPM.F90:181` / `:228` (`continuity_PPM`):

  ```fortran
  !$omp target enter data map(alloc: h_W, h_E, h_S, h_N)
  ... ! zonal/meridional edge reconstruction + mass flux calls
  !$omp target exit data map(delete: h_W, h_E, h_S, h_N)
  end subroutine continuity_PPM
  ```

  and `MOM_continuity_PPM.F90:658-660` (`zonal_mass_flux`, multi-line continuation):

  ```fortran
  !$omp target enter data &
  !$omp   map(alloc:uhbt_t,uh_t,duhdu,du,du_min_CFL,du_max_CFL,duhdu_tot_0,uh_tot_0, &
  !$omp     visc_rem_max,do_I,visc_rem,simple_OBC_pt)
  ```

- **Dummy-argument scratch arrays owned by the caller but mapped by the callee's caller** —
  `MOM_dynamics_split_RK2.F90:435-436` (`step_MOM_dyn_split_RK2`, entered right after locals are
  declared, before any use):

  ```fortran
  !$omp target enter data map(alloc: u_bc_accel, v_bc_accel, eta_pred, uh_in, vh_in)
  !$omp target enter data map(alloc: up, vp, hp, dz, h_tmp)
  ```

  balanced at the very end of the same subroutine, `:1193-1194`, using **two different map kinds**
  for the two groups:

  ```fortran
  !$omp target exit data map(release: u_bc_accel, v_bc_accel, eta_pred, uh_in, vh_in)
  !$omp target exit data map(delete: hp, up, vp, dz, h_tmp)
  ```

  (`up, vp` are additionally deleted early, mid-routine, at `:1253`, once their last use in the
  corrector has passed — an example of shrinking device residency lifetime below the whole
  subroutine when memory pressure or reuse patterns warrant it.)

### 1.4 Balance bugs and their fixes (git history)

The enter/exit pairing is maintained **by hand** and has drifted out of sync more than once:

- **`15ca2a25f` "vertvisc: add missing b_denom_1 map delete"** — `src/parameterizations/vertical/MOM_vert_friction.F90`.
  The enter-data list `map(alloc: b1, c1, d1, Ray, b_denom_1)` was *replaced* (not matched) by a
  narrower `map(delete: b1, c1, d1, Ray)` at teardown, silently leaking `b_denom_1`'s device
  allocation every call. Fix: mirror the full original list:

  ```diff
  -  !$omp target enter data map(alloc: b1, c1, d1, Ray, b_denom_1)
     !$omp target enter data map(to: visc%Ray_v) if (allocated(visc%Ray_v))
     ...
  -  !$omp target exit data map(delete: b1, c1, d1, Ray)
  +  !$omp target exit data map(delete: b1, c1, d1, Ray, b_denom_1)
  ```

  (The enter-data line for `b1,c1,d1,Ray,b_denom_1` itself was later removed as redundant — it
  duplicated an allocation done elsewhere in the routine — but the lesson generalizes: any edit that
  changes an enter-data variable list must grep the same routine for the paired exit-data list.)

- **`bc05a6a89` "Explicitly allocate h_tmp to prevent expensive transfers during initialization"** —
  `MOM_dynamics_split_RK2.F90`. Before this commit `h_tmp` (an automatic/scratch array) had **no**
  enter/exit data pair at all in `initialize_dyn_split_RK2`, so every `do concurrent` touching it
  paid an implicit per-loop host-device transfer (or relied on unified memory). Fix bracketed its
  one use site with an explicit pair and converted the surrounding manual triple-nested loops to
  `do concurrent`:

  ```fortran
  !$omp target enter data map(alloc: h_tmp )
  if (CS%store_CAu) then
    ...
    do concurrent (k=1:nz, j=jsd:jed, i=isd:ied)
      h_tmp(i,j,k) = h(i,j,k)
    enddo
    call continuity(CS%u_av, CS%v_av, h, h_tmp, uh, vh, dt, G, GV, US, CS%continuity_CSp, CS%OBC, pbv)
    ...
  endif
  !$omp target exit data map(delete: h_tmp )
  ```

  Generalizable rule: **any array touched inside a `do concurrent`/`omp target` region must have an
  explicit enter-data before the first touch**, even if it is only "temporary" scratch — omitting it
  does not fail to compile, it silently reintroduces per-statement host-device traffic.

**Practical balance-checking method used in this codebase:** grep the variable name for both
`enter data` and `exit data` inside the same subroutine (see §1.3's `up, vp` example: entered
*once* at `:436` — `map(alloc: up, vp, hp, dz, h_tmp)` — but deleted in *two* separate exit-data
statements, at `:1194` — `map(delete: hp, up, vp, dz, h_tmp)` — and again at `:1253` —
`map(delete: up, vp)`. The extra early delete at `:1253` is intentional early release once the
corrector's last use of `up, vp` has passed, not a bug — but this is exactly the kind of
one-enter-vs-two-exit count mismatch a balance check must be able to *explain* rather than flag
blindly). Because a second `map(delete:)` on an already-removed variable is a no-op under
nvfortran, this double-delete is safe; the danger a balance check guards against is the opposite —
an enter with no matching exit (the `15ca2a25f` leak).

---

## 2. Mapping whole derived-type shells before members ("partial presence")

nvfortran (like most OpenMP implementations) requires a struct's own memory to be "present" on
device before any of its allocatable/pointer members can be attached. The codebase's fix is to map
the **bare CS shell** with `map(alloc:)` *before* calling the child module's `_init` routine (which
then maps its own members internally):

> **Mechanism (attach/detach under nvfortran).** When you `map(to:/alloc:)` a derived-type variable,
> only its *scalar* fields and its descriptor words travel — an allocatable/pointer array **member**
> is a separate device allocation whose device descriptor must then be *attached* (pointed) to the
> parent's device copy. That attach only happens if the parent shell is already present, which is why
> the shell map must strictly precede the member maps (violating the order is the "ambiguous partial
> presence" error `81680c15d` fixed). Corollary rules an agent can apply mechanically: (1) map order
> is always **outermost shell → … → innermost array**, and teardown is the exact reverse; (2) each
> attach is a per-member device operation, so mapping an *array of* member-bearing structs one element
> at a time is O(#elements) attaches — the cost `1865612de` eliminated by flattening (§3.4); (3) a
> plain scalar CS member (an `id_*` diagnostic ID, a block size) needs **no** separate map — it rides
> along inside the shell's `map`, and is refreshed with whole-struct `update to(CS)` (§2.2), never its
> own `enter data`.

`src/core/MOM_dynamics_split_RK2.F90:1689-1712` (`initialize_dyn_split_RK2`):

```fortran
!$omp target enter data map(alloc: CS%continuity_CSp)
call continuity_init(Time, G, GV, US, param_file, diag, CS%continuity_CSp, CS%OBC)
...
!$omp target enter data map(alloc: CS%PressureForce_CSp)
call PressureForce_init(Time, G, GV, US, param_file, diag, CS%PressureForce_CSp, CS%ADp, &
                        CS%SAL_CSp, CS%tides_CSp)

!$omp target enter data map(alloc: CS%hor_visc)
call hor_visc_init(Time, G, GV, US, param_file, diag, CS%hor_visc, ADp=CS%ADp)

allocate(CS%vertvisc_CSp)
!$omp target enter data map(alloc: CS%vertvisc_CSp)
call vertvisc_init(MIS, Time, G, GV, US, param_file, diag, CS%ADp, dirs, &
                   ntrunc, CS%vertvisc_CSp, CS%fpmix)
...
!$omp target enter data map (alloc: CS%barotropic_CSp)
call barotropic_init(u, v, h, Time, G, GV, US, param_file, diag, &
                     CS%barotropic_CSp, restart_CS, calc_dtbt, CS%BT_cont, &
                     CS%OBC, CS%SAL_CSp, HA_CSp)
```

The pointer-typed child (`vertvisc_CSp`) needs an explicit host `allocate(CS%vertvisc_CSp)` first
(pointers have no storage until allocated), whereas the by-value children (`hor_visc`,
`continuity_CSp`, `PressureForce_CSp`, `barotropic_CSp` — embedded `type(x_CS) :: x` members) already
have storage as part of the parent and only need the shell attached.

**Top-level analogue** in `MOM.F90` — the whole `MOM_control_struct` and both viscosity CSs, added by
commit `81680c15d` ("Allocate MOM CS and both viscosity CS on GPU") specifically to fix "ambiguous
partial presence" errors:

```fortran
! config_src/drivers/solo_driver/MOM_driver.F90
!$omp target enter data map(alloc: MOM_CSp)
...
call MOM_end(MOM_CSp)
!$omp target exit data map(delete: MOM_CSp)
```

```fortran
! src/core/MOM.F90, initialize_MOM — allocate(CS%visc) is now required because vertvisc_type
! was changed from an embedded value member to `type(vertvisc_type), allocatable :: visc`
allocate(CS%visc)
!$omp target enter data map(alloc: CS%visc)
call set_visc_register_restarts(HI, G, GV, US, param_file, CS%visc, restart_CSp, use_ice_shelf)
```

Commit `81680c15d` also changed `G_in` (`ocean_grid_type`) from an embedded value member to
`allocatable`, "to prevent excessive grid transfers" — i.e. the shell-map pattern only works cleanly
when the member is a pointer/allocatable; embedded-by-value derived-type members inside another
mapped struct are harder to manage independently and were converted to allocatable specifically to
decouple their device lifetime from the parent's.

### 2.1 Nested sub-types: `CS%pbv` and its members

`src/core/MOM.F90:3225-3231` — a two-level nest (`MOM_control_struct` → `porous_barrier_type pbv` →
four allocatable arrays), mapped shell-then-members in one block, non-conditionally:

```fortran
allocate(CS%pbv%por_face_areaU(IsdB:IedB,jsd:jed,nz), source=1.0)
allocate(CS%pbv%por_face_areaV(isd:ied,JsdB:JedB,nz), source=1.0)
allocate(CS%pbv%por_layer_widthU(IsdB:IedB,jsd:jed,nz+1), source=1.0)
allocate(CS%pbv%por_layer_widthV(isd:ied,JsdB:JedB,nz+1), source=1.0)
!$omp target enter data map(to: CS%pbv)
!$omp target enter data map(to: CS%pbv%por_face_areaU, CS%pbv%por_face_areaV)
!$omp target enter data map(to: CS%pbv%por_layer_widthU, CS%pbv%por_layer_widthV)
```

Order matters: `CS%pbv` (the shell) must be entered before `CS%pbv%por_face_areaU` etc. (the
members), exactly mirroring the parent-CS-then-child-CS ordering in §2's dycore example — this is
the same rule applied one level deeper.

### 2.2 Whole-struct bulk `update`

Rather than updating individual scalar members one at a time after a batch of host-side parameter
computation, the codebase sometimes updates the **entire mapped CS** in one directive:

`src/core/MOM_barotropic.F90:6576` (`barotropic_init`, after ~30 `register_diag_field` calls that
set scalar `CS%id_*` diagnostic-ID members) and `:6588`:

```fortran
!$omp target update to (CS)
...
!$omp target enter data map (to: CS%frhatu, CS%frhatv)
!$omp target enter data map (to: CS%eta_cor)
call set_dtbt(G, GV, US, CS, gtot_est=gtot_estimate, SSH_add=SSH_extra)
...
!$omp target update to (CS%dtbt)
```

and `src/core/MOM.F90:3104` (`initialize_MOM`, right after `CS%G_in`'s grid metrics have been
uploaded, and again after `set_visc_init` in `MOM_set_viscosity.F90` at the analogous `CS` update
site introduced by `81680c15d`):

```fortran
call tracer_registry_init(param_file, CS%tracer_Reg)

!$omp target update to(CS)
```

The commit message for `81680c15d` flags this as an experimental departure from the codebase's usual
member-by-member discipline: *"one change here breaks our derived type handling rules: we do an
`update(CS)` after the grid has been uploaded... this needs exploration."* Treat whole-struct
`update to(CS)` as a **pragmatic escape hatch** used when a batch of scalar CS members (mostly
diagnostic IDs / dtbt-like scalars) must reach the device and enumerating each one is impractical,
not as the general convention — the general convention is per-array `map`/`update` next to each
`ALLOC_`/`DEALLOC_`.

---

## 3. Proven-works mapping pattern catalogue

### 3.1 Macro-allocatable array inside a CS (the dominant pattern)

See §1.1/§1.2 in full. One-line summary: `ALLOC_(CS%x(...)) ; CS%x = <init>` then
`!$omp target enter data map(to: CS%x)` in `*_init`; `DEALLOC_(CS%x)` then
`!$omp target exit data map(delete: CS%x)` in `*_end`, same order both directions is not required but
strongly conventional.

### 3.2 Nested CS shell ("partial presence")

See §2 in full — `map(alloc: CS%child_CSp)` before calling `child_init`, both for pointer children
(needs `allocate()` first) and by-value embedded children.

### 3.3 Conditional maps: `if(associated())` / `if(<condition>)`

Two distinct places the `if` can go — as a Fortran `if` block around the whole directive, or as an
OpenMP `if()` clause on the directive itself (compiled unconditionally, skipped at runtime):

- OpenMP `if()` clause, `src/core/MOM_dynamics_split_RK2.F90:472` / `:939` (`p_surf` is a pointer
  that may alias either `p_surf_end` or `forces%p_surf`, and is only sometimes associated):

  ```fortran
  !$omp target enter data map(to: p_surf) if (associated(p_surf))
  ...
  !$omp target exit data map(delete: p_surf) if (associated(p_surf))
  ```

- Same idiom for a hybrid allocatable/pointer CS member. The **live** current example is the pair
  of guarded whole-field updates at `src/parameterizations/vertical/MOM_vert_friction.F90:1440-1441`:

  ```fortran
  !$omp target update to(visc%Kv_shear) if (associated(visc%Kv_shear))
  !$omp target update to(visc%Kv_shear_Bu) if (associated(visc%Kv_shear_Bu))
  ```

  (Historically the same routine also carried
  `!$omp target enter data map(to: visc%Ray_v) if (allocated(visc%Ray_v))` — visible in the diff
  context of commit `15ca2a25f`. That directive has since been **removed**: in current source
  `visc%Ray_v` is read pointwise inside the tridiagonal loop with an inline
  `if (allocated(visc%Ray_v)) Ray = visc%Ray_v(i,J,k)` guard — `MOM_vert_friction.F90:942,954,1259,1267`
  — and the associated `b1,c1,d1,Ray,b_denom_1` scratch became loop-`private(...)` rather than
  device-mapped. Do not cite `:437` for a `map` — line 437 is the `!$omp declare target` of
  `find_coupling_coef_gl90`.)

  `vertvisc_type` (`MOM_variables.F90:258`) is documented in `00-architecture.md` §2.3 as **hybrid**
  — some fields allocatable, some pointer (because they must be restart-registry targets) — so every
  map of one of its optional fields must be guarded (`allocated()` for allocatable fields,
  `associated()` for pointer fields), since the field may legitimately be unallocated for a given
  configuration (e.g. `Kv_shear` only exists if KPP/shear mixing is active).

- Fortran `if` block guarding a diagnostic-ID-gated device→host round trip (see §6),
  `src/tracer/MOM_tracer_hor_diff.F90:665-680`:

  ```fortran
  if (CS%id_KhTr_u > 0) then
    !$omp target exit data map(from: Kh_u)
    do j=js,je ; do I=is-1,ie
      Kh_u(I,j,:) = G%mask2dCu(I,j)*Kh_u(I,j,1)
    enddo ; enddo
    ...
    call post_data(CS%id_KhTr_u, Kh_u, CS%diag)
  endif
  ```

### 3.4 Flat array vs. array-of-structs ("struct of arrays" pitfall)

**Anti-pattern (pre-fix):** an array of small derived types each holding its own pointer/allocatable
member (`type(p2d), dimension(SZJ_(G)) :: deep_wt_Lu` — `p2d` wraps a single `real, pointer :: p(:,:)`
component), mapped **one struct at a time inside a `do j` loop** — commit `cdd3de9e3` ("add data
mapping for tracer_epipycnal_ML_diff"), `src/tracer/MOM_tracer_hor_diff.F90` (pre-flatten form):

```fortran
do j=js,je
  k_size = max(2*max_srt(j),1)
  allocate(deep_wt_Lu(j)%p(IsdB:IedB,k_size))
  ...
  !$omp target enter data map(alloc: deep_wt_Lu(J)%p, deep_wt_Ru(J)%p, hP_Lu(J)%p, hP_Ru(J)%p, &
  !$omp   k0a_Lu(j)%p, k0a_Ru(j)%p, k0b_Lu(j)%p, k0b_Ru(j)%p)
enddo
```

This requires one `attach`/`detach` operation *per j-row, per array* — `SZJ_(G)` separate small
device allocations and pointer attachments instead of one big one.

**Fix (proven pattern):** commit `1865612de` ("Convert structs of arrays to flat arrays") — replace
`type(p2d), dimension(SZJ_(G)) :: deep_wt_Lu` with a single
`real, dimension(:,:,:), allocatable :: deep_wt_Lu` (index order `(I,k,j)`), sized once by a
`do concurrent ... DO_LOCALITY(reduce(max:k_size))` over all rows, allocated and mapped **once**:

```fortran
k_size = 1
do concurrent (j=js-1:je+1) DO_LOCALITY(reduce(max:k_size))
  k_size = max(k_size, 2*max_srt(j))
enddo
allocate(k0a_Lu(IsdB:iedB,k_size,jsd:jed))
allocate(k0a_Ru(IsdB:iedB,k_size,jsd:jed))
allocate(deep_wt_Lu(IsdB:iedB,k_size,jsd:jed))
allocate(deep_wt_Ru(IsdB:iedB,k_size,jsd:jed))
...
!$omp target enter data map(alloc: deep_wt_Lu, deep_wt_Ru, hP_Lu, hP_Ru, k0a_Lu, k0a_Ru, k0b_Lu, &
!$omp   k0b_Ru)
```

Measured effect (commit message, `1865612de`): **halved GPU compile-region time in
`MOM_tracer_hor_diff`**, at the cost of ~20% more memory (worst-case per-array element count rose
from ~700k to ~830k, because every `j`-row now allocates the same `k_size` instead of its own
tighter `max(2*max_srt(j),1)`). **Rule of thumb for new ports: never map a derived-type array whose
element is itself a pointer/allocatable-holding struct inside a loop — flatten to one contiguous
array with the loop index as a trailing dimension first.**

**Accepted, unfixed instance of the same anti-pattern:** the tracer registry `Reg%Tr(:)`
(`tracer_type`, one array element per tracer, each independently shaped/sized) is *not* flattened —
each element's members are mapped individually inside a host loop over tracers, using the `!$`
free-form conditional-compilation sentinel (compiled only when OpenMP is enabled, so the loop
variable `m` doesn't need to exist in a non-OpenMP build) — commit `97629c240`,
`src/tracer/MOM_tracer_hor_diff.F90:207-210`:

```fortran
! MOM_tracer_hor_diff.F90:209-216
!$omp target enter data map(to: Reg, Reg%Tr, CS) map(alloc: khdt_x, khdt_y, kh_u, kh_v)
!$ do m = 1, Reg%ntr
  !$omp target enter data map(to: Reg%Tr(m)%t)
  !$omp target enter data map(to: Reg%Tr(m)%df_x)   if(associated(Reg%Tr(m)%df_x))
  !$omp target enter data map(to: Reg%Tr(m)%df_y)   if(associated(Reg%Tr(m)%df_y))
  !$omp target enter data map(to: Reg%Tr(m)%df2d_x) if(associated(Reg%Tr(m)%df2d_x))
  !$omp target enter data map(to: Reg%Tr(m)%df2d_y) if(associated(Reg%Tr(m)%df2d_y))
!$ enddo
```

Note how the mandatory `Reg%Tr(m)%t` field is mapped unconditionally while every *optional* pointer
field (`df_x`, `df_y`, `df2d_x`, `df2d_y`) gets its own `if(associated(...))`-guarded directive — the
per-element/per-field guarding is precisely what makes this awkward to flatten. Also note the
`map(to: Reg, Reg%Tr, CS)` on the first line: the registry shell **and** the `Reg%Tr(:)` array of
`tracer_type` must be present before any `Reg%Tr(m)%...` member can attach, the same shell-before-member
ordering as §2. This is tolerated (rather than flattened like `deep_wt_Lu` in
`tracer_epipycnal_ML_diff`) because `ntr` and each tracer's presence of optional fields vary
per-configuration and per-element — flattening would require a redesign of
`tracer_type` itself, a larger change than the local scratch-array flattening in `1865612de`. It
remains a candidate for the same fix if the registry loop is ever shown to dominate profile time.

### 3.5 `map(alloc)` vs `map(to)` vs `map(from)` vs `map(delete)` vs `map(release)` — when each is used

| Kind | Used when | Example |
|---|---|---|
| `map(to:)` | Host has meaningful initial data the device kernel reads before ever writing it (zeroed/`GV%Angstrom_H`-initialized CS arrays, grid metrics, restart-read fields) | `MOM_dynamics_split_RK2.F90:1354` `map(to: CS%diffu, CS%diffv)` (after explicit host zero-init) |
| `map(alloc:)` | Device-only scratch, or a CS array whose first write happens on-device and host content is irrelevant (the `MOM_dynamics_split_RK2.F90:1350` `TODO` explicitly asks "if not [needed], do `map(alloc:)` rather than `map(to:)`") | `MOM_dynamics_split_RK2.F90:1637` `map(alloc: CS%uhbt, CS%vhbt)`; nested CS shells (§2) always use `alloc` |
| `map(from:)` | One-shot device→host copy-out, typically for a diagnostic about to be posted or a value about to feed host-only code | `MOM_tracer_hor_diff.F90:666` `map(from: Kh_u)` right before `post_data` |
| `map(delete:)` | Paired teardown of a `map(to:)/map(alloc:)` at `*_end`/end-of-scope. Forces the device reference count to **zero** and deallocates regardless of prior count — **does not copy back** | `MOM_dynamics_split_RK2.F90:2066` `map(delete: CS%diffu, CS%diffv)` |
| `map(release:)` | Teardown that **decrements** the device reference count by one (deallocating only if it hits zero) — also does not copy back. Used where the mapped variable may have been mapped from more than one place, or where the exact map state across conditional branches is harder to track statically | `MOM_dynamics_split_RK2.F90:1193` `map(release: u_bc_accel, v_bc_accel, eta_pred, uh_in, vh_in)`; `MOM_tracer_hor_diff.F90:723` `map(release: khdt_x, khdt_y, Kh_u, Kh_v) map(release: CS)` |

**Critical copyback rule:** neither `delete` nor `release` copies the device value back to the host —
they only tear down the device allocation. Any array whose *final host value matters* after the device
region (e.g. it will be written to a restart, read by host-only code, or checksummed) must be brought
back with `!$omp target update from(...)` or `!$omp target exit data map(from:)` **before** the
`delete`/`release`. In this codebase, CS work arrays like `CS%diffu` are pure device-side intermediates
recomputed every timestep, so `map(delete:)` with no copy-back is correct; the diagnostic/restart
copy-outs are handled separately by the `map(from:)`/`update from` sites in §6.

Practical distinction actually driving the choice in this codebase between `delete` and `release`:
`delete` is used for **CS members**, where the enter/exit pairing is unconditional and exactly
mirrored (§1.2); `release` is used for **local scratch** whose allocate/map calls may occur inside
conditional branches (`if (CS%store_CAu)`, `if (dyn_p_surf)` etc.) where the author was not
100%-confident every code path mapped the variable exactly once — `release`'s decrement-not-force
semantics degrade gracefully in that case.

> **FABLE-CHECK (reviewed 2026-07-14 — resolution or current status in KNOWLEDGE.md §8a/§8b):** Is the `delete` vs `release` split here genuinely load-bearing (reference-count
> correctness) or merely stylistic convention? The reference-count distinction only produces different
> behaviour when a variable is mapped more than once (nested/overlapping regions). If every scratch
> array in these routines is mapped exactly once per call, `release` and `delete` are behaviourally
> identical and the choice is cosmetic. Worth confirming against one multiply-mapped case (e.g. `CS`
> itself, which is entered at `MOM_tracer_hor_diff.F90:209` and released at `:723`, but may also be
> shell-mapped by a caller) before presenting the split as a hard rule.

---

## 4. No central mapping utility — hand-written, co-located with `ALLOC_`/`DEALLOC_`

There is **no** wrapper macro, interface, or subroutine that performs "map this CS member" as a
single call. `MOM_memory_macros.h` defines only the *host*-side allocation macros
(`ALLOCABLE_`, `PTR_`, `ALLOC_(x)` → `allocate(x)`, `DEALLOC_(x)`, `TO_NULL_`); there is no
`MAP_ENTER_(x)`/`MAP_EXIT_(x)` counterpart anywhere in the tree (confirmed: no `omp` directives
appear inside `MOM_memory_macros.h`, and no `*.h`/module in `src/framework/` wraps a mapping
directive in a subroutine or macro — grep for `subroutine.*map\(` and `#define.*omp target` both
return nothing).

Every `!$omp target enter/exit data`/`update` in the tree is a hand-written directive placed
immediately next to the corresponding `ALLOC_`/`DEALLOC_`/local-declaration line, as shown throughout
§1-§3. This is a **deliberate design choice**, not an oversight in progress:

1. **Per-array map kind varies** (`to` vs `alloc` vs conditional — §3.5) based on whether the host
   initializes the array before first device use, which a generic macro cannot infer without an
   extra parameter that would have to be threaded through every call site anyway.
2. **Balance is visually auditable** only when the two directives sit next to their matching
   `ALLOC_`/`DEALLOC_` — burying the mapping inside a macro/subroutine would hide exactly the
   information (`15ca2a25f`'s missing `b_denom_1`) that a code reviewer needs to spot an imbalance.
3. **Struct-shell-before-members ordering (§2)** is call-site-specific — it depends on where in the
   child module's own `_init` the member arrays get allocated, so a parent-side generic "map this CS"
   helper would need the same insider knowledge a hand-written directive already encodes.
4. Bug history (`bc05a6a89`, `15ca2a25f`, `cdd3de9e3`→`1865612de`) shows the team iterating on
   *placement and granularity* of directives per call site — premature abstraction into a shared
   utility would have made these fixes harder, not easier, since each fix changed the *shape* of
   what's being mapped (added a variable to a list, flattened a struct array, added an early-release
   point), not a parameter to a generic call.

Net effect: adding a GPU-resident array to any CS is a **copy-paste-and-adapt** operation from the
nearest analogous existing pattern in this document, not a call into shared infrastructure.

---

## 5. `!$omp declare target` convention for device-callable helpers

All 21 occurrences (see table in the summary) mark small, `pure`/`elemental`/side-effect-free
**column or point kernels** called from inside `do concurrent`/`omp target teams loop` regions:

| File:line | Routine | Notes |
|---|---|---|
| `src/framework/MOM_coms.F90:69` | module data `pr, I_pr` (parameter arrays) | `!$omp declare target(pr, I_pr)` — the only *data* (not routine) declare-target in the inventory; these EFP precision-lookup tables must be resident for `efp_decompose` to read on device |
| `src/framework/MOM_coms.F90:779` | `efp_decompose` | `pure subroutine`; called per-real inside the block-reduction `do concurrent` of the reproducing sum (see `00-architecture.md` §7.2) |
| `src/framework/MOM_intrinsic_functions.F90:51` | `cuberoot` | `elemental function`; avoids `modulo()`/`pow()`-like intrinsics not implemented on all device targets |
| `src/framework/MOM_intrinsic_functions.F90:133,181,246` | `nth_root` family | bit-stable Newton iteration, explicitly documented (`:120-129`) as replacing `x**(1.0/n)` because device libm/libdevice differ in last-bit rounding from host |
| `src/parameterizations/vertical/MOM_vert_friction.F90:437,2101,2611` | `find_coupling_coef_gl90` and two others | column kernels called per-(i,j) inside the tridiagonal solve's `omp target teams loop collapse(2)` |
| `src/parameterizations/vertical/MOM_set_viscosity.F90:1251,1258,1294,1316,1389,1438,1718,1742,1802,1836,1966,2012` | `find_L_open_*` family (porous-topography open-fraction kernels) and BBL/ML column helpers | 12 of the 21 total — the single largest concentration; several routines carry the directive **twice** (once near the top of the subroutine body, once repeated just before the declarations end, e.g. `MOM_set_viscosity.F90:1251` and `:1258` inside the same `find_L_open_uniform_slope`) — harmless duplication, not two different routines |

**The rule (from `00-architecture.md` §0.4/§7.5):** cross-module calls inside a device loop are
"painful" and must either be (a) inlined via `!NVF$ INLINE` / `-Minline=name:<routine>` compiler
flags, or (b) exposed as a `!$omp declare target` free routine with no polymorphic/`class(*)`
arguments and no unresolved external calls of its own — i.e. **declare target is necessary but not
sufficient; the routine must also actually inline or itself be fully declare-target reachable**.
This is why the EOS layer (`00-architecture.md` §7.1) could not simply add `!$omp declare target` to
its existing polymorphic dispatch — nvfortran cannot resolve the v-table on device regardless of the
directive, and duplicating each kernel as a `_loc` free function (`7c7af5572`) was required instead.

---

## 6. `!$omp target update to/from(...)` — forced host↔device round-trips

`update` is used, not to establish/tear down device residency (that's `enter`/`exit data`), but to
force a **fresh copy** in one direction while both host and device copies already exist. Three
recurring reasons appear in the inventory:

### 6.1 Diagnostics posting (guarded by `id_* > 0`)

The dominant use. `MOM_diag_mediator.F90`'s `post_data` is host-only and unchanged on `dev/gpu`
(`00-architecture.md` §7.4), so any field about to be posted must be pulled back first, and — since
that transfer is only needed if the diagnostic is actually requested this run — every such `update
from` is guarded by the diagnostic's registered ID being positive:

```fortran
! src/tracer/MOM_tracer_hor_diff.F90:722
!$omp target update from(khdt_x, khdt_y) if(CS%debug .or. CS%id_khdt_x>0 .or. CS%id_khdt_y>0)
!$omp target exit data map(release: khdt_x, khdt_y, Kh_u, Kh_v) map(release: CS)
...
if (CS%id_khdt_x > 0) call post_data(CS%id_khdt_x, khdt_x, CS%diag)
if (CS%id_khdt_y > 0) call post_data(CS%id_khdt_y, khdt_y, CS%diag)
```

and the Fortran-`if`-block variant, `MOM_tracer_hor_diff.F90:665-679`:

```fortran
if (CS%id_KhTr_u > 0) then
  !$omp target exit data map(from: Kh_u)
  do j=js,je ; do I=is-1,ie
    Kh_u(I,j,:) = G%mask2dCu(I,j)*Kh_u(I,j,1)   ! host-side post-processing before posting
  enddo ; enddo
  ...
  call post_data(CS%id_KhTr_u, Kh_u, CS%diag)
endif
```

Both forms exist: an OpenMP `if()` clause on `update` when the guard is a simple boolean
disjunction evaluated once, vs. a full Fortran `if` block (using `exit data map(from:)` rather than
`update from`) when the guarded region also does non-trivial host-side arithmetic before `post_data`.

### 6.2 Host-only physics call bracketed by round-trips

The split-RK2 driver (`MOM_dynamics_split_RK2.F90`) is riddled with `update to`/`update from` pairs
bracketing calls into modules/phases that are only partially ported, or bracketing debug-checksum
calls (§6.3) — e.g. around `PressureForce`/`CorAdCalc`/`vertvisc` transitions:

```fortran
! :613-616
!$omp target update from(CS%CAu_pred, CS%CAv_pred)
!$omp target update from(CS%PFu, CS%PFv, CS%pbce)
!$omp target update from(CS%diffu, CS%diffv)
!$omp target update from(u_bc_accel, v_bc_accel)
```

and the tracer underflow clean-up in `MOM_tracer_hor_diff.F90:1655-1660`, host-only scalar-threshold
logic wrapped in ordinary (unguarded) `update`:

```fortran
if (Tr(m)%conc_underflow > 0.0) then
  !$omp target update from(Tr(m)%t)
  !$OMP parallel do default(shared)
  do k=1,nz ; do j=js,je ; do i=is,ie
    if (abs(Tr(m)%t(i,j,k)) < Tr(m)%conc_underflow) Tr(m)%t(i,j,k) = 0.0
  enddo ; enddo ; enddo
  !$omp target update to(Tr(m)%t)
endif
```

### 6.3 Debug/checksum round-trips

Guarded by `CS%debug`, e.g. the `CS%debug` disjunct in §6.1's `id_khdt_x` example, and generally
anywhere `MOM_state_chksum`/`uvchksum`/`hchksum` (`MOM_checksums.F90`, `00-architecture.md` §7.2) is
called on a field that lives on-device the rest of the time — these checksum calls are host-only, so
a `debug`-gated `update from` precedes them.

### 6.4 Batch scalar refresh (`update to(CS)`)

Whole-struct `update to(CS)`, distinct from the per-array round-trips above — see §2.2. Used after a
burst of host-side scalar/diag-ID computation on the CS, as a coarser-grained alternative to
enumerating each scalar member.

---

## 7. Cross-references

- `00-architecture.md` §2 (CS pattern, member styles), §6 (quantified inventory), §7.1 (EOS
  polymorphism vs. declare target), §7.2 (EFP reproducing sums / `efp_decompose`), §7.3 (halos /
  `omp_offload`), §7.4 (diagnostics still host-only), §7.5 (compiler workarounds).
- Key commits referenced here: `81680c15d` (CS/visc shell allocation, top-level partial-presence
  fix), `1865612de` (struct-of-arrays → flat arrays, attach/detach cost), `bc05a6a89` (explicit
  `h_tmp` map to avoid implicit transfers), `97629c240`/`cdd3de9e3` (incremental data-mapping
  additions to `tracer_epipycnal_ML_diff`, later superseded in structure by `1865612de`),
  `15ca2a25f` (missing `map(delete:)` balance bug), `7c7af5572` (EOS `_loc` free-function pattern,
  referenced for the declare-target sufficiency rule in §5).

---

## Verification notes

Verified against source and git on branch `dev/gpu` (source + `git show`/`git log` only; no build/run).

**Confirmed (spot-checked against the actual bytes):**

- Directive counts across `src/ config_src/` reproduce exactly: **213** `enter data`, **168**
  `exit data`, **398** `update`, **21** `declare target` (all 21 in `src/`; distribution
  set_viscosity 12, vert_friction 3, intrinsic 4, coms 2 — sums to 21). The `168 (≈167)` hedge was
  wrong-way-round (167 is the architecture doc's stale figure) and was corrected to `168`.
- All eight cited commit hashes resolve with the quoted subject lines (`81680c15d`, `1865612de`,
  `bc05a6a89`, `15ca2a25f`, `cdd3de9e3`, `7c7af5572`, `97629c240`, `5b5f6b2b1`).
- §1.1/§1.2 lifecycle quotes are byte-exact at `MOM_dynamics_split_RK2.F90:1350-1368`
  (`register_restarts_dyn_split_RK2`, 1329-1415) and `:2065-2083` (`end_dyn_split_RK2`).
- §1.3 local-scratch citations exact: `:435-436`, `:1193-1194`, `:1253`; continuity `:181`/`:228`,
  `:658-660`.
- §1.4 `bc05a6a89` (h_tmp) and `15ca2a25f` (b_denom_1) diffs match the described before/after exactly.
- §2 shell-before-member sequence exact at `MOM_dynamics_split_RK2.F90:1689-1712`; `CS%pbv` block
  exact at `MOM.F90:3225-3231`; `MOM_driver.F90:282`/`:636`; `MOM.F90:3277-3278` (`allocate(CS%visc)`).
- §2.2 `update to (CS)` at `MOM_barotropic.F90:6576`, `frhatu/frhatv`/`eta_cor` at `:6579-6580`,
  `update to (CS%dtbt)` at `:6588`, and `MOM.F90:3104` all exact.
- §5 declare-target table: every line number verified (`MOM_coms.F90:69,779`;
  `MOM_intrinsic_functions.F90:51,133,181,246`; `MOM_vert_friction.F90:437,2101,2611`;
  `MOM_set_viscosity.F90:1251…2012` — 12 lines).
- §6 tracer_hor_diff quotes exact: `:665-680`, `:722-723`, `:1653-1662`.

**Corrected:**

1. **§3.3** — the `map(to: visc%Ray_v) if (allocated(visc%Ray_v))` example was cited at
   `MOM_vert_friction.F90:437`, but `:437` is the `!$omp declare target` of `find_coupling_coef_gl90`,
   and that `map` directive **no longer exists in current source** (it appears only in the diff
   context of `15ca2a25f`; `Ray_v` is now read inline at `:942,954,1259,1267` and its scratch became
   loop-`private`). Rewrote the bullet to make the live example the `Kv_shear` updates at `:1440-1441`
   and flag the Ray_v form as historical.
2. **§3.4** — the `Reg%Tr(m)` snippet showed a single combined `map(to: …%t, …%df_x, …%df_y, …)`;
   current source (`MOM_tracer_hor_diff.F90:209-216`) maps `%t` unconditionally and each optional
   pointer field on its own `if(associated(...))`-guarded line. Replaced with the actual code.
3. **§1.4** — the balance-check example miscounted (`up, vp` "in three exit-data statements … two
   enter-data statements"); actually entered once at `:436`, deleted in two statements (`:1194`,
   `:1253`). Corrected the arithmetic and added why the double-delete is safe under nvfortran.

**Enhanced:** added an attach/detach mechanism box in §2 (why shell-before-member, three mechanical
map-ordering rules, scalar members need no map); added a "critical copyback rule" to §3.5
(`delete`/`release` never copy back — use `map(from:)`/`update from` first if the host needs the
value); sharpened the `delete`/`release` table rows to the correct force-to-zero vs decrement OpenMP
semantics; noted `allocated()` vs `associated()` guard selection for hybrid `vertvisc_type` fields.

**FABLE-CHECK markers added:** 1 (§3.5 — whether the `delete`/`release` split is load-bearing
reference-count correctness or cosmetic convention).

**Confidence:** High. Every file:line and every commit cited in the document was opened and matched;
the three corrections were the only substantive drifts (two stale-vs-current-code snippets and one
counting slip), and none affect the document's core patterns, which are all accurate.
