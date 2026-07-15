# Pointer Usage Across MOM6 (dev/gpu) — Aliasing Hazards for GPU Offload

> Companion to `00-architecture.md` §2 (CS pattern) and §3 (memory conventions). This document
> inventories every recurring `pointer` idiom in the tree, ties each to a concrete device-mapping
> consequence, and reconstructs the multi-GPU/pointer bugs that have already been hit and fixed on
> `dev/gpu`. Read `00-architecture.md` first.

---

## 1. Why pointers exist at all in this codebase

`grep -rn "pointer" src/ --include=*.F90` matches in essentially every module; the busiest files are
`MOM.F90` (108), `MOM_diag_mediator.F90` (107), `MOM_open_boundary.F90` (89), `MOM_restart.F90` (62),
`MOM_variables.F90` (54), `MOM_dynamics_split_RK2.F90` (53). Three genuinely distinct motives account
for nearly all of them; a fourth is incidental and should be removed opportunistically.

1. **Restart-registry targets (unavoidable, by construction of `MOM_restart.F90`).** Every
   `register_restart_field_ptr*d` entry point takes the field as a `target, intent(in)` dummy and
   stores a bare Fortran pointer alias to it:
   ```fortran
   ! src/framework/MOM_restart.F90:207-239 (register_restart_field_ptr3d)
   real, dimension(:,:,:), target, intent(in) :: f_ptr
   ...
   CS%var_ptr3d(CS%novars)%p => f_ptr
   ```
   `MOM_restart_CS` (`:124-134`) holds `type(p0d/p1d/p2d/p3d/p4d), pointer :: var_ptrNd(:)` arrays —
   heterogeneous, dynamically-sized collections of aliases to arbitrary host arrays scattered across
   many CSs, so they **must** be pointers (or `target` actuals bound at call time); there is no
   allocatable equivalent that can alias a *pre-existing* array owned by someone else. This is the
   one class of pointer that cannot be designed away without redesigning the restart mechanism
   itself.
   - Because the registry only needs a *host*-side alias for read/write I/O, restart-registered
     fields do not need special device treatment purely on that account — but any field that is
     *conditionally* allocated (only when a physics option is on) is forced to be declared
     `pointer` in its owning derived type so `associated()` can gate both the registration call and
     later device mapping (see §5).

2. **Cross-module aliasing of shared "bag of state" containers (`MOM_variables.F90`).** Several
   derived types exist purely to let unrelated modules see the same prognostic/diagnostic arrays
   without copying:
   - `thermo_var_ptrs` (`MOM_variables.F90:79`) — `T`, `S`, `p_surf` are pointers; `tv%T => CS%T` and
     `tv%S => CS%S` are set once in `MOM.F90:3119` (`CS%tv%T => CS%T ; CS%tv%S => CS%S`, repeated at
     `MOM.F90:3432-3433` after a possible reallocation) so that any routine holding only `tv` (not
     `CS`) — the whole EOS/diabatic/ALE call chain — reads/writes the *same* memory as the dycore's
     `CS%T`/`CS%S`.
   - `ocean_internal_state` (`MOM_variables.F90:138`) — **every** member is `pointer`, aliasing
     `T,S,u,v,h,uh,vh,CAu,CAv,PFu,PFv,diffu,diffv,pbce,u_accel_bt,v_accel_bt,u_av,v_av,u_prev,v_prev`.
     Populated in one shot in `MOM.F90:3195-3212`:
     ```fortran
     MOM_internal_state%u => CS%u ; MOM_internal_state%v => CS%v
     MOM_internal_state%h => CS%h
     MOM_internal_state%uh => CS%uh ; MOM_internal_state%vh => CS%vh
     ...
     CS%CDp%uh => CS%uh ; CS%CDp%vh => CS%vh
     ```
     It exists solely so a single "give me everything for diagnostics" structure can be handed to
     ensemble/diagnostic code without copying 3D prognostic arrays.
   - `accel_diag_ptrs` (`:167`) / `cont_diag_ptrs` (`:241`) — **all-pointer** diagnostic alias
     structs (`CS%ADp`, `CS%CDp`) that let `MOM_vert_friction.F90`, `MOM_CoriolisAdv.F90`,
     `MOM_PressureForce_FV.F90`, `MOM_continuity_PPM.F90` each write into a shared accumulator array
     for later energy-budget diagnostics, without every producer routine owning the array itself.

3. **Optional/feature-gated fields (`associated()` as a runtime "is this feature on" flag).**
   `vertvisc_type` (`MOM_variables.F90:258`) is explicitly **hybrid**: drag fields
   (`bbl_thick_u`, `kv_bbl_u`, `Ray_u`, …) are plain `allocatable` because they always exist, but
   `MLD`, `h_ML`, `sfc_buoy_flx`, `Kd_shear`, `Kv_shear`, `Kv_shear_Bu`, `Kv_slow`, `TKE_turb` are
   `pointer` with an explicit comment:
   ```fortran
   ! The following elements are pointers so they can be used as targets for pointers in the
   ! restart registry.
   real, pointer, dimension(:,:) :: MLD => NULL()
   ...
   real, pointer, dimension(:,:,:) :: Kd_shear => NULL()
   ```
   These are allocated only if the relevant parameterization (KPP/ePBL/CVMix shear/`RiNo_mix`) is
   selected at init (`MOM_set_viscosity.F90:2899-2909`, `register_restart_field(visc%Kv_shear, ...)`
   guarded by `if (useKPP .or. useEPBL .or. use_CVMix_shear .or. ...)`). Downstream code (e.g.
   `vertvisc_coef` in `MOM_vert_friction.F90`) then must use `associated(visc%Kv_shear)` both as a
   host feature-flag *and* as an OpenMP `if()` map guard — see §3.

4. **Incidental pointers that are not load-bearing for aliasing.** Many `local pointer => CS%member`
   assignments exist purely as a **naming convenience** so a long subroutine can write `u`,`v`,`h`
   instead of `CS%u`, `CS%v`, `CS%h` — they alias, are never reassigned, and could be plain
   dummy-argument passing (or associate blocks) instead:
   - `MOM.F90:629` (`step_MOM`): `u => CS%u ; v => CS%v ; h => CS%h` (verified).
   - `MOM.F90:1272` (`step_MOM_dynamics`): same pattern (verified).
   - `MOM_dynamics_split_RK2.F90:426` (`step_MOM_dyn_split_RK2`):
     `u_av => CS%u_av ; v_av => CS%v_av ; h_av => CS%h_av ; eta => CS%eta`. These four are the true
     incidental aliases in the dycore; the `eta` pointer even carries an explicit source comment
     "*This pointer is just used as shorthand for `CS%eta`*" (declared `:385-395`, bound `:426`).
     **Correction:** `u,v,h` themselves do *not* appear as local `=> CS%…` aliases in this routine —
     they enter as dummy arguments (`u_inst`, `v_inst`, `h`; the velocities carry a `target`
     attribute at `:309/:311` because `btstep`/`u_ptr` build pointer views of them). The earlier draft's
     "`u,v,h => NULL()` declared at `:151-153`, bound at `:165`" was wrong: `:151-153` are *CS-type
     members* (`taux_bot`, `tauy_bot`), and no `u => CS%u` binding exists anywhere in this file.
   These are the ones the port doc calls "incidental": they don't gate any `associated()` logic and
   aren't restart targets themselves (the *targets*, `CS%u_av`/`CS%eta`/…, are macro-allocatables, not
   pointers). They complicate reading the code and (in principle) complicate alias analysis for a
   compiler trying to prove no-aliasing for `do concurrent`, but they carry no unique semantic
   weight and are candidates for straightforward removal once a porting pass touches these routines.

   **Contrast — reassigned aliases are *not* incidental.** In the same routine,
   `u_ptr, v_ptr, uh_ptr, vh_ptr` (declared `:395-400`, comment "*used to alter which fields are
   passed to `btstep` with various options*") and the local `p_surf` (§3) are pointers whose **target
   changes at runtime** depending on options — they select *which* array `btstep` operates on. These
   carry real semantic weight: their `associated()` status and device attachment must be
   re-established every call (the map guard cannot be hoisted, see §3's `p_surf` case). Do not lump
   them in with the removable `u_av => CS%u_av` shorthand.

**Rule of thumb going forward:** if a pointer's only job is "give this array a short local name," it
is incidental and removable. If `associated()` on it changes control flow (feature gating) or it
appears as `target` in a `register_restart_field` call, it is load-bearing and must stay a pointer
(or an allocatable held by `target`, see §4).

---

## 2. `tracer_type` / `tracer_registry_type` — an array-of-structs-of-pointers

`src/tracer/MOM_tracer_types.F90`:
- `tracer_type` (`:13`) has **~20 pointer members** — `t` (the tracer concentration, `:15`), plus
  advective/diffusive diagnostic arrays `ad_x, ad_y, ad2d_x, ad2d_y, df_x, df_y, hbd_dfx, hbd_dfy,
  advection_xy, t_prev, …` — all `real, dimension(...), pointer => NULL()`. Most of these are
  optionally allocated depending on which diagnostics are requested (`associated()` gates their use
  throughout `MOM_tracer_advect.F90` and `MOM_tracer_hor_diff.F90`, e.g. `if (associated(Reg%Tr(m)%ad_x))`
  at `MOM_tracer_advect.F90:244,777`).
- `tracer_registry_type` (`:129`): `type(tracer_type) :: Tr(MAX_FIELDS_)` — a **fixed-size array of
  structs, each stuffed with pointers** (`MAX_FIELDS_ = 50`, `config_src/memory/*/MOM_memory.h:26`).
  This is the single densest pointer-in-derived-type-array structure in the tree and is the direct
  cause of the bug in §3.

This AoS-of-pointers pattern recurred as a **performance** problem too, independent of correctness:
commit `1865612de` ("Convert structs of arrays to flat arrays", `src/tracer/MOM_tracer_hor_diff.F90`)
replaced `type(p2d), dimension(SZJ_(G)) :: deep_wt_Lu, hP_Lu, ...` (one pointer-to-2D-array *per j-row*,
each individually `!$omp target enter data`'d — `allocate(deep_wt_Lu(j)%p(...)); !$omp target enter
data map(alloc: deep_wt_Lu(J)%p, ...)` inside a `do j=js,je` loop) with flat
`real, dimension(:,:,:), allocatable :: deep_wt_Lu` arrays mapped once. Commit message: *"Lots of
time was being spent 'attaching' and 'detaching' the member arrays to/from each struct on the
GPU"* — flattening **halved GPU compile/attach time** at the cost of ~20% more memory (700k → 830k
elements allocated per array in the benchmark case). This generalizes §2.3 of `00-architecture.md`:
**any array of pointer-bearing structs is a per-element attach/detach tax; flatten it if it's on a
hot path.**

---

## 3. Aliasing patterns and their `associated()` map guards

The dominant defensive idiom on `dev/gpu` is: pointer members that may or may not be allocated are
mapped/updated with an `if (associated(...))` clause so the OpenMP directive is a no-op when the
feature is off, instead of erroring on a null/unassociated pointer:

- `MOM_dynamics_split_RK2.F90:472`: `!$omp target enter data map(to: p_surf) if (associated(p_surf))`
  and the matching `MOM_dynamics_split_RK2.F90:939`: `!$omp target exit data map(delete: p_surf) if
  (associated(p_surf))`. `p_surf` here is a **local pointer**, resolved just above (`:461-467`) to
  either `p_surf_end` (dynamic surface pressure) or `forces%p_surf`, i.e. the alias target changes
  every call — the map guard has to be re-evaluated every call because `associated()` can flip.
- `MOM_vert_friction.F90` (`vertvisc_coef`): `!$omp target update to(visc%Kv_shear) if
  (associated(visc%Kv_shear))` and `!$omp target update to(visc%Kv_shear_Bu) if
  (associated(visc%Kv_shear_Bu))` — see the full history in §5, this exact pair was buggy twice.
- `MOM_set_viscosity.F90` (`set_visc_init`): `!$omp target enter data map(to: visc%Kv_shear) if
  (associated(visc%Kv_shear))` / `... map(to: visc%Kv_shear_Bu) if (associated(visc%Kv_shear_Bu))`
  — the *allocating* side's mirror of the same guard.
- Throughout `MOM_tracer_advect.F90`: `if (associated(Reg%Tr(m)%ad_x))`,
  `if (associated(Reg%Tr(m)%ad_y))`, `if (associated(Reg%Tr(m)%advection_xy))`,
  `if (associated(Reg%Tr(m)%ad2d_x))`, `if (associated(Reg%Tr(m)%ad2d_y))` (`:244-266`) gate
  device-side zeroing of optional diagnostic accumulators inside `do concurrent`.

**The pattern to internalize:** `associated()` is not just a host-side null check here — it is a
*device control-flow condition* embedded directly in OpenMP `map`/`update` clauses and inside
`do concurrent` bodies. For it to give correct answers on device, the pointer's association status
(and, once associated, its target's contents) must be **faithfully mirrored to device** — which is
exactly what `map(alloc:)` does *not* do (next section).

**`allocated()` vs `associated()` — match the intrinsic to the member kind.** `vertvisc_type` is
*hybrid* (§1.3): its always-present drag fields are `allocatable`, its feature-gated fields are
`pointer`. The guard intrinsic differs accordingly and the two are not interchangeable:
`allocated()` for allocatable members, `associated()` for pointer members. Both are used as device
map/update guards side by side — e.g. `MOM.F90:1807-1812` guards the allocatable drag fields with
`!$omp target update from(CS%visc%Ray_u) if (allocated(CS%visc%Ray_u))`,
`... bbl_thick_u) if (allocated(...))`, while the pointer fields (`Kv_shear`, `Kv_shear_Bu`) are
guarded with `if (associated(...))` (above). **Porting rule:** before writing a map guard, check the
member's declaration — an `if (associated(x))` on an `allocatable` array (or vice-versa) is a
compile error at best and silently wrong control flow at worst.

---

## 4. `map(alloc:)` vs `map(to:)` on pointer/derived-type members — the `Reg%Tr(:)` bug

Two same-day commits by Edward Yang fixed distinct multi-GPU answer-change bugs in
`advect_tracer` (`src/tracer/MOM_tracer_advect.F90`); both are cited together in the branch history
but have **different root causes** and are worth keeping separate.

### 4a. `e182de310` — "advect_tracer: fix multi gpu answer change" (race on a shared array element)

Not actually a pointer-aliasing bug, but the immediate predecessor in the same file and commonly
conflated with 4b. The reduction variable `domore_k(k)` was being written from inside two separate
`do concurrent` loops as a plain assignment:
```fortran
domore_k(k) = 0
do concurrent (j=jsv:jev, domore_u(j,k))
  domore_k(k) = 1        ! every iteration in the do-concurrent writes the SAME element
enddo
```
This is a write-write race on a single shared array element with no reduction semantics — undefined
under `do concurrent` and answer-dependent on team/thread scheduling, hence non-reproducible
**across GPUs** (and, more subtly, across different launch configurations on the same GPU). The fix
introduces a scalar temporary and an explicit reduction clause:
```fortran
domore_k_tmp = 0
do concurrent (j=jsv:jev, domore_u(j,k)) DO_LOCALITY(reduce(max:domore_k_tmp))
  domore_k_tmp = 1
enddo
...
domore_k(k) = domore_k_tmp   ! single, well-defined write after the reduction completes
```
The commit note explains the scalar temporary is required because "`do concurrent` can't use array
elems yet" as the reduction variable. Lesson for porting: **never let a `do concurrent` body write an
indexed array element as an implicit reduction target — always reduce into a scalar first.**

### 4b. `a774eb331` — "tracer_advect: fix map of Reg%Tr(:)" (the actual pointer/AoS mapping bug)

```diff
- !$omp target enter data map(to: OBC) map(alloc: domore_u, domore_v, uhr, vhr, uh_neglect, &
- !$omp   vh_neglect, hprev, local_advect_scheme, Reg, Reg%Tr(:))
+ !$omp target enter data map(to: OBC, Reg, Reg%Tr(:)) map(alloc: domore_u, domore_v, uhr, vhr, &
+ !$omp   uh_neglect, vh_neglect, hprev, local_advect_scheme)
```
(and the matching `exit data`: `map(release: hprev, ...)` no longer separately `map(from: hprev)`).

**Mechanism.** `Reg%Tr(:)` is `tracer_registry_type%Tr(MAX_FIELDS_)` — an array of `tracer_type`,
each element carrying ~20 `pointer` members (§2) plus plain scalars (`advect_scheme`, `ntr`-derived
metadata, diagnostic IDs). The routine immediately (and throughout) tests
`associated(Reg%Tr(m)%ad_x)`, `associated(Reg%Tr(m)%ad_y)`, `associated(Reg%Tr(m)%advection_xy)`,
etc. on device (`:244-266`). `map(alloc: Reg, Reg%Tr(:))` only **reserves device storage** for the
struct array — it does *not* copy the host struct's bytes over, so every scalar and every pointer
descriptor inside `Reg%Tr(m)` starts as **whatever bit pattern the device allocator happened to hand
back** (uninitialized device memory). `associated()` reads that pointer descriptor; with garbage
bits the answer is unpredictable — sometimes true, sometimes false, and the pattern differs by GPU
architecture, driver, and allocator state, i.e. exactly a "multi-GPU answer change": the same source
produces different results depending on which device (or even which run) executes it, because it's
reading uninitialized memory to decide whether to zero a diagnostic array. `map(to: Reg, Reg%Tr(:))`
instead performs the copy: the host's true association status and (for later individually-attached
members like `Reg%Tr(m)%t`, which is separately mapped at `:237` `map(to: Reg%Tr(m)%t)`) correct
target linkage are mirrored to device before any kernel reads them.

**Generalization — the porting rule:** for a derived type (or array of derived types) that is later
read with `associated()` on device, or whose scalar members feed device control flow, **you must
`map(to:)` it, never `map(alloc:)`.** `map(alloc:)` is only safe for arrays whose entire initial
content is written by device code before being read (pure "workspace" arrays like `domore_u`, `uhr`,
`hprev` in the same directive, which is exactly why those remain `map(alloc:)` in the same line).
Mapping a `pointer`-bearing struct with `alloc` silently converts a feature-flag check into a read of
uninitialized memory — this is a structurally hard bug to catch because it can pass on one GPU/driver
combination and fail (or silently diverge) on another.

---

## 5. `vertvisc_type` pointer members: `Kv_shear`/`Kv_shear_Bu`/`MLD` as a recurring hazard class

`vertvisc_type` (`MOM_variables.F90:258`) is the paradigm case of §1.3 (feature-gated pointers) and
has been the site of **three separate device-mapping bugs**, all variations on "the pointer's
`associated()` status and/or target contents were not correctly on device":

1. **`d75e4870e` "vertvisc: Do not pass CS as pointers"** (Marshall Ward). Several
   `MOM_vert_friction.F90` entry points (`vertvisc`, `vertvisc_remnant`, `vertvisc_limit_vel`) took
   `type(vertvisc_CS), pointer :: CS` and began with an `if (.not.associated(CS)) call MOM_error(...)`
   guard. The patch changes the dummy argument to a plain `type(vertvisc_CS) :: CS` (no `pointer`)
   and drops the `associated()` guard (keeping only the `CS%initialized` logical check). Rationale
   (from the commit message): *"This reduces some of the implicit 'microtransfers' required when
   passing derived type point[er]s to and from the device."* Passing a CS **by pointer** into a
   routine that then touches its members on device forces the compiler/runtime to re-resolve the
   pointer's device address on every call (a "microtransfer" of the descriptor) instead of using a
   plain by-reference/by-value derived-type argument whose device mapping was already established at
   a higher scope. Net effect: same semantics, fewer descriptor round-trips per call. Also folded
   into this commit: removed a redundant `!$omp target enter data map(alloc: b1, c1, d1, Ray,
   b_denom_1)` / matching `exit data` pair around the tridiagonal solve in `vertvisc_remnant` — local
   scratch arrays that don't need to persist across the call don't need explicit device
   alloc/dealloc at all if they're firstprivate/private in the enclosing `target teams` region.
   Guidance distilled: **once a CS's device mapping is established once (at `initialize_MOM`/child
   `_init` time), pass it down the call tree as an ordinary (non-pointer) derived-type argument; only
   the *owning* module should hold it as `pointer`/`allocatable` and manage its `enter
   data`/`exit data` lifecycle.**

2. **`c82e1254a` "vertvisc: Fix CS memory management"** (the most recent commit on `dev/gpu`,
   in the log preceding this one). Root cause per commit message: *"The visc object was being
   allocated twice, which was overwriting information about Kv_shear and Kv_shear_Bu that was
   defined in `set_visc_register_restarts()`. This was causing errors in kernels which needed both
   the arrays and the `associated()` state for flow control."* **Corrected mechanism (verified
   against the diff and the parent tree):** the "allocated twice" is *not* a host double-`allocate`.
   The host `allocate(CS%visc)` happens exactly once (`MOM.F90:3277`, unchanged by this commit). What
   happened twice was the **device mapping** of the `visc` derived type: `!$omp target enter data
   map(alloc: CS%visc)` at `MOM.F90:3278`, followed ~430 lines later by a redundant `!$omp target
   enter data map(to: CS%visc, CS%set_visc_CSp)` at `:3709` (parent-commit line numbers). In between
   those two, `set_visc_register_restarts` (called at `:3279`) `safe_alloc_ptr`'d and *device-attached*
   `visc%Kv_shear`/`Kv_shear_Bu` (its own now-removed `!$omp target enter data map(alloc:
   visc%Kv_shear)`). The second whole-struct `map(to: CS%visc)` at `:3709` **re-established /
   overwrote the device image of the parent `visc` object**, clobbering those child pointer
   attachments and the `associated()` state that had just been set up. Symptom: a device-side
   "addressing error" in `double_gyre` runs, because subsequent `associated(visc%Kv_shear)` map guards
   read a device descriptor that no longer matched the host object. The lesson is subtler and more
   important than "don't allocate twice": **re-mapping a parent derived type with `map(to:)`/`map(alloc:)`
   after its allocatable/pointer members have been individually device-attached can silently detach or
   corrupt those member attachments — map the parent *once*, then attach members, and refresh the
   parent's scalar contents with `target update to(...)`, never a second `enter data`.** Fix, in
   three parts:

   > **FABLE-CHECK (reviewed 2026-07-14 — resolution or current status in KNOWLEDGE.md §8a/§8b):** The precise OpenMP/nvfortran semantics behind this "second `map(to:)` clobbers
   > member attachments" claim deserve the strongest-model check. Under a strict OpenMP 5.x reading, a
   > `map(to:)` on a variable already *present* on device should only bump the reference count and copy
   > nothing (present → no re-alloc, no member re-attach). The observed bug implies nvfortran instead
   > re-runs the derived-type mapper (or re-copies the descriptor block) on the second `enter data`,
   > detaching the separately-attached `Kv_shear`/`Kv_shear_Bu`. Is the correct root-cause framing "(i)
   > nvfortran does not honor present-check semantics for derived types with allocatable/pointer
   > components and re-copies the descriptor", or "(ii) the two directives mapped *different*
   > storage (`map(alloc:)` of the whole struct vs `map(to:)` including members) so refcounts/attach
   > state genuinely diverged"? The distilled porting rule (map parent once, never re-`enter data`)
   > holds either way, but the *why* should be stated correctly for the compiler-workarounds doc.
   - `CS%set_visc_CSp` changed from embedded-by-value (`type(set_visc_CS) :: set_visc_CSp`) to
     `type(set_visc_CS), allocatable :: set_visc_CSp` in `MOM.F90:417` (consistency with other
     pointer/allocatable child CSs, and to make the allocate/deallocate lifecycle explicit and
     single-sourced).
   - `CS%visc` is updated to device *inside* `set_visc_init()` (`!$omp target update to(visc)`) right
     after all its scalar members are finalized, **before** the conditional `Kv_shear`/`Kv_shear_Bu`
     arrays are separately entered:
     ```fortran
     !$omp target update to(visc)
     ...
     !$omp target enter data map(to: visc%Kv_shear) if (associated(visc%Kv_shear))
     !$omp target enter data map(to: visc%Kv_shear_Bu) if (associated(visc%Kv_shear_Bu))
     ```
   - In `vertvisc_coef` (`MOM_vert_friction.F90`), `Kv_shear_Bu` handling changed from a **persistent
     `map(alloc:)` at `set_viscosity_init` + a fresh `map(to:)`/`map(release:)` pair every call** to
     the same `target update to(...) if (associated(...))` idiom already used for `Kv_shear` — i.e.
     the fix explicitly *removed* an inconsistency where `Kv_shear` was persistently mapped and
     refreshed with `target update`, but `Kv_shear_Bu` was instead re-entered/released every call
     (`map(to: visc%Kv_shear_Bu) ... map(release: visc%Kv_shear_Bu)`), which is wasteful and, per the
     surrounding comment removed in the diff, was based on a stale assumption (*"Kv_shear is
     persistently mapped on device via map(alloc:) in set_viscosity_init, so map(to:) here would not
     copy host updates"*) that no longer matched the actual mapping strategy once `Kv_shear_Bu` was
     unified with the same pattern.
   Lesson: **a pointer member that is both (a) conditionally allocated, (b) a restart-registry
   target, and (c) persistently mapped to device is fragile against *any* re-mapping (or
   re-allocation) of its owning derived type.** If the owning struct (`visc`) is re-mapped or
   reallocated anywhere in the init sequence after its members are attached, every pointer inside it
   must be re-established and re-mapped in the correct order: allocate host → `map(alloc:)` the parent
   *once* → register restart (sets pointer) → `target update to` the parent's scalars/descriptors →
   `map(to:)` the conditionally-present array members guarded by `associated()`. Never issue a second
   whole-parent `enter data map(to:)`/`map(alloc:)` after the member attach step.

3. `remotes/origin/vertvisc-no-ptr-transfer` (local remote branch, tip commit `f327f04c0`, subject
   line identical to `d75e4870e`) is the same "do not pass CS as pointers" fix living on a
   differently-based branch — evidence this fix was reapplied/rebased at least once, underscoring
   that the CS-by-pointer anti-pattern recurs as code is merged from `dev-gfdl`.

**Combined guidance for `vertvisc_type` and similarly-shaped types:** when a struct mixes always-on
`allocatable` arrays with feature-gated `pointer` arrays that are also restart targets, (1) allocate
the owning struct exactly once, ideally as `allocatable` at the top-level CS so its lifecycle is
unambiguous; (2) register restarts (which binds the pointers) before any device mapping of those
pointer members; (3) `target update to` the struct itself (its scalars + pointer descriptors) before
individually mapping/updating the conditionally-associated array members with an `if
(associated(...))` guard; (4) keep the guard identical in both the allocating routine (`set_visc_init`)
and every consuming routine (`vertvisc_coef`) — an asymmetric strategy (one map(alloc)-and-hold, the
other map(to)-and-release) is exactly what went wrong in point 2 above.

---

## 6. Enumerated porting hazards (pointer-heavy structures, ranked)

| Structure | File:line | Why hazardous | Status |
|---|---|---|---|
| `tracer_registry_type%Tr(MAX_FIELDS_)` | `MOM_tracer_types.F90:129`, `tracer_type` `:13` | AoS of ~20-pointer-member structs; `associated()` used as device control flow throughout `MOM_tracer_advect.F90`/`MOM_tracer_hor_diff.F90`; already caused the `map(alloc:)→map(to:)` bug (§4b) | Fixed on `dev/gpu` for advect path; watch any *new* code that maps `Reg%Tr(:)` |
| `ocean_internal_state` | `MOM_variables.F90:138` | All-pointer alias struct over the entire prognostic+accel state; populated once (`MOM.F90:3195-3212`) and handed to diagnostics/ensemble code — a large "view" object that must never itself be separately device-mapped (its members already are, via their true owners) | Not GPU-mapped itself (used for host-side diagnostics/ensembles); low risk if it stays that way |
| `vertvisc_type` (`MLD`, `Kd_shear`, `Kv_shear`, `Kv_shear_Bu`, `Kv_slow`, `TKE_turb`, `h_ML`, `sfc_buoy_flx`) | `MOM_variables.F90:258` | Feature-gated pointer + restart target + persistently mapped on device; three bugs already fixed here (§5) | Actively fixed/hardened; still fragile to any future re-map or re-allocation of `CS%visc` after its members are attached |
| `thermo_var_ptrs` (`tv%T`, `tv%S`, `tv%p_surf`) | `MOM_variables.F90:79` | Cross-module alias of the dycore's own `CS%T/CS%S`; re-established at two points (`MOM.F90:3119`, `:3432-3433`) — any code path that reallocates `CS%T`/`CS%S` without re-running the `tv%T => CS%T` assignment silently detaches `tv` from current data | No known bug yet, but structurally analogous to the `visc`/`Kv_shear` re-map bug (§5.2) — worth auditing anywhere `CS%T`/`CS%S` are reallocated (the alias must be re-run *and* the device image refreshed) |
| `accel_diag_ptrs` / `cont_diag_ptrs` (`ADp`, `CDp`) | `MOM_variables.F90:167,241` | All-pointer diagnostic aliases written by many producer modules (`vertvisc`, `CorAdCalc`, `PressureForce_FV`, `continuity`); each producer's `associated()` check on its own diagnostic slot must see a faithfully-mapped pointer | **`CS%ADp` *is* mapped `map(alloc: CS%ADp)` at `MOM.F90:3190`** (not `map(to:)`), and `associated(ADp%sal_u/tides_u/…)` is read as control flow in `MOM_PressureForce_FV.F90:913-931,2044-2058` — this is exactly the §4b shape. See FABLE-CHECK below. |
| `MOM_restart_CS%var_ptrNd(:)` (`p0d..p4d`) | `MOM_restart.F90:130-134` | Heterogeneous pointer-array registry over arbitrary host arrays; host-only by design (I/O), but any future "GPU-resident restart" work would hit the same AoS-of-pointers attach cost documented in `1865612de` | Host-only today (`00-architecture.md` §7.4); a future hazard, not a current one |
| Any *future* `type(p2d)/type(p2di) dimension(SZJ_(G))` (array-of-pointer-to-2D-array, one alloc per row) | pattern retired in `MOM_tracer_hor_diff.F90` by `1865612de`, defined at `:104-110` (now dead code — no remaining users in that file, verified) | Per-row `enter data` inside a loop is the concrete anti-pattern that cost 2x compile/attach time; the type definitions remain in-file as a fossil/warning | Fixed here; **do not reintroduce this pattern elsewhere** (e.g. `MOM_set_diffusivity.F90`, `MOM_CVMix_KPP.F90`, `MOM_energetic_PBL.F90` are still unported and may contain the same idiom — check before porting) |

> **FABLE-CHECK (reviewed 2026-07-14 — resolution or current status in KNOWLEDGE.md §8a/§8b):** `CS%ADp` is mapped with `!$omp target enter data map(alloc: CS%ADp)` at
> `MOM.F90:3190` (an all-pointer `accel_diag_ptrs`), while `associated(ADp%sal_u)`, `associated(ADp%tides_u)`,
> etc. are read for control flow in `MOM_PressureForce_FV.F90:913-931` and `:2044-2058`. This is the
> exact `map(alloc:)`-on-a-pointer-struct shape that §4b/`a774eb331` identified as a multi-GPU bug for
> `Reg%Tr(:)`. Is `CS%ADp` a latent version of the same bug, or is it safe here? Two possible reasons
> it may be safe — please adjudicate: (a) those `associated(ADp%...)` reads sit in *plain* `do k/do j`
> host loops (not `do concurrent`/`target`), so the check may execute host-side where the host
> descriptor is authoritative; (b) `map(alloc:)` on the *parent* `ADp` may be harmless as long as the
> pointer *members* actually read on device (`du_dt_visc`, etc.) are separately `map(to:)`-attached and
> the parent's own pointer descriptors are never dereferenced on device. Confirm which (if either)
> holds, and whether `map(alloc: CS%ADp)` should be `map(to:)` for safety/consistency.

---

## 7. Summary of concrete guidance for a porting agent

1. Before mapping any derived-type instance or array-of-derived-types to device, ask: **does any
   code path read `associated()` on one of its pointer members, or a scalar member, from inside
   device code?** If yes, it must be `map(to:)` (or `target update to`), never `map(alloc:)`
   (§4b, exemplified by `a774eb331`).
2. **Never write to a shared array element from inside a `do concurrent` without a `DO_LOCALITY`
   reduction clause** — use a scalar temporary and assign the array element once afterward (§4a,
   `e182de310`).
3. **Don't pass a CS as `pointer` into leaf routines just to check `associated(CS)`** — pass by
   ordinary derived-type argument once its device mapping is established at a higher scope; this
   avoids repeated pointer-descriptor "microtransfers" (§5.1, `d75e4870e`).
4. **Map a parent derived type to device exactly once, then attach its members; never re-map the
   parent afterward.** A second whole-parent `enter data map(to:)`/`map(alloc:)` issued after its
   allocatable/pointer members are individually device-attached clobbers those attachments (§5.2,
   `c82e1254a` — the "allocated twice" was a *double device mapping* at `MOM.F90:3278`+`:3709`, not a
   host double-`allocate`). Correct order: allocate host → `map(alloc:)` parent once → register
   restart (binds pointers) → `target update to` the parent's scalars/descriptors → `map(to:)` the
   pointer members guarded by `associated()`.
5. **Match the guard intrinsic to the member kind:** `if (allocated(...))` for allocatable members,
   `if (associated(...))` for pointer members (`vertvisc_type` uses both — `MOM.F90:1807-1812` vs the
   `Kv_shear` guards). Mixing them is a compile error or silent control-flow bug (§3).
6. **Flatten arrays-of-pointer-structs on hot paths** (`type(p2d/p2di), dimension(SZJ_(G))`) to flat
   allocatable arrays — the per-row attach/detach cost is real and measured (`1865612de`, ~2x compile
   time saved for ~20% more memory).
7. Pointers whose only job is a short local alias to a macro-allocatable CS member (`u => CS%u`,
   `eta => CS%eta`) are not aliasing hazards in the mapping sense — but they're not free either, since
   a compiler doing alias analysis for `do concurrent`/`target` regions has to prove non-aliasing
   through them. Retire them opportunistically when a routine is otherwise being touched, not as a
   dedicated pass. **But** distinguish these from pointers whose target is *reassigned* at runtime
   (`p_surf`, `u_ptr`/`v_ptr` feeding `btstep`): those are load-bearing and must keep their
   per-call `associated()`-guarded map (§1.4, §3).

---

## 8. Verification notes

Every commit reconstruction and every file:line citation in this doc was checked against the actual
source and `git show`/`git diff` on branch `dev/gpu`.

**Confirmed correct as written:**
- `e182de310` (§4a, `domore_k` write-write race → scalar-temp + `reduce(max:)`): diff matches exactly,
  including the removal of `domore_k` from the map lists and the three call sites.
- `a774eb331` (§4b, `Reg%Tr(:)` `map(alloc:)`→`map(to:)`): diff and the paired `exit data`
  (`map(from: hprev)`→`map(release: hprev, …)`) match; `map(to: Reg%Tr(m)%t)` at `:237` and the
  `associated(Reg%Tr(m)%…)` guards at `:244-266` confirmed.
- `d75e4870e` (§5.1, "do not pass CS as pointers"): all three signatures (`vertvisc`,
  `vertvisc_remnant`, `vertvisc_limit_vel`) changed from `pointer` to plain `type(vertvisc_CS)`, the
  `associated(CS)` guards dropped, and the redundant `map(alloc: b1,c1,d1,Ray,b_denom_1)`/`map(delete:)`
  pair around the `vertvisc_remnant` tridiagonal solve removed — all confirmed.
- `1865612de` (§2, AoS-of-pointers flattening): commit message figures (~700k→830k, ~20%, "halves"
  compile time) quoted verbatim; pre-commit form `type(p2d), dimension(SZJ_(G)) :: deep_wt_Lu` with
  per-`j` `allocate(...(j)%p(...))` + in-loop `map(alloc: deep_wt_Lu(J)%p, …)` confirmed at parent; the
  `p2d`/`p2di` types are now genuinely dead (no users) — confirmed.
- All `MOM_variables.F90` type line numbers (`:79/:138/:167/:241/:258/:317`), the `tv%T => CS%T`
  assignments (`MOM.F90:3119`, `:3432-3433`), the `ocean_internal_state` population block
  (`MOM.F90:3195-3212`), `register_restart_field_ptr3d` (`MOM_restart.F90:207-239`), `var_ptrNd`
  (`:130-134`), `tracer_type`/`tracer_registry_type` (`:13/:15/:129`), `MAX_FIELDS_=50`, and the
  `p_surf` local-pointer resolution (`MOM_dynamics_split_RK2.F90:461-467`, mapped `:472`, released
  `:939`) — all confirmed.

**Corrected:**
1. **§1.4 (incidental pointers):** the draft's dycore example `u,v,h => NULL()` "declared at `:151-153`,
   bound at `:165`" was fabricated — `:151-153` are CS-type members (`taux_bot`/`tauy_bot`) and no
   `u => CS%u` binding exists in `MOM_dynamics_split_RK2.F90`. Replaced with the real aliases
   (`u_av/v_av/h_av => CS%… ; eta => CS%eta` at `:426`) and noted `u,v,h` arrive as dummy arguments.
   Added the contrasting *reassigned-alias* class (`u_ptr`/`v_ptr`/`p_surf`).
2. **§5.2 (`c82e1254a`):** the draft called this "a plain host-side double-`allocate` bug." That is
   wrong — the host `allocate(CS%visc)` occurs once (`MOM.F90:3277`, untouched by the commit). The real
   defect is a **double *device mapping*** of the parent struct: `map(alloc: CS%visc)` at `:3278` then a
   redundant `map(to: CS%visc)` at `:3709` (parent line numbers), the second of which overwrote the
   device image and clobbered the `Kv_shear`/`Kv_shear_Bu` attachments created in between by
   `set_visc_register_restarts`. Rewrote the mechanism and the derived rule accordingly (map parent
   once, never re-`enter data`).

**Enhancements added:** the `allocated()` vs `associated()` guard-intrinsic distinction (§3, with
`MOM.F90:1807-1812`); the reassigned-alias hazard class (§1.4); tightened §7 rules (now 7 rules); and
a flagged latent-hazard finding that `CS%ADp` is itself mapped `map(alloc:)` at `MOM.F90:3190`.

**Open items (FABLE-CHECK):** 2 markers — (1) the nvfortran present-check/derived-type re-map
semantics behind the §5.2 clobber; (2) whether `map(alloc: CS%ADp)` at `MOM.F90:3190` is a latent
`a774eb331`-style bug or safe.

**Confidence:** High on all commit reconstructions and line numbers (directly verified against
source/git). Medium on the two FABLE-CHECK items, which turn on nvfortran-specific runtime behavior
that cannot be settled from source alone.
