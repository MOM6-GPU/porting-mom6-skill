# Diagnostics / IO Path and the Remaining Host-Only Surface (dev/gpu)

> **Purpose.** The diagnostics/IO stack (`MOM_diag_mediator`, `MOM_diag_remap`, restarts) is the
> largest *unported, host-only* subsystem still touched every timestep. Every posted diagnostic and
> every restart write forces the model state across the PCIe/NVLink boundary. This document maps
> the `post_data` call path, catalogues the guarded `!$omp target update from(...)` transfer sites
> that keep those crossings conditional, evaluates the in-flight `diag_map_mediator_port` branch,
> and covers profiling (nvtx-on-clocks) and restart IO. Read `00-architecture.md` §4.1 and §7.4
> first.

---

## 1. The diagnostics path — confirmed host-only mediator

`src/framework/MOM_diag_mediator.F90` is **byte-for-byte unchanged on `dev/gpu`**:

```
$ git diff --stat dev-gfdl...dev/gpu -- src/framework/MOM_diag_mediator.F90
(empty)
```

`src/framework/MOM_restart.F90` is likewise **completely unchanged** (empty diff, zero directives)
— the restart registry is fully host-staged; see §6.

### 1.1 Call map

- **`post_data` generic** (`MOM_diag_mediator.F90:73-75`) dispatches on rank:
  `module procedure post_data_3d, post_data_2d, post_data_1d_k, post_data_0d`.
- **`post_data_2d`** (`:1408`) → asserts a valid registered id, then loops the diag "variants"
  linked list (CMOR aliases etc.) calling **`post_data_2d_low`** (`:1436`) for each.
- **`post_data_3d`** (`:1585`) → same pattern → **`post_data_3d_low`** (`:1750`).
- `post_data_2d_low`/`post_data_3d_low` do unit conversion (`field * diag%conversion_factor` into a
  freshly host-`allocate`d `locfield`), optional masking, optional downsampling
  (`downsample_diag_field`), vertical remapping via `diag_remap_do_remap` (in
  `MOM_diag_remap.F90`), and finally `send_data_infra` (→ FMS `diag_manager`, host-only by
  construction — FMS has no device awareness in this fork).
- `post_data_3d_by_column` (`:1927`) / `post_data_3d_by_point` (`:1945`) / `post_data_3d_final`
  (`:1964`) are narrower host-side entry points used by column physics for point/column diagnostics.
- **`diag_update_remap_grids`** (`:3655-3759`) — snapshots `h`/`T`/`S` (or `alt_h/alt_T/alt_S`) as
  plain Fortran pointer aliases (`h_diag => diag_cs%h`) and drives per-coordinate remap-grid
  updates; **no OpenMP anywhere in the routine** — pure host pointer arithmetic and (per-coordinate)
  calls into `MOM_diag_remap`.
- **`diag_copy_diag_to_storage`** (`:4130-4146`) / **`diag_copy_storage_to_diag`** (`:4149-4164`) —
  plain whole-array Fortran assignment (`grid_storage%h_state(:,:,:) = h_state(:,:,:)`), host-side,
  used to snapshot/restore the diagnostic grid across the dynamics/thermo sync boundary
  (`MOM.F90:1100` calls `diag_copy_diag_to_storage(CS%diag_pre_sync, h, CS%diag)`).
- **`calculate_diagnostic_fields`** (`MOM_diagnostics.F90`) is the big host-side diagnostics
  computation entered from `MOM.F90:1096`; it fans out to dozens of `post_data`/`post_product_u`/
  `post_product_v` calls guarded individually by `if (CS%id_xxx > 0)` (see e.g. lines 305-320 for
  `id_u`, `id_v`, `id_h`, `id_usq`, `id_vsq`, `id_uv`).
- **`post_transport_diagnostics`** (`MOM_diagnostics.F90:1822`) posts transport diagnostics
  (`umo`, `vmo`, dynamics `h`-tendency) after remapping storage is restored.

### 1.2 The implication (Q1 answer)

Because `MOM_diag_mediator.F90` has **zero** OpenMP directives, every array handed to `post_data_*`
is assumed **already host-resident**. Since the prognostic state (`u,v,h,uh,vh,uhtr,vhtr,T,S,...`)
lives device-resident for essentially the whole timestep (§4.2 of `00-architecture.md`), *the
transfer burden is pushed onto the caller*: each producing module must do its own
`!$omp target update from(...)` immediately before invoking `post_data`, or (as `MOM.F90:1091` does)
the driver does one blanket transfer of the whole synchronized state right before
`calculate_diagnostic_fields` is entered. **Posting any diagnostic is a device→host transfer either
way** — the only design freedom is *how much* is transferred and *how often* (whole-state blanket
vs. per-field guarded).

---

## 2. Transfer-audit: guarded vs. unconditional `!$omp target update from(...)`

`grep -rn "omp target update from" src/` returns **249** sites across `src/`. The large majority are
**not** diagnostics-specific — they cross OpenMP-target-region boundaries inside the dycore/barotropic
solver for algorithmic reasons (e.g. handing a partial result to a subsequent host-computed group
halo pass) or are unconditional `CS%debug`-class checksum staging. Only a **minority are explicitly
gated on a diagnostic id** (`if (CS%id_... > 0)`), which is the pattern this document is asked to
audit. Per-file raw counts of `target update from`: `MOM.F90` 56, `MOM_barotropic.F90` 52,
`MOM_hor_visc.F90` 53, `MOM_dynamics_split_RK2.F90` 45, `MOM_CoriolisAdv.F90` 11,
`MOM_set_viscosity.F90` 7, `MOM_tracer_hor_diff.F90` 6, `MOM_PressureForce_FV.F90` 5,
`MOM_vert_friction.F90` 4, `MOM_interface_heights.F90` 3, `MOM_diagnostics.F90` 3,
`MOM_lateral_mixing_coeffs.F90` 2, `MOM_tracer_advect.F90` 1, `MOM_state_initialization.F90` 1.

### 2.1 Diagnostic-ID-guarded sites (the "only transfer if a diagnostic is active" pattern)

| Site | Fields | Guard condition |
|---|---|---|
| `MOM_tracer_hor_diff.F90:722` | `khdt_x, khdt_y` | `if(CS%debug .or. CS%id_khdt_x>0 .or. CS%id_khdt_y>0)` |
| `MOM_diagnostics.F90:1825` | `uhtr` | `if (any([IDs%id_umo_2d, IDs%id_umo, IDs%id_uhtr] > 0))` |
| `MOM_diagnostics.F90:1826` | `vhtr` | `if (any([IDs%id_vmo_2d, IDs%id_vmo, IDs%id_vhtr] > 0))` |
| `MOM_diagnostics.F90:1827` | `h` | `if (IDs%id_dynamics_h_tendency > 0)` |
| `MOM_PressureForce_FV.F90:1301` | `e` (interface heights) | `if ((use_ALE .and. CS%Recon_Scheme > 0) .or. ...)` (debug/diag combo) |

Quoted excerpt (`MOM_tracer_hor_diff.F90:719-733`):
```fortran
call post_data(CS%id_KhTr_h, Kh_h, CS%diag)
endif

!$omp target update from(khdt_x, khdt_y) if(CS%debug .or. CS%id_khdt_x>0 .or. CS%id_khdt_y>0)
!$omp target exit data map(release: khdt_x, khdt_y, Kh_u, Kh_v) map(release: CS)

if (CS%debug) then
  call uvchksum("After tracer diffusion khdt_[xy]", khdt_x, khdt_y, ...)
endif

if (CS%id_khdt_x > 0) call post_data(CS%id_khdt_x, khdt_x, CS%diag)
if (CS%id_khdt_y > 0) call post_data(CS%id_khdt_y, khdt_y, CS%diag)
```
The transfer and the `post_data` call are **decoupled**: the `target update from` fires once
(covering both the checksum debug path and either diag id), then the two `if (CS%id_...>0)` guards
individually decide whether to actually call `post_data`. This avoids doing the transfer twice.

Quoted excerpt (`MOM_diagnostics.F90:1822-1831`, `post_transport_diagnostics`):
```fortran
call diag_save_grids(diag)
call diag_copy_storage_to_diag(diag, diag_pre_dyn)

!$omp target update from(uhtr) if (any([IDs%id_umo_2d, IDs%id_umo, IDs%id_uhtr] > 0))
!$omp target update from(vhtr) if (any([IDs%id_vmo_2d, IDs%id_vmo, IDs%id_vhtr] > 0))
!$omp target update from(h) if (IDs%id_dynamics_h_tendency > 0)

if (IDs%id_umo_2d > 0) then
  umo2d(:,:) = 0.0
  do k=1,nz ; do j=js,je ; do I=is-1,ie
```
Here a **single** `uhtr`/`vhtr`/`h` transfer covers *several* downstream diagnostics (`umo_2d`,
`umo`, `uhtr` raw output) that would otherwise each want their own guard — the `any([...]>0)`
collapses multiple ids into one gate.

`MOM_diagnostics.F90:967-975` (`calculate_vertical_integrals`) shows the same idea applied to a
derived (not raw-state) field:
```fortran
if (CS%id_col_ht > 0) then
  !$omp target update to(h)
  !$omp target enter data map(alloc: z_top)
  call find_eta(h, tv, G, GV, US, z_top)
  !$omp target exit data map(from: z_top)
```
Note the direction here is `to(h)` (host→device, pushing possibly-stale-on-device `h` up) followed
by a device-computed `find_eta` and a `from: z_top` pull — the transfer is bidirectional around a
single-diagnostic-only device kernel.

### 2.2 Unconditional / blanket transfer sites (the dominant pattern in practice)

These are **not** individually diagnostic-gated; they transfer the whole synchronized prognostic
state once per relevant call so that every downstream `if (id>0)` check inside
`calculate_diagnostic_fields`/`post_*` sees host-valid data, trading a larger transfer for much
simpler code:

- `MOM.F90:1091` — before `calculate_diagnostic_fields`:
  ```fortran
  if (MOM_state_is_synchronized(CS)) then
    !$omp target update from(u, v, h, CS%uhtr, CS%vhtr)
    call cpu_clock_begin(id_clock_other) ; call cpu_clock_begin(id_clock_diagnostics)
    call enable_averages(CS%t_dyn_rel_diag, Time_local, CS%diag)
    call calculate_diagnostic_fields(u, v, h, CS%uh, CS%vh, CS%tv, CS%ADp, &
                        CS%CDp, p_surf, CS%t_dyn_rel_diag, CS%diag_pre_sync, &
                        G, GV, US, CS%diagnostics_CSp)
  ```
  This single line covers the entire fan-out of dozens of `if (CS%id_xxx>0) call post_data(...)`
  branches inside `calculate_diagnostic_fields` — cheaper to reason about than gating each one, at
  the cost of always paying for the transfer whenever the state is synchronized (every coupling
  step boundary), whether or not *any* diagnostic in that big list is actually active.
- `MOM.F90:1113` — a **dead/disabled** duplicate of the same transfer, commented out with `!**`:
  `!**!$omp target update from(u, v, h, CS%uhtr, CS%vhtr)` with a `TODO: This appears safe to remove
  but needs verification.` — evidence the team is actively trying to prune redundant blanket
  transfers.
- `MOM.F90:1036/1038` — around ALE remap/regrid (host-only stack, §4): `target update from(u,v,h)`
  before `ALE_regridding_and_remapping`, `target update to(u,v,h)` after — this is **not**
  diagnostics-related but is the same idiom (bracket a host-only region with from/to).
- `MOM.F90:1043-1046` (commit `ff86497d5`, already merged into `dev/gpu`) — see §2.3.
- `MOM_barotropic.F90` / `MOM_dynamics_split_RK2.F90` / `MOM_hor_visc.F90` / `MOM_CoriolisAdv.F90`:
  their ~50 `target update from` sites each are almost entirely **algorithmic** (moving partial
  sums/intermediate arrays across a host-mediated halo pass or a `!$omp declare target`-incompatible
  branch), not diagnostics-guarded — do not conflate these with the diagnostics-transfer pattern.

### 2.3 A fixed regression: `ff86497d5` "Fix small transfers in post_diabatic_halo_updates" (#174)

Already merged into `dev/gpu`. Before the fix, `post_diabatic_halo_updates` (`MOM.F90:2105-2113`)
implicitly triggered a stream of small per-member transfers for `CS%tv%T`/`CS%tv%S` (accessed through
derived-type dereference inside a group-pass call) each time it ran. The fix wraps the call with an
explicit bulk map:
```fortran
! UMW NOTE: These transfers are needed to prevent excessive transfers in the group
! updates within this subroutine
!$omp target enter data map(to: CS%tv, CS%tv%T, CS%tv%S)
call post_diabatic_halo_updates(CS, G, GV, US, u, v, h, CS%tv)
!$omp target exit data map(from: CS%tv%T, CS%tv%S)
!$omp target exit data map(release: CS%tv)
```
and removes a stale `! TODO: Safe? what about T and S?` comment next to the
`call do_group_pass(pass_uv_T_S_h, G%Domain, clock=id_clock_pass, omp_offload=.true.)` inside that
routine. This is a **derived-type deep-copy trap** (see `00-architecture.md` §2.3): touching
`CS%tv%T`/`CS%tv%S` through the `tv` derived type inside an `omp_offload` group pass was silently
causing the OpenMP runtime to materialize many small implicit transfers instead of one bulk one —
exactly the class of bug flagged generically in §7.5 ("implicit copy ... which cannot yet be
prevented").

---

## 3. In-flight: `diag_map_mediator_port` — what is actually being offloaded

The branch touches the diag path in these commits (`git log dev/gpu..origin/diag_map_mediator_port
-- MOM_diag_mediator.F90 MOM_diag_remap.F90`): `b121aecbf` "separate" (+267/−73 across both files),
`fa796e4e0` "do concurrent the omp loops" (+82/−121 net, mostly loop-syntax cleanup on
`MOM_diag_mediator.F90` only), and `072db9355` "do concurrent the last loop I missed" (+5/−7,
`MOM_diag_remap.F90` only — a small follow-up converting one remaining plain loop to `do concurrent`).
**Verification note:** the *local* `diag_map_mediator_port` ref in this checkout has only the first
two of these (net `dev/gpu...diag_map_mediator_port` = +276/−121); `072db9355` lives only on
`origin/diag_map_mediator_port`, which is one commit ahead (net `dev/gpu...origin/diag_map_mediator_port`
= +277/−124). Cite the remote ref when quoting the third commit.

**This branch does *not* offload `send_data_infra`/FMS diag_manager writes, nor the top-level
`post_data`/`post_data_2d`/`post_data_3d` dispatch** — those remain plain host Fortran. What it
*does* offload, confirmed by `git diff dev/gpu...diag_map_mediator_port -- MOM_diag_mediator.F90
MOM_diag_remap.F90`:

1. **Mask setup** — `set_masks_for_axes` (`MOM_diag_mediator.F90:~800`, one-time init, not per
   timestep): the 8 per-coordinate `mask3d` arrays (`mTL, mCuL, mCvL, mBL, mTi, mCui, mCvi, mBi`) are
   now built with `do concurrent` directly on device-resident (`!$omp target enter data map(alloc:
   ...)`) arrays instead of plain host loops, then pulled back with one bulk
   `!$omp target exit data map(from: mTL, mCuL, mCvL, mBL, mTi, mCui, mCvi, mBi)` at the end.
2. **`diag_remap_calc_hmask`** (`MOM_diag_remap.F90:~496-550`) gained an explicit `h` argument (was
   implicitly `remap_cs%h`) and is rewritten as two `do concurrent` kernels — a plain
   `do concurrent(k, j, i)` zero-init and a `do concurrent(j, i) DO_LOCALITY(local(h_tot, h_err, k))`
   vanished-layer mask loop with a serial inner `k` loop. (The intermediate commit `b121aecbf`
   introduced these as `!$omp target teams distribute parallel do collapse(...)` kernels; the
   follow-up `fa796e4e0` "do concurrent the omp loops" converted them, so the **net branch state is
   `do concurrent`, not `omp target teams distribute`** — the whole branch mediator has **zero**
   `target teams distribute` and 29 `do concurrent`.) The routine carries the doc comment *"Both mask
   and h must already be present on the device (via prior enter data map)"* — i.e. it is now called
   with device-resident arguments and does no transfer inside itself.
3. **Downsampling** — `downsample_field_2d/3d`, `downsample_mask_2d/3d`, and the driver
   `downsample_diag_masks_set` are reworked to keep everything device-resident across the `dl=2,
   MAX_DSAMP_LEV` loop, with new local pointer aliases (`m2dT, m3dTL, ...`) used specifically **"to
   avoid derived-type deep-copy issues"** in `omp target` map clauses (the same trap as §2.3) — a
   comment states explicitly: *"downsample_mask expects field_in on device"* / *"outputs stay on
   device until the bulk map(from:) at the end of the c loop."*
4. **`post_data_2d_low`/`post_data_3d_low`** (`:1551-1582`, `:1892-1924`) — when a `conversion_factor`
   forces a host-side copy into `locfield`, that copy is now explicitly pushed onto device
   (`!$omp target enter data map(to: locfield)`) *before* `downsample_diag_field`/`downsample_field_*`
   run, and explicitly removed (`!$omp target exit data map(delete: locfield)`) afterward — avoiding
   an **implicit** map(to:)/map(from:) pair that the downsample kernels would otherwise trigger on
   their own, one field at a time.

> **FABLE-CHECK (reviewed 2026-07-14 — resolution or current status in KNOWLEDGE.md §8a/§8b):** The rewritten `diag_remap_calc_hmask`/`downsample_*` routines now *assume* their
> array arguments are already device-resident (comments "must already be present on the device",
> "expects field_in on device") and do no transfer themselves. Confirm every caller on
> `origin/diag_map_mediator_port` actually establishes that residency (a dominating `target enter data
> map(alloc/to:)` on the exact array passed) — a caller that hands in a host-only array would read
> uninitialized device memory silently. The `set_masks_for_axes` path aliases `axes%mask3d` to `mTL`
> and maps `mTL`; check the `h` argument threaded into `diag_remap_calc_hmask` is likewise mapped at
> every call site, not just the mask.

### 3.1 Answer to Q3 — offload the mediator, or just cut transfers?

**It is the latter, not the former.** The actual per-timestep hot path — `post_data_2d`/`post_data_3d`
dispatch, `post_data_2d_low`/`post_data_3d_low`'s masking/remap/`send_data_infra` sequence for the
*common* (no-conversion-factor, no-downsampling) case — is untouched. The branch instead targets the
**auxiliary, still-frequently-called machinery around** `post_data`: the one-time mask
setup/downsample-mask setup at init (§3.1/3.2 items 1–3, called once or per-restart, not per
timestep) and the **conversion+downsample side path inside `post_data_*_low`** (item 4, which only
fires for fields that have a non-1/0 unit conversion factor *and* are being downsampled). The goal
stated implicitly by the code comments throughout ("to avoid derived-type deep-copy issues", "expects
field_in on device", "keep field_in device-resident") is **transfer elimination for the
downsample/mask subsystem**, not turning the mediator into a GPU-resident diagnostics engine — the
FMS `diag_manager` write path (`send_data_infra`) fundamentally cannot be offloaded since FMS is
host-only in this fork, so a full "offload of the mediator" is not on the table; only its
device-adjacent pre-processing (masking, downsampling, unit conversion) is being made
transfer-free.

### 3.2 Sibling branches: `feat/new-diag-manager`, `port-fms-diags`

- `remotes/edoyango/port-fms-diags` diffs against `dev/gpu` at only **2 added lines** in
  `MOM_diag_mediator.F90:1122`: `!$omp target enter data map(to: axes, axes%mask3d)` at the end of
  `define_axes_group` — an exploratory first step toward keeping axis masks device-resident, clearly
  a precursor idea to (and superseded by) `diag_map_mediator_port`'s more complete mask handling.
- `remotes/edoyango/feat/new-diag-manager` is mostly build-system work (FMS submodule/yaml
  `configure.ac` support) plus one directly relevant commit, `b2a30750a` "Step_MOM_tracer_dyn: Remove
  data transfers" (not yet on `dev/gpu`): it **deletes** the blanket
  `!$omp target update from(h, CS%uhtr, CS%vhtr)` / `to(...)` pair that used to unconditionally
  bracket every call to `step_MOM_tracer_dyn` in `MOM.F90:996-1002`, and **relocates** the transfers
  *inside* `step_MOM_tracer_dyn` gated behind the actual conditions that need host data:
  `CS%debug` (checksums), `CS%use_particles .and. CS%use_uh_particles` (Lagrangian particle
  tracking, host-only), `associated(CS%OBC)` (open boundary tracer reservoirs), and
  `allocated(CS%tv%SpV_avg)` (derived thermodynamics). It also converts `CS%uhtr(:,:,:) = 0.0` /
  `CS%vhtr(:,:,:) = 0.0` resets into `do concurrent` device kernels. This is the same "push the
  transfer down to the actual conditional consumer" strategy as §2.1/§2.2, applied one call frame
  deeper than where `dev/gpu` currently does it — a preview of where the blanket
  `MOM.F90:1091`-style transfers are headed.

---

## 4. Remaining host-only regions in the timestep and the transfers they force (Q4)

Per `00-architecture.md` §6.3, the following are **untouched on `dev/gpu`** (diff line counts against
`dev-gfdl`): `MOM_diabatic_driver.F90` (3 lines), `MOM_set_diffusivity.F90` (0),
`MOM_CVMix_KPP.F90` (0), `MOM_energetic_PBL.F90` (0), `MOM_mixed_layer_restrat.F90` (0),
`MOM_ALE.F90` (3), `MOM_regridding.F90` (0), `MOM_remapping.F90` (0),
`MOM_diag_mediator.F90` (0), and (confirmed here) `MOM_restart.F90` (0).

Each host-only region is bracketed in `MOM.F90`/`step_MOM_thermo` by an explicit
`target update from(...)` / `target update to(...)` pair so the device-resident state stays
consistent around the host detour:

| Host-only region | Bracketing transfer (in) | Bracketing transfer (out) |
|---|---|---|
| `diabatic`/`layered_diabatic` (`MOM_diabatic_driver.F90`) | implicit — dycore state already host-visible via surrounding brackets | `MOM.F90:1043` explicit `map(to: CS%tv, CS%tv%T, CS%tv%S)` before `post_diabatic_halo_updates` |
| ALE regrid/remap (`ALE_regridding_and_remapping` → `MOM_ALE.F90`/`MOM_regridding.F90`/`MOM_remapping.F90`) | `MOM.F90:1036` `!$omp target update from(u, v, h)` | `MOM.F90:1038` `!$omp target update to(u, v, h)` |
| `write_energy` — T/S consumption (`MOM_sum_output.F90:762`) | `!$omp target update to(tv%S, tv%T)` (host-modified-by-diabatic T/S pushed back to device) | n/a (device-resident kernel follows) |
| `find_eta`/diagnostics needing `z_top`,`Z_0APE` (`MOM_diagnostics.F90:968`, `MOM_sum_output.F90:705`) | `target update to(h)` / `to(Z_0APE)` | `target exit data map(from: z_top)` |
| MEKE / thickness-diffuse / lateral mixing coeffs | `MOM_lateral_mixing_coeffs.F90:273` (`if (CS%calculate_cg1)`) and `:300` (`if (CS%BS_use_sqg_struct .or. ... .or. CS%id_sqg_struct>0)`), each `!$omp target update from(h)` before the host-only `wave_speed`/`calc_sqg_struct` | none in this file — `h` is pulled read-only, never pushed back |
| `calculate_diagnostic_fields` fan-out (all `id_xxx` diagnostics) | `MOM.F90:1091` blanket `target update from(u, v, h, CS%uhtr, CS%vhtr)` | none (read-only consumption) |
| Restart write/read (`MOM_restart.F90`, all of it) | **entirely implicit** — see §6 | **entirely implicit** |

**MEKE** (`MOM_MEKE.F90`) is not in the untouched list in §6.3 of the architecture doc but is
mentioned as a target-of-interest here; it was not found to have its own `target update from`
sites distinguishing it from the general dycore pattern — its interaction with diagnostics/host
regions is via the same `CS%visc`/`CDp` shared containers audited above (`MOM.F90:1380-1381`,
`1807-1812` guard `CS%visc%Ray_u/v`, `bbl_thick_u/v`, `Kv_bbl_u/v` — all `if (allocated(...))`
guarded rather than diagnostic-id guarded, since these are set-viscosity/BBL outputs, not
attached to a specific diagnostic id):
```fortran
!$omp target update from(CS%visc%Ray_u) if (allocated(CS%visc%Ray_u))
!$omp target update from(CS%visc%Ray_v) if (allocated(CS%visc%Ray_v))
!$omp target update from(CS%visc%bbl_thick_u) if (allocated(CS%visc%bbl_thick_u))
!$omp target update from(CS%visc%bbl_thick_v) if (allocated(CS%visc%bbl_thick_v))
!$omp target update from(CS%visc%Kv_bbl_u) if (allocated(CS%visc%Kv_bbl_u))
!$omp target update from(CS%visc%Kv_bbl_v) if (allocated(CS%visc%Kv_bbl_v))
```
(`MOM.F90:1807-1812`) — an `allocated()`-guarded variant of the same "only transfer if it's actually
going to be used" idiom, one level removed from a diagnostic id (these fields are allocated only
when the corresponding BBL/viscosity option is active, so the guard is a config-time rather than a
diagnostic-active-time gate).

---

## 5. Profiling: nvtx markers piggybacking on `cpu_clock` (Q5)

Branch `remotes/edoyango/benchmark_ALE_nvtx_clocks`, commit `ae67665d3` "add nvtx markers to clocks"
(+13/−0 twice, identical patch to both `config_src/infra/FMS1/MOM_cpu_clock_infra.F90` and
`config_src/infra/FMS2/MOM_cpu_clock_infra.F90`):

```fortran
use nvtx
...
integer, parameter :: MAX_NVTX_CLOCKS = 4096
character(len=64), save :: nvtx_clock_names(MAX_NVTX_CLOCKS) = ""
...
subroutine cpu_clock_begin(id)
  integer, intent(in) :: id
  if (id > 0 .and. id <= MAX_NVTX_CLOCKS) then
    if (len_trim(nvtx_clock_names(id)) > 0) call nvtxStartRange(trim(nvtx_clock_names(id)))
  endif
  call mpp_clock_begin(id)
end subroutine cpu_clock_begin

subroutine cpu_clock_end(id)
  integer, intent(in) :: id
  call mpp_clock_end(id)
  if (id > 0 .and. id <= MAX_NVTX_CLOCKS) then
    if (len_trim(nvtx_clock_names(id)) > 0) call nvtxEndRange
  endif
end subroutine cpu_clock_end

integer function cpu_clock_id(name, sync, grain)
  ...
  cpu_clock_id = mpp_clock_id(name, flags=clock_flags, grain=grain)
  if (cpu_clock_id > 0 .and. cpu_clock_id <= MAX_NVTX_CLOCKS) then
    nvtx_clock_names(cpu_clock_id) = name
  endif
end function cpu_clock_id
```

**Mechanism:** MOM6 already instruments essentially every named phase of the timestep with
`cpu_clock_id("name")` at init + `cpu_clock_begin`/`cpu_clock_end` pairs around the corresponding
code (this predates the GPU port — it is the pre-existing FMS `mpp_clock` profiling infra used for
the text-based clock summary at the end of a run). This commit **transparently wraps** that existing
infra: every clock name registered via `cpu_clock_id` is cached, and every subsequent
`begin`/`end` call also opens/closes an NVTX range of the same name — **with zero changes anywhere
else in the codebase**. This means `id_clock_diagnostics`, `id_clock_diag_mediator`,
`id_clock_pass`, `id_clock_thermo`, `id_clock_tracer`, `id_clock_dynamics`, `id_clock_other`, etc.
(the clocks bracketing exactly the diagnostics/IO regions audited in §1–§4) automatically show up
as named regions in an Nsight Systems (`nsys`) timeline, without writing any new instrumentation.

**What this reveals:** because the granularity is whatever the *existing* CPU-profiling clock
hierarchy already had, the nvtx timeline directly exposes (a) how much wall time
`id_clock_diagnostics`/`id_clock_diag_mediator` consume relative to the dycore clocks, and (b) via
the visual gaps/overlaps in the Nsight timeline, where a `target update from`/`to` pair (§2, §3, §4)
stalls the GPU stream waiting on a host-side diagnostics or restart region. It reuses the *pre-existing
naming convention* rather than requiring an nvtx-specific taxonomy, so anyone already familiar with
MOM6's clock summary output can read the nvtx timeline without relearning names.

`remotes/edoyango/devgpu-w-viscmlclock`, commit `5bbf92a3a` "add tmp clock" (+7 lines,
`MOM_set_viscosity.F90` only) is a much smaller, single-file exploratory addition of one extra
`cpu_clock_id`/`begin`/`end` pair around (implied by the branch name) the mixed-layer viscosity
computation — a manual instrumentation add-on rather than an infra change, presumably to get
finer-grained nvtx visibility into `set_viscous_ML` specifically once `ae67665d3`'s wrapper is in
place upstream of it.

---

## 6. Restart IO: fully host-staged (Q6)

`src/framework/MOM_restart.F90` has **zero** diff against `dev-gfdl` and **zero** OpenMP directives
of any kind (`grep -c "target update\|omp target\|do concurrent" MOM_restart.F90` → 0/0/0 across all
three patterns). The restart registry (`register_restart_field`, `save_restart`, `restore_state`) is
untouched: it operates purely on whatever host-resident copy of a field it is handed. This means
**every restart write requires the relevant device arrays to already have been pulled to host** by
whatever caller invoked the restart save — restart IO does not do its own transfer, it inherits
host-valid data from the surrounding `target update from` brackets already present in `MOM.F90`
(e.g. the same blanket `u,v,h,uhtr,vhtr` transfers at synchronization points, §2.2, cover most of
what a restart file needs) or, if none happens to be in scope at the restart-write call site, would
silently write stale host memory — a latent correctness risk worth flagging for anyone porting a
new CS member that also participates in the restart registry (`MOM_variables.F90:294` comment on
pointer members being restart-registry targets, cross-ref `00-architecture.md` §2.1). No branch in
this repo currently makes any part of the restart path device-aware; it is treated purely as a
serial, host-side bookkeeping concern (consistent with `.testing/tools/track_gpu_port.py`'s
`!@start noport` sentinel category described in `00-architecture.md` §8, though `MOM_restart.F90`
itself carries no such sentinels — it simply has nothing device-related to mark).

> **FABLE-CHECK (reviewed 2026-07-14 — resolution or current status in KNOWLEDGE.md §8a/§8b):** The "would silently write stale host memory" risk is an *inference*, not an
> observed bug. Confirm it: locate the actual `save_restart`/`restart_registry` write call sites in
> `MOM.F90`/`config_src/drivers/solo_driver/MOM_driver.F90` and check whether each is dominated by a
> preceding `target update from(...)` covering every registered device-resident field (not just
> `u,v,h,uhtr,vhtr`). If a restart-registered array that is *only* written on device (e.g. a
> pointer-member `Kd_shear`/`MLD` restart target from `MOM_variables.F90` §2.1) has no transfer before
> the write, the risk is real *today*; if all restart writes currently happen at synchronization
> points already covered by the §2.2 blanket transfers, it is only a latent trap for *future* ports.

---

## 7. Summary table — file status at a glance

| File | Diff vs `dev-gfdl` | Directives | Status |
|---|---|---|---|
| `MOM_diag_mediator.F90` | 0 (unchanged) | 0 | Fully host; `diag_map_mediator_port` in flight (§3) |
| `MOM_diag_remap.F90` | 0 on `dev/gpu` (changed only on `diag_map_mediator_port`) | 0 on `dev/gpu` | Same |
| `MOM_restart.F90` | 0 (unchanged) | 0 | Fully host, no in-flight branch |
| `MOM_diagnostics.F90` | +7/−0 | 3 guarded `target update from` | Minimal, transfer-minimization only |
| `MOM_sum_output.F90` (`write_energy`) | +34/−22 (56 lines touched) | multiple `do concurrent` + 2 `target update to` (`:705` `Z_0APE`, `:762` `tv%S,tv%T`) | **Compute is GPU-resident**; bridges `to(tv%S,tv%T)`/`to(Z_0APE)` from host-modified-by-diabatic inputs |
| `MOM.F90` (driver-level brackets) | part of +260/−40 | 56 `target update from`, ~6 guarded | Mostly unconditional blanket transfers around host-only regions |

---

## 8. Prescriptive rules for a porting agent (distilled)

When you port a module that computes device-resident fields and *also* posts diagnostics or feeds
the restart registry, apply these rules. They encode the trade-off the codebase has actually made:
**blanket `target update from` at coarse synchronization points, not per-field guards everywhere.**

### 8.1 Handling `post_data` calls in a newly ported module

1. **Never assume `post_data_*` sees device data.** `MOM_diag_mediator.F90` has **zero** OpenMP
   directives and is unchanged on `dev/gpu` (§1). Any array you pass to `post_data` must already be
   **host-valid**. The transfer is *your* responsibility as the producer, not the mediator's.
2. **Decouple the transfer from the post.** Do **one** `!$omp target update from(<fields>)` covering
   *all* diagnostics that consume those fields, then keep the individual `if (CS%id_xxx > 0) call
   post_data(...)` guards. Do not put a separate `update from` inside each `if (id>0)`. The canonical
   shape is `MOM_tracer_hor_diff.F90:719-733` (§2.1): a single guarded transfer covering both the
   `CS%debug` checksum path and both diag ids, followed by two independent `post_data` guards.
3. **Collapse multiple ids into one gate** with `any([...] > 0)` when several diagnostics derive from
   the same raw field — see `MOM_diagnostics.F90:1825-1827` (`uhtr`/`vhtr`/`h` each gate several
   downstream ids). This is strictly better than N separate transfers of the same array.
4. **Where to place the transfer:** immediately before the *first* consumer, at the widest scope
   where the field is still known host-valid. For a whole fan-out of unrelated diagnostics (the
   `calculate_diagnostic_fields` case) the codebase deliberately does **one blanket** transfer at the
   driver level (`MOM.F90:1091`, `target update from(u, v, h, CS%uhtr, CS%vhtr)`) rather than gating
   each of dozens of `if (id>0)` posts — simpler to reason about, at the cost of always paying the
   transfer at every synchronization boundary. **This blanket-at-sync-points choice is the codebase
   default; match it unless you have a measured reason to push guards deeper.**
5. **Know the guarded ideal and where it's headed.** The finer-grained alternative — relocate each
   transfer down to the *actual conditional consumer* (`CS%debug`, `associated(CS%OBC)`,
   `allocated(CS%tv%SpV_avg)`, a specific `id`) — is previewed on `feat/new-diag-manager`
   (`b2a30750a`, §3.2) and is the direction of travel, but is **not** what merged `dev/gpu` does at
   the `MOM.F90:1091` frame today. Prefer the blanket form for new work at sync points; use guarded
   push-down only when profiling (§5) shows the blanket transfer is a real stall.

### 8.2 Restart-registered arrays a module mutates on device

1. If your module `register_restart_field`s an array **and** mutates it inside a `target` region,
   that array is device-resident between mutation and the next restart write. `MOM_restart.F90` does
   **no** transfers of its own (§6) — it writes whatever host memory it is handed.
2. Ensure a `target update from(<array>)` dominates every `save_restart` call site that will write
   it. In practice the §2.2 blanket transfers at synchronization points already cover the core
   prognostic set; a **new** restart-registered device array outside that set (especially a
   pointer-member restart target, `MOM_variables.F90:294`/§2.1) needs its own transfer or it will be
   written stale. This is the concrete latent trap flagged in §6.
3. Do **not** try to make `MOM_restart.F90` device-aware — no branch does, and it is treated as
   serial host bookkeeping (the `!@start noport` category, `00-architecture.md` §8).

### 8.3 Checking you haven't left a stale-host trap

- **Grep discipline:** for every `post_data`/`save_restart`/host-only `call` in your ported module,
  confirm a `target update from` for its argument fields exists on every path reaching it. The
  transfer must be *upstream* (dominate) the consumer, not merely present in the file.
- **Bracket host-only detours:** any host-only region you cannot avoid (ALE remap, diabatic, energy
  sums) must be wrapped `from(...)` before / `to(...)` after so device state is restored — pattern
  `MOM.F90:1036/1038` (ALE) and `MOM_sum_output.F90:762` (`to(tv%S,tv%T)` pushing diabatic-modified
  host T/S back up). Direction matters: `from` when the host will *read* device data, `to` when the
  host has *modified* data the device must see next.
- **Verify with checksums, not eyeballs:** a stale-host transfer bug is invisible unless a
  diagnostic/restart field diverges. Compare `MOM_checksums` hchksum/uchksum and reproducing-sum
  energy output CPU vs GPU (`00-architecture.md` §7.2, §9) — a stale field shifts the bitcount
  checksum. `CS%debug`-gated `uvchksum` calls (e.g. `MOM_tracer_hor_diff.F90:725`) exist precisely to
  catch this class.
- **Watch the derived-type deep-copy trap:** touching `CS%tv%T`/`CS%tv%S` (or any derived-type member
  array) inside an `omp_offload` group pass or a `map` clause silently materializes many small
  implicit transfers instead of one bulk transfer (§2.3, `ff86497d5`). Wrap such calls in an explicit
  `map(to: CS%tv, CS%tv%T, CS%tv%S)` / `map(from: ...)` bracket, or alias the member to a bare pointer
  first (the `diag_map_mediator_port` `mTL`/`m2dT` pattern, §3).

## 9. Cross-references

- `00-architecture.md` §2.3 (derived-type deep-copy cost — the same trap fixed in `ff86497d5` and
  worked around throughout `diag_map_mediator_port`'s pointer-alias pattern), §7.4, §8 (port-coverage
  tooling / `!@start noport` sentinels, relevant to why `MOM_restart.F90` has none).
- `03-openmp-mapping.md` for the general enter-data/exit-data lifecycle idiom reused throughout §2–§3
  here.

---

## Verification notes

Verified against source + git on branch `dev/gpu` (baseline `dev-gfdl`); no code built or run.

**Confirmed exactly (checked against code/git):**
- `MOM_diag_mediator.F90` and `MOM_restart.F90` empty `dev-gfdl...dev/gpu` diffs; `MOM_restart.F90`
  0/0/0 for `target update`/`omp target`/`do concurrent`; `MOM_diag_remap.F90` empty on `dev/gpu`.
- Total `omp target update from` across `src/` = **249**, and every per-file count in §2 (MOM.F90 56,
  MOM_barotropic 52, MOM_hor_visc 53, dyn_split_RK2 45, CoriolisAdv 11, set_viscosity 7,
  tracer_hor_diff 6, PressureForce_FV 5, vert_friction 4, interface_heights 3, diagnostics 3,
  lateral_mixing_coeffs 2, tracer_advect 1, state_initialization 1) — all exact.
- All guarded-transfer sites and their guard conditions: `MOM_tracer_hor_diff.F90:722`,
  `MOM_diagnostics.F90:1825/1826/1827`, `MOM_PressureForce_FV.F90:1301`. Blanket sites `MOM.F90:1091`,
  the commented-out dead duplicate near `:1113`, ALE brackets `MOM.F90:1036/1038`. `allocated()`-guarded
  visc transfers `MOM.F90:1807-1812`; `lateral_mixing_coeffs.F90:273/300`.
- Commit `ff86497d5` (#174, author uwagura): reconstructed from the diff — 5 insertions/1 deletion,
  adds `map(to: CS%tv, CS%tv%T, CS%tv%S)` before / `map(from:)` + `map(release:)` after the
  `post_diabatic_halo_updates` call (`MOM.F90:1042-1046`) and removes the `! TODO: Safe? what about T
  and S?` comment at the group pass. Derived-type deep-copy characterization is sound.
- `write_energy` transfers: `MOM_sum_output.F90:705` `to(Z_0APE)`, `:762` `to(tv%S, tv%T)` — direction
  and host-diabatic-mutation rationale confirmed.
- `diag_map_mediator_port` content: pointer-alias mask setup (`mTL/mCuL/mCvL/mBL`, `m2dT/m3dTL/...`),
  `diag_remap_calc_hmask` gained explicit `h` arg + "must already be present on the device" comment,
  `downsample_diag_masks_set` rework, `locfield` `map(to:)`/`map(delete:)` around the
  conversion+downsample side path in `post_data_2d_low`/`post_data_3d_low`. Branch does **not** touch
  `send_data_infra`/FMS or the top-level `post_data` dispatch (Q3 answer "cut transfers, not offload
  the mediator" is correct). Per-commit stats `b121aecbf` +267/−73, `fa796e4e0` +82/−121,
  `072db9355` +5/−7 all match.
- Sibling branches: `port-fms-diags` = 2 added lines (`map(to: axes, axes%mask3d)` at
  `define_axes_group`); `feat/new-diag-manager` `b2a30750a` touches MOM.F90 + diagnostics +
  tracer_advect + tracer_hor_diff (consistent with the "push transfers to conditional consumer"
  description). nvtx commit `ae67665d3` +13/+13 to both FMS1/FMS2 `MOM_cpu_clock_infra.F90` — quoted
  code (MAX_NVTX_CLOCKS=4096, nvtxStartRange/EndRange wrapping) matches verbatim. `5bbf92a3a` +7 lines,
  `MOM_set_viscosity.F90` only.
- All §1 call-map line numbers (post_data generic `:74`, `post_data_2d :1408`, `_2d_low :1436`,
  `post_data_3d :1585`, `_3d_low :1750`, `by_column :1927`, `by_point :1945`,
  `diag_update_remap_grids :3655-3759`, `diag_copy_diag_to_storage :4130-4146`,
  `diag_copy_storage_to_diag :4149-4164`) and the `calculate_diagnostic_fields` `id_u/id_v/id_h/id_usq`
  guards at `MOM_diagnostics.F90:305-320`. Summary-table diffstats `MOM_diagnostics +7/−0`,
  `MOM.F90 +260/−40` confirmed.

**Corrected:**
1. **§3 branch/commit provenance:** the *local* `diag_map_mediator_port` ref here has only 2 of the 3
   commits; `072db9355` exists only on `origin/diag_map_mediator_port` (remote is one commit ahead).
   Doc now qualifies the ref and gives both net diffstats (local +276/−121, remote +277/−124).
2. **§3 item 2 `diag_remap_calc_hmask` kernel form:** the *net* branch state uses two **`do concurrent`**
   kernels, not `!$omp target teams distribute parallel do collapse(...)`. The omp-teams form existed
   only in the intermediate commit `b121aecbf`; `fa796e4e0` converted it (branch mediator: 0
   `target teams distribute`, 29 `do concurrent`). Corrected in place with the history noted.
3. **§7 summary table `MOM_sum_output.F90`:** was `+56/−22`; the true diffstat is **+34/−22** (56 lines
   *touched*). Corrected.

**Confidence:** High. Every load-bearing factual claim (empty diffs, the 249 count and all per-file
counts, the five guarded sites and their conditions, `ff86497d5`, the `write_energy` transfers, the
`diag_map_mediator_port` scope and its four offload items, the nvtx wrapper code, all §1 line numbers)
was verified directly against source or git and matches. The two open items are flagged as
FABLE-CHECK: (a) whether the restart "stale host" risk is live *today* vs latent for future ports, and
(b) whether the `diag_map_mediator_port` device-residency contract is honored by all callers.
