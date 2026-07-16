# Halos, Domain Decomposition, and Multi-GPU Communication (dev/gpu)

> **Purpose.** How MOM6's domain decomposition and halo-exchange infrastructure
> (`MOM_domains.F90` → `config_src/infra/FMS2/MOM_domain_infra.F90` → FMS `mpp_domains`) was adapted
> for GPU residency, and what breaks when the domain is spread across more than one GPU. Companion to
> `00-architecture.md` §3.3 (index/halo conventions) and §7.3 (halo/`omp_offload` summary), and to
> `09-barotropic-solver.md` (deep dive on the wide-halo `CS%BT_Domain` sub-cycle — this document only
> summarizes that mechanism and instead concentrates on the cross-module `omp_offload` inventory, the
> nonblocking pattern, and the multi-GPU tracer-advection bugfix). All line numbers are against
> `dev/gpu` unless a commit hash is given.

---

## 1. Domain decomposition and the halo model

### 1.1 Data domain vs. computational domain

Every PE (MPI rank) owns a rectangular tile of the global grid. `hor_index_type`
(`src/framework/MOM_hor_index.F90:18`), replicated into `ocean_grid_type` (`src/core/MOM_grid.F90:28`),
carries two sets of bounds per staggering:

- **Computational domain** — `isc:iec`, `jsc:jec` — the cells this PE actually owns and updates.
- **Data domain** — `isd:ied`, `jsd:jed` — computational domain **+ halo** — the cells this PE can
  read, filled by neighbor exchange.
- **Global domain** — `isg:ieg` — indices into the whole simulation, used only for I/O/diagnostics.

`NIHALO_ = NJHALO_ = 2` is the default halo width (`00-architecture.md` §3.1, `config_src/memory/`).
In **symmetric memory** mode, the B/C-grid (velocity/corner) index `IsdB` starts one lower than `isd`
(`MOM_hor_index.F90:90-96`: `HI%IsdB = HI%isd ; ... ; if (HI%symmetric) HI%IsdB = HI%isd - 1`), so a
`u`-point array is `u(IsdB:IedB, jsd:jed)` and a corner array is `q(IsdB:IedB, JsdB:JedB)`. This is
purely an indexing convention (`MOM_memory_macros.h` `NIMEMB_`/`NIMEMB_SYM_`); it does not change how
many halo rings are physically exchanged.

A `MOM_domain_type` (`config_src/infra/FMS2/MOM_domain_infra.F90`) wraps an FMS `domain2D`
(`mpp_domain` member) plus MOM-specific bookkeeping (`nihalo`/`njhalo`, `symmetric`,
`nonblocking_updates`, `thin_halo_updates`). `G%Domain` is the "normal" halo=2 domain used by nearly
everything; `CS%BT_Domain` in `MOM_barotropic.F90` is a **second, wider-halo** domain cloned from it
(§5).

### 1.2 `create_group_pass` — batching multiple fields into one exchange

`MOM_domains.F90:51` re-exports the whole halo-update API from
`config_src/infra/FMS2/MOM_domain_infra.F90`:

```fortran
public :: create_group_pass, do_group_pass, group_pass_type, start_group_pass, complete_group_pass
```

`create_group_pass` is a generic interface (`MOM_domain_infra.F90:91-96`) over 2D/3D scalar and vector
field registration (`create_var_group_pass_2d/3d`, `create_vector_group_pass_2d/3d`). The batching
trick: callers pass the **same** `group_pass_type` variable (a CS member, e.g. `CS%pass_h`,
`CS%pass_uv`) across several calls, each with a different field:

```fortran
! src/core/MOM_dynamics_split_RK2.F90:512-514
call create_group_pass(CS%pass_hp_uv, hp, G%Domain, halo=cor_stencil)
call create_group_pass(CS%pass_hp_uv, u_av, v_av, G%Domain, halo=max(cor_stencil,vel_stencil))
call create_group_pass(CS%pass_hp_uv, uh(:,:,:), vh(:,:,:), G%Domain, halo=max(cor_stencil,vel_stencil))
```

Internally (`create_var_group_pass_3d`, `MOM_domain_infra.F90:985-1026`):

```fortran
if (mpp_group_update_initialized(group)) then
  call mpp_reset_group_update_field(group, array)      ! append another field to an existing group
elseif (present(halo) .and. MOM_dom%thin_halo_updates) then
  call mpp_create_group_update(group, array, MOM_dom%mpp_domain, flags=dirflag, position=position, &
                               whalo=halo, ehalo=halo, shalo=halo, nhalo=halo)
else
  call mpp_create_group_update(group, array, MOM_dom%mpp_domain, flags=dirflag, position=position)
endif
```

The first call on a given `group_pass_type` initializes it (`mpp_create_group_update`); every
subsequent call with a *different array* on the *same* `group` variable is detected as
"already initialized" and appends the field via `mpp_reset_group_update_field` instead of
re-creating the group. One `do_group_pass(group, ...)` later then issues **a single underlying
MPI exchange batching all registered fields** (u, v, h, T, S, ... in one message per neighbor
direction), instead of one exchange per field. This is a pre-existing FMS/MOM6 optimization (predates
the GPU port) that the GPU port reuses unchanged — the `create_group_pass` calls in
`MOM_dynamics_split_RK2.F90:506-517` are re-executed every dycore call (cheap: they only touch the
group's field list, not the data), and `do_group_pass` is what actually moves halo data.

Per-pass halo width is tailored to the *consuming* kernel's stencil rather than always using the
maximum `NIHALO_=2`: `cor_stencil = CoriolisAdv_stencil(...)`, `vel_stencil = max(2, obc_stencil,
hor_visc_vel_stencil(...))`, `cont_stencil = continuity_stencil(...)`
(`MOM_dynamics_split_RK2.F90:498-504`), and each `create_group_pass` call passes `halo=` a `max(...)`
of whichever stencils touch that particular field. `pass_eta` uses `halo=1` (`:506`) because only a
1-point stencil ever reads `eta`'s halo. Smaller halo ⇒ smaller message ⇒ less data crossing the
device/host and PE/PE boundary per exchange — directly relevant to GPU communication cost.

---

## 2. The `omp_offload` mechanism — GPU-aware halo exchange

### 2.1 The infra change

Baseline (`dev-gfdl`) `do_group_pass` took no offload argument. Commit `656e09013` ("add flag for
gpu2gpu do_group_update mpi transfers for latest fms", the only commit touching
`config_src/infra/FMS2/MOM_domain_infra.F90` between `dev-gfdl` and `dev/gpu` —
`git log --oneline dev-gfdl..dev/gpu -- config_src/infra/FMS2/MOM_domain_infra.F90` shows only a
license-relicense commit and a `dev/gfdl` merge besides it) added the parameter:

```fortran
! config_src/infra/FMS2/MOM_domain_infra.F90:1143-1162
subroutine do_group_pass(group, MOM_dom, clock, omp_offload)
  type(group_pass_type), intent(inout) :: group
  type(MOM_domain_type), intent(inout) :: MOM_dom
  integer,     optional, intent(in)    :: clock
  logical,     optional, intent(in)    :: omp_offload !< Whether the data to be transferred is
                                                      !! offloaded to the GPU with OpenMP.
  real :: d_type
  if (present(clock)) then ; if (clock>0) call cpu_clock_begin(clock) ; endif
  call mpp_do_group_update(group, MOM_dom%mpp_domain, d_type, omp_offload)
  if (present(clock)) then ; if (clock>0) call cpu_clock_end(clock) ; endif
end subroutine do_group_pass
```

`mpp_do_group_update` itself lives in the external FMS `mpp_domains` library (fetched at build time via
`ac/deps/Makefile.fms.in`, not vendored in this repo), so the actual GPU-aware implementation (CUDA-
aware MPI send/recv directly on device pointers, vs. staging through a host buffer) cannot be
inspected here — only the MOM-side call contract.

> **Resolved (2026-07-14):** the FMS `omp_offload` path is a genuine device path with **no** fallback.
> In the sibling FMS checkout, `mpp_group_update.fh` device-packs halos (`target teams distribute …
> if(use_device_ptr)` into a device buffer) and `mpp_transmit_mpi.fh` posts `MPI_ISEND`/`IRECV` inside
> `!$omp target data use_device_ptr(...)` — real CUDA-aware MPI on device pointers. There is no
> capability check: a non-GPUDirect MPI stack means a crash or corruption, **not** graceful host
> staging. The nonblocking variants hardcode `use_device_ptr = .false. ! placeholder`, which confirms
> the gated/unconditional split in §4 from the FMS side.

This parameter exists **only** in the FMS2
infra shim; `config_src/infra/FMS1/MOM_domain_infra.F90:1144` still has the old
`do_group_pass(group, MOM_dom, clock)` signature with no `omp_offload`. `ac/configure.ac:238-241`
auto-selects FMS2 vs FMS1 based on whether the linked FMS provides `fms2_io_mod`, so a GPU build
implicitly requires a modern-enough FMS with the offload-aware `mpp_do_group_update` overload — this
is a build-time external dependency, not something visible in `src/`.

`pass_var`/`pass_vector` (single-field, non-grouped passes) have **no** `omp_offload` parameter at
all (`MOM_domain_infra.F90:173,220`). Only the *batched, group* path was extended for GPU-awareness;
single-field passes remain host-mediated by construction (§4).

### 2.2 What the change replaced — before/after in `MOM_barotropic.F90`

The same commit (`656e09013`) simultaneously deleted a large amount of manual host-staging code in
`MOM_barotropic.F90`, which is the clearest illustration of what `omp_offload=.true.` buys. Before:

```fortran
! before 656e09013 (illustrative, from the diff's "-" lines)
!$omp target update from(bt_rem_u, bt_rem_v, eta_src)
!$omp target update if(integral_BT_cont) from(eta_IC)
! ... 5 more conditional "from" staging lines ...
call do_group_pass(CS%pass_eta_bt_rem, CS%BT_Domain)
!$omp target update to(bt_rem_u, bt_rem_v, eta_src)
!$omp target update if(integral_BT_cont) to(eta_IC)
! ... 5 more conditional "to" staging lines ...
if (.not.use_BT_cont) then
  !$omp target update from(Datu, Datv)
  call do_group_pass(CS%pass_Dat_uv, CS%BT_Domain)
  !$omp target update to(Datu, Datv)
endif
```

After:

```fortran
! src/core/MOM_barotropic.F90:1772-1774 (dev/gpu)
call do_group_pass(CS%pass_eta_bt_rem, CS%BT_Domain, omp_offload=.true.)
if (.not.use_BT_cont) call do_group_pass(CS%pass_Dat_uv, CS%BT_Domain, omp_offload=.true.)
call do_group_pass(CS%pass_force_hbt0_Cor_ref, CS%BT_Domain, omp_offload=.true.)
```

Ten-plus `!$omp target update from/to` directives (each a D2H then H2D transfer of every field in the
group, done unconditionally around the pass) collapsed into three plain `omp_offload=.true.` calls.
The same pattern recurs at four more spots in the same commit/file (`:929/:1439/:1641/:1896` in the
pre-fix numbering). **Data-residency implication:** when `omp_offload=.true.`, the halo exchange reads
and writes the fields *in place on the device* — the arrays are expected to already be
`!$omp target enter data`-mapped, and the exchange does not require (and the MOM-side code no longer
performs) a round-trip through a host mirror. When the flag is absent/false (the nonblocking branches,
§4, and all bare `pass_var`/`pass_vector` calls), the exchange is host-mediated and the caller must
stage the data there-and-back manually with `!$omp target update from(...)` / `to(...)`.

One TODO-turned-real-fix, in the same commit's diff at the inner sub-cycle loop
(`MOM_barotropic.F90:2757`, `btstep_timeloop`):

```fortran
-      ! TODO: direct GPU-to-GPU transfer
-      !$omp target update from(ubt, vbt, eta)
-      call do_group_pass(CS%pass_eta_ubt, CS%BT_Domain, clock=id_clock_pass_step)
-      !$omp target update to(ubt, vbt, eta)
+      call do_group_pass(CS%pass_eta_ubt, CS%BT_Domain, clock=id_clock_pass_step, omp_offload=.true.)
```

— the barotropic inner-loop halo pass (the one amortized by the wide halo, §5) is exactly the "direct
GPU-to-GPU transfer" the TODO asked for.

---

## 3. Every `omp_offload=.true.` call site

`grep -rn omp_offload src/ config_src/` returns **29** lines: 3 belong to the single FMS2
`do_group_pass` definition (the `subroutine` line `:1143`, the `omp_offload` argument declaration
`:1152`, and the `mpp_do_group_update` call `:1158` — all in `config_src/infra/FMS2/`, since FMS1 has
no such parameter), leaving **26** call sites (the architecture doc's "~25" estimate, §7.3).
*(Verified: `grep -rn omp_offload src/ config_src/ | wc -l` = 29.)* Grouped by module:

| # | File:line | Group-pass handle | Domain | Fields (from `create_group_pass`) | Context / stencil |
|---|---|---|---|---|---|
| 1 | `MOM.F90:742` | `pass_tau_ustar_psurf` | `G%Domain` | `forces%taux,tauy` (`:732`), `ustar` (`:734`), `tau_mag` (`:736`), `p_surf` (`:738`) — the latter two only `if (associated(...))` | top of `step_MOM`; **gated** — this `omp_offload=.true.` blocking call is the `else` branch of `if (nonblocking_p_surf_update)` (`:739-743`). Note `:681` enter-data maps only `forces, forces%taux/tauy/ustar` (**not** `tau_mag`/`p_surf`) |
| 2 | `MOM.F90:2112` | `pass_uv_T_S_h` | `G%Domain` | `u,v` (`:2106`), `tv%T`,`tv%S` (`:2108,2110`), `h` (`:2111`) | `halo=dynamics_stencil = min(3,nihalo,njhalo)` (`:2105`); the "GPU-aware" pass cited in `00-architecture.md` §4.1 |
| 3 | `MOM_dynamics_split_RK2.F90:663` | `pass_eta` | `G%Domain` | `eta` (`halo=1`, `:506`) | predictor stage, blocking branch (paired with nonblocking start at `:580`/complete `:658`) |
| 4 | `MOM_dynamics_split_RK2.F90:664` | `pass_visc_rem` | `G%Domain` | `CS%visc_rem_u,visc_rem_v` (`:507`, `halo=max(1,cont_stencil)`) | predictor stage |
| 5 | `MOM_dynamics_split_RK2.F90:841` | `pass_visc_rem` | `G%Domain` | same as #4 | predictor stage, second occurrence (after `vertvisc_remnant`) |
| 6 | `MOM_dynamics_split_RK2.F90:847` | `pass_uvp` | `G%Domain` | `up,vp` (`:509`, `halo=max(1,cont_stencil)`) | blocking branch (nonblocking start `:829`, complete `:844`) |
| 7 | `MOM_dynamics_split_RK2.F90:860` | `pass_hp_uv` | `G%Domain` | `hp` (`halo=cor_stencil`), `u_av,v_av` and `uh,vh` (`halo=max(cor_stencil,vel_stencil)`) (`:512-514`) | after predictor `continuity` |
| 8 | `MOM_dynamics_split_RK2.F90:1131` | `pass_visc_rem` | `G%Domain` | same as #4 | corrector stage |
| 9 | `MOM_dynamics_split_RK2.F90:1137` | `pass_uv` | `G%Domain` | `u_inst,v_inst` (`:510`, `halo=max(2,cont_stencil)`) | blocking branch (nonblocking start `:1118`, complete `:1134`) |
| 10 | `MOM_dynamics_split_RK2.F90:1153` | `pass_h` | `G%Domain` | `h` (`:515`, `halo=max(cor_stencil,cont_stencil)`) | after corrector `continuity` |
| 11 | `MOM_dynamics_split_RK2.F90:1165` | `pass_av_uvh` | `G%domain` | `u_av,v_av`, `uh,vh` (`:516-517`, `halo=max(cor_stencil,vel_stencil)`) | blocking branch (nonblocking start `:1163`) |
| 12 | `MOM_barotropic.F90:1008` | `CS%pass_q_DCor` | `CS%BT_Domain` | `q` (position `CORNER`), `DCor_u,DCor_v` (`:822-823`) | wide-halo; blocking branch |
| 13 | `MOM_barotropic.F90:1550` | `CS%pass_gtot` | `CS%BT_Domain` | `gtot_E,gtot_N` / `gtot_W,gtot_S` (`:828-830`) | wide-halo |
| 14 | `MOM_barotropic.F90:1551` | `CS%pass_ubt_Cor` | `G%Domain` | `ubt_Cor,vbt_Cor` (`:862`) | normal halo |
| 15 | `MOM_barotropic.F90:1772` | `CS%pass_eta_bt_rem` | `CS%BT_Domain` | `eta_src`, `bt_rem_u/v`, `eta_PF*`, `eta_IC`, `dyn_coef_eta`, `Rayleigh_u/v` (`:834-848`) | wide-halo |
| 16 | `MOM_barotropic.F90:1773` | `CS%pass_Dat_uv` | `CS%BT_Domain` | `Datu,Datv` (`:857`) | wide-halo; only if `.not. use_BT_cont` |
| 17 | `MOM_barotropic.F90:1774` | `CS%pass_force_hbt0_Cor_ref` | `CS%BT_Domain` | `BT_force_u/v`, `uhbt0/vhbt0`, `Cor_ref_u/v` (`:853-855`) | wide-halo |
| 18 | `MOM_barotropic.F90:2022` | `CS%pass_e_anom` | `G%Domain` | `e_anom` (`:868`) | normal halo |
| 19 | `MOM_barotropic.F90:2070` | `CS%pass_ubta_uhbta` | `G%Domain` | `CS%ubtav,vbtav`, `uhbtav,vhbtav` (`:869-870`) | normal halo |
| 20 | `MOM_barotropic.F90:2757` | `CS%pass_eta_ubt` | `CS%BT_Domain` | `eta`, `ubt,vbt` (`:2740-2741`) | **the inner sub-cycle pass**, amortized by wide halos (§5) |
| 21 | `MOM_barotropic.F90:5290` | `BT_cont%pass_polarity_BT` | `BT_Domain` | `u_polarity,v_polarity`, `uBT_EE/vBT_NN`, `uBT_WW/vBT_SS` (`:5279-5281`) | face-area closure setup |
| 22 | `MOM_barotropic.F90:5291` | `BT_cont%pass_FA_uv` | `BT_Domain` | `FA_u_EE/FA_v_NN`, `FA_u_E0/FA_v_N0`, `FA_u_W0/FA_v_S0`, `FA_u_WW/FA_v_SS` (`:5283-5286`) | face-area closure setup |
| 23 | `MOM_tracer_advect.F90:277` | `CS%pass_uhr_vhr_t_hprev` | `G%Domain` | `uhr,vhr` (`:180`), `hprev` (`:181`), `Reg%Tr(m)%t` for each tracer (`:183`) | inner advection iteration `do itt=1,max_iter` |
| 24 | `MOM_tracer_hor_diff.F90:573` | `CS%pass_t` | `G%Domain` | `Reg%Tr(m)%t` per tracer (`:244`) | neutral-diffusion iteration |
| 25 | `MOM_tracer_hor_diff.F90:882` | `CS%pass_t` | `G%Domain` | same | mixed-layer/buffer-layer density-coordinate setup |
| 26 | `MOM_tracer_hor_diff.F90:1283` | `CS%pass_t` | `G%Domain` | same | along-surface diffusion iteration `do itt=1,num_itts` |

**Corrected module counts** (recounted from the grep above — the fields sum to the 26 total):
**11** sites are in `MOM_barotropic.F90` (**8** on the wide-halo `BT_Domain` — rows 12,13,15,16,17,20
plus rows 21,22 whose `BT_Domain` dummy is `CS%BT_Domain`, passed at `MOM_barotropic.F90:1217/1219`),
9 in `MOM_dynamics_split_RK2.F90`, 2 in `MOM.F90`, 1 in `MOM_tracer_advect.F90`, 3 in
`MOM_tracer_hor_diff.F90` ⇒ 11+9+2+1+3 = 26. *(An earlier draft said "15 (7 on `CS%BT_Domain`)"; that
was wrong — the table itself has only 11 barotropic rows, 8 of them wide-halo, and 15+9+2+1+3=30≠26.)*
See `09-barotropic-solver.md` §2.2 for a barotropic-only version of this table with additional
`btstep`-internal commentary.

### 3.1 Where halos are still done on host

Every call above is a **grouped** pass. Bare, single-field `pass_var`/`pass_vector` calls never carry
`omp_offload` (the parameter doesn't exist on that entry point, §2.1) and are therefore host-mediated
by default — if the field is device-resident, the caller must bracket the call with manual
`!$omp target update from(...)` / `to(...)`, or the pass simply operates on a host-only field never
mapped to device in the first place. Confirmed examples in modules the architecture doc lists as
"largely untouched" on `dev/gpu` (§6.3):

```
src/parameterizations/lateral/MOM_thickness_diffuse.F90:2234:    call pass_var(CS%khth2d, G%domain)
src/parameterizations/lateral/MOM_mixed_layer_restrat.F90:365:    call pass_var(mle_fl_2d, G%domain, halo=1)
src/parameterizations/lateral/MOM_mixed_layer_restrat.F90:680:    call pass_var(h, G%domain, To_West+To_South+Omit_Corners, halo=1)
src/parameterizations/lateral/MOM_mixed_layer_restrat.F90:864:    call pass_var(bflux, G%domain, halo=1)
src/parameterizations/lateral/MOM_mixed_layer_restrat.F90:1150:   call pass_var(h, G%domain, To_West+To_South+Omit_Corners, halo=1)
src/parameterizations/lateral/MOM_mixed_layer_restrat.F90:1479:   call pass_var(h, G%domain, To_West+To_South+Omit_Corners, halo=1)
src/parameterizations/lateral/MOM_mixed_layer_restrat.F90:1752:   call pass_var(CS%MLD_Tfilt_space, G%domain)
src/parameterizations/lateral/MOM_mixed_layer_restrat.F90:1764:   call pass_var(CS%Cr_space, G%domain)
src/parameterizations/lateral/MOM_mixed_layer_restrat.F90:1954-1956: pass_var(CS%MLD_filtered/_slow/wpup_filtered, G%domain)
```

`MOM_mixed_layer_restrat.F90` and `MOM_thickness_diffuse.F90` are both `+0`-diff (untouched) on
`dev/gpu` per `00-architecture.md` §6.3 — these `h`, `bflux`, `khth2d` halo updates run entirely on
the CPU today; if/when those modules are ported (branch `edoyango/port/thickness_diffuse`,
`bodner-naive-port`), these calls are exactly where `create_group_pass`/`omp_offload=.true.` would
need to be introduced to keep the halo exchange on-device.

---

## 4. The nonblocking group-pass pattern (`start_group_pass`/`complete_group_pass`)

`MOM_domains.F90:49` re-exports the non-blocking single-field entry points
(`pass_var_start/complete`, `pass_vector_start/complete`) and `:51` the non-blocking **group** entry
points used on the GPU-relevant hot path. Both are gated at runtime by `NONBLOCKING_UPDATES`
(`src/framework/MOM_domains.F90:215`, `G%nonblocking_updates = G%Domain%nonblocking_updates`,
`MOM_grid.F90:300`; in `MOM_barotropic.F90` the same flag is cached as `nonblock_setup =
G%nonblocking_updates`, `:789`).

**Key finding: `start_group_pass`/`complete_group_pass` have no `omp_offload` parameter at all**
(`MOM_domain_infra.F90:1165,1186`) — only the blocking `do_group_pass` was extended. So whenever
`G%nonblocking_updates` is true, the code path is *not* GPU-aware and must stage data through the host
manually around the non-blocking send/receive, exactly like the pre-`656e09013` barotropic code
(§2.2). The pattern, repeated at every dycore group-pass site:

```fortran
! src/core/MOM_dynamics_split_RK2.F90:656-665
if (G%nonblocking_updates) then
  call complete_group_pass(CS%pass_eta, G%Domain)
  !$omp target update to(eta)
  !$omp target update from(CS%visc_rem_u, CS%visc_rem_v)
  call start_group_pass(CS%pass_visc_rem, G%Domain)
else
  call do_group_pass(CS%pass_eta, G%Domain, omp_offload=.true.)
  call do_group_pass(CS%pass_visc_rem, G%Domain, omp_offload=.true.)
endif
```

i.e. **the nonblocking path and the GPU-offload path are mutually exclusive branches of the same
`if`.** Where such an `if (G%nonblocking_updates)/(nonblock_setup)/(nonblocking_p_surf_update)` guard
exists, turning on `NONBLOCKING_UPDATES` reverts *that* site to host-staged communication (extra D2H
before `start_group_pass`, extra H2D after `complete_group_pass`), trading GPU-resident halo exchange
for compute/communication overlap on the host side.

> **CORRECTED — scope of the revert.** An earlier draft claimed enabling `NONBLOCKING_UPDATES`
> "silently reverts **every one** of the 26 sites in §3." That is **not true**. Only the sites that
> physically sit in an `if (…nonblocking…)/else` block have a nonblocking counterpart; the rest are
> **unconditional** `do_group_pass(…, omp_offload=.true.)` calls that stay GPU-aware regardless of the
> flag. Recounted against the code:
>
> | | Gated (reverts to host-staging when `NONBLOCKING_UPDATES=.true.`) | Always `omp_offload=.true.` (flag-independent) |
> |---|---|---|
> | `MOM.F90` | `:742` `pass_tau_ustar_psurf` (else of `nonblocking_p_surf_update`, `:739`) | `:2112` `pass_uv_T_S_h` (no guard) |
> | `MOM_dynamics_split_RK2.F90` | `:663` `pass_eta`, `:664` `pass_visc_rem`, `:847` `pass_uvp`, `:1137` `pass_uv`, `:1165` `pass_av_uvh` | `:841` `pass_visc_rem`, `:860` `pass_hp_uv`, `:1131` `pass_visc_rem`, `:1153` `pass_h` |
> | `MOM_barotropic.F90` | `:1008` `pass_q_DCor` (else of `:1004`), `:1550` `pass_gtot`, `:1551` `pass_ubt_Cor` (else of `:1539`), `:1772/:1773/:1774` `pass_eta_bt_rem`/`pass_Dat_uv`/`pass_force_hbt0_Cor_ref` (else of `:1763`), `:2022` `pass_e_anom` (else of `:2017`), `:2070` `pass_ubta_uhbta` (else of `:2064`) | `:2757` `pass_eta_ubt` (inner sub-cycle, §5), `:5290` `pass_polarity_BT`, `:5291` `pass_FA_uv` |
> | `MOM_tracer_advect.F90` | — | `:277` `pass_uhr_vhr_t_hprev` (no nonblocking in file) |
> | `MOM_tracer_hor_diff.F90` | — | `:573`, `:882`, `:1283` `pass_t` (no nonblocking in file) |
>
> **14 gated, 12 unconditional (14+12 = 26).** So `NONBLOCKING_UPDATES` disables the GPU-aware path at
> the 14 gated sites only — notably the barotropic *inner sub-cycle* pass (`:2757`, the hottest
> exchange, §5) and **all four tracer passes** keep `omp_offload=.true.` no matter what. The flag
> therefore *weakens* GPU-residency for the outer dynamics/barotropic-setup exchanges but does **not**
> globally defeat GPU-aware exchange.

One gated site (`MOM.F90:732-744`, `pass_tau_ustar_psurf`) is the exception that proves the staging
rule: its nonblocking branch (`:739-740`) has *no* explicit `target update from` before
`start_group_pass`, because `forces%taux/tauy/ustar` were only just
`!$omp target enter data map(to: ...)`'d at `:681` (that map covers `forces, forces%taux, forces%tauy,
forces%ustar` — **not** `tau_mag`/`p_surf`, which are host-resident here) — host and device copies are
still identical at that point, so starting the nonblocking send from the (still valid) host mirror
needs no extra staging.

`MOM.F90:743`/`:818` (cited in the task) are exactly `pass_tau_ustar_psurf`'s
`do_group_pass(...,omp_offload=.true.)` (blocking branch, `:742`) and its nonblocking counterpart's
`complete_group_pass` (`:818`), gated by the same `nonblocking_p_surf_update` flag
(`:727-729`, a refinement of `G%nonblocking_updates` that also requires `p_surf`/`SpV_avg`/`T` not to
be simultaneously in play).

**Conclusion for Q6:** the nonblocking API is preserved and actively used (it predates the GPU port
and is wired through *most* dynamics/barotropic group passes), but it is **not GPU-offload-aware** — it
is a CPU-communication-overlap feature that, when enabled, disables the `omp_offload` GPU-aware path
**at the 14 gated sites** (falling back to explicit host staging) while leaving the 12 unconditional
sites — including the barotropic inner sub-cycle and all tracer passes — on the GPU-resident path.

---

## 5. The wide-halo barotropic domain — minimizing exchange frequency

Full algorithmic treatment in `09-barotropic-solver.md` §1.2/§2.2; summary here for the
communication/multi-GPU angle.

`CS%BT_Domain` (`MOM_barotropic.F90:338`) is created once, in `barotropic_init`, by cloning the normal
domain with a larger minimum halo:

```fortran
! src/core/MOM_barotropic.F90:6103-6104
! Initialize a version of the MOM domain that is specific to the barotropic solver.
call clone_MOM_domain(G%Domain, CS%BT_Domain, min_halo=wd_halos, symmetric=.true.)
```

`wd_halos` comes from the runtime parameters `BT_USE_WIDE_HALOS` (default `.true.`) and `BTHALO`
(minimum halo size, default 0 ⇒ under dynamic memory `wd_halos = bt_halo_sz` as configured; under
`STATIC_MEMORY_` it is fixed by the `WHALOI_`/`WHALOJ_` macros, `:47-51`,
`WHALOI_ = MAX(BTHALO_-NIHALO_, 0)`). `clone_MD_to_d2D` (`MOM_domain_infra.F90:1717-1780`) takes
`max(existing_halo, min_halo)` — the wide-halo domain has the **same PE layout / decomposition** as
`G%Domain`, only a bigger halo ring width.

Why this matters for communication volume: `btstep_timeloop` (`MOM_barotropic.F90:2376`) runs
`nstep+nfilter` small barotropic sub-steps per call, each of which only needs a 1–2 point stencil
update of `eta`/`ubt`/`vbt`. Rather than issuing a halo exchange on every sub-step, the valid
(non-communicated) index range is allowed to **shrink by `stencil` points per sub-step** ("march
inward"), and a `do_group_pass(..., omp_offload=.true.)` is issued only once the valid range would no
longer cover the true computational domain:

```fortran
! src/core/MOM_barotropic.F90:2621-2630, 2754-2758
stencil = max(1, CS%min_stencil)
num_cycles = 1
if (CS%use_wide_halos) &
  num_cycles = min((is-CS%isdw) / stencil, (js-CS%jsdw) / stencil)
isvf = is - (num_cycles-1)*stencil ; ievf = ie + (num_cycles-1)*stencil
jsvf = js - (num_cycles-1)*stencil ; jevf = je + (num_cycles-1)*stencil
...
do n=1,nstep+nfilter
  ...
  if ((iev - stencil < ie) .or. (jev - stencil < je)) then
    call do_group_pass(CS%pass_eta_ubt, CS%BT_Domain, clock=id_clock_pass_step, omp_offload=.true.)
    isv = isvf ; iev = ievf ; jsv = jsvf ; jev = jevf
  else
    isv = isv - stencil ; iev = iev + stencil ; jsv = jsv - stencil ; jev = jev + stencil  ! march inward, no comm
  endif
enddo
```

With `num_cycles` sub-steps amortized per exchange, the number of `omp_offload=.true.` MPI/NVSHMEM-
style exchanges across the barotropic sub-cycle drops from `O(nstep)` to `O(nstep/num_cycles)`. This
is the single biggest lever for reducing communication (and hence device-buffer synchronization)
overhead in the whole dycore, because `nstep` (set by `set_dtbt`, `MOM_barotropic.F90:3797`,
cited in `00-architecture.md` §4.3) is typically O(10)–O(30) sub-cycles per outer dynamics step. Every
exchange on `CS%BT_Domain` in the table in §3 (rows 12,13,15,16,17,20,21,22) benefits from this same
wider halo even outside the inner sub-cycle loop — e.g. `pass_q_DCor` and `pass_eta_bt_rem` are
one-shot per `btstep` call but still use the wide halo so their downstream consumers (`btstep_timeloop`
itself) can march inward before needing another exchange.

---

## 6. Device↔host transfer boundaries and data residency

Putting §2–§4 together, the rule of thumb on `dev/gpu` is:

- **Grouped, `omp_offload=.true.` passes** (§3, 26 sites): operate directly on device-resident arrays.
  No corresponding `!$omp target update` is needed immediately around the call — the calling code is
  expected to already have the fields `enter data`-mapped (typically once, at CS init or at the top of
  `step_MOM`/`step_MOM_dyn_split_RK2`).
- **Nonblocking passes** (`start_group_pass`/`complete_group_pass`, §4): always host-mediated;
  explicit `!$omp target update from(...)` precedes `start_group_pass` and `!$omp target update
  to(...)` follows `complete_group_pass`, except where the host mirror is already known-valid
  (the `pass_tau_ustar_psurf` case, §4).
- **Bare `pass_var`/`pass_vector`** (§3.1): host-mediated by construction; used almost exclusively in
  modules that are still entirely CPU (`MOM_mixed_layer_restrat.F90`, `MOM_thickness_diffuse.F90`),
  so in practice these fields (`h`, `bflux`, `khth2d`, MLD filters) simply never leave the host in the
  first place inside those routines — no transfer is "forced" so much as the whole subroutine is
  outside the mapped region.
- **`redistribute_array_*`/`global_field`/`broadcast_domain`** (`MOM_domain_infra.F90:1207-1261` and
  neighboring routines, re-exported at `MOM_domains.F90:20,44`): used for domain-to-domain
  redistribution (e.g. coarsening, I/O gather) — not touched by the GPU port at all, and by
  construction operate on host arrays (`mpp_redistribute`, no offload argument), so any GPU-resident
  field passed to them requires a manual `target update from` beforehand — none of the omp_offload
  call sites in §3 route through this family.

The forcing-field entry point at `step_MOM` is the cleanest illustration of the residency contract:
`forces%taux/tauy/ustar` arrive from the coupler on the host, are `!$omp target enter data map(to:
...)`'d exactly once (`MOM.F90:681`), and every subsequent group pass over them
(`pass_tau_ustar_psurf`, §3 row 1) runs `omp_offload=.true.` for the remainder of that `step_MOM` call
— the H2D transfer happens once per coupling step, not once per halo exchange.

---

## 7. Multi-GPU: the `MOM_tracer_advect.F90` answer-change bugfix

Branch `remotes/edoyango/bugfix-traceradvection-multigpu` (rebased commits `d28afbf32`,
`680f927e7`), merged into `dev/gpu` as `a774eb331` and `e182de310`. Both fix real correctness bugs
that **only manifest when the domain is split across more than one GPU** (i.e. more than one MPI rank,
each bound to its own device) — a single-GPU/single-rank run does not exercise the code paths that
expose either bug. `git show e182de310` / `git show a774eb331` for the full diffs; key excerpts below.

### 7.1 Bug 1 — unreduced write to `domore_k(k)` (`e182de310`, "advect_tracer: fix multi gpu answer change")

`advect_tracer` (`MOM_tracer_advect.F90`) tracks, per vertical layer `k`, whether any more advection
iterations are needed on this PE via an integer flag array `domore_k(1:nz)`. Before the fix:

```fortran
domore_k(k) = 0
do concurrent (j=jsv:jev, domore_u(j,k))
  domore_k(k) = 1
enddo
do concurrent (J=jsv+stencil-1:jev-stencil, domore_v(J,k))
  domore_k(k) = 1
enddo
```

Every iteration that fires writes the **same** value (`1`) to the **same** array element
`domore_k(k)`, with no `reduce`/`local` locality-spec on the `do concurrent`. Per ISO Fortran 2018
semantics, a `do concurrent` construct must not have iterations that access the same variable unless
that access pattern is declared (`local`/`local_init`/`reduce`); an unguarded shared write like this
is technically nonconforming, and on `nvfortran`'s GPU lowering of `do concurrent` the compiler needs
an explicit `reduce` locality-spec to generate correct synchronized/atomic accumulation into a shared
scalar — without it, nothing guarantees the write from an arbitrary GPU thread actually lands in
global memory before the loop's implicit barrier, or that a value written by one thread block isn't
overwritten by another block's stale copy. The fix:

```fortran
! e182de310, src/tracer/MOM_tracer_advect.F90
domore_k_tmp = 0
do concurrent (j=jsv:jev, domore_u(j,k)) DO_LOCALITY(reduce(max:domore_k_tmp))
  domore_k_tmp = 1
enddo
do concurrent (J=jsv+stencil-1:jev-stencil, domore_v(J,k)) DO_LOCALITY(reduce(max:domore_k_tmp))
  domore_k_tmp = 1
enddo
domore_k(k) = domore_k_tmp
```

using a **new scalar temporary** `domore_k_tmp` rather than the array element `domore_k(k)` directly,
because (per the commit message) "do concurrent can't use array elems yet" as a reduction target —
Fortran's `reduce` locality-spec (and `nvfortran`'s support for it, gated by
`HAVE_FC_DO_CONCURRENT_LOCAL` / the `DO_LOCALITY` macro, `src/framework/do_concurrent_compat.h`) only
accepts scalar variables. Three call sites in `advect_tracer` had this pattern
(`:298-311`, `:335-347`, `:362-374` in the pre-fix file); all three were converted identically. The
same commit also removed now-redundant staging (`!$omp target` bracket around a plain
`domore_k(k) = 0` reset, and a `!$omp target update from(domore_k)` before the `sum_across_PEs`
reduction across PEs) since `domore_k` no longer needs a manual round trip once the device-side
reduction is correct.

**Why only multi-GPU:** the write pattern is undefined-behavior-adjacent on *any* GPU execution, but
whether it produces a wrong answer depends on how many independent thread blocks/teams the compiler
launches to cover the `j`/`J` range and whether their partial results are ever lost before the implicit
end-of-loop synchronization. A single-GPU (single-rank) run's per-rank `jsv:jev` range spans the whole
(large) global domain, likely scheduled by the runtime as one arrangement of teams the compiler
happens to handle correctly (or the CPU fallback masks it entirely). A multi-GPU run decomposes the
domain into many smaller per-rank tiles — different loop trip counts, different team/block counts per
kernel launch — which is exactly the situation where a missing `reduce` clause is more likely to
surface as a dropped update (`domore_k(k)` silently staying `0` when it should be `1`), causing
`advect_tracer` to terminate its outward iteration early on some layers/PEs and silently
under-advecting a tracer — an "answer change" that differs by GPU count rather than being wrong on
every run. (This mechanism is inferred from the code and Fortran locality semantics; the commit
message states only the symptom — "leading to answer changes on multiple GPUs" — and the fix, not the
precise compiler-internal cause.)

### 7.2 Bug 2 — `Reg%Tr(:)` mapped `alloc` instead of `to` (`a774eb331`, "tracer_advect: fix map of Reg%Tr(:)")

```fortran
! before (e182de310's state)
!$omp target enter data map(to: OBC) map(alloc: domore_u, domore_v, uhr, vhr, uh_neglect, &
!$omp   vh_neglect, hprev, local_advect_scheme, Reg, Reg%Tr(:))
! after (a774eb331)
!$omp target enter data map(to: OBC, Reg, Reg%Tr(:)) map(alloc: domore_u, domore_v, uhr, vhr, uh_neglect, &
!$omp   vh_neglect, hprev, local_advect_scheme)
```

`Reg` (the tracer registry, `tracer_registry_type`) and its array-of-derived-type component `Reg%Tr(:)`
(one `tracer_type` per registered tracer) were being `map(alloc:)`'d — i.e. the device gets freshly
allocated, **uninitialized** device memory for the registry structure, with no host→device copy. But
`advect_tracer` reads host-set scalar members of `Reg%Tr(m)` **inside** a `do concurrent` immediately
after this enter-data region:

```fortran
! MOM_tracer_advect.F90:149-152
do concurrent (m = 1:ntr)
   local_advect_scheme(m) = Reg%Tr(m)%advect_scheme     ! <-- reads a host-set scalar member
   if (local_advect_scheme(m) < 0) local_advect_scheme(m) = CS%default_advect_scheme
   ...
```

`Reg%Tr(m)%advect_scheme` is set once at tracer registration time on the host and never written on
device — with `map(alloc:)`, the device's copy of this scalar is whatever garbage happened to occupy
that freshly-allocated device memory, not the registered value. With `map(to:)`, it is the correct
host value copied down. (The commit's companion `map(release:)` vs `map(from:)` cleanup on the exit
side, and moving `hprev` from a `from`-mapped release to a plain `release`, are related tidying of the
same enter/exit-data region but not the correctness fix itself.)

**Why only multi-GPU:** whether stale/garbage device memory for `advect_scheme` "happens to" produce
the right branch outcome depends on whatever was previously resident in that memory region on that
specific device — a function of allocator history, prior kernel launches, and each GPU's own
allocation pool. A single-GPU/single-rank test run exercises exactly one allocator instance, and it is
plausible for it to coincidentally return zeroed or otherwise-benign memory (e.g. a freshly-allocated
region on a lightly used device, or a value that happens to be a valid enum member of
`ADVECT_PLM/PPM/PPMH3`). Once the same code runs across multiple ranks/GPUs, each device's allocator
has an independent (and generally different) history, so the garbage value read back differs **per
GPU** — producing inter-rank inconsistency in `local_advect_scheme`, hence a different advection
scheme selected on different PEs for what should be the same tracer, hence a genuine multi-GPU-only
answer change (and, being a garbage-memory read, potentially also nondeterministic run-to-run).
Notably this fix is co-authored by the current user (`Co-authored-by: Jorge Luis Gálvez Vallejo
<jorgegalvez1694@gmail.com>` in the `a774eb331` commit trailer).

---

## 8. Summary answers

1. **Domain/halo model:** computational (`isc:iec`) vs. data (`isd:ied` = computational + `NIHALO_=2`
   halo) domains, symmetric-memory `IsdB=isd-1` B-grid offset, `create_group_pass` batches many fields
   onto one `group_pass_type` handle so `do_group_pass` issues a single exchange per group; halo width
   per pass is tailored to the consuming stencil (`cor_stencil`/`vel_stencil`/`cont_stencil`), not
   always the full `NIHALO_`.
2. **`omp_offload` end-to-end:** `do_group_pass`'s optional `omp_offload` (`MOM_domain_infra.F90:1143`)
   forwards straight to FMS's `mpp_do_group_update(...,omp_offload)` (external library, not in this
   repo); when true, the exchange reads/writes device-resident buffers directly (no host round-trip);
   commit `656e09013` shows the conversion from ~10 manual `!$omp target update from/to` pairs around
   a bare `do_group_pass` to a single `omp_offload=.true.` call at 5+ sites in `MOM_barotropic.F90`,
   including replacing a literal `! TODO: direct GPU-to-GPU transfer` comment.
3. **26 `omp_offload=.true.` call sites** across `MOM.F90` (2), `MOM_dynamics_split_RK2.F90` (9),
   `MOM_barotropic.F90` (**11**, **8** of them on the wide-halo `BT_Domain`), `MOM_tracer_advect.F90`
   (1), `MOM_tracer_hor_diff.F90` (3) — full table in §3. Bare `pass_var`/`pass_vector` (no
   `omp_offload` parameter exists on that entry point) remain host-only, concentrated in the
   still-unported `MOM_mixed_layer_restrat.F90` and `MOM_thickness_diffuse.F90`.
4. **Multi-GPU bug:** two independent, sequential fixes in `MOM_tracer_advect.F90` — (a) `domore_k(k)`
   written from multiple `do concurrent` iterations without a `reduce` locality-spec, fixed by
   reducing into a scalar temporary (`reduce(max:domore_k_tmp)`) because array elements aren't valid
   `do concurrent` reduction targets yet; (b) `Reg`/`Reg%Tr(:)` mapped `alloc` instead of `to`, so a
   host-set scalar (`Reg%Tr(m)%advect_scheme`) read inside a device `do concurrent` got uninitialized
   device memory instead of its registered value. Both are latent on any GPU run but only produce
   observable answer changes when multiple independent devices/allocators are involved (more/different
   team-launch configurations for bug (a); independent per-device garbage-memory contents for bug (b)).
5. **Device↔host boundaries:** forced at (i) coupler ingest (`forces%*` `enter data` once per
   `step_MOM`, `MOM.F90:681`), (ii) every nonblocking group pass (`start_group_pass`/
   `complete_group_pass` have no `omp_offload` — always host-staged), (iii) every bare
   `pass_var`/`pass_vector` call, and (iv) redistribution/global-field routines (host-only, untouched
   by the port). The wide-halo `CS%BT_Domain` (cloned via `clone_MOM_domain(..., min_halo=wd_halos)`,
   `MOM_barotropic.F90:6104`) minimizes *how often* GPU-aware exchanges happen in the barotropic
   sub-cycle by letting `btstep_timeloop` "march inward" for `num_cycles` sub-steps between
   `do_group_pass(..., omp_offload=.true.)` calls, turning `O(nstep)` exchanges into
   `O(nstep/num_cycles)`.
6. **Nonblocking pattern:** `start_group_pass`/`complete_group_pass` are preserved and wired to *many*
   dycore/barotropic group passes, gated by runtime flag `NONBLOCKING_UPDATES` /
   `G%nonblocking_updates` (`nonblock_setup` in barotropic), but they have no offload awareness —
   enabling them reverts the **14 gated** sites (of 26) to manual host staging (`!$omp target update
   from` before `start`, `to` after `complete`), the opposite of the GPU-resident `omp_offload=.true.`
   blocking path in the same `if/else`. The other **12 sites** — including the barotropic inner
   sub-cycle (`:2757`) and all four tracer passes — are unconditional `do_group_pass(…,
   omp_offload=.true.)` calls unaffected by the flag (breakdown table in §4).

---

## 9. Prescriptive rules for a porting agent

Distilled from §§1–8 and cross-checked against the merged commits. Every rule is grounded in a
citation you can re-open.

### 9.1 You are porting a module that calls `pass_var`/`pass_vector`

These single-field entry points have **no** `omp_offload` parameter (`MOM_domain_infra.F90:173`
`pass_var_3d`, `:220` `pass_var_2d`, `:516/:662` `pass_vector_*` — none take it, and FMS1 lacks the
group offload arg entirely, `FMS1/MOM_domain_infra.F90:1144`). If the field the pass touches becomes
device-resident, you have two options:

- **Option A — leave it host-staged, bracket the pass.** Keep the `call pass_var(field, G%domain,
  …)` and wrap it: `!$omp target update from(field)` immediately before, `!$omp target update
  to(field)` immediately after. This is exactly what the pre-`656e09013` barotropic code did (§2.2)
  and what every *nonblocking* branch still does (§4). *Precondition:* `field` is already
  `enter data`-mapped. *When to prefer:* a one-off pass, a rarely-hit path, or a field only
  transiently on device — the round-trip cost is paid once and the diff stays tiny.

- **Option B — promote to a grouped, offload-aware pass.** Add a `type(group_pass_type)` handle as a
  CS member (e.g. `CS%pass_x`), register the field(s) with `create_group_pass(CS%pass_x, field,
  G%Domain, halo=<stencil>)` (batch several fields onto the *same* handle to get one MPI message,
  §1.2), then replace the pass with `call do_group_pass(CS%pass_x, G%Domain, omp_offload=.true.)`.
  The fields are then exchanged in place on the device with no host round-trip (§2.2, §6).
  *Preconditions:* (i) the fields are `!$omp target enter data`-mapped for the lifetime of the pass;
  (ii) the build links an FMS whose `mpp_do_group_update` accepts the offload flag — this is implicit
  in any FMS2 build (`ac/configure.ac:238-241` selects FMS2 when `fms2_io_mod` is present) but is an
  **external** dependency not vendored here (§2.1); (iii) if you place the offload call in an
  `if (G%nonblocking_updates)/else`, remember the nonblocking branch is **not** offload-aware and must
  still be hand-staged (§4) — or omit the guard entirely (as the tracer passes and barotropic inner
  sub-cycle do) to stay unconditionally GPU-aware. *When to prefer:* a hot-path pass on fields that
  live on-device across the whole routine — this is the blessed pattern (all 26 sites in §3).

### 9.2 Halo-width-vs-stencil rule

Pass `halo=` sized to the *consuming* kernel's stencil, never reflexively the full `NIHALO_=2`.
The dynamics driver computes `cor_stencil`/`vel_stencil`/`cont_stencil` once
(`MOM_dynamics_split_RK2.F90:498-504`) and each `create_group_pass` requests `halo=max(...)` of only
the stencils that read *that* field (`:506-517`); `pass_eta` uses `halo=1` because only a 1-point
stencil reads `eta`'s halo. Smaller halo ⇒ smaller message ⇒ less data crossing the PE/PE (and, when
`omp_offload`, device) boundary (§1.2). Corollary: if your ported kernel widens a stencil, widen the
matching `create_group_pass` `halo=` or you will read stale halo cells.

### 9.3 `map(to:)` (not `map(alloc:)`) for registry-like structs read on device

Any derived type — especially an **array-of-derived-type** component — whose *host-set scalar members*
are read inside a device region must be `map(to:)`, so the host values are copied down;
`map(alloc:)` gives the device freshly-allocated **uninitialized** memory. This is precisely the
`a774eb331` bug: `Reg`/`Reg%Tr(:)` were `map(alloc:)`'d, and `Reg%Tr(m)%advect_scheme` (set once at
registration, never on device) was read in a `do concurrent` at `MOM_tracer_advect.F90:149-152`,
yielding garbage (§7.2). The fix moved `Reg, Reg%Tr(:)` into the `map(to:)` clause. Rule: *if a device
loop reads it and the host wrote it, it is `to`, not `alloc`.*

### 9.4 `reduce` locality-spec for shared-scalar accumulation in `do concurrent`

A `do concurrent` whose iterations all write the same scalar/array-element needs an explicit
`DO_LOCALITY(reduce(op:var))` locality-spec, or `nvfortran`'s GPU lowering may drop updates (the
`e182de310` `domore_k` multi-GPU answer-change, §7.1). Two sub-rules from that fix: (i) the reduction
target must be a **scalar** — array elements like `domore_k(k)` are not yet valid `reduce` targets, so
reduce into a scalar temp and assign back (`domore_k(k) = domore_k_tmp`); (ii) once the device-side
reduction is correct, drop the now-redundant `!$omp target update from(...)` staging that previously
existed only to let the host recompute the value (`e182de310` also removed `domore_k` from the
`enter data` map).

### 9.5 Why these two classes of bug only surface multi-GPU

Both `MOM_tracer_advect.F90` fixes are latent on *any* GPU run but only change answers across
multiple devices (§7). The porting lesson: **single-GPU correctness does not prove a port correct.**
A missing `reduce` (9.4) depends on the team/block launch geometry, which changes with per-rank tile
size; an `alloc`-vs-`to` slip (9.3) depends on per-device allocator history. Validate ports at
≥2 ranks/GPUs, and compare `MOM_checksums` (`00-architecture.md` §7.2) across GPU counts, not just
CPU-vs-single-GPU.

---

## Verification notes

Verified against `dev/gpu` source and git (baseline `dev-gfdl`) on 2026-07-14. No code was built or
run. Line/commit references below were re-derived independently of the original draft.

### Confirmed (spot-checked in code/git, correct as written)

- **Domain/halo model.** `IsdB = isd-1` in symmetric mode: `MOM_hor_index.F90:90-96` (assignment
  `HI%IsdB = HI%isd` then `if (HI%symmetric) HI%IsdB = HI%isd-1` at `:92-95`) — confirmed. `NIHALO_=2`
  default (per `00-architecture.md` §3.1) — consistent.
- **Group-pass batching.** `create_group_pass` generic + `mpp_create_group_update`/
  `mpp_reset_group_update_field` append-on-reinit mechanics — confirmed at
  `MOM_domain_infra.F90` (create routines) and the `MOM_dynamics_split_RK2.F90:498-517` stencil-sized
  registration (halos `cor_/vel_/cont_stencil`, `pass_eta` `halo=1`) — all confirmed verbatim.
- **`omp_offload` plumbing.** FMS2 `do_group_pass(group, MOM_dom, clock, omp_offload)` at `:1143`
  forwards to `mpp_do_group_update(..., omp_offload)` at `:1158` — confirmed. `start_group_pass`
  (`:1165`) / `complete_group_pass` (`:1186`) and `pass_var_3d`/`pass_var_2d` (`:173`/`:220`),
  `pass_vector_*` (`:516`/`:662`) take **no** `omp_offload` — confirmed. FMS1 `do_group_pass` (`:1144`)
  has the old 3-arg signature — confirmed.
- **Commit `656e09013`.** Subject, and that it is the *only* substantive commit touching
  `FMS2/MOM_domain_infra.F90` between `dev-gfdl` and `dev/gpu` (besides a relicense and a `dev/gfdl`
  merge) — confirmed via `git log`. Its barotropic before/after (staging→`omp_offload`) at pre-fix
  hunks `:929/:1439/:1641/:1896` (plus `:2578` btstep_timeloop TODO and `:5023`
  set_local_BT_cont_types) — confirmed in the diff.
- **Multi-GPU bugfixes.** `e182de310` (`domore_k` → scalar `domore_k_tmp` with
  `DO_LOCALITY(reduce(max:...))`, `domore_k` dropped from the `enter data` map) and `a774eb331`
  (`Reg, Reg%Tr(:)` moved `map(alloc:)`→`map(to:)`, `hprev` `from`→`release`) — both diffs confirmed
  verbatim; `e182de310` is an ancestor of `a774eb331` (order in §7 correct); Jorge Gálvez co-author
  trailer on `a774eb331` confirmed.
- **Wide-halo clone + march-inward.** `clone_MOM_domain(G%Domain, CS%BT_Domain, min_halo=wd_halos,
  symmetric=.true.)` at `MOM_barotropic.F90:6103-6104`, and the `btstep_timeloop` march-inward /
  `do_group_pass(CS%pass_eta_ubt,...,omp_offload=.true.)` at `:2757` — confirmed.
- **`ac/configure.ac` FMS selection.** `AX_FC_CHECK_MODULE([fms2_io_mod], ...)` selecting FMS2 vs FMS1
  at `:238-241` — confirmed (draft's citation accurate).
- **Critical-claim mechanism.** `start/complete_group_pass` lack `omp_offload` and sit in the opposite
  branch of `if (G%nonblocking_updates)` from the `omp_offload=.true.` `do_group_pass` — **confirmed**
  verbatim at `MOM_dynamics_split_RK2.F90:657-665` and the barotropic `if (nonblock_setup)/else` blocks.

### Corrected

1. **Barotropic site count.** Draft said "**15** sites in `MOM_barotropic.F90` (**7** on
   `CS%BT_Domain`)". `grep` shows **11** barotropic call sites, **8** of them wide-halo (`BT_Domain`);
   11+9+2+1+3 = 26 (the draft's 15+9+2+1+3 = 30 ≠ 26 was internally inconsistent, and its own §5 lists
   8 wide-halo rows). Fixed in §3 summary and §8.3. (Rows 21/22's `BT_Domain` dummy = `CS%BT_Domain`,
   passed at `:1217/1219`, so they *are* wide-halo.)
2. **Scope of the `NONBLOCKING_UPDATES` revert (the strongest correction).** Draft said enabling it
   "silently reverts **every one** of the 26 sites." Recounting against the code: only **14** sites
   sit in an `if(…nonblocking…)/else` guard and revert to host staging; the other **12** are
   **unconditional** `do_group_pass(…, omp_offload=.true.)` calls (dynamics `:841/:860/:1131/:1153`,
   `MOM.F90:2112`, barotropic inner-sub-cycle `:2757` and setup `:5290/:5291`, and **all four** tracer
   passes) that stay GPU-aware regardless of the flag. Rewrote §4 (with a gated-vs-unconditional
   table), and the §4 conclusion / §8.6.
3. **`grep` accounting.** Draft: "the two infra definitions plus 26 call sites." It is **one** FMS2
   `do_group_pass` definition spanning **3** grep-matched lines (`:1143/:1152/:1158`) + 26 call sites =
   29 lines. Clarified in §3.
4. **`:681` enter-data contents.** Draft implied `forces%…/p_surf` are all mapped at `MOM.F90:681`.
   That line maps only `forces, forces%taux, forces%tauy, forces%ustar` — **not** `tau_mag`/`p_surf`.
   Corrected in §3 row 1 and §4.

### Enhancements added

- New **§9 "Prescriptive rules for a porting agent"**: (9.1) two options for porting a
  `pass_var`/`pass_vector` module — host-staged bracket vs. promote-to-grouped-offload — with
  preconditions each; (9.2) halo-width-vs-stencil rule; (9.3) `map(to:)`-not-`map(alloc:)` rule for
  registry-like structs; (9.4) `reduce` locality-spec rule (scalar-temp workaround); (9.5) why single-
  GPU correctness is insufficient. All grounded in the citations verified above.

### Confidence

**High** for everything checked directly against source/git (all §§1–7 code excerpts, the two commits,
the 26-site table, the gated/unconditional split, the corrected counts). Also **high** for the claims
resting on the external FMS library: the `omp_offload` device path has since been verified against the
FMS source (§2.1). **Medium** for the *inferred* "why only multi-GPU"
causal mechanisms in §7, which the draft already flags as inference from Fortran/compiler semantics
rather than from commit messages — that framing is appropriate and left as-is.
