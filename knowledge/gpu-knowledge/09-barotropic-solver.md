# The Barotropic Solver — `MOM_barotropic.F90` (dev/gpu)

> **Purpose.** Deep dive on `btstep`, the barotropic (fast-mode) sub-cycle that `MOM_dynamics_split_RK2.F90`
> calls twice per baroclinic step (predictor + corrector, `00-architecture.md` §4.2 step 6/12). At
> 6868 lines and 242 `do concurrent` / 106 `omp target` occurrences it is the single most
> directive-dense module on `dev/gpu`, and it is the only module using a **wide-halo domain**
> (`CS%BT_Domain`) to amortize MPI cost over many fast time steps. Read `00-architecture.md` §4.2–4.3
> first. All line numbers below are against `src/core/MOM_barotropic.F90` on `dev/gpu` unless noted.

---

## 1. Algorithm structure and the sub-cycle

### 1.1 Call shape

`btstep` (`:480`) is called from `MOM_dynamics_split_RK2.F90:726` (predictor) and `:1023` (corrector).
Its job: given baroclinic accelerations (`bc_accel_u/v`), layer velocities/pressure-force terms, and
the provisional free-surface `eta_in`, integrate the 2‑D (depth-averaged) shallow-water equations
through many small time steps `dtbt` to cover one full dynamics step `dt`, returning barotropic
accelerations (`accel_layer_u/v`), the final `eta_out`, and time-averaged transports (`uhbtav`,
`vhbtav`). Internally it:

1. Sets up wide-halo copies of `eta`, Coriolis PV `q`, and various coefficient arrays (`:850`–`:1300`
   region) — `linearized_BT_PV` branch pre-computes `q` from `CS%q_D` "quite early... to start the
   halo update that needs to be completed before the next calculations" (comment at `:883`).
2. Calls `btstep_find_Cor` (`:3159`) to build the Coriolis coefficient arrays `f_4_u`/`f_4_v` used in
   the C-grid Coriolis bracket.
3. Calls `btstep_ubt_from_layer` (`:3671`) to project the 3-D layer velocities down onto an initial
   barotropic velocity (`ubt`, `vbt`) using the weights `wt_u`/`wt_v` (see §3 below for the
   reproducibility bug in this weight calculation).
4. Calls `BT_cont_to_face_areas` / `set_local_BT_cont_types` to build the face-area closure
   (`BT_cont_type`, see §6) used for the nonlinear continuity option `use_BT_cont`.
5. Calls `btstep_timeloop` (`:2376`) — **the actual sub-cycle**, described in §1.2.
6. Calls `btstep_layer_accel` (`:3723`) to project the resulting barotropic acceleration back onto
   each layer, weighted by `visc_rem_u/v`, producing `accel_layer_u/v`.

### 1.2 `btstep_timeloop` — why the wide halo matters

`btstep_timeloop` (`:2376`) runs `do n=1,nstep+nfilter` (`:2751`, closing `:3143`), alternately
updating `eta`, then `u`/`v` (order alternates by parity: `v_first = (MOD(n+G%first_direction,2)==1)`,
`:2804`), then the transports `uhbt`/`vhbt`. Each of these local updates only needs a 1- or 2-point
stencil (`stencil = max(1, CS%min_stencil)`, `:2622`, bumped to 2 for nonlinear continuity with a
finite update period, `:2623-2624`).

The key trick is in the "march inward" logic at `:2621-2630` and `:2754-2763`:

```fortran
! Figure out the fullest arrays that could be updated.
stencil = max(1, CS%min_stencil)
...
num_cycles = 1
if (CS%use_wide_halos) &
  num_cycles = min((is-CS%isdw) / stencil, (js-CS%jsdw) / stencil)
isvf = is - (num_cycles-1)*stencil ; ievf = ie + (num_cycles-1)*stencil
jsvf = js - (num_cycles-1)*stencil ; jevf = je + (num_cycles-1)*stencil
...
do n=1,nstep+nfilter
  ...
  ! Update the range of valid points, either by doing a halo update or by marching inward.
  if ((iev - stencil < ie) .or. (jev - stencil < je)) then
    call do_group_pass(CS%pass_eta_ubt, CS%BT_Domain, clock=id_clock_pass_step, omp_offload=.true.)
    isv = isvf ; iev = ievf ; jsv = jsvf ; jev = jevf
  else
    isv = isv+stencil ; iev = iev-stencil
    jsv = jsv+stencil ; jev = jev-stencil
  endif
```
(`:2621-2630`, `:2754-2763`)

Because `CS%BT_Domain` is cloned from `G%Domain` with a **much wider halo** than the normal `NIHALO_=2`
(`clone_MOM_domain(G%Domain, CS%BT_Domain, min_halo=wd_halos, symmetric=.true.)`, `:6104`; `wd_halos`
sized from `BT_halo_sz`/`use_wide_halos` params), a single MPI halo exchange at the top of the loop
(or none, if `num_cycles>1`) fills in enough valid data that the algorithm can run `num_cycles`
barotropic steps "marching inward" — shrinking the valid index range by `stencil` points per step —
before it runs out of valid halo data and must do another real halo exchange. This is exactly why the
barotropic solver, whose per-step physics is trivial (shallow-water update), doesn't become
communication-bound even though it may sub-cycle 10s of steps per baroclinic step: `num_cycles` steps
run for the cost of one exchange. `CS%isdw/iedw/jsdw/jedw` (set at `:6125-6126`) are the wide-halo
bounds; `BT_USE_WIDE_HALOS` (param, `:5817`) toggles the feature and `BT_HALO` sizes the halo.

### 1.3 `set_dtbt` — choosing the barotropic step

`set_dtbt` (`:3797`) computes `CS%dtbt`, the sub-step length, from a CFL-type stability estimate.
Given `pbce` (baroclinic pressure-anomaly sensitivity) or a rough `gtot_est`, it builds `gtot_E/W/N/S`
(the effective reduced gravity felt at each face, `:3880-3900`), combines them with face areas
`Datu`/`Datv` (from `BT_cont_to_face_areas` or `find_face_areas`) and grid metrics into a local
squared-timestep bound (`:3903-3912`):

```fortran
do concurrent (j=js:je, i=is:ie) DO_LOCALITY(reduce(min:min_max_dt2))
  Idt_max2 = 0.5 * (1.0 + 2.0*CS%bebt) * (G%IareaT(i,j) * &
    (((gtot_E(i,j)*Datu(I,j)*G%IdxCu(I,j)) + (gtot_W(i,j)*Datu(I-1,j)*G%IdxCu(I-1,j))) + &
     ((gtot_N(i,j)*Datv(i,J)*G%IdyCv(i,J)) + (gtot_S(i,j)*Datv(i,J-1)*G%IdyCv(i,J-1)))) + &
    ((G%Coriolis2Bu(I,J) + G%Coriolis2Bu(I-1,J-1)) + &
     (G%Coriolis2Bu(I-1,J) + G%Coriolis2Bu(I,J-1))) * CS%BT_Coriolis_scale**2 )
  if (Idt_max2 * min_max_dt2 > 1.0) min_max_dt2 = 1.0 / Idt_max2
enddo
```
(`:3903-3912`) — a `do concurrent` with a `DO_LOCALITY(reduce(min:...))` clause (the GPU-portable
min-reduction idiom, see `00-architecture.md` §9), followed by a **global** `min_across_PEs(dtbt_max)`
(`:3916`, an MPI allreduce — the one place per baroclinic step where all PEs must synchronize on the
barotropic step size). `CS%dtbt = CS%dtbt_fraction * dtbt_max` (`:3919`), where `dtbt_fraction`
defaults to 0.98 (or the negative of param `DTBT` if the user supplies a safety fraction; `:6404`).
Back in `btstep` (`:807`): `nstep = CEILING(dt/CS%dtbt - 0.0001)` — the number of barotropic steps
needed to cover the full baroclinic `dt`. `set_dtbt` is called once at `barotropic_init` (`:6581`) to
get a startup estimate and once per baroclinic step from `MOM_dynamics_split_RK2.F90:715/719`; if
`DTBT>0` (fixed timestep) it's skipped at runtime and `dtbt` is just read from the input parameter or
restart file (`:6583-6588`).

Note `set_dtbt` brackets its `gtot_*`/`Datu`/`Datv` scratch arrays in explicit
`!$omp target enter data map(alloc: ...)` / `map(release: ...)` (`:3864`, `:3913`) rather than relying
on `barotropic_init`-time mapping, since these are transient per-call locals, not CS members.

---

## 2. GPU status on mainline `dev/gpu`

**Directive density:** 242 `do concurrent`, 106 `omp target` in this one file — the highest raw count
of any ported module (`00-architecture.md` §4.3, §6.1: `+1003/−804` lines vs `dev-gfdl`). `git log
--oneline dev-gfdl..dev/gpu -- src/core/MOM_barotropic.F90` shows **100 commits** touching this file,
almost entirely small, single-purpose "port this loop" / "alloc this array" commits (`c2b162eda
btcalc: promote hatu to 3d` → ... → `7616a34c5 btstep: port btstep_ubt_from_layer subroutine` →
`65cf95a6f btstep: port btstep_find_Cor` → `1159cbd23 btstep: port eta_out loop` →
`8c62b0d37 send wt_* to gpu for btstep_timeloop` → `9013e8514 kji -> jki some loops` →
`e68e0e83a Dycore: Move halo updates to GPU`), i.e. the module was ported loop-by-loop and
array-by-array over many small, individually-reviewable commits rather than one large rewrite —
consistent with the "bitwise reproducibility is mandatory" guiding principle (`00-architecture.md` §0.2).

### 2.1 `do concurrent` vs `omp target`

- `do concurrent` is the default idiom for all the elementwise/columnwise updates inside `btstep`,
  `btcalc`, `bt_mass_source`, `btstep_timeloop`'s per-step update, `btstep_find_Cor`,
  `btstep_ubt_from_layer`, and `btstep_layer_accel` — e.g. the velocity update loops
  `btloop_update_u`/`btloop_update_v` (ported by commits `063f8da03`/`70b25b3c3`) and the pressure-force
  loop `btloop_find_PF` (`c2c4d4670`).
- `!$omp target` appears almost exclusively as **data-movement directives**
  (`target enter/exit data`, `target update to/from`) around the wide-halo scratch arrays and CS
  members, plus the 11 `do_group_pass(..., omp_offload=.true.)` calls (§2.2) and a few explicit
  `!$omp target data`/`!$omp end target data` regions bracketing `btcalc`'s per-column loop (see the
  `5f413739b` diff, §3). There is essentially no `!$omp target teams`/`loop` compute offload in this
  file — unlike `MOM_continuity_PPM.F90`'s k-blocked hybrid kernels (`00-architecture.md` §5), btstep
  relies on `do concurrent` alone for compute and on OpenMP purely for host/device data-transfer
  bookkeeping.

### 2.2 The ~11 group passes, all `omp_offload=.true.`

`grep -n omp_offload src/core/MOM_barotropic.F90`:

| Line | Pass | Domain | Purpose |
|---|---|---|---|
| `:1008` | `CS%pass_q_DCor` | `CS%BT_Domain` | wide-halo update of PV `q`/`DCor_u`/`DCor_v` before Coriolis-coefficient calc (clock `id_clock_pass_pre`) |
| `:1550` | `CS%pass_gtot` | `CS%BT_Domain` | wide-halo update of `gtot_E/W/N/S` |
| `:1551` | `CS%pass_ubt_Cor` | `G%Domain` | normal-halo update of reference Coriolis velocities |
| `:1772` | `CS%pass_eta_bt_rem` | `CS%BT_Domain` | wide-halo update of `eta`, `bt_rem_u/v`, `Rayleigh_u/v`, etc. |
| `:1773` | `CS%pass_Dat_uv` | `CS%BT_Domain` | face-area `Datu`/`Datv` (only when `.not. use_BT_cont`) |
| `:1774` | `CS%pass_force_hbt0_Cor_ref` | `CS%BT_Domain` | `BT_force_u/v`, `uhbt0/vhbt0`, `Cor_ref_u/v` |
| `:2022` | `CS%pass_e_anom` | `G%Domain` | eta anomaly used for time-averaged diagnostics |
| `:2070` | `CS%pass_ubta_uhbta` | `G%Domain` | time-averaged `CS%ubtav/vbtav`, `uhbtav/vhbtav` (the `omp_offload=.true.` was added in `9f4e48be1` "offload pass_uta_uhbta") |
| `:2757` | `CS%pass_eta_ubt` | `CS%BT_Domain` | **the inner sub-cycle halo pass** inside `btstep_timeloop`'s `do n=1,nstep+nfilter` loop (clock `id_clock_pass_step`) — this is the one amortized by wide halos, §1.2 |
| `:5290` | `BT_cont%pass_polarity_BT` | `BT_Domain` | face-polarity for the BT_cont closure |
| `:5291` | `BT_cont%pass_FA_uv` | `BT_Domain` | face areas `FA_u_*`/`FA_v_*` for the BT_cont closure |

All 11 pass `omp_offload=.true.` so the halo exchange operates on device-resident buffers without a
round-trip to host (`00-architecture.md` §7.3, `do_group_pass` optional arg,
`config_src/infra/FMS2/MOM_domain_infra.F90:1143`). This is the same count and same call sites on the
`edoyango/acc-btstep` experimental branch (§5) — that branch does not touch the halo-exchange strategy,
only the compute-kernel scheduling around it.

### 2.3 CS array members mapped in `barotropic_init` (`:6579-6676`)

Representative CS-member enter-data calls, in the order they appear at the end of `barotropic_init`:

```fortran
!$omp target enter data map (to: CS%frhatu, CS%frhatv)
!$omp target enter data map (to: CS%eta_cor)
...
!$omp target enter data map(to: CS%bathyT)
!$omp target enter data map(to: CS%D_u_Cor, CS%D_v_Cor)
!$omp target enter data map(to: CS%dx_Cv, CS%dy_Cu)
!$omp target enter data map(to: CS%IareaT, CS%IareaT_OBCmask)
!$omp target enter data map(to: CS%IDatu, CS%IDatv)
!$omp target enter data map(to: CS%IdxCu, CS%IdyCv)
!$omp target enter data map(to: CS%OBCmask_u, CS%OBCmask_v)
!$omp target enter data map(to: CS%q_d)
!$omp target enter data map(to: CS%ua_polarity, CS%va_polarity)
!$omp target enter data map(to: CS%ubtav, CS%vbtav)
```
(`:6579-6580`, `:6667-6676`)

`CS%frhatu`/`CS%frhatv` (`:118-121`) are the fraction-of-column-thickness-per-layer arrays computed by
`btcalc` and read every barotropic step by `set_dtbt` and the layer-projection routines;
`CS%eta_cor`/`CS%D_u_Cor`/`CS%D_v_Cor`/`CS%q_d` (i.e. `CS%q_D`) are all **wide-halo** arrays
(`isdw:iedw,jsdw:jedw` bounds, allocated at `:6135-6142`, `:6171-6180`, `:6308-6314`) — these are the
static/near-static per-timestep-invariant fields (bathymetry-derived depths, planetary vorticity) that
the sub-cycle reads every barotropic step without re-deriving, so they are mapped once at init and
persist device-resident for the life of the run. `CS%ubtav`/`CS%vbtav` are the running
time-averaged barotropic velocities accumulated across the sub-cycle (`:2954`, `:2959`) and consumed
by `barotropic_get_tav` (`:6681`) for the next predictor step's reference Coriolis velocities.

---

## 3. The `frhat[uv]` HYBRID repro fix (`5f413739b`)

Commit `5f413739b` ("Barotropic: frhat[uv] HYBRID repro fix", Marshall Ward) fixes **a bitwise
reproducibility regression under ifort**, not nvfortran — but it's directly relevant to the GPU port
because the loop structure it touches is exactly the kind of "split a loop for parallelism" refactor
the port does everywhere. `btcalc`'s `HYBRID` interpolation scheme (u-face thickness interpolation)
had been split into two separate `do concurrent` loops: one computing `CS%frhatu(I,j,k)` per layer,
and a second, separate loop summing `hatutot(I,j) = hatutot(I,j) + CS%frhatu(I,j,k)` across layers.
The fix folds the sum back into the same loop nest as the per-layer computation:

```diff
             CS%frhatu(I,j,k) = wt_arith*h_arith + (1.0-wt_arith)*h_harm
           endif
         endif
-      enddo
-    enddo
-    !$omp end target data
-    do concurrent (j=js:je, I=is-1:ie)
-      do k=1,nz
         hatutot(I,j) = hatutot(I,j) + CS%frhatu(I,j,k)
       enddo
     enddo
+    !$omp end target data
```
(same pattern mirrored for `frhatv`). The commit message explains: "this patch fixes a minor bit
reproducibility regression in the calculation of `hat[uv]tot` and, consequently, `frhat[uv]`... it
seems possible that Intel has added a reduction-like optimization, even at `-O0`... this was notably
subtle, since many tests only call `btcalc()` once without the BTCONT `h_[uv]` inputs, was only
observed in `frhatu`, and did not actually change solution answers. Nonetheless, this is a genuine
answer change in an actively used solver, so we want to preserve bit repro until discussed and
approved by the consortium." The underlying lesson for the GPU port: **splitting a reduction-style
accumulation loop (`hatutot`) away from the loop that produces its addends is not always a
transparent refactor** — a compiler (ifort here, but nvfortran is equally capable of it) can reorder
or vectorize the split accumulation loop differently than it would the fused one, changing rounding.
The fix re-fuses the two loops rather than trying to force a specific evaluation order in the split
form, and the accompanying `!$omp end target data` scoping moved with it (the `target data` region
now wraps the single fused loop instead of the first of the two split ones).

---

## 4. nvfortran-specific issues

### 4.1 A100 / nvfortran 25.5 crash — `2108e0eba`

Commit `2108e0eba` ("Remove eta_bt transfer from find_eta_2d that was causing crashes on stellar A100s
with nvfortran 25.5", Utheri Wagura) touches `src/core/MOM_interface_heights.F90`, not
`MOM_barotropic.F90` itself, but `find_eta`/`find_eta_2d` is called from the barotropic-adjacent SSH
diagnostics path in `MOM.F90`'s `step_MOM` (the `CS%eta_av_bc` / `ssh` bookkeeping that consumes the
barotropic solver's time-averaged output) and is one of the few remaining `target enter data ... if
(present(...))` constructs in the code, so it's catalogued here as a directly relevant nvfortran
compiler-bug workaround for code immediately downstream of `btstep`. The whole
`!$omp target enter data map(to: eta_bt) if (present(eta_bt))` directive — added a few weeks earlier
in `7003282d3` ("find_eta: Port to GPU and wrap calls") — was simply deleted:

```diff
   dZ_ref = 0.0 ; if (present(dZref)) dZ_ref = dZref

-  !$omp target enter data map(to: eta_bt) if (present(eta_bt))
-
   if (GV%Boussinesq) then
     if (present(eta_bt)) then
       do concurrent (j=js:je, i=is:ie)
```
(`src/core/MOM_interface_heights.F90:247-249`, commit `2108e0eba`). The bug: an `if (present(...))`
clause guarding a `map(to:)` on an **optional dummy argument** crashed nvfortran 25.5 specifically on
Stellar's A100 GPUs. There's no replacement directive — the workaround is simply "don't map it there";
presumably the caller already ensures `eta_bt` is mapped before the call (all six `find_eta` call
sites touched by `7003282d3` bracket the call with their own `!$omp target enter data
map(alloc:)`/`exit data map(from:)` around the whole call, e.g.
`MOM_ALE.F90:492-496`, `MOM.F90:1073-1077`). This is a clean example of "document nvfortran bugs...
genuine compiler bugs are hit and worked around" (`00-architecture.md` §0.5) — the fix is a deletion
with no compensating logic, purely because the conditional-map construct itself was what crashed.

### 4.2 Checksum-transfer discipline — `b29b27150`

Commit `b29b27150` ("btstep: Update GPU checksum transfers", Marshall Ward) is not a bug workaround
but shows the debugging-transfer pattern that has to be maintained by hand as fields move to/from
device: every `CS%debug`-gated `Bchksum`/`uvchksum`/`hchksum` call inside `btstep` needs an explicit
`!$omp target update from(...)` immediately before it, because the checksum routines run on the host
and read whatever is in the host-side copy of the array — which is stale once the corresponding
device buffer has diverged. The commit adds these one at a time to calls that had been missed:

```diff
     if (CS%linearized_BT_PV) then
+      !$omp target update from(CS%q_D)
       call Bchksum(CS%q_D, "BT PV (q_D)", ...)
     else
+      !$omp target update from(q)
       call Bchksum(q, "BT PV (q)", ...)
     endif
+    !$omp target update from(DCor_u, DCor_v)
     call uvchksum("BT DCor_[uv]", DCor_u, DCor_v, ...)
+    !$omp target update from(Cor_ref_u, Cor_ref_v)
     call uvchksum("BT Cor_ref_[uv]", Cor_ref_u, Cor_ref_v, ...)
+    !$omp target update from(uhbt0, vhbt0)
     call uvchksum("BT [uv]hbt0", uhbt0, vhbt0, ...)
     ...
+    !$omp target update from(visc_rem_u, visc_rem_v)
     call uvchksum("BT visc_rem_[uv]", visc_rem_u, visc_rem_v, ...)
+    !$omp target update from(bc_accel_u, bc_accel_v)
     call uvchksum("BT bc_accel_[uv]", bc_accel_u, bc_accel_v, ...)
+    !$omp target update from(CS%IDatu, CS%IDatv)
     call uvchksum("BT IDat[uv]", CS%IDatu, CS%IDatv, ...)
```
(`:1815-1850` region, commit `b29b27150`). This is not a compiler bug — it is a recurring maintenance
tax of the port: **every** debug/verification code path (`00-architecture.md` §9 "Verify a port")
has to be re-audited whenever a variable's residency changes, since a missed `target update from` will
silently checksum stale host data and can mask a real divergence (or manufacture a false one). Given
how frequently `btstep` was touched (100 commits), checksum transfers evidently drifted out of sync
with the mapping state repeatedly, hence a dedicated cleanup commit.

---

## 5. The `edoyango/acc-btstep` OpenACC + async experiment

`remotes/edoyango/acc-btstep` (2 commits ahead of `dev/gpu` on this file: `5bd8bb66d "add acc kernels
loop"`, `1ff44b2c5 "add asyncs"`; `git diff --stat dev/gpu...remotes/edoyango/acc-btstep --
src/core/MOM_barotropic.F90` = `+289/−50`) is an **additive, non-competing** experiment: it does not
touch the `do concurrent` bodies, the `omp_offload=.true.` halo passes (identical 11 call sites,
verified by diff — same line-for-line `do_group_pass(..., omp_offload=.true.)` calls), or the
overall algorithm. It layers `!$acc kernels loop` directives *around* the existing `do concurrent`
loops and assigns them to one of **three OpenACC async queues** (`async(1)`, `async(2)`, `async(3)` —
counts on the branch: 185 `!$acc kernels`, 177 `async(`, 40 `!$acc wait`), with explicit `!$acc wait`
/ `!$acc wait(N)` barriers inserted wherever a downstream consumer (a halo pass, an `!$omp target
update`, an OBC calculation) needs the result:

```fortran
! Zero out various wide-halo arrays.
!$acc kernels loop collapse(2) async(1)
do concurrent (j=CS%jsdw:CS%jedw, i=CS%isdw:CS%iedw)
  gtot_E(i,j) = 0.0 ; gtot_W(i,j) = 0.0
  gtot_N(i,j) = 0.0 ; gtot_S(i,j) = 0.0
  eta(i,j) = 0.0 ; eta_PF(i,j) = 0.0
  ! ... (also eta_PF_1/d_eta_PF, eta_IC, dyn_coef_eta under their flags)
enddo
!$acc kernels loop collapse(2) async(2)
do concurrent (j=CS%jsdw:CS%jedw, I=CS%isdw-1:CS%iedw)
  Cor_ref_u(I,j) = 0.0 ; BT_force_u(I,j) = 0.0 ; ubt(I,j) = 0.0
  Datu(I,j) = 0.0 ; bt_rem_u(I,j) = 0.0 ; uhbt0(I,j) = 0.0
enddo
!$acc kernels loop collapse(2) async(3)
do concurrent (J=CS%jsdw-1:CS%jedw, i=CS%isdw:CS%iedw)
  Cor_ref_v(i,J) = 0.0 ; BT_force_v(i,J) = 0.0 ; vbt(i,J) = 0.0
  Datv(i,J) = 0.0 ; bt_rem_v(i,J) = 0.0 ; vhbt0(i,J) = 0.0
enddo
```
(`:1018-1044`, commit `1ff44b2c5`), and later, before something that depends on all three streams:

```fortran
!$acc wait
if (id_clock_calc_pre > 0) call cpu_clock_end(id_clock_calc_pre)
if (nonblock_setup) then
  !$omp target update from(q, DCor_u, DCor_v)
```
(`:1006`-area). The pattern extends into the hot sub-cycle: 35 `!$acc kernels`/`async(` occurrences
and 12 `!$acc wait`s fall inside `btstep_timeloop` itself, so the experiment reaches the actual
per-barotropic-step loop, not just the one-time setup section of `btstep`.

**What this adds over mainline:** on `dev/gpu`, `do concurrent` loops are scheduled by whatever the
nvfortran runtime's default heuristic picks (typically one CUDA stream, synchronous-looking from
Fortran's point of view even though `do concurrent` has no sequencing guarantee by the standard). The
ACC branch takes explicit control: independent zero-init / setup loops that don't depend on each
other (three separate wide-halo zero-init loops — one h-point group, one u-point group, one v-point
group, in the example above) are put on **three
different async queues** so the GPU can overlap their kernel launches/execution instead of the runtime
serializing them, with `!$acc wait` inserted only at true data dependencies (a halo pass, an OBC
calc, a `target update`).

**What it suggests about the current path's performance:** the fact that this branch exists at all —
and that it required manually auditing dozens of small independent loops in `btstep`'s setup and
sub-cycle to assign them to 3 concurrent streams — implies the author (Ed Yang) suspected the mainline
`do concurrent`-only approach was leaving overlap opportunities on the table: nvfortran's default
scheduling of `do concurrent` may serialize logically-independent kernels that don't share data, so
small, independent setup loops (of which `btstep` has many, e.g. the three wide-halo zero-init loops at
`:1018-1044`) pay full kernel-launch latency serially rather than overlapping. Because `btstep_timeloop`'s
inner loop is itself a chain of *dependent* steps (eta update → PF → Coriolis → velocity update →
transport, each step consuming the last), the value of async queues there is more about overlapping
the small independent per-step housekeeping (multiple wide-halo array zero/copy operations at the top
of each iteration) than about restructuring the sequential physics — consistent with `!$acc wait`
appearing right before each real dependency point rather than only at the end of the subroutine. No
performance numbers accompany either commit (no build/run per this study's constraints), so this
remains a plausible-but-unquantified hypothesis: OpenACC's explicit async model is being tried as a
finer-grained alternative to relying on `do concurrent`'s implicit (and possibly conservative)
scheduling, on the same underlying computation.

**Evidence discipline (what the diff does and does not prove).** What is *verified from source*: the
branch is purely additive (`+289/−50`, no `do concurrent` body or halo-pass call site changed — the 11
`omp_offload=.true.` sites are byte-identical), it introduces 185 `!$acc kernels loop`, 177 `async(N)`
clauses over queues 1/2/3, and 40 `!$acc wait`s, of which 35 kernels and 12 waits land inside
`btstep_timeloop` (`:2528-3350` on the branch). What is *not* in the diff: any benchmark, timing, or
profile. So the claim "nvfortran serializes independent `do concurrent` kernels and async recovers the
overlap" is an *inference from the shape of the intervention* (someone bothered to hand-assign queues),
not a measured result. State it as a hypothesis, not a finding.

> **FABLE-CHECK (reviewed 2026-07-14 — resolution or current status in KNOWLEDGE.md §8a/§8b):** Does nvfortran actually launch consecutive `do concurrent` loops on a single
> in-order CUDA stream (making logically-independent loops serialize), such that OpenACC `async(N)`
> queues are the intended remedy? This is the load-bearing assumption of §5 and can only be settled by
> an NVHPC-runtime/`nsys` timeline, not by the repo. Look for any profiling notes on
> `edoyango/acc-btstep` or `benchmark_ALE_nvtx_clocks` before repeating the serialization claim as fact.

---

## 6. Device-resident barotropic state: what and how

Two categories of state must be device-resident for `btstep` to run without excessive host↔device
traffic:

1. **`BT_cont_type`** (`MOM_variables.F90:317`) — the barotropic face-area closure. All-allocatable
   members (`FA_u_EE/E0/W0/WW`, `uBT_WW/EE`, `FA_v_NN/N0/S0/SS`, `vBT_SS/NN`, optional `h_u`/`h_v`).
   `alloc_BT_cont_type` (`MOM_variables.F90:567`) maps the **struct pointer itself** first
   (`!$omp target enter data map(to: BT_cont)`, `:582`) then each member array individually right after
   its `allocate(..., source=0.0)` (`:583-591`, `:593-601`, `:603-607`) — the member-by-member
   "attach" pattern flagged as expensive in `00-architecture.md` §2.3 (cf. commit `1865612de`'s
   flat-array refactor in `MOM_tracer_hor_diff` for the same reason). `BT_cont` is threaded through
   `btstep`'s call signature as a `pointer` dummy argument (`:537`) and consumed by
   `BT_cont_to_face_areas`/`set_local_BT_cont_types`, with its own wide-halo group passes
   `pass_polarity_BT`/`pass_FA_uv` (`:5279-5286`, both `omp_offload=.true.` at `:5290-5291`).

2. **Wide-halo `barotropic_CS` arrays** — the "always allocated with symmetric memory and wide halos"
   locals declared at `:593-620` inside `btstep` (`q`, `ubt`, `bt_rem_u`, `BT_force_u`, `u_accel_bt`,
   `uhbt`, `uhbt0`, `Cor_ref_u`, `Rayleigh_u`, `DCor_u`, `Datu`, and their v-counterparts) plus the
   persistent CS members with `isdw:iedw,jsdw:jedw`-type bounds: `CS%bathyT`, `CS%IareaT`,
   `CS%IareaT_OBCmask`, `CS%IdxCu`/`CS%IdyCv`, `CS%dx_Cv`/`CS%dy_Cu`, `CS%OBCmask_u`/`CS%OBCmask_v`,
   `CS%D_u_Cor`/`CS%D_v_Cor`, `CS%q_D`, `CS%ua_polarity`/`CS%va_polarity` — all mapped once at the tail
   of `barotropic_init` (`:6667-6676`, listed in full in §2.3) and never re-mapped per call, since they
   are either static grid-derived metrics or run-persistent accumulators. The per-call locals (`q`,
   `ubt`, `Datu`, etc., declared inside `btstep`/`btstep_timeloop`) instead get their own
   `!$omp target enter data map(alloc: ...)` blocks scattered through `btstep`
   (e.g. `:2663-2665`, `:2694-2695`) that are torn down before return, since their contents don't need
   to persist across baroclinic steps — only within one `btstep` invocation. `CS%frhatu`/`CS%frhatv`
   (the per-layer thickness fractions computed once per baroclinic step by `btcalc`, consumed every
   barotropic sub-step by the layer-projection routines and by `set_dtbt`) sit in between: they persist
   across the barotropic sub-cycle but are recomputed once per baroclinic step, and are mapped with
   `to:` at `barotropic_init` (`:6579`, initial allocation) then refreshed via ordinary device-resident
   writes inside `btcalc` each call (no repeated host round-trip).

The unifying design point: **`CS%BT_Domain` (the wide-halo domain clone) is itself just a `MOM_domain_type`
pointer** (`:338`) — it carries no device-mapped array data of its own; what has to be device-resident
is every array whose valid range is described relative to its wide bounds (`CS%isdw/iedw/jsdw/jedw`),
because those are exactly the arrays the "march inward" trick (§1.2) reads and writes across many
barotropic steps between the relatively rare `do_group_pass(..., CS%BT_Domain, omp_offload=.true.)`
calls — if any one of them silently fell back to host residency, every barotropic sub-step touching it
would force a device→host→device round trip, defeating the entire wide-halo optimization.

---

## 6.5 Transferable lessons for a porting agent

Distilled from the commits above; each is a rule you can carry to the next module, with the barotropic
evidence that grounds it.

1. **Loop-fusion is a reproducibility decision, not just a performance one.** When you split a loop to
   expose parallelism, any accumulation you carry out of it (`hatutot += frhatu`, a running sum, a
   dot-product) becomes a *separate* reduction the compiler is free to re-associate — even at `-O0`,
   even under ifort (`5f413739b`). If the original fused loop set the bit pattern, keep the accumulate
   inside that loop nest; do not "clean up" by hoisting it into its own `do concurrent`. The repro fix
   is re-fusion, never a directive that pins evaluation order in the split form. Rule of thumb: **a
   producer loop and the reduction that consumes its outputs must stay fused unless you have re-verified
   the checksum after splitting them.**

2. **Never `map(...) if (present(optional_arg))`.** A conditional data-map keyed on an optional dummy
   argument crashed nvfortran 25.5 on A100 (`2108e0eba`). The fix is deletion, not repair — push the
   mapping responsibility to the caller, which already brackets the call with its own
   `enter data map(alloc:)/exit data map(from:)` around the whole callee. General pattern: **map
   optional-argument buffers at the call site where presence is unambiguous, never inside the callee on
   an `if (present(...))` guard.**

3. **Every debug/verify path is a device→host transfer you must maintain by hand.** Checksums, `chksum0`,
   and `[uv]/hchksum` run on the host and read the *host* copy; once a field is device-resident its host
   copy is stale, so each debug call needs a matching `!$omp target update from(...)` immediately before
   it (`b29b27150`). This drifts constantly (a module touched 100 times will silently checksum stale
   data somewhere), so audit it whenever a variable's residency changes — a missing transfer both hides
   real divergence *and* manufactures false ones.

4. **Wide-halo device residency is all-or-nothing.** The "march inward" trick (§1.2) only pays off if
   *every* array indexed on the wide bounds (`CS%isdw:iedw,jsdw:jedw`) is device-resident across the
   whole sub-cycle. Static grid-derived metrics and run-persistent accumulators are mapped once at
   `barotropic_init` and never re-mapped (§2.3); per-call scratch gets `enter/exit data map(alloc:)`
   scoped to one `btstep`. A single wide-halo array that silently falls back to host residency turns
   every one of the many barotropic sub-steps that touches it into a host round-trip — defeating the
   entire optimization. When you port a wide-/deep-halo solver, treat "prove every halo-scoped array is
   mapped" as a checklist item, not an afterthought.

5. **Split the directives by job: `do concurrent` for compute, OpenMP `target` for data.** In this file
   OpenMP is used almost exclusively for data movement (`target enter/exit data`, `target update`) and
   for the `omp_offload=.true.` halo passes; the actual elementwise/columnwise math is `do concurrent`
   (§2.1). Reductions are the documented exception — `do concurrent (...) DO_LOCALITY(reduce(min:...))`
   in `set_dtbt` (§1.3), followed by an MPI `min_across_PEs`. This is the opposite of
   `MOM_continuity_PPM`'s k-blocked `!$omp target teams` compute kernels (`00-architecture.md` §5): the
   barotropic per-step physics is a cheap 2-D stencil, so it needs no manual team tuning, whereas
   continuity's per-column reconstruction did. **Pick the compute idiom by kernel arithmetic intensity,
   and keep OpenMP for the residency/transfer bookkeeping either way.** The `acc-btstep` experiment (§5)
   is a *third* axis — explicit async scheduling — layered on top without disturbing either split.

## 7. Cross-references

- `00-architecture.md` §4.2 (call sequence into/out of `btstep`), §4.3 (directive-count summary), §7.3
  (halo/`omp_offload` infrastructure), §7.5 (compiler-workaround catalogue — the A100/nvfortran-25.5
  bug belongs there too).
- `03-openmp-mapping.md` for the general CS-member enter-data/exit-data lifecycle pattern that
  `barotropic_init`/`barotropic_end` follow.
- Branches: `remotes/origin/btstep-halo-control`, `remotes/origin/bbl-cleanup-almost-step` (adjacent,
  not reviewed here — halo-control and BBL cleanup respectively); `remotes/edoyango/acc-btstep` (§5).

---

## Verification notes

Verified against `src/core/MOM_barotropic.F90`, `src/core/MOM_interface_heights.F90`,
`src/core/MOM_dynamics_split_RK2.F90`, `src/core/MOM_variables.F90`, and git history on `dev/gpu`
(source + git only; no build/run, per constraints).

**Confirmed exactly:**
- File scale and directive density: 6868 lines, 242 `do concurrent`, 106 `omp target`, 100 commits
  touching the file on `dev-gfdl..dev/gpu`.
- All 11 `omp_offload=.true.` group-pass call sites and their line numbers (`:1008, :1550, :1551,
  :1772, :1773, :1774, :2022, :2070, :2757, :5290, :5291`) — table in §2.2 is byte-accurate.
- `btstep` `:480`, called from `MOM_dynamics_split_RK2.F90:726/:1023`; `btstep_timeloop` `:2376`,
  `btstep_find_Cor` `:3159`, `btstep_ubt_from_layer` `:3671`, `btstep_layer_accel` `:3723`,
  `barotropic_get_tav` `:6681`.
- `num_cycles`/march-inward math at `:2621-2630`/`:2754-2763`, `v_first` at `:2804`, `stencil` logic —
  code matches the doc's quoted block.
- `set_dtbt` (`:3797`): `reduce(min:min_max_dt2)` `do concurrent` at `:3903`, `min_across_PEs(dtbt_max)`
  at `:3916`, `CS%dtbt = CS%dtbt_fraction*dtbt_max` at `:3919`, `map(alloc:)`/`map(release:)` at
  `:3864`/`:3913`, `nstep = CEILING(dt/CS%dtbt - 0.0001)` at `:807`, `dtbt_fraction=0.98` at `:6404`;
  called at `barotropic_init:6581` and `MOM_dynamics_split_RK2.F90:715/:719`.
- `clone_MOM_domain(..., min_halo=wd_halos, symmetric=.true.)` at `:6104`; `BT_USE_WIDE_HALOS` param
  `:5817`; `barotropic_init` CS-member `map(to:)` block `:6579-6676` (list in §2.3 matches).
- `BT_cont_type` `MOM_variables.F90:317`; `alloc_BT_cont_type:567` maps `BT_cont` at `:582` then
  member arrays right after their `allocate(...,source=0.0)`.
- Commit diffs `5f413739b` (ifort HYBRID re-fusion, 2 ins/10 del), `2108e0eba` (2 deletions,
  nvfortran 25.5 / Stellar A100, `MOM_interface_heights.F90`), `b29b27150` (8 checksum
  `target update from` insertions), `9f4e48be1` (added `omp_offload=.true.` to `pass_ubta_uhbta`),
  `7003282d3` (added the deleted directive) — all verified line-for-line.
- `edoyango/acc-btstep`: 2 commits (`5bd8bb66d`, `1ff44b2c5`), `+289/−50`; 185 `!$acc kernels loop`,
  177 `async(`, 40 `!$acc wait`, 11 `omp_offload=.true.` (identical); 35 kernels + 12 waits inside
  `btstep_timeloop`. All 10 small "port this loop" commit hashes in §2 exist with the quoted messages.

**Corrected:**
- §1.2: wide-halo bounds set at `:6125-6126`, not `:6125-6127`.
- §2.2: the `:2070` row cross-referenced "§4", where `9f4e48be1` is not discussed; reworded to name the
  commit's actual role (adding `omp_offload=.true.`).
- §5: the `async(1)` zero-init loop in the quoted snippet was truncated — it zeros an h-point *group*
  (`gtot_*`, `eta`, `eta_PF`, and flag-gated `eta_PF_1/d_eta_PF/eta_IC/dyn_coef_eta`), not just the four
  `gtot_*` arrays; snippet annotated and the miscount "four wide-halo zero-inits" corrected to three
  loops (h/u/v groups), line range `:1018-1044`.

**Enhancements:** added §6.5 (five sharpened transferable lessons — loop-fusion repro rule,
conditional-map-on-optional hazard, checksum-transfer tax, all-or-nothing wide-halo residency, the
`do concurrent`-for-compute / OpenMP-for-data split); added an "evidence discipline" paragraph to §5
separating what the `acc-btstep` diff proves (additive, counts) from the unmeasured performance
inference.

**FABLE-CHECK markers:** 1 — on the load-bearing §5 assumption that nvfortran serializes independent
`do concurrent` kernels onto one stream (the premise that motivates the async experiment), which is not
settleable from source and needs an NVHPC/`nsys` timeline.

**Confidence:** High. Every line number, commit hash, diff, and directive count in the document was
checked against the tree and matched (modulo the three minor corrections above). The one genuinely
unverifiable claim — the OpenACC-async performance rationale — was already appropriately hedged in the
draft and is now flagged explicitly.
