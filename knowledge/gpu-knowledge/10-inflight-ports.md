# Naive vs. Blessed: A Contrast Study from In-Flight Branches

> **Purpose.** `00-architecture.md` §0 defines the "blessed" port strategy (CPU-preserving
> k-blocking, mandatory bitwise reproducibility, `do concurrent` as the default idiom, no device
> polymorphism) in the abstract. This document makes it concrete by contrasting two unmerged
> branches that touch the *same* code (`MOM_density_integrals.F90` / EOS 3D interfaces / ALE PLM)
> but diverge on a *different* subsystem (Bodner MLE). Read `00-architecture.md` §0, §5, §6.2 first.
>
> Studied via `git log`/`git diff` only, no build. Branches: `bodner-naive-port` (forked from
> `dev/gpu` at `b40fe2d766` "add submodule support", **8 commits behind** the current `dev/gpu` tip
> `b8c471cfa` — it predates the merged EOS 2D/3D work `7c7af5572`, the block-repro-sum `8593a732a`,
> the k-block continuity `93dbbd36e`, the CorAdCalc k-block `b8c471cfa`, plus four earlier commits)
> and `port/pressureforce-benchmark_ALE` (forked at `c82e1254a6` "vertvisc: Fix CS memory
> management" — this **was** the `dev/gpu` tip when this study began, but `dev/gpu` has since
> advanced 4 commits past it, so it is now 4 behind). Where the two branches' *own* diffs disagree on
> `MOM_coms.F90` / `MOM_set_viscosity.F90` / `MOM_vert_friction.F90` / `MOM.F90`, it is **only
> baseline drift** (each forked at a different point on `dev/gpu`, and those files changed on
> `dev/gpu` in between). Verified the right way: **neither branch touches those files in its own
> work** — `git diff <fork>..<branch>` on them is empty for `port/pressureforce-benchmark_ALE`, and
> `bodner-naive-port` has only a 3-line real edit to `MOM.F90` for its MLE port. (Do *not* verify
> this by diffing pf against the *current* `dev/gpu`: because `dev/gpu` advanced 4 commits — one of
> which, `8593a732a`, rewrote `MOM_coms.F90` — that diff is now large and misleading.) Do not read
> those files as branch content below.

---

## 0. Branch topology — what is actually shared

The commit graphs are not independent: both branches carry the **same ~24-commit PLM lineage** by
Edward Yang — from `submodulify` (`c75ebddaf`/`83628d990`), through "add j dimension" and the
`int_density_dz_generic_plm` tiling series, up to "move block size to user input"
(`b864bafba`/`1038a4921`) — that rewrites `int_density_dz_generic_plm` from a per-layer scalar-`k`
call into a k-blocked, submodule-split kernel. Verified by `git patch-id`: **21 of those commits are
byte-identical patches** across the two branches; the remaining 3 (`cec1c8d0a`/`8bbd4c085`/`207c0ce34`
— "use 2d calc dens" / "move k loops inside" / "block loop in set_pbvce") differ **only by EOS
baseline drift**: `bodner-naive-port` forked *before* the merged EOS 2D/3D commit, so on it these
three commits carry the `calculate_density_*_3d` additions inline (they each touch `MOM_EOS.F90` /
`MOM_EOS_Wright.F90` / `MOM_EOS_Roquet_rho.F90` / `MOM_EOS_base_type.F90`), whereas on
`port/pressureforce-benchmark_ALE` that machinery already arrived via its first commit `8eb41475b`
"EOS: 2D and 3D density implementations of methods". They then diverge at the **same timestamp**
(`Tue Jun 2 11:36:45 2026 -0400`, both authored by Edward Yang, both parented on the matched "move
block size to user input" commit) on one commit — `bodner-naive-port`'s
`8744bf362 set plm block size defaults to 32x4` vs. `port/pressureforce-benchmark_ALE`'s
`fae6c9a5c set plm block size defaults to 0x1` — a **literal fork point**, not two unrelated efforts.
(So "26 identical commits" is an overcount: the shared lineage is ~24 commits, 21 byte-identical and
3 differing only by the EOS drift above.) After the fork:

- `bodner-naive-port` stops touching `MOM_density_integrals.F90`/PLM and instead layers the
  **Bodner MLE (`MOM_mixed_layer_restrat.F90`) naive port** on top (`6f854c5aa` → `08be6d130`,
  authored by "Jorge" with `Co-Authored-By: Claude Opus 4.8`).
- `port/pressureforce-benchmark_ALE` keeps refining the **same PLM density-integral kernel**
  (`desubmodule` cleanup only).

**Implication for judging "naive vs. blessed": the PLM/EOS k-blocking substrate is common,
in-flight, blessed-style work by a different author than the Bodner MLE port.** The genuinely
*naive* code under study is specifically `MOM_mixed_layer_restrat.F90` on `bodner-naive-port`. The
PLM/EOS work on both branches is judged separately below (§3) and is closer to blessed on both,
with `port/pressureforce-benchmark_ALE` slightly ahead.

| | `bodner-naive-port` | `port/pressureforce-benchmark_ALE` |
|---|---|---|
| Forked from | `b40fe2d766` (**8** commits behind current `dev/gpu` `b8c471cfa`) | `c82e1254a6` (was tip at study time; now **4** behind) |
| Commits ahead of its fork | 30 | 27 |
| Files touched (own work) | `MOM_mixed_layer_restrat.F90` (**+174/−91**), `MOM_density_integrals.F90`/`_s.F90` (submodule split), EOS 3D interfaces (via the shared PLM lineage), `MOM_PressureForce_{FV,Montgomery}.F90` | `MOM_density_integrals.F90`, EOS 3D interfaces (deeper — Roquet_rho +158/−16), `MOM_PressureForce_{FV,Montgomery}.F90`, `MOM_ALE.F90`/`PLM_functions.F90` |
| Distinctive new file | `MOM_density_integrals_s.F90` (**submodule**, not a scalar variant — see §5) | none (same submodule pattern, later `desubmodule`d back) |

---

## 1. The naive pattern, concretely — `MOM_mixed_layer_restrat.F90` on `bodner-naive-port`

### 1.1 First cut (`6f854c5aa "naive omp offload working"`)

Per-loop `!$omp target teams distribute parallel do collapse(2)` regions, each with its own
explicit `map(to/from/tofrom:)` clause, wrapped around loops copied verbatim from the CPU code.
Debug `print *` statements are left in (`print *, "Bodner"`, `print *, "MLD grid"`, `print *, "Else"`,
`print *, "anwer date"`, `print *, " if lfbod "`, … — **ten** newly-added debug prints, verified
against the fork baseline `b40fe2d766` which had only one pre-existing unit-test banner). They
survive through three further MLE commits and are stripped only by the last MLE commit
`08be6d130 "print removal"`. Note also the commit-message typos (`"denstiy"`, `"btiwse"`,
`"anwer date"`) consistent with fast, unreviewed iteration:

```fortran
elseif (CS%use_Bodner) then
  print *, "Bodner"
    ! Implementation of Bodner et al., 2023
    call mixedlayer_restrat_Bodner(...)
```

```fortran
tau_bgrow = CS%BLD_growing_Tfilt ; tau_bdecay = CS%BLD_decaying_Tfilt
h_MLD_l(:,:) = h_MLD(:,:)
MLDf_l(:,:) = CS%MLD_filtered(:,:)
!$omp target teams distribute parallel do collapse(2) &
!$omp   map(to: h_MLD_l) map(tofrom: MLDf_l) map(from: little_h)
do j=js-1,je+1 ; do i=is-1,ie+1
  little_h(i,j) = rmean2ts(h_MLD_l(i,j), MLDf_l(i,j), tau_bgrow, tau_bdecay, dt)
  MLDf_l(i,j) = little_h(i,j)
enddo ; enddo
CS%MLD_filtered(:,:) = MLDf_l(:,:)
```

Two things to flag: (a) **`omp target teams distribute`** is used where the architecture doc says
`do concurrent` should be the default (principle #3) — this gets fixed two commits later; (b) a
**plain local copy `h_MLD_l`/`MLDf_l`** of every CS member is made before mapping "for clean device
mapping" (comment added later). This sidesteps CS-member/derived-type mapping headaches (§2.3 of
`00-architecture.md`) at the cost of doubling host memory and adding host-side copy loops that don't
exist in the CPU code path at all — a real, if small, CPU regression risk introduced by porting.
Density itself is still computed on the **host** in this commit:

```fortran
! Active path: density at p=0 (sigma_0) computed on the HOST into rho3d (the polymorphic EOS
! dispatch stays on the host), then the per-column mixed-layer integral is offloaded.
do k=1,nz ; do j=js-1,je+1
  call calculate_density(tv%T(:,j,k), tv%S(:,j,k), p0, rho3d(:,j,k), tv%eqn_of_state, EOSdom)
enddo ; enddo
```
followed by a `!$omp target teams distribute parallel do collapse(2)` per-column integral kernel
that maps `rho3d` in — i.e. a host EOS call feeding a device integral, a transfer the blessed
pattern (§1.2) tries to design away.

### 1.2 Second cut (`02bf308f3 "persistent target data region + do concurrent for in-region loops"`)

Wraps the *tail* of the routine (mixed-layer integral, U/V components, h update) in **one**
`!$omp target data` region and converts those loops to `do concurrent` +
`DO_LOCALITY(local(...))`:

```fortran
!$omp target data &
!$omp   map(to: little_h, big_H, wpup, Cr_l) map(alloc: rho3d) &
!$omp   map(from: vol_dt_avail, htot, buoy_av, uhml, vhml, uDml_diag, vDml_diag)
...
do concurrent (j=js-1:je+1, i=is-1:ie+1) DO_LOCALITY(local(k, dh, Rml_i, htot_i))
  ...
enddo
...
!$omp end target data
```

This is a genuine step toward blessed (principle #1's "one source form" and principle #3's
`do concurrent` default), but it is applied only to the back third of the subroutine — the
earlier filter/w'u' loops (little_h, big_H, wpup) each keep their **own** `target enter
data`/`exit data` pair (confirmed: 5 `enter data`, 4 `exit data`, 2 `target data` blocks in the
final file), i.e. several small host↔device round-trips remain where the blessed pattern in
`MOM_continuity_PPM.F90`/`MOM_CoriolisAdv.F90` uses a single region spanning the whole hot path.

### 1.3 Third cut (`09d1b93f7 "another do concurrent"`)

Converts the remaining `omp target teams distribute` loops (little_h, big_H, wpup, wpup filter) to
`do concurrent`, each now bracketed by its own `target enter data ... exit data` (still not folded
into the one persistent region from §1.2). By branch tip, `omp target teams` no longer appears
anywhere in the file (0 occurrences) — principle #3 is fully satisfied in the end state, it just
took three iterations to get there, with the intermediate commits shipping a mixed idiom.

**What is missing throughout, and never added:** `MOM_mixed_layer_restrat.F90` has **zero**
`niblock`/`njblock`/`nkblock` CS parameters anywhere (checked: no `block` hits besides unrelated
`openParameterBlock` calls). There is no CPU-cache-blocking story at all — every device loop runs
over the whole horizontal domain unconditionally, with no `#ifdef __NVCOMPILER_OPENMP_GPU` /
CPU-default split like the one added to `MOM_PressureForce_FV.F90` (§3). Compare to principle #1:
the blessed pattern requires *one source form that is also tuned for CPU*; this port has one source
form that was **never tuned for CPU at all** — the whole-domain assumption is baked in with no
runtime override, unlike every merged/blessed kernel in `00-architecture.md` §5/§6.1.

---

## 2. "Call calculate density on the GPU bitwise" (`2271af66e`) — how it stays bit-identical

Before this commit, the p=0 (sigma_0) density used for the Bodner mixed-layer integral was computed
by a **host** loop calling the existing 1-D `calculate_density` interface column-by-column, then
`!$omp target update to(rho3d)` pushed the result to device. `2271af66e` replaces that with a
single **3-D** EOS call issued from inside the target-data region:

```fortran
T_l(:,:,:) = tv%T(:,:,:) ; S_l(:,:,:) = tv%S(:,:,:)
p3d(:,:,:) = 0.0
EOSdom3d(1,:) = EOS_domain(G%HI, halo=1)
EOSdom3d(2,:) = [(js-1) - (G%jsd-1), (je+1) - (G%jsd-1)]
EOSdom3d(3,:) = [1, nz]
...
! Active path: density at p=0 (sigma_0) computed ON THE DEVICE via the 3D EOS interface (the
! polymorphic dispatch is resolved host-side, the per-element evaluation runs in do concurrent),
! then the per-column mixed-layer integral is offloaded.
call calculate_density(T_l, S_l, p3d, rho3d, tv%eqn_of_state, EOSdom3d)
```

This only compiles/works because a **`calculate_density_3d` generic** and matching 3-D type-bound
procedures already exist in the EOS layer of this branch. **They were *not* added by `2271af66e`
itself** — that commit only edits `MOM_mixed_layer_restrat.F90` (+17/−8) to *call* the 3-D
interface. The 3-D EOS machinery arrived earlier, inside the shared Edward-Yang PLM lineage
(`cec1c8d0a`/`8bbd4c085`/`207c0ce34` — the same three commits flagged as EOS baseline drift in §0).
Because `bodner-naive-port` forked *before* the merged EOS 2D/3D commit `7c7af5572`, the 3-D
interface on this branch is carried by that PLM lineage rather than inherited from the merge; it
mirrors the same 2D pattern that `7c7af5572` merged onto `dev/gpu`:

- `MOM_EOS.F90`: `calculate_density_3d` added to the `calculate_density` interface; it resolves
  scaling (`EOS%RL2_T2_to_Pa` etc.) then calls `EOS%type%calculate_density_array_3d(...)` — the
  **one place** the polymorphic v-table dispatch happens, on the host, before any device region is
  entered.
- `MOM_EOS_base_type.F90`: default fallback `a_calculate_density_array_3d` — still polymorphic
  (`class(EOS_base), intent(in) :: this`), calls the elemental `this%density_elem(...)` in
  whole-array syntax. **This fallback is not GPU-safe** — only concrete EOS classes with their own
  override are.
- `MOM_EOS_Wright.F90` / `MOM_EOS_Roquet_rho.F90`: concrete overrides
  `calculate_density_array_3d_buggy_Wright` / `..._Roquet_rho` implement the device-safe path with
  `do concurrent` over a free `_loc` function (no `this`):

```fortran
! NOTE: There is an implicit copy of `this` which cannot yet be prevented.
!   Possibly because Nvidia cannot associate `this` with `EOS%type`.
if (present(rho_ref)) then
  do concurrent (k=ks:ke, j=js:je, i=is:ie)
    rho(i,j,k) = density_anomaly_elem_buggy_Wright(this, T(i,j,k), S(i,j,k), pressure(i,j,k), rho_ref)
  enddo
else
  do concurrent (k=ks:ke, j=js:je, i=is:ie)
    rho(i,j,k) = density_elem_buggy_Wright_loc( T(i,j,k), S(i,j,k), pressure(i,j,k))
  enddo
endif
```

**Why this is bitwise-safe:** the `rho_ref`-absent branch calls exactly the same per-element free
function (`density_elem_buggy_Wright_loc`) that the already-merged 2D path calls — same polynomial,
same operation order, only the surrounding loop nest and *where* (host vs. device) it runs have
changed. No reduction, no reordering, no fused-multiply-add reassociation. The commit re-uses the
architecture's established "resolve the v-table once on the host, then call a free `_loc` kernel
inside the parallel region" idiom from `07-eos` and matches the "implicit copy of `this`" nvfortran
limitation already catalogued in `00-architecture.md` §7.5 — the `present(rho_ref)` branch above
*still* passes `this` and is annotated with that exact caveat, i.e. it is a **known, not-yet-fixed**
gap even in this "GPU bitwise" commit: the rho_ref path is not actually proven device-safe, only the
no-rho_ref path (the one Bodner MLE actually uses) is.

---

## 3. `port/pressureforce-benchmark_ALE`'s k-blocking of `int_density_dz_generic_plm`

This is the blessed template — CS-level block-size parameters with CPU/GPU-conditional defaults,
resolved at `_init`, `0` meaning "whole domain":

```fortran
#ifdef __NVCOMPILER_OPENMP_GPU
integer, parameter :: default_nkblock = 0  !< 0 = full domain
integer, parameter :: default_njblock_plm = 0
#else
integer, parameter :: default_nkblock = 1  !< 1 = one layer per CPU cache-blocking pass
integer, parameter :: default_njblock_plm = 1
#endif
integer, parameter :: default_niblock_plm = 0 !< i is never cache-blocked here
```
(`fae6c9a5c "set plm block size defaults to 0x1"` — the "0x1" in the commit title is
`niblock=0 (full), njblock=1 (one row)`, with `nkblock` fixed at 1 on CPU / 0 on GPU by the earlier
`#ifdef` block. Contrast `bodner-naive-port`'s stalled `8744bf362 "set plm block size defaults to
32x4"`, i.e. `niblock=32, njblock=4` — a direct copy of the `MOM_continuity_PPM.F90` "32/4/1"
convention cited in `00-architecture.md` §5, before further benchmarking on this branch revised it
down to row/full/layer tiling more suited to the EOS-heavy PLM kernel.)

The interface gains `niblock`/`njblock` optional arguments (`1038a4921`/`b864bafba "move block size
to user input"`, identical on both branches) that are threaded down into
`generic_plm_update_{dpa,intx_dpa,inty_dpa}`'s `TILE_SIZE_X`/`TILE_SIZE_Y` locals.

**"move k loops inside" (`3b8525a54` on pf / `8bbd4c085` on bodner — the *same* logical step on both
branches; the two differ only by the EOS baseline drift noted in §0, since this commit also touches
`MOM_EOS*.F90`)** is the substantive k-blocking step: scratch arrays (`T5`,`S5`,`p5`,`r5`,`u5`, …) grow a third `kstart:kend`
dimension, and the per-`k` `calculate_density` call over a 2-D `(5*i,j)` domain is replaced by
**one** `calculate_density` call over the whole 3-D `(5*i,j,k)` block:

```fortran
real :: T5(5*TILE_SIZE_X,TILE_SIZE_Y,kstart:kend)   ! was (5*TILE_SIZE_X,TILE_SIZE_Y)
...
EOSdom_h5(3,1) = 1 ; EOSdom_h5(3,2) = kend-kstart+1
if (use_rho_ref) then
  call calculate_density(T5, S5, p5, r5, EOS, EOSdom_h5, rho_ref=rho_ref)   ! one 3D call, not a k-loop of 2D calls
else
  call calculate_density(T5, S5, p5, r5, EOS, EOSdom_h5)
  do concurrent (k=kstart:kend, j=jstart:jend, i=istart:iend, n=1:5)
    ...
  enddo
endif
```
i.e. exactly the amortize-the-EOS-dispatch-over-k pattern that `00-architecture.md` §7.1 identifies
as the point of the 3D EOS interface, applied here to the PGF-integral hot path rather than to the
Bodner MLE code.

**"optimise a little bit" (`2a99c9dd1` on pf = `98cb748c6` on `bodner-naive-port` — patch-id
identical, so present on *both* branches)** applies the vertvisc-style tridiagonal idiom (§4.3/§5 of
`00-architecture.md`: outer parallel loop, serial recurrence loop, inner parallel loop) to the three
places in `PressureForce_FV_Bouss` that have a genuine `k`-recurrence (the interface-height `e` and
the `intx_pa`/`inty_pa` cumulative sums) and previously used an outer-serial-`k` /
inner-`do-concurrent` nesting the wrong way round:

```fortran
! before: do k=nz,1,-1 ; do concurrent (j=...,i=...) ... enddo ; enddo
do concurrent(j=Jsq:Jeq+1)
  do k=nz,1,-1                      ! true recurrence in k stays serial
    do concurrent (i=Isq:Ieq+1)
      e(i,j,K) = e(i,j,K+1) + h(i,j,k)*GV%H_to_Z
    enddo
  enddo
enddo
```
This is a genuine "did we get the k-blocking direction right" fix. The commit's *entire* content is:
the three recurrence-direction reversals above (25 changed lines in `MOM_PressureForce_FV.F90`), a
`private(k)` fix on one `!$omp target teams loop`, one added
`!$omp target enter data map(to: tv_tmp, tv_tmp%T, tv_tmp%S)`, and a 1-line change to `MOM_ALE.F90`.
**Because `98cb748c6` and `2a99c9dd1` are byte-identical, `bodner-naive-port` carries this same fix**
— it is *not* a pf-vs-bodner differentiator.

> **CORRECTION (verified).** An earlier draft attributed to this commit the removal of a
> `! defensive update - not sure if it works` directive and a narrowing of an
> `!$omp target update from(e)` guard from `Recon_Scheme > 0` to `Recon_Scheme == 2`. **Neither is
> real.** The string `defensive`/`not sure if it works` appears *nowhere* in either branch's `src/`,
> and the `!$omp target update from(e) if(...)` guard is the identical `Recon_Scheme == 2` form on
> both branches. Directly diffing the two branches' `MOM_PressureForce_FV.F90` yields **only** the
> 9-line block-size-defaults hunk (`32x4` vs `0x1`) — no transfer-guard or defensive-directive
> divergence exists.

**Is this the blessed template applied to pressure integrals? Yes**, modulo one caveat: unlike
`MOM_continuity_PPM.F90`'s manual `nteams` team-count tuning (`00-architecture.md` §5, working
around commit `5b5f6b2b1`'s under-launch bug), this branch relies entirely on `do concurrent` — no
`!$omp target teams num_teams(...)` anywhere in the diff — so if nvfortran under-launches teams
for the 5×/15×-widened `T5`/`T15`/etc. arrays the way it did for continuity, that workaround has not
yet been ported over here. Flag for whoever picks this up for merge.

> **Open (reviewed 2026-07-14):** does the PLM density-integral hot path actually need continuity's
> manual `num_teams(ceiling(...))` workaround, or does the tile geometry here (a `5*TILE_SIZE_X` inner
> dimension) keep nvfortran's default team launch adequate? The review could not settle this from
> source — it needs a benchmark of the PLM kernel against the under-launch symptom that motivated
> `5b5f6b2b1` in continuity. See KNOWLEDGE.md §9.

---

## 4. Same code, two branches — what diverges

| Aspect | `bodner-naive-port` (PLM/EOS portion) | `port/pressureforce-benchmark_ALE` |
|---|---|---|
| PLM k-blocking depth | **Identical.** Carries the full lineage incl. "move k loops inside" (EOS amortized over the whole k-block) and "optimise a little bit" (recurrence-direction fix), `98cb748c6` = pf's `2a99c9dd1` byte-for-byte | Same lineage; diverges from bodner *only* at the block-size-defaults commit and the later `desubmodule` |
| Block-size defaults (CPU side of the `#ifdef`; GPU side is `0/0/0` on both) | `32x4` (`niblock=32, njblock=4`) | `0x1` (`niblock=0, njblock=1`) — revised after benchmarking this specific EOS-heavy kernel |
| Submodule split | Kept (`_s.F90` present at tip) | Reversed (`a3e889601 desubmodule`, `_s.F90` deleted) |
| Roquet_rho diff-from-fork | +77 | +158/−16 (deeper — but its diff-from-fork *also* absorbs the `8eb41475b` EOS-merge equivalent that bodner instead folds into the PLM-lineage commits) |
| **Closer to merge-quality on this shared file** | — | **Marginally.** The only substantive edges are the benchmarked-down block defaults and the `desubmodule` cleanup; the k-blocking body (incl. the recurrence-direction fix) is identical on both. |

> **Open (reviewed 2026-07-14):** is `port/pressureforce-benchmark_ALE` genuinely the more merge-ready
> branch on the shared PLM code, or merely *different*? Its only edges are the `0x1` CPU default and
> the `desubmodule`. The review could not settle this from source — it needs a benchmark (does `0x1`
> beat `32x4` on CPU?) and a maintainer decision on whether desubmoduling is the intended end-state.
> See KNOWLEDGE.md §9.

The one thing `bodner-naive-port` has that the other branch doesn't touch at all is the actual
**Bodner MLE port** — but that work, per §1, is the naive contrast case, not a competing
implementation of the same feature.

---

## 5. `MOM_density_integrals_s.F90` — what the `_s` suffix actually means

**It is not a scalar/structured duplicate and not a device-hazard workaround.** `git show
bodner-naive-port:src/core/MOM_density_integrals_s.F90 | head` shows:

```fortran
!> Provides integrals of density
submodule (MOM_density_integrals) MOM_density_integrals_s
```

`_s` = **submodule**. The `submodulify`/`desubmodule` commit pair (`c75ebddaf` on `bodner-naive-port`,
`83628d990`/`a3e889601` on `port/pressureforce-benchmark_ALE`) is a Fortran `module`/`submodule`
split: `MOM_density_integrals.F90` shrinks to public declarations + an `interface … module subroutine
… end interface` block, and the executable bodies move verbatim into
`MOM_density_integrals_s.F90` (`submodule (MOM_density_integrals) MOM_density_integrals_s`,
`module subroutine int_density_dz_generic_plm(...)` matching the interface). The
`git log --graph` history shows a **sibling, unrelated branch** doing the identical maneuver
project-wide (`"+Convert all modules to module+submodule pairs for compile-speed testing"`) —
this is a **build/compile-time separation technique** (submodules let the interface-only file be
a stable compilation unit that downstream users don't need to recompile when the implementation
changes), not a GPU-porting device-hazard fix. `port/pressureforce-benchmark_ALE` in fact
**`desubmodule`s it back** (`a3e889601`) at the very end of its own history, i.e. the split was
provisional/exploratory scaffolding on both branches, later abandoned on the more mature branch.
Net effect on the diff stat (`MOM_density_integrals.F90` −518 / `MOM_density_integrals_s.F90` +589
on `bodner-naive-port`) is overwhelmingly code *moved*, not duplicated.

---

## 6. Lessons — what a naive port gets wrong that the blessed pattern fixes

1. **No CPU-blocking story at all vs. a tuned one.** `MOM_mixed_layer_restrat.F90` ships with zero
   `niblock`/`njblock`/`nkblock` parameters — the device version *is* the only version, so there is
   no way to reason about (or preserve) CPU cache performance. The PLM work on both branches, by
   contrast, exposes `PGF_PLM_NKBLOCK`/`NIBLOCK`/`NJBLOCK` runtime params with `#ifdef
   __NVCOMPILER_OPENMP_GPU`-gated defaults from day one, and iterates the CPU-side default
   (`32x4` → `0x1`) as understanding improves. **Lesson: add the block-size CS parameters and the
   `#ifdef` default split in the *first* commit, not as an afterthought — it is what makes "one
   source form for CPU and GPU" (principle #1) achievable instead of aspirational.**

2. **`omp target teams distribute` before `do concurrent`.** The naive branch's first commit reaches
   for classic OpenMP target constructs; two commits later it's rewritten to `do concurrent` +
   `DO_LOCALITY`. Both compile and (per the branch's own commit message) are "bitwise", but the
   churn shows the default idiom (principle #3) wasn't the first instinct — a porting agent
   following the doc should reach for `do concurrent` immediately and reserve `omp target teams`
   for the reduction/underperformance cases the architecture doc names explicitly.

3. **Many small `target enter/exit data` pairs vs. one persistent region.** Even at branch tip,
   `MOM_mixed_layer_restrat.F90` has 5 `enter data` / 4 `exit data` sites plus 2 `target data`
   blocks — most of the routine still round-trips host↔device multiple times per call. Only the
   integral/U/V/h-update tail was folded into a single persistent region. The blessed exemplars
   (`MOM_continuity_PPM.F90`, `MOM_CoriolisAdv.F90`) hoist communication/mapping to the driver and
   keep one region per hot path. **Lesson: design the data-region boundary before writing loops,
   not loop-by-loop.**

4. **Host EOS calls feeding device integrals, fixed by extending the interface, not duplicating
   logic.** The first naive commit computes `rho3d` on the host and `target update to`s it in — an
   extra round trip and, worse, a hidden serialization point between "compute density" and
   "consume density" that the second (`2271af66e`) commit removes by switching the MLE code to a
   genuine 3-D `calculate_density` entry point (the EOS-layer machinery for which the shared PLM
   lineage had already added — mirroring the merged 2D pattern) rather than hand-inlining the
   Wright/Roquet math into `MOM_mixed_layer_restrat.F90`. This preserved
   bitwise results because the same `_loc` free-function kernel is reused; it would not have if the
   formula had been retyped locally.

5. **Polymorphism is still not fully closed.** Even in the "GPU bitwise" commit, the
   `present(rho_ref)`-true branch of `calculate_density_array_3d_buggy_Wright` still passes `this`
   into a `do concurrent` loop with an explicit unresolved-limitation comment. The base-type
   fallback `a_calculate_density_array_3d` is polymorphic outright and silently unsafe for any EOS
   form that doesn't override it (per `00-architecture.md` §7.1, most forms still don't). A porting
   agent should treat "we added a 3D interface" as necessary but not sufficient — check whether the
   *specific* branch taken at runtime (which `rho_ref`/EOS-form combination) actually reaches a
   `_loc`-based override before calling it device-safe.

6. **Debug artifacts linger without review discipline.** Ten `print *` debug statements survived
   through three further MLE commits before `08be6d130 "print removal"` stripped them, and several
   commit messages carry typos (`"denstiy"`, `"btiwse"`, `"anwer date"`) — the signature of fast,
   unreviewed iteration. (Note: contrary to an earlier draft, the naive branch does *not* carry a
   leftover `! defensive update` directive or a coarser transfer guard than pf — its
   `MOM_PressureForce_FV.F90` differs from pf's by exactly the 9-line block-size-defaults hunk and
   nothing else; the "CPU-tuning is a distinct phase" point is real but lives in the `32x4 → 0x1`
   default change of Lesson 1, not in any transfer-directive cleanup.) **Lesson: strip debug prints
   and fix message typos before proposing merge — they are the cheapest possible signal that a port
   has not been reviewed.**

---

## Naive-vs-blessed checklist (derived from this comparison)

| Trait | Naive (`bodner-naive-port` MLE code) | Blessed (PLM/EOS work, both branches; merged exemplars in §5/§6.1 of `00-architecture.md`) |
|---|---|---|
| Default parallel idiom | Starts `omp target teams distribute`, migrates to `do concurrent` over 3 commits | `do concurrent` (+ `DO_LOCALITY`) from the start; `omp target teams` reserved for reductions/underperformance |
| CPU block-size parameters | None | `niblock`/`njblock`/`nkblock` CS params, `#ifdef`-gated defaults, tuned by benchmarking |
| Data-region granularity | Many small per-loop `enter/exit data` + one late persistent region for the tail only | One region per hot path/subroutine call |
| EOS density calls | First commit: host loop + `target update to`; fixed only in a follow-up commit | 3-D interface added at the EOS layer, dispatched once (host, v-table) then `do concurrent` + `_loc` kernel |
| Polymorphism (`this`) | N/A here (fixed via the EOS-layer fix, but the fix itself still has one unresolved `this`-copy branch) | Documented, tracked, not fully eliminated even on the "blessed" branch — an open item, not a solved one |
| Cleanliness | Ten `print *` debug statements committed and removed only in the last commit; typos in commit messages (`"denstiy"`, `"btiwse"`) | No debug prints; k-recurrences nested correctly (recurrence-direction fix — though that fix is *shared* with the naive branch, not exclusive to the blessed one) |
| Bitwise care | Claimed ("Bitwise" in commit message) and plausible by construction (same `_loc` formulas, relocated loops) — not independently checksum-verified in this study | Same standard; also not independently checksum-verified here (source+git only, per constraints) |

---

## Review rubric — run against your own diff *before* proposing a merge

Eight pass/fail gates distilled from the contrast above. Each cites the concrete evidence a reviewer
can re-check, so "why does this matter?" always has an answer in this repo's history.

1. **Block-size CS parameters present and `#ifdef`-gated?**
   `grep` your diff: every hot-loop file must add `ni/nj/nkblock`-style CS integers, `get_param`'d
   (e.g. `PGF_PLM_NKBLOCK` / `_NIBLOCK` / `_NJBLOCK`, `MOM_PressureForce_FV.F90:2297-2308`), behind an
   `#ifdef __NVCOMPILER_OPENMP_GPU` default split — GPU `0` (whole domain), CPU a real tile size
   (`MOM_PressureForce_FV.F90:41-48`). **Fail:** `MOM_mixed_layer_restrat.F90` on `bodner-naive-port`
   — *zero* block params (verified: no `niblock`/`njblock`/`nkblock` hits), whole-domain assumption
   baked in with no CPU override. → Add these in the **first** commit, not as an afterthought.

2. **One persistent data region per hot path — not many small `enter/exit data` pairs?**
   Count `enter data` / `exit data` / `target data` per touched file; they should collapse toward a
   single region spanning the call. **Fail:** MLE tip still has **5 `enter data` / 4 `exit data` / 2
   `target data`** (verified) — most of the routine round-trips host↔device per call. **Pass:** the
   continuity/CorAdCalc drivers hoist mapping up and keep one region per hot path.

3. **`do concurrent` (+`DO_LOCALITY`) the default idiom — `omp target teams` only for
   reductions/underperformance?**
   `grep 'target teams distribute'`; there should be none left except documented reduction/column
   cases. **Fail signal from history:** MLE first cut shipped **9** `target teams distribute` and 0
   `do concurrent`; it took three commits to reach 0/10 (verified counts). Reach for `do concurrent`
   immediately.

4. **EOS density via the extended 2D/3D interface, v-table resolved once host-side?**
   No host `calculate_density` loop feeding a `!$omp target update to`. The call must be a single
   2D/3D `calculate_density(...)` that resolves scaling on the host (`EOS%RL2_T2_to_Pa …`,
   `MOM_EOS.F90`) then runs `do concurrent` over a free `_loc` kernel (no `this`). **Evidence:**
   `2271af66e` (MLE) and the "move k loops inside" PLM commit both do this; the naive first cut did
   the host-loop→`target update to` anti-pattern.

5. **Bitwise safety — argue it, and check *which* runtime path you hit.**
   For every relocated loop confirm the per-element arithmetic is byte-identical: same `_loc`
   function, same operand order, no new reduction / FMA reassociation. **Caveat to check
   explicitly:** the `present(rho_ref)` branch of `calculate_density_array_3d_buggy_Wright` *still*
   passes polymorphic `this` (verbatim comment: "implicit copy of `this` … cannot yet be prevented").
   Verify your runtime `rho_ref`/EOS-form combination reaches a `_loc`-based override, **not** the
   polymorphic base-type fallback `a_calculate_density_array_3d` (unsafe for any EOS form lacking an
   override — per `00-architecture.md` §7.1 most forms still lack one).

6. **No debug prints, no commit-message typos going into the merge?**
   `grep 'print \*'`; strip them. **Evidence:** ten debug prints survived three MLE commits before
   removal — the cheapest possible signal that a diff has not been reviewed.

7. **CPU-tuning phase actually done — not just "compiles + bitwise once"?**
   Block-size CPU defaults must be *benchmarked*, not copy-pasted. **Evidence:** the CPU default was
   iterated `32x4 → 0x1` on the PLM kernel because the `MOM_continuity_PPM.F90` `32/4/1` convention
   did not suit the EOS-heavy PLM loop. A diff whose CPU default is a verbatim copy of another
   kernel's is a red flag until benchmarked.

8. **`k`-recurrences nested the right way?**
   Any genuine `k`-recurrence (cumulative sums, `e(:,:,K)=e(:,:,K+1)+…`) must be
   `do concurrent(j) → serial do k → do concurrent(i)`, never a `serial do k` wrapping a full-2D
   `do concurrent(j,i)`. **Evidence:** the "optimise a little bit" commit (present on *both* branches)
   reverses exactly this in `PressureForce_FV_Bouss` for `e`, `intx_pa`, and `inty_pa`.

---

## Verification notes

Verified by an Opus agent against `git` (branches `bodner-naive-port`, `port/pressureforce-benchmark_ALE`,
`dev/gpu`) and source only; no build/run. Temp artifacts under `tmp_local_artifacts/` (none retained).

**Confirmed (spot-checked against source/git):**
- Same-timestamp literal fork point: `8744bf362` (32x4) and `fae6c9a5c` (0x1), both
  `Tue Jun 2 11:36:45 2026 -0400`, both by Edward Yang, both parented on the patch-id-matched "move
  block size to user input" pair (`b864bafba`/`1038a4921`).
- MLE idiom migration counts: `target teams distribute` 9→4→0→0→0; `do concurrent` 0→5→9→10→10 across
  `6f854c5aa`→`02bf308f3`→`09d1b93f7`→`2271af66e`→`08be6d130`. Tip: 0 `omp target teams`, 5 `enter
  data`, 4 `exit data`, 2 `target data`, **zero** block-size params. All verbatim.
- §2 EOS code (host `RL2_T2_to_Pa` dispatch, base-type polymorphic fallback
  `a_calculate_density_array_3d`, Wright override with `rho_ref`→`this` / else→`_loc`, and the
  "implicit copy of `this` … cannot yet be prevented" comment) — all present verbatim on bodner tip.
- §5 submodule facts: `submodule (MOM_density_integrals) MOM_density_integrals_s` (line 8);
  `port/pressureforce-benchmark_ALE` deletes `_s.F90` via `a3e889601 desubmodule`; sibling branch
  `ecbf83a8d "+Convert all modules to module+submodule pairs for compile-speed testing"` exists;
  cumulative bodner diff `−518 / +589`.
- §3 `#ifdef __NVCOMPILER_OPENMP_GPU` block, `PGF_PLM_*` params, the recurrence-direction code, the
  T5 `(…,kstart:kend)` 3-D scratch arrays + single 3-D `calculate_density`, and the §1.2
  `target data … DO_LOCALITY(local(...))` region — all present as quoted (§3 quoted call signatures
  are lightly idealized paraphrases, but the transformation they depict is real).
- MLE commits authored by "Jorge" with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

**Corrected:**
1. Fork positions: bodner is **8** commits behind current `dev/gpu` tip `b8c471cfa` (not 3); pf's fork
   `c82e1254a6` is **no longer** the tip (now 4 behind). The "pf diff vs `dev/gpu` is empty" drift
   test is stale — the correct test is pf's diff vs its *own* fork (empty), and bodner has a 3-line
   real `MOM.F90` edit.
2. Shared lineage is **~24 commits (21 byte-identical by `git patch-id` + 3 EOS-drift)**, not "26
   identical". The 3 drifting commits (`cec1c8d0a`/`8bbd4c085`/`207c0ce34`) carry the EOS 3-D
   additions inline on bodner because it forked before the merged EOS commit.
3. **The 3-D EOS interface was not added by `2271af66e`** (which only edits MLE, +17/−8, to *call*
   it) — it came from the shared PLM lineage.
4. **Fabricated evidence removed:** no `! defensive update - not sure if it works` directive exists
   in either branch, and there is no `Recon_Scheme > 0 → == 2` transfer-guard narrowing — the two
   branches' `MOM_PressureForce_FV.F90` differ by *only* the 9-line block-size-defaults hunk.
5. **§4 "PLM k-blocking depth" divergence was false:** bodner carries the *full* lineage including
   the "optimise a little bit" recurrence-direction fix (`98cb748c6` = pf's `2a99c9dd1`, patch-id
   identical). Real post-fork divergence = block-size defaults (`32x4` vs `0x1`) + pf's `desubmodule`.
6. Numbers: MLE own-work stat `+174/−91` (not `+265`); newly-added debug prints = **ten** (not five);
   Roquet_rho pf diff `+158/−16`.

**Confidence:** High on all branch-forensics (patch-id, timestamps, file diffs re-derived
independently). High on the §2/§3/§5 code-content confirmations (read on the branch blobs). The open
items in §3 and §4 are genuinely benchmark-dependent judgments (team-launch adequacy; whether pf
is materially more merge-ready) that a source-only study cannot settle.
