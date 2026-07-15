# K-blocking / Tiling — the Blessed CPU-and-GPU-Preserving Refactor

> Companion to `00-architecture.md` §5 (its one-paragraph summary is the seed for this document).
> This document reconstructs the *exact* mechanical transformation from git history across three
> cases — continuity (`MOM_continuity_PPM.F90`, fully merged, i/j **and** k blocking), CoriolisAdv
> (`MOM_CoriolisAdv.F90`, merged `b8c471cfa`, k-blocking only), and horizontal viscosity
> (`MOM_hor_visc.F90`, in-flight on `kblock-hor-visc`, k-blocking only) — and distills a template.
> Read `00-architecture.md` first.

---

## 1. The mechanical transformation, step by step

There are two distinct blocking axes in this codebase, and it is important not to conflate them:

- **Horizontal (i/j) tiling** — only in `MOM_continuity_PPM.F90`. A 2-D (or implicitly 3-D, looping
  `k` innermost/serially) computation is chopped into rectangular `(niblock × njblock)` tiles; the
  host loops over tiles, and each tile's body becomes a small, separately-launched device kernel.
- **Vertical (k) blocking** — in all three modules. A `do k=1,nz` **serial outer loop** whose body
  is a 2-D horizontal calculation (with 2-D scratch/work arrays reused every iteration) is turned
  into a loop over **blocks of `nkblock` adjacent layers**, so that up to `nkblock` layers can be
  exposed simultaneously to `do concurrent`/`omp target`. Scratch arrays grow one extra dimension of
  size `nkblock` (not `nz`) so device memory footprint stays bounded regardless of column depth.

Both axes follow the same recipe; k-blocking is the more general and more widely used one, so it is
the primary subject below.

### 1.1 Vertical (k) blocking — before/after (continuity reconstruction)

This is commit `93dbbd36e` ("Use blocking in k dimension for continuity reconstruction (#165)"),
`src/core/MOM_continuity_PPM.F90`, subroutine `PPM_reconstruction_x`. Before:

```fortran
! integer :: k                                          ! vertical grid index
real, dimension(SZI_(G),SZJ_(G),SZK_(GV))  :: slp        ! full-column scratch slope array
...
do concurrent (k=1:nz, j=jsl:jel, i=isl-1:iel+1)
  ...
  slp(i,j,k) = sign(1.,slp(i,j,k)) * min(abs(slp(i,j,k)), 2. * min(dMx, dMn))
enddo
...
do concurrent (k=1:nz, j=jsl:jel, i=isl:iel)
  h_im1 = G%mask2dT(i-1,j) * h_in(i-1,j,k) + (1.0-G%mask2dT(i-1,j)) * h_in(i,j,k)
  h_ip1 = G%mask2dT(i+1,j) * h_in(i+1,j,k) + (1.0-G%mask2dT(i+1,j)) * h_in(i,j,k)
  h_W(i,j,k) = 0.5*( h_im1 + h_in(i,j,k) ) + oneSixth*( slp(i-1,j,k) - slp(i,j,k) )
  h_E(i,j,k) = 0.5*( h_ip1 + h_in(i,j,k) ) + oneSixth*( slp(i,j,k) - slp(i+1,j,k) )
enddo
if (monotonic) then
  call PPM_limit_CW84(h_in, h_W, h_E, G, GV, isl, iel, jsl, jel, nz)
else
  call PPM_limit_pos(h_in, h_W, h_E, h_min, G, GV, isl, iel, jsl, jel, nz)
endif
```

After (`src/core/MOM_continuity_PPM.F90:2703-2832` today):

```fortran
integer :: k, kk  ! vertical grid and k-block index
real, dimension(SZI_(G),SZJ_(G),max(1,nkblock)) :: slp   ! one k-BLOCK of scratch, not the full column
...
do ks = 1, nz, nkblock
  ke = min(ks + nkblock - 1, nz)
  ...
  do concurrent (k=ks:ke, j=jsl:jel, i=isl-1:iel+1) DO_LOCALITY(local(dMx,dMn,kk))
    kk = k - ks + 1
    slp(i,j,kk) = sign(1.,slp(i,j,kk)) * min(abs(slp(i,j,kk)), 2. * min(dMx, dMn))
  enddo
  ...
  do concurrent (k=ks:ke, j=jsl:jel, i=isl:iel) DO_LOCALITY(local(h_im1,h_ip1,kk))
    kk = k - ks + 1
    h_im1 = G%mask2dT(i-1,j) * h_in(i-1,j,k) + (1.0-G%mask2dT(i-1,j)) * h_in(i,j,k)
    h_ip1 = G%mask2dT(i+1,j) * h_in(i+1,j,k) + (1.0-G%mask2dT(i+1,j)) * h_in(i,j,k)
    h_W(i,j,k) = 0.5*( h_im1 + h_in(i,j,k) ) + oneSixth*( slp(i-1,j,kk) - slp(i,j,kk) )
    h_E(i,j,k) = 0.5*( h_ip1 + h_in(i,j,k) ) + oneSixth*( slp(i,j,kk) - slp(i+1,j,kk) )
  enddo
  if (monotonic) then
    call PPM_limit_CW84(h_in, h_W, h_E, G, GV, isl, iel, jsl, jel, ks, ke)
  else
    call PPM_limit_pos(h_in, h_W, h_E, h_min, G, GV, isl, iel, jsl, jel, ks, ke)
  endif
enddo
```

The five-part mechanical recipe visible in this diff (repeated verbatim in CoriolisAdv and
hor_visc):

1. **Add a block-size CS member** (`nkblock`, or `niblock`/`njblock`) and thread it into the
   subroutine as an argument (`MOM_continuity_PPM.F90:76-78`, `2693`).
2. **Wrap the serial/parallel `k=1:nz` axis in an outer host loop over block starts**:
   `do ks = 1, nz, nkblock ; ke = min(ks+nkblock-1, nz)`. For i/j tiling the equivalent is
   `do j_start=jsh,jeh,njblock ; do i_start=ish-1,ieh,niblock` (`:696`).
3. **Shrink full-extent scratch arrays to block extent**: `SZK_(GV)` → `max(1,nkblock)` for k-blocks
   (`:2706`), or `SZI_/SZJ_(G)` → `niblock,njblock` for i/j tiles (`:621-636`). This is the memory
   payoff: device/stack footprint is bounded by the *block* size, not the *domain* size.
4. **Introduce a block-local index** (`kk = k - ks + 1`, or `ii = i - i_start + 1 ; jj = j - j_start + 1`)
   and rewrite every scratch-array reference from the global index to the local one, while grid
   metric and full-size in/out arrays (`h_in`, `h_W`, `G%mask2dT`, ...) keep using the **global**
   index unchanged (`:2754-2764`, `:717-718`).
5. **Pass the active sub-range `(ks,ke)` or `(i_start,i_end,j_start,j_end)` down** to any helper
   subroutine that previously took the full range (`PPM_limit_pos(...,ks,ke)` vs the old
   `PPM_limit_pos(...,nz)`; `zonal_flux_adjust(...,i_start,i_end,j_start,j_end,...)`).

### 1.2 The i/j-tiled hybrid kernel (continuity `zonal_mass_flux`)

`zonal_mass_flux`, `src/core/MOM_continuity_PPM.F90:696-736`, is the file's most fully evolved
example, combining i/j tiling with an explicit device-team launch:

```fortran
do j_start=jsh,jeh,njblock ; do i_start=ish-1,ieh,niblock
  i_end = min(i_start+niblock-1,ieh)
  j_end = min(j_start+njblock-1,jeh)

  ! calculate number of teams
  !$ nteams = ceiling(real((j_end-j_start+1)*(i_end-i_start+1))/128.)

  do concurrent (jj=1:j_end-j_start+1, ii=1:i_end-i_start+1)
    do_I(ii,jj) = .true.
  enddo
  ! Set uh and duhdu.
  !$omp target teams num_teams(nteams)
  do k=1,nz
    if (use_visc_rem) then
      !$omp loop collapse(2) private(ii,jj)
      do j=j_start,j_end ; do I=i_start,i_end
        ii=I-i_start+1 ; jj=j-j_start+1
        visc_rem(ii,jj,k) = visc_rem_u(I,j,k)
      enddo ; enddo
    endif
    !$omp loop collapse(2) private(ii,jj)
    do j=j_start,j_end ; do i=i_start,i_end
      ii=i-i_start+1 ; jj=j-j_start+1
      call flux_elem(u(i,j,k),h_in(i,j,k),h_in(i+1,j,k),h_W(i,j,k),h_W(i+1,j,k),h_E(i,j,k),&
                     h_E(i+1,j,k),uh_t(ii,jj,k),duhdu(ii,jj,k),visc_rem(ii,jj,k),G%dy_Cu(i,j),&
                     G%IareaT(i,j),G%IareaT(i+1,j),G%IdxT(i,j),G%IdxT(i+1,j),dt,CS%vol_CFL,&
                     por_face_areaU(I,j,k))
      ...
    enddo ; enddo
  enddo
  !$omp end target teams
  ...
enddo ; enddo
```

`flux_elem`/`flux_elem_OBC` are `elemental subroutine`s carrying `!DIR$ ATTRIBUTES FORCEINLINE`
(`MOM_continuity_PPM.F90:1086`, `:1149`); `ratio_max` is a plain `pure function`
(`MOM_continuity_PPM.F90:3086`) with **no** inline attribute of its own (nvfortran inlines it via the
`-Minline=name:ratio_max` build flag catalogued in `00-architecture.md` §7.5, not a source directive).
Either way they are called from inside the tile kernel — this is the "extract into `pure`/`elemental`
subroutine" idiom from `00-architecture.md` guiding principle 2, and it is what lets the tile body call
into shared code without a device-side v-table/module boundary. (Verified: the current merged form
uses Intel `!DIR$ ATTRIBUTES FORCEINLINE`, having replaced the earlier `!NVF$ INLINE` directive in
`93dbbd36e`; the hybrid kernel itself spans `:696-738`, ending at the `!$omp end target teams`.)

### 1.3 Historical evolution of the same transformation (continuity)

The current form is the end of a multi-commit evolution, all on `src/core/MOM_continuity_PPM.F90`
(`git log dev-gfdl..dev/gpu --oneline -- src/core/MOM_continuity_PPM.F90`, chronological):

| Commit | What changed |
|---|---|
| `26b8b4da7` "implement tiling of zonal/meridional_mass_flux main loops" | **First** i/j tiling. Used **OpenACC** (`!$acc parallel loop`, `!$acc loop seq`) with hard-coded `TILE_SIZE_X=32,TILE_SIZE_Y=4` constants; introduced the `ii,jj` block-local index and block-shaped scratch arrays for the first time. |
| `3cb184edd` "use openmp instead of openacc" | Same tile structure, directives translated from `!$acc parallel/loop` to `!$omp target [teams] loop`. |
| `5b5f6b2b1` "add teams spec to problematic target region" | Added the hand-computed `nteams` and `num_teams(nteams)` — see §6. |
| `a327bbcf2` "continuity: runtime tile sizes user params" | `TILE_SIZE_X/Y` became runtime CS members read via `get_param`, defaulting to 0; `omp_get_num_devices()>0` checked at runtime to pick whole-domain sizing on GPU. |
| `bf3a6c3c3`, `e8b0ecfbf` "omp target teams loop -> do concurrent" | Reverted some `omp target teams loop` regions back to `do concurrent` where it performed better/was simpler. |
| `51c2f4032` "rename TILE_SIZE_[XY] to niblock/njblock" | Cosmetic rename to the current, more MOM6-idiomatic names. |
| `7f182139f` "proper ni/jblock selection in 3d_fluxes and adjust_vel" | Fixed missed call sites that weren't resolving 0→whole-domain. |
| `93dbbd36e` "Use blocking in k dimension for continuity reconstruction (#165)" | Added the **third, independent** blocking axis `nkblock` to the reconstruction routines (§1.1); switched the GPU/CPU default selection from a runtime `omp_get_num_devices()` check to a compile-time `#ifdef __NVCOMPILER_OPENMP_GPU` (§2). |

Two lessons embedded in this history: (a) the port started on OpenACC and was mechanically
translated to OpenMP target once that became the house style — the tiling structure itself did not
change across that translation; (b) tile-size selection went through three different mechanisms
(hardcoded constant → runtime `omp_get_num_devices()` check → compile-time `#ifdef` + `if(block==0)`
resolution) before settling on the scheme described in §2.

### 1.4 CoriolisAdv: the "serial-k-loop-with-2D-scratch" variant

`MOM_CoriolisAdv.F90` (`b8c471cfa`, "Kblock coradcalc #167") starts from a different but very common
pre-existing pattern: a `do k=1,nz` **outer serial loop** (not a `do concurrent`) whose body computes
2-D (`SZIB_(G),SZJB_(G)`-shaped) work arrays fresh every iteration — the classic CPU-cache-friendly
"process one layer, reuse small 2-D scratch" idiom used throughout MOM6's dynamical core. Before
(`src/core/MOM_CoriolisAdv.F90`, pre-`b8c471cfa`):

```fortran
real, dimension(SZIB_(G),SZJB_(G)) :: dvdx, dudy, ...   ! 2-D, one layer's worth
...
do k=1,nz
  do concurrent (J=Js_q:Je_q, I=Is_q:Ie_q)
    dvdx(I,J) = (v(i+1,J,k)*G%dyCv(i+1,J)) - (v(i,J,k)*G%dyCv(i,J))
    dudy(I,J) = (u(I,j+1,k)*G%dxCu(I,j+1)) - (u(I,j,k)*G%dxCu(I,j))
  enddo
  ...
enddo
```

After (`src/core/MOM_CoriolisAdv.F90:148-202` of the diff; current source `:373-`):

```fortran
real, dimension(SZIB_(G),SZJB_(G),merge(GV%ke,CS%nkblock,CS%nkblock==0)) :: dvdx, dudy, ...
...
do k_start=1,nz,nkblock
  k_end = min(k_start+nkblock-1, nz)
  kmax = k_end - k_start + 1
  ...
  do concurrent (kk=1:kmax, J=Js_q:Je_q, I=Is_q:Ie_q) DO_LOCALITY(local(k))
    k = k_start + kk - 1
    dvdx(I,J,kk) = (v(i+1,J,k)*G%dyCv(i+1,J)) - (v(i,J,k)*G%dyCv(i,J))
    dudy(I,J,kk) = (u(I,j+1,k)*G%dxCu(I,j+1)) - (u(I,j,k)*G%dxCu(I,j))
  enddo
  ...
enddo
```

Same five-step recipe as §1.1, with one addition: `k` itself is no longer the `do concurrent`
control variable — the **block-local** index `kk` is (range `1:kmax`), and the *global* layer index
`k` is a per-iteration computed scalar carried via `DO_LOCALITY(local(k))`. This matters for
correctness (see §3) and was itself revised mid-flight: the local branch `kblock-coradcalc`
(single commit `0b5c00333`, predates the merged form) iterates the other way — `do concurrent
(k=kstart:kend, ...) ; kk = k - kstart + 1` — and a later commit inside `b8c471cfa`
("Iterate k-block loops on the block-local index directly") flipped it to iterate on `kk` with `k`
computed, which is the form now in `dev/gpu`. Both are bitwise-equivalent; the flip was a
style/performance preference, not a correctness fix (see §5 for why nvfortran can care about which
variable is the literal `do concurrent` control variable).

Parts of `CorAdCalc` not yet portable (OBC segment handling, WENO, `KE_UP3`) are explicitly left as
serial `do k=k_start,k_end ! TODO: port (OBC GPU path not yet implemented)` loops with a manually
computed `kk`, i.e. the k-block **loop nest is always installed**, but its body may still be
call-by-call serial Fortran where a `do concurrent`/OpenMP rewrite hasn't happened yet. This is a key
structural point: k-blocking and "porting the body to a parallel construct" are separable steps.

### 1.5 Horizontal viscosity: same template, still in-flight

`kblock-hor-visc` (`36f51ff84` "block k in horizontal_viscosity") applies the *identical* recipe to
`MOM_hor_visc.F90`, and says so in its own commit message: *"Restructure the outer k-loop in
horizontal_viscosity into a ks-block loop using the same pattern as PPM_reconstruction_x/y."*
Representative before/after (`src/parameterizations/lateral/MOM_hor_visc.F90`, diff of `36f51ff84`):

```fortran
! before
do k=1,nz
  do concurrent (j=Jsq-1:Jeq+2, i=Isq-1:Ieq+2)
    dudx(i,j) = CS%DY_dxT(i,j)*((G%IdyCu(I,j) * u(I,j,k)) - (G%IdyCu(I-1,j) * u(I-1,j,k)))
  enddo
  ...
enddo

! after
do kstart=1,nz,nkblock
  kend = min(kstart+nkblock-1, nz)
  do concurrent (k=kstart:kend, j=Jsq-1:Jeq+2, i=Isq-1:Ieq+2) DO_LOCALITY(local(kk))
    kk = k - kstart + 1
    dudx(i,j,kk) = CS%DY_dxT(i,j)*((G%IdyCu(I,j) * u(I,j,k)) - (G%IdyCu(I-1,j) * u(I-1,j,k)))
  enddo
  ...
enddo
```

Note this file iterates on the *global* `k` (like `kblock-coradcalc`, not like the final merged
`CorAdCalc`), confirming both index styles are in active use and considered equivalent. Not-yet-
ported sub-blocks (e.g. `use_Leithy` smoothed-velocity terms, `id_normstress` diagnostic capture) are
left as literal serial `do k=kstart,kend ! TODO: port` loops nested inside the k-block loop —
exactly the same incremental-porting pattern seen in CoriolisAdv (§1.4).

Follow-on commits on the same branch:
- `d6fe494e6` "pass nkblock as argument to horizontal_viscosity" — adds a `hor_visc_nkblock()`
  accessor function so callers outside the module (which cannot see the `private` CS members
  directly) can compute `merge(GV%ke, CS%nkblock, CS%nkblock==0)` and pass it in explicitly, rather
  than the subroutine reaching into `CS%nkblock` itself. This is purely an encapsulation cleanup, not
  a behavior change.
- `28eb296f4` "rearrange arrays to improve cpu perf" — see §4. (Not a *pure* declaration move — it
  also fuses a short run of `do concurrent` loops and normalizes one array's dimension expression;
  corrected there.)

Commit order on the branch is confirmed `36f51ff84 → d6fe494e6 → 28eb296f4` (via `git merge-base
--is-ancestor`). **Caveat on staleness:** the `kblock-hor-visc` branch has since advanced well past
`28eb296f4` — its tip (`9b69fc581` at time of verification) carries ~15 further commits that extract
Leith/QG-Leith, EY24 backscatter, GME setup and the Leith+E update into named subroutines, add
tailored `DO_LOCALITY` clauses, and make `nkblock` a runtime parameter. Treat the three commits above
as the *illustrative* k-blocking slice of a still-evolving branch, not its final state.

---

## 2. How `niblock`/`njblock`/`nkblock` get set

All three block-size families follow one convention, first established for continuity and copied
verbatim to CoriolisAdv (`nkblock` only) and hor_visc (`nkblock` only):

**(a) User-tunable runtime parameters**, declared as `integer` CS members and read with `get_param`
in the module's `*_init` routine:

```fortran
! MOM_continuity_PPM.F90:3202-3213
call get_param(param_file, mdl, "CONTINUITY_NIBLOCK", CS%niblock, ..., default=default_niblock, layoutParam=.true.)
call get_param(param_file, mdl, "CONTINUITY_NJBLOCK", CS%njblock, ..., default=default_njblock, layoutParam=.true.)
call get_param(param_file, mdl, "CONTINUITY_NKBLOCK", CS%nkblock, ..., default=default_nkblock, layoutParam=.true.)
if (CS%niblock < 0) call MOM_error(FATAL, "CONTINUITY_NIBLOCK must be nonnegative; use 0 to select the default block size.")
```
(Analogously `CORIOLIS_ADV_NKBLOCK` in `MOM_CoriolisAdv.F90:2108-2112`, `HORVISC_NKBLOCK` in
`MOM_hor_visc.F90:2781-2785`.) Negative values are a fatal error; **0 is reserved to mean "dynamic /
whole extent."**

**(b) Compile-time CPU-vs-GPU default divergence**, selected by the same preprocessor guard in every
module's `*_init` (`MOM_continuity_PPM.F90:3120-3129`):

```fortran
#ifdef __NVCOMPILER_OPENMP_GPU
  integer, parameter :: default_niblock = 0 !< whole domain / no cache blocking
  integer, parameter :: default_njblock = 0
  integer, parameter :: default_nkblock = 0
#else
  ! These were found to give best performance in limited tests.
  integer, parameter :: default_niblock = 32
  integer, parameter :: default_njblock = 4
  integer, parameter :: default_nkblock = 1
#endif
```
CoriolisAdv and hor_visc only have the k-axis, and their CPU default is `nkblock = 1` (i.e. exactly
the original per-layer serial loop, byte-for-byte the pre-port cache behavior) while GPU default is
`nkblock = 0`.

**(c) Resolving `0` → whole extent, at call time, not at init time.** `CS%niblock`/`njblock`/`nkblock`
stay `0` in the CS; every call site that actually sizes an array or drives a loop resolves it locally:
- Continuity, i/j axis (`MOM_continuity_PPM.F90:186-189`, repeated at 4 call sites covering both
  advection directions and both x-first/y-first branches):
  ```fortran
  LB = set_continuity_loop_bounds(G, CS, i_stencil=.false., j_stencil=.true.)
  ! set whole-domain block sizes when ni/jblock is 0
  if (niblock == 0) niblock = LB%ieh-LB%ish+2
  if (njblock == 0) njblock = LB%jeh-LB%jsh+1
  ```
  The **resolved size depends on which of the four advection phases is being computed** (zonal vs.
  meridional, symmetric-halo offset `+2` vs `+1`) — this is why the same `CS%niblock` is re-resolved
  at every call site rather than once in `_init`.
- Continuity, k axis (`MOM_continuity_PPM.F90:512-513`, `557-558`, inside `zonal_edge_thickness` /
  `meridional_edge_thickness`): `nkblock = CS%nkblock ; if (nkblock == 0) nkblock = nz`.
- CoriolisAdv (`MOM_CoriolisAdv.F90:275`): `nkblock = merge(GV%ke, CS%nkblock, CS%nkblock==0)` — a
  one-line equivalent of the same `if`, used because it appears inside an array-dimension expression
  (`real, dimension(SZIB_(G),SZJB_(G),merge(GV%ke,CS%nkblock,CS%nkblock==0))`, `:169`) as well as a
  scalar assignment.
- hor_visc (`MOM_hor_visc.F90:500`, and — after `d6fe494e6` — via the external accessor
  `hor_visc_nkblock(CS)` so callers outside the module can compute it too).

This 3-stage design (user override → compile-time default → runtime 0-resolution) is what lets
*one source file* serve both roles: on CPU, tiles are small and cache-resident (`32×4` points,
1 layer); on GPU, "tiles" *are* the whole horizontal domain / whole column, so the tiling loops
degenerate to a single iteration and the inner kernel body becomes one big device-parallel region.

**Evolutionary note:** this exact scheme is the fourth iteration of tile-size selection for
continuity (see §1.3): hardcoded constant → `omp_get_num_devices()` runtime check
(`a327bbcf2`) → `#ifdef __NVCOMPILER_OPENMP_GPU` compile-time default + runtime `if(block==0)`
resolution (`93dbbd36e`, the version now in the tree). The runtime `omp_get_num_devices()` check was
dropped because it doesn't compose with the k-axis default (nkblock had no such check at the time)
and because compile-time defaults are simpler to reason about when the same binary is not expected
to run on both host and device targets interchangeably.

---

## 3. Why k/i/j-blocking preserves bitwise results

The core argument, true in every one of the three cases studied:

**Blocking only changes which *loop* an operation is nested inside and which *scratch buffer* holds
an intermediate value — it never changes what is computed from what.** For a fixed grid point
`(i,j,k)`:
- The set of *inputs* read (`h_in(i±1,j,k)`, `u(I,j,k)`, `G%mask2dT(i,j)`, ...) is identical before
  and after blocking — blocking never changes a stencil's footprint.
- The *order of floating-point operations* within the expression computing that point's output
  (`slp(...) = sign(1.,...) * min(abs(...), 2.*min(dMx,dMn))`, `h_W(i,j,k) = 0.5*(...) +
  oneSixth*(...)`) is copied verbatim; only the array subscript used to store/load the intermediate
  (`slp(i,j,k)` → `slp(i,j,kk)`) changes, and `kk` is a bijection of `k` within a block
  (`kk = k - ks + 1`), so it addresses the *same logical value*, just at a different physical
  offset.
- Each grid point's result therefore depends only on that point's own block-local computation, never
  on which other points share its block or on block iteration order — blocks (and, within GPU
  kernels, `do concurrent`/team iterations) can therefore be evaluated in *any* order, or all at
  once, without changing any individual result. This is precisely why `do concurrent` (whose
  standard-mandated semantics already forbid inter-iteration order dependence) is a legal target for
  the transformation in the first place.
- Reductions are the one place order-independence needs an explicit argument, and the code handles it
  two ways: (i) keep any cross-layer accumulation as a **sequential** `do k=1,nz` inside the tile,
  so order is pinned by construction rather than left to a compiler reduction. Two distinct
  accumulations in `zonal_mass_flux` do this: `visc_rem_max(ii,jj) = max(visc_rem_max, visc_rem(...,k))`
  (`MOM_continuity_PPM.F90:746-752`) — a `max`, associative regardless of order — **and, more
  importantly for the bitwise argument, a genuine floating-point vertical sum**
  `uh_tot_0(ii,jj) = uh_tot_0 + uh_t(ii,jj,k)` / `duhdu_tot_0 += duhdu(ii,jj,k)`
  (`MOM_continuity_PPM.F90:776-782`). The float sum is left as a literal serial `do k=1,nz`
  precisely so its summation order is fixed; note the i/j tiling never splits this loop (each
  `(ii,jj)` column accumulates its own independent partial sum over the *full* `1:nz`), so tiling
  cannot reorder it — and continuity's k-axis blocking (`nkblock`) is confined to
  `PPM_reconstruction_x/y`, which contains **no** cross-layer sum, so it cannot reorder it either.
  (ii) where a
  true reduction is used (`any_simple_OBC`, `MOM_continuity_PPM.F90:870`,
  `DO_LOCALITY(reduce(.or.:any_simple_OBC))`), it is a boolean OR, which is associative/commutative
  bit-for-bit regardless of grouping — unlike floating-point sums, boolean/integer reductions are
  reorder-safe by construction. (Real floating-point reproducing sums are handled by an entirely
  separate exact-integer mechanism in `MOM_coms.F90`; see `00-architecture.md` §7.2 and
  `07-reproducibility.md` — none of the k/i/j-blocking transforms in this document touch a
  floating-point reduction.)

**Where care was genuinely needed** (documented in the diffs themselves):

1. **Which variable is the `do concurrent` control variable vs. a private-computed scalar.**
   CoriolisAdv's final form iterates on `kk` (block-local, `1:kmax`) and computes the global `k`
   inside the loop body via `DO_LOCALITY(local(k))` — i.e. `k` must be declared with `local` locality
   or every device thread would race on a shared `k`. Getting this wrong (declaring `k` shared instead
   of `local`) would not change *which* value is stored where, but would be a data race / undefined
   behavior on GPU, not a silent bitwise mismatch — still worth flagging because such bugs are easy to
   introduce when converting a "compute index, then use it" idiom to a parallel loop. The
   `DO_LOCALITY(local(...))` macro (`src/framework/do_concurrent_compat.h`) expands to the standard
   `local()`/`reduce()` locality-specifier list when the compiler supports it
   (`HAVE_FC_DO_CONCURRENT_LOCAL`), and to nothing otherwise — so on older compilers correctness
   relies on the compiler's default (usually correct, but undocumented) treatment of loop-body scalars.
2. **Passing the active sub-range, not the full range, to helpers.** `PPM_limit_pos`/`PPM_limit_CW84`
   changed signature from `(...,nz)` to `(...,ks,ke)` (`MOM_continuity_PPM.F90:2830` today vs. the
   pre-`93dbbd36e` `(...,nz)`) specifically so a helper never iterates outside the block it was given
   scratch data for — if it had kept iterating `1:nz` while `slp` was sized `nkblock`, it would read
   garbage/out-of-bounds, not merely reorder arithmetic. This is a shape-safety concern introduced
   *by* blocking, not a bitwise-order concern, but it is the actual bug class the PR's second commit
   ("add error for -ve block size") and its general carefulness with `min(1,nkblock)`-sized arrays
   guard against.
3. **Scratch array sizing must accommodate `nkblock==0` semantics.** `real, dimension(...,max(1,nkblock))`
   (`MOM_continuity_PPM.F90:2706`) — the `max(1,...)` guards against a zero-sized array declaration
   before `nkblock` has been resolved from `0` to `nz` inside the same subroutine (the resolution
   happens a few lines earlier at `:512-513`, so in practice this is defensive, but it shows the
   discipline expected: never let a `0` sentinel leak into an array bound unexamined).
4. **`-Minline` / `!NVF$ INLINE` on the per-point helpers.** `flux_elem`/`flux_elem_OBC`/`ratio_max`
   are called from inside the blocked kernel and must be inlined onto the device. Commit
   `93dbbd36e`'s "remove nvf inline and replace with intel forceinline" step is documented **only as a
   performance change** — its verbatim message is *"Significantly improves performance of blocked
   zonal/meridional_mass_flux at -O2"*, with no claim about correctness. (The earlier draft of this
   doc attributed "Otherwise results are incorrect" to this commit; that phrase is **not** in the
   commit message and has been removed.) The *correctness*-critical form of the inlining requirement
   lives elsewhere: `00-architecture.md` §7.5 catalogues `-Minline=name:ratio_max,name:flux_elem` /
   `!NVF$ INLINE` as "mandatory or wrong answers," and branch `kblock-hor-visc` even carries a
   dedicated commit `4e3f1b758` "add !NVF$ INLINE to ratio_max flux_elem." So the load-bearing point
   stands — the blessed pattern silently depends on force-inlining these helpers — but it should be
   cited to the §7.5 workaround and `4e3f1b758`, not to `93dbbd36e`'s performance step.
   > **FABLE-CHECK (reviewed 2026-07-14 — resolution or current status in KNOWLEDGE.md §8a/§8b):** Is there a primary source (commit message, code comment, or issue) that
   > *directly* states missing inline of `flux_elem`/`ratio_max` produced **wrong numerical answers**
   > (as opposed to a slowdown)? `00-architecture.md` §7.5 asserts "mandatory or wrong answers," but I
   > could not locate the originating evidence in `git log`/source — check the PR discussion for #165
   > and the history of `4e3f1b758`/the `-Minline` flag in the build config.

---

## 4. Preserving CPU performance

The mechanism is deliberately structural, not incidental: **when a block size resolves to a value
that reconstructs the pre-port loop nest exactly, CPU performance is preserved because the compiler
sees (almost) the same code.**

- **`nkblock=1` on CPU (CoriolisAdv, hor_visc)** literally reconstructs `do k_start=1,nz,1 ; k_end =
  k_start` — a trivial one-layer-at-a-time outer loop identical in trip count and body to the
  pre-blocking `do k=1,nz`, with 2-D-sized scratch arrays (`nkblock=1` makes the 3rd dimension size 1,
  which the compiler can treat as effectively 2-D). No cache-blocking gain is *lost* because the
  original code was already "blocked" at the finest possible granularity (one layer) by construction.
- **`niblock=32, njblock=4, nkblock=1` on CPU (continuity)** — the code comment is explicit: *"These
  were found to give best performance in limited tests"* (`MOM_continuity_PPM.F90:3125`). This is a
  genuine, non-default cache-blocking regime (32×4-point horizontal tiles processed one at a time)
  chosen empirically, not a degenerate case of "whole domain" — i.e. for continuity, CPU performance
  is preserved by *actually* cache-blocking, not merely by not-blocking.
- **Commit `26b8b4da7`'s own log entry** ("seems to be roughly same gpu perf, but much better CPU
  perf") is the origin evidence that this i/j-tiling axis exists *specifically* for CPU cache
  behavior — GPU performance was roughly a wash, but CPU improved markedly, which is exactly the
  trade the "preserve both" strategy is built to capture.

**The `28eb296f4` "rearrange arrays to improve cpu perf" commit** (on `kblock-hor-visc`, 72 insertions
/ 80 deletions in `MOM_hor_visc.F90`) is **overwhelmingly, but not purely,** a declaration-order
change. I re-derived its full diff (verification); it does three things:
1. **Declaration reordering (the bulk).** Large `real, dimension(...,nkblock)` blocks
   (`Del2u/h_u/...`, `dvdx/dudy/...`, `div_xx/sh_xx/str_xx/...`, `Ah/Kh/Shear_mag/...`) are lifted out
   of their old positions and re-emitted, verbatim, lower in the local-variable block (adjacent to the
   `Ah_q/Ah_h` full-`SZK` arrays). No statement that computes a value is touched by this part.
2. **One dimension-expression normalization.** `str_xy_BS` changes from
   `dimension(SZIB_(G),SZJB_(G),merge(GV%ke,CS%nkblock,CS%nkblock==0))` to
   `dimension(SZIB_(G),SZJB_(G),nkblock)`. This is **semantically identical** *because*
   `d6fe494e6` already made `nkblock` a dummy argument the caller sets to exactly that `merge(...)`
   value — so the two expressions have the same runtime extent. It is a cleanup, not a shape change.
3. **A genuine loop fusion (the one real structural change).** Three consecutive
   `do concurrent (k=kstart:kend, j=Jsq-1:Jeq+2, i=Isq-1:Ieq+2)` loops — computing `dudx(i,j,kk)`,
   then `dvdy(i,j,kk)`, then `sh_xx(i,j,kk) = dudx(i,j,kk) - dvdy(i,j,kk)` in three separate passes —
   are **merged into a single `do concurrent`** with all three assignments in one body (the
   `@@ -724,16 +724,8 @@` hunk, −8 net lines). So the earlier characterization of "zero algorithmic
   diff / no change to any statement" was **incorrect**: the loop *nesting* changed.

**This fusion is nonetheless bitwise-safe**, and it is worth stating why explicitly (this is exactly
the "fused vs. split loops" hazard flagged during verification): the three fused loops share the
*identical* iteration space `(k=kstart:kend, j=Jsq-1:Jeq+2, i=Isq-1:Ieq+2)`; each writes a *distinct*
array (`dudx`, `dvdy`, `sh_xx`); and the only intra-set dependence — `sh_xx(i,j,kk)` reading
`dudx(i,j,kk)` and `dvdy(i,j,kk)` — is at the **same** `(i,j,kk)` computed earlier in the same fused
iteration, never at a neighbor or a different layer. Fusing therefore neither reorders any
floating-point operation within an expression nor introduces a cross-iteration read-after-write, so
every point's result is bit-identical. (A fusion that instead pulled a *reduction* or a
*neighbor-stencil* read across the merged boundary would **not** be safe — that is the case to watch
for when imitating this commit.)

The declaration-order half of the commit is real and load-bearing on its own. The pre-existing source
warns about it (comment quoted verbatim below, confirmed present in the tree):

```fortran
! NOTE: The position of these declarations can impact performance, due to the
!   very large number of stack arrays in this function.  Move with caution!
```

This confirms that for a routine with dozens of large automatic (stack) arrays now carrying an extra
`nkblock` dimension, **the compiler's stack layout/alignment decisions are sensitive to declaration
order**, and this sensitivity is large enough to be worth a dedicated commit — i.e. k-blocking's
memory-footprint increase (2-D scratch → 3-D scratch-of-size-`nkblock`) can itself regress CPU
performance through stack-layout effects unrelated to the algorithm, and the fix is non-obvious
(reordering declarations, not touching logic). This is the single most surprising CPU-perf lesson in
the three case studies: **not every regression from k-blocking is about cache blocking per se — some
are about how many/how-large stack arrays a single Fortran procedure declares and in what order.**

---

## 5. Contrasting the three cases

| | Continuity (`MOM_continuity_PPM.F90`) | CoriolisAdv (`MOM_CoriolisAdv.F90`) | hor_visc (`MOM_hor_visc.F90`) |
|---|---|---|---|
| Status | Merged, most mature | Merged (`b8c471cfa`) | In-flight (`kblock-hor-visc`) |
| Blocking axes | i, j, **and** k (three independent CS members) | k only | k only |
| Pre-blocking loop shape | `do concurrent(k,j,i)` over the whole 2-D reconstruction plane per call, or a hand-tiled `!$omp target teams` region | serial `do k=1,nz` outer loop, 2-D scratch reused each iteration | serial `do k=1,nz` outer loop, 2-D scratch reused each iteration (identical shape to CoriolisAdv pre-port) |
| Block-local index style | `ii,jj` for i/j tiles; `kk` for k-blocks, computed as `kk = k - ks + 1` inside a `do concurrent(k=ks:ke,...)` (k is the control variable) | Iterates on `kk` (`1:kmax`) as the control variable, computing `k = k_start + kk - 1` via `DO_LOCALITY(local(k))` — the *opposite* convention from continuity | Iterates on the *global* `k` as control variable (same convention as continuity), like the pre-merge `kblock-coradcalc` branch |
| Device directive at the outer/whole-tile level | Explicit `!$omp target teams num_teams(nteams)` with hand-computed team count (§6) | Plain `do concurrent`, no explicit team/thread directives seen at this level | Plain `do concurrent`, matching CoriolisAdv |
| Incremental-porting markers | None needed — reconstruction is fully ported | `! TODO: port (OBC GPU path not yet implemented)` serial fallback loops nested inside the k-block loop | `! TODO: port` serial fallback loops (`use_Leithy`, `id_normstress`) nested inside the k-block loop, same idiom |
| CPU-perf-specific follow-up commit | None beyond the empirically tuned `32/4/1` defaults | None found | `28eb296f4` "rearrange arrays to improve cpu perf" — mostly declaration reordering + one bitwise-safe loop fusion (§4) |
| Param names | `CONTINUITY_NIBLOCK`/`NJBLOCK`/`NKBLOCK` | `CORIOLIS_ADV_NKBLOCK` | `HORVISC_NKBLOCK` |

**What is common (the blessed template, restated):** one (or three) CS-level integer block-size
parameter(s), defaulting to a small CPU-tuned constant (or `1` for pure k-blocking) on CPU builds and
`0` ("whole extent") on `__NVCOMPILER_OPENMP_GPU` builds; an outer host loop striding over blocks;
block-sized (not domain-sized) scratch arrays; a block-local index computed as an offset from the
block start; the active sub-range threaded explicitly into any helper subroutine; `do concurrent`
(with `DO_LOCALITY` macros for locality clauses) as the default parallel construct for the block
body.

**What differs:** (a) continuity alone combines all three axes and is the only one with an explicit
manual-team-count `omp target teams` region — the other two modules haven't (yet, as of this
writing) needed to hand-tune team counts; (b) the two k-block-only modules disagree with each other
on which of `k`/`kk` is the literal `do concurrent` control variable, showing this is a
non-load-bearing style choice, not part of the "blessed" contract; (c) only hor_visc has hit a
declaration-order CPU-perf regression, plausibly because it is the largest/most stack-array-heavy of
the three routines.

---

## 6. `!$omp target teams num_teams(nteams)` vs. plain `do concurrent`

The default parallel idiom across the whole port is `do concurrent` (`00-architecture.md` guiding
principle 3). The blocked kernels in continuity's `zonal_mass_flux`/`meridional_mass_flux` are the
one place in these three case studies where an explicit `!$omp target teams num_teams(nteams)` region
wraps a serial-looking `do k=1,nz` with `!$omp loop collapse(2)` inside — and the reason is recorded
verbatim in the commit that introduced it, `5b5f6b2b1` ("add teams spec to problematic target
region"):

> For some reason omp runtime was only starting a kernel with 17 blocks when the openacc version
> would start it with 238 or something like that. Manually calculating number of teams sped it up.

I.e. this is a **documented nvfortran OpenMP-runtime under-launch bug**, not a general preference for
`target teams` over `do concurrent`. The workaround:

```fortran
! calculate number of teams
!$ nteams = ceiling(real((j_end-j_start+1)*(i_end-i_start+1))/128.)
...
!$omp target teams num_teams(nteams)
do k=1,nz
  !$omp loop collapse(2) private(ii,jj)
  do j=j_start,j_end ; do i=i_start,i_end
    ...
```
`128` is chosen as a thread-block-size divisor (an early revision of this same commit briefly also
carried `thread_limit(128)`, later dropped — current source at `:707` uses `num_teams(nteams)` alone
and relies on the runtime's default thread count per team). The `!$` sentinel is a conditional
OpenMP-compilation comment, so `nteams` is only computed/used when actually building with OpenMP.

**Decision rule observed in the source, in priority order:**
1. **Default to `do concurrent`** for any loop whose iteration space can be expressed that way and
   where the compiler's own scheduling has not been shown to misbehave (this is the overwhelming
   majority of loops in all three modules, including most of the blocked bodies themselves, e.g. the
   `do concurrent(jj=1:...,ii=1:...)` initializations flanking the teams region above).
2. **Escalate to `!$omp target [teams] loop`** when a region needs a reduction `do concurrent` can't
   (yet, portably) express, or when `do concurrent` alone under-parallelizes/misschedules on
   nvfortran (the general pattern named in `00-architecture.md` guiding principle 3 and its §7.5
   compiler-workaround catalogue, e.g. `e8b0ecfbf` "omp target teams loop -> do concurrent" shows the
   traffic runs both directions depending on measured behavior).
3. **Escalate further to an explicit, hand-computed `num_teams(nteams)`** only when even
   `target teams loop`'s automatic team count is empirically wrong (measured: 17 launched vs. ~238
   expected) — i.e. this is a last-resort, bug-specific fix applied to exactly the one region in
   continuity where it was needed, not a house style to imitate by default. Neither CoriolisAdv nor
   hor_visc's k-blocking (as of the branches studied here) needed this escalation, consistent with
   §5's observation that the manual team count is a continuity-specific quirk, not part of the
   general k-blocking template.

---

## 7. The blessed k-blocking recipe — a followable template

This is the prescriptive form of §1–§6: a numbered procedure for k-blocking one serial-`k` loop nest,
with the exact code shapes to copy. Every step cites a merged (or near-merged) example. Substitute
your module's name for `MYMOD` and choose a `MYMOD_NKBLOCK` param name. (For i/j tiling, the same
skeleton applies with `niblock`/`njblock`; only continuity currently needs it — see §1.2.)

### 7.0 Pre-flight checklist — does this loop nest qualify?

A loop nest is a k-blocking candidate **iff all** of these hold:

- [ ] It is (or can be) an **outer `do k=1,nz`** whose body is a horizontal (2-D, `i`/`j`) calculation
      — the classic "process one layer, reuse small 2-D scratch" idiom (`MOM_CoriolisAdv.F90` and
      `MOM_hor_visc.F90` pre-port). A nest that is already a single `do concurrent (k,j,i)` over the
      whole cube (continuity reconstruction) also qualifies — you are bounding its scratch, not
      serializing it.
- [ ] The per-point computation is **pointwise in `k`**: point `(i,j,k)`'s output depends only on
      inputs at layer `k` (any horizontal stencil is fine). **No cross-layer coupling** — no
      `k`↔`k±1` dependence, no running vertical integral, no tridiagonal-in-`k` solve. (Those belong to
      the `vert_friction` "teams-loop with serial inner k" pattern, `00-architecture.md` §4.3, **not**
      here.)
- [ ] Any **vertical reduction** in the body is either (a) associative-by-construction (`max`, `.or.`,
      integer) or (b) a float sum you can keep as a **sequential `do k=1,nz`** that blocking will
      *not* split (see §3(i)). If a float sum would have to be partitioned across blocks, **stop** —
      that reorders arithmetic and breaks bitwise reproducibility.

**Disqualifiers** (do *not* k-block; port differently or leave serial): implicit vertical solves
(tridiagonal), cumulative `k` integrals, remapping/regridding across layers, or anything where a
block boundary would fall *inside* a summation or a `k`-stencil.

### 7.1 Step-by-step

**Step 1 — Add the CS member.** In `MYMOD_CS`:
```fortran
integer :: nkblock  !< The k block size used in <...> calculations [nondim].
```
(`MOM_continuity_PPM.F90:78`, `MOM_CoriolisAdv.F90:59`, `MOM_hor_visc.F90:125`.)

**Step 2 — Compile-time CPU/GPU default divergence**, in `MYMOD_init`, before the `get_param`:
```fortran
#ifdef __NVCOMPILER_OPENMP_GPU
  integer, parameter :: default_nkblock = 0  !< whole column / no cache blocking
#else
  integer, parameter :: default_nkblock = 1  !< one layer at a time = pre-port CPU behavior
#endif
```
(`MOM_continuity_PPM.F90:3120-3129`, `MOM_CoriolisAdv.F90:2094-2098`, `MOM_hor_visc.F90:2759-2763`.)
Continuity's i/j axis uses `32`/`4` instead of `1`; the k axis is always `0` (GPU) / `1` (CPU).

**Step 3 — `get_param`, with the `0`-is-fatal-if-negative guard:**
```fortran
call get_param(param_file, mdl, "MYMOD_NKBLOCK", CS%nkblock, &
       "The k-direction block size ... the default 0 setting is dynamic and fits the "//&
       "full vertical column.", default=default_nkblock, layoutParam=.true.)
if (CS%nkblock < 0) call MOM_error(FATAL, "MYMOD_NKBLOCK must be >= 0.")
```
(`MOM_continuity_PPM.F90:3210-3220`, `MOM_CoriolisAdv.F90:2108-2112`, `MOM_hor_visc.F90:2781-2785`.)
`0` is reserved to mean "dynamic / whole extent"; negative is a hard error.

**Step 4 — Resolve `0` → whole extent at the point of use** (never store the resolved value back into
`CS`). Two equivalent forms:
```fortran
nkblock = CS%nkblock ; if (nkblock == 0) nkblock = nz          ! scalar form (continuity :512-513)
nkblock = merge(GV%ke, CS%nkblock, CS%nkblock==0)              ! expression form (CoriolisAdv :275)
```
Use the `merge(...)` form when the value must also appear in an **array-dimension expression** (Step 5).
If the loop nest lives in a *different* module from the CS, expose an accessor rather than reaching
into private members — `hor_visc_nkblock(CS)` returns `merge(GV%ke,CS%nkblock,CS%nkblock==0)` for
external callers (`d6fe494e6`, `MOM_hor_visc.F90:274,298`).

**Step 5 — Shrink full-column scratch to one block.** Every `SZK_(GV)`/`nz`-deep automatic work array
gets its 3rd dimension replaced by the block size, guarded against a zero-size declaration:
```fortran
real, dimension(SZIB_(G),SZJB_(G),merge(GV%ke,CS%nkblock,CS%nkblock==0)) :: dvdx, dudy, ...  ! CoriolisAdv :169
! or, when nkblock is already a resolved local/argument:
real, dimension(SZI_(G),SZJ_(G),max(1,nkblock)) :: slp                                        ! continuity :2706
```
The `merge(...)`/`max(1,...)` guard exists so a `0` sentinel can never reach an array bound (§3(3)).
**Stack-layout caution:** in a procedure with many such arrays, *where* you declare them can itself
move CPU performance — keep related blocks together and expect to tune order empirically (§4,
`28eb296f4`; the source comment "Move with caution!").

**Step 6 — Wrap the `k` axis in an outer host loop over block starts:**
```fortran
do k_start=1,nz,nkblock
  k_end = min(k_start+nkblock-1, nz)
  kmax  = k_end - k_start + 1
  ...
enddo
```
(`MOM_CoriolisAdv.F90:373-374`, `MOM_continuity_PPM.F90` reconstruction uses `ks/ke`, `MOM_hor_visc.F90`
uses `kstart/kend`.) i/j tiling nests two such loops (`do j_start=...; do i_start=...`,
`MOM_continuity_PPM.F90:696`).

**Step 7 — The device kernel body + block-local index.** Convert the body to `do concurrent` with a
block-local index; **either** convention is accepted (they are bitwise-equivalent, §1.4/§5):
```fortran
! (a) iterate the block-local index (merged CoriolisAdv b8c471cfa :382):
do concurrent (kk=1:kmax, J=Js_q:Je_q, I=Is_q:Ie_q) DO_LOCALITY(local(k))
  k = k_start + kk - 1
  dvdx(I,J,kk) = (v(i+1,J,k)*G%dyCv(i+1,J)) - (v(i,J,k)*G%dyCv(i,J))
enddo
! (b) iterate the global k (continuity :2753, hor_visc, kblock-coradcalc):
do concurrent (k=ks:ke, j=jsl:jel, i=isl:iel) DO_LOCALITY(local(h_im1,h_ip1,kk))
  kk = k - ks + 1
  h_W(i,j,kk-or-k) = ...
enddo
```
**Index-mapping rule (do not get this wrong):** *scratch/work* arrays use the **block-local** third
index (`kk`); *full-size in/out and grid-metric* arrays (`u`, `v`, `h_in`, `h_W`, `G%dyCv`,
`G%mask2dT`) keep the **global** `k`. Whichever variable is *not* the `do concurrent` control variable
must be declared with `DO_LOCALITY(local(...))` so each device iteration owns a private copy (a shared
scalar here is a data race on GPU — §3(1)). The `DO_LOCALITY` macro
(`src/framework/do_concurrent_compat.h`) degrades to nothing where `HAVE_FC_DO_CONCURRENT_LOCAL` is
unset.

**Step 8 — Thread the active sub-range into every helper** the body calls. Change signatures that took
the full range to take `(k_start,k_end)` (or `ks,ke` / `i_start,i_end,j_start,j_end`):
```fortran
call PPM_limit_pos(h_in, h_W, h_E, h_min, G, GV, isl, iel, jsl, jel, ks, ke)   ! was (...,nz)
call gradKE(u, v, h, KE, KEx, KEy, k_start, k_end, nkblock, G, GV, US, CS)     ! CoriolisAdv :781
```
A helper that kept iterating `1:nz` against block-sized scratch would read out of bounds — this is a
shape-safety bug, not merely a reordering (§3(2)).

**Step 9 — Incremental porting is allowed.** If a sub-block of the body isn't ready for `do
concurrent` (OBC paths, WENO, `use_Leithy`, diagnostic capture), leave it as a **serial** loop *inside*
the k-block loop with a hand-computed `kk`, tagged for follow-up:
```fortran
do k=k_start,k_end   ! TODO: port (OBC GPU path not yet implemented)
  kk = k - k_start + 1
  ...
enddo
```
The k-block *nest* is installed unconditionally; porting each body to a parallel construct is a
separable later step (§1.4, §1.5).

**Step 10 — Device data mapping** for any new CS arrays follows the standard pattern (unchanged by
blocking): `ALLOC_(CS%x(...)); CS%x=0.0; !$omp target enter data map(to: CS%x)` in `_init`, mirrored
`!$omp target exit data map(delete: CS%x)` beside `DEALLOC_` in `_end` (`00-architecture.md` §9).
Block-sized *automatic* scratch (Step 5) needs no explicit map when used only inside `do concurrent`;
an explicit hand-launched `!$omp target teams` region maps its scratch with `enter/exit data
map(alloc:/release:)` around the tile loop (continuity `slp`, `MOM_continuity_PPM.F90:2708` /
`:2835`).

**Step 11 (escalation, rarely needed) — hand-tuned team count.** Default to `do concurrent`. Only if
profiling shows nvfortran under-launching teams for a hand-written `!$omp target teams` region do you
add a manual `num_teams` (continuity's `zonal_mass_flux` is the *sole* case in these three modules,
§6): `!$ nteams = ceiling(real((j_end-j_start+1)*(i_end-i_start+1))/128.)` then
`!$omp target teams num_teams(nteams)`. This is a bug-specific last resort (`5b5f6b2b1`), not a
default to imitate.

### 7.2 Verification step (mandatory before claiming a port)

k-blocking is a bitwise-preserving refactor, so it is verifiable *exactly* — there is no "close
enough":

1. Build the **pre-change** binary and the **post-change** binary at the same optimization level.
2. Run both (CPU is sufficient for the arithmetic check; `nkblock=1` on CPU reconstructs the pre-port
   loop, §4) and compare `MOM_checksums` field checksums (`hchksum`/`uchksum`/`vchksum`,
   `popcnt`-based, `MOM_checksums.F90`, `00-architecture.md` §7.2) at matching timesteps. **Every
   checksum must be bit-identical**; a single differing bit means the transform reordered arithmetic
   (most likely a split float sum, a mishandled reduction, or a helper still iterating the full range).
3. Compare the EFP reproducing-sum energy output (`write_energy`) — also required to be bit-identical.
4. Confirm the `nkblock=0` (GPU-default) and `nkblock=nz` (explicit) resolutions agree with each other
   and with `nkblock=1`, since all three must produce identical results by construction.
5. Sanity-check the `!NVF$ INLINE`/`-Minline` requirement for any per-point helper called from the
   kernel — dropping it can change answers on nvfortran (§3(4), `00-architecture.md` §7.5).

If any checksum diverges, the port is wrong — do not "accept" a nonzero diff as rounding.

---

## 8. Where to look next

- `src/core/MOM_continuity_PPM.F90:76-78,164-421,502-736,2684-2989,3110-3223` — CS block-size
  members, `niblock`/`njblock` resolution at all four `continuity_PPM`/`continuity_3d_fluxes`/
  `continuity_adjust_vel` call sites, the hybrid tiled kernel, `PPM_reconstruction_x/y`, and
  `continuity_PPM_init` defaults.
- `src/core/MOM_CoriolisAdv.F90:59,169-275,373-` — `nkblock` CS member, `merge(GV%ke,CS%nkblock,...)`
  resolution pattern, the k-block main loop.
- `src/parameterizations/lateral/MOM_hor_visc.F90` on branch `kblock-hor-visc` (commits `36f51ff84`,
  `d6fe494e6`, `28eb296f4`) — same template applied to the largest of the three routines, plus the
  declaration-order CPU-perf lesson.
- `src/framework/do_concurrent_compat.h` — the `DO_LOCALITY(X)` macro used throughout to make
  `local()`/`reduce()` locality clauses conditional on `HAVE_FC_DO_CONCURRENT_LOCAL`.
- Git: `git log dev-gfdl..dev/gpu --oneline -- src/core/MOM_continuity_PPM.F90` for the full
  continuity evolution; `git show b8c471cfa`; `git diff dev/gpu...kblock-coradcalc`; `git log
  dev/gpu..kblock-hor-visc --oneline` and `git diff dev/gpu...kblock-hor-visc -- src/parameterizations/lateral/MOM_hor_visc.F90`.

---

## Verification notes

Opus verification pass against source + git (no build/run). Baseline `dev-gfdl`, branch `dev/gpu`.

### Confirmed (checked directly in code/git)

- **Continuity CS members** `niblock`/`njblock`/`nkblock` at `MOM_continuity_PPM.F90:76-78`. ✓
- **`#ifdef __NVCOMPILER_OPENMP_GPU` default divergence** `0/0/0` (GPU) vs `32/4/1` (CPU) at
  `:3120-3129`; `get_param` for `CONTINUITY_NIBLOCK/NJBLOCK/NKBLOCK` with negative-is-FATAL at
  `:3202-3223`. ✓
- **`if (niblock==0) niblock = LB%ieh-LB%ish+2` whole-domain fill** at `:186-189`, and the claim that
  the resolved size is re-derived per advection phase (`+2` vs `+1`) — confirmed, four phases. ✓
- **Hybrid `num_teams` kernel** at `:696-738` (doc previously said `:696-736`; corrected to `:738`,
  the `!$omp end target teams`), with `!$ nteams = ceiling(real((...)*(...))/128.)`. ✓
- **PPM_reconstruction blocking** (`93dbbd36e`): `nkblock` added as subroutine argument
  (`:2684`), `slp` shrunk to `dimension(...,max(1,nkblock))` (`:2706`), outer `do ks=1,nz,nkblock`,
  `kk=k-ks+1`, and `PPM_limit_pos/CW84` signature changed `(...,nz)` → `(...,ks,ke)` — all verified
  against the pre-`93dbbd36e` source (which had `slp(SZI,SZJ,SZK)` and `PPM_limit_pos(...,nz)`). ✓
- **k-axis `0`→`nz` resolution** at `:512-513`, `:557-558`; the "one k-BLOCK of scratch" bitwise
  argument (pointwise in `k`, no cross-layer sum in reconstruction) — re-derived and holds. ✓
- **CoriolisAdv `b8c471cfa`** iterates `kk=1:kmax`, `k=k_start+kk-1`, `DO_LOCALITY(local(k))`
  (`:373-410`); `merge(GV%ke,CS%nkblock,CS%nkblock==0)` in both array dims (`:169,181,195,203`) and
  scalar (`:275`); `CORIOLIS_ADV_NKBLOCK` at `:2108-2112`. ✓
- **kk-vs-k divergence** vs branch `kblock-coradcalc` (`0b5c00333`): confirmed the branch iterates
  `do concurrent (k=kstart:kend,...) ; kk=k-kstart+1` (opposite convention), and a commit inside
  `b8c471cfa` flipped it to iterate `kk`. Both bitwise-equivalent. ✓
- **hor_visc branch** commit order `36f51ff84 → d6fe494e6 → 28eb296f4` (via `merge-base
  --is-ancestor`); `36f51ff84` blocks k iterating the *global* `k`; `d6fe494e6` adds the
  `hor_visc_nkblock` accessor and makes `nkblock` a dummy arg; `HORVISC_NKBLOCK` at `:2781-2785`. ✓
- **"Move with caution!" source comment** — exact text confirmed present in the branch tree. ✓
- **`5b5f6b2b1` num_teams under-launch** rationale — commit message quote confirmed. ✓

### Corrected

1. **`28eb296f4` is NOT "zero algorithmic diff" (critical).** Re-deriving its full 72/80-line diff
   shows, besides the declaration reordering: (a) `str_xy_BS`'s dimension normalized
   `merge(GV%ke,CS%nkblock,...)` → `nkblock` (equivalent post-`d6fe494e6`), and (b) a **genuine loop
   fusion** — three separate `do concurrent` loops (`dudx`, `dvdy`, `sh_xx`) merged into one
   (`@@ -724,16 +724,8 @@`). Rewrote §4 to state this and to prove the fusion is nonetheless
   bitwise-safe (identical iteration space, distinct output arrays, only a same-index intra-iteration
   dependence). This is exactly the "fused vs split loops" hazard the task asked to hunt for; the
   fusion is safe *here*, but the doc now spells out what would make it unsafe.
2. **§3(4) misquoted `93dbbd36e`.** The phrase "Otherwise results are incorrect" is **not** in the
   commit message (which says only "Significantly improves performance ... at -O2"). Reattributed the
   correctness-critical inlining claim to `00-architecture.md` §7.5 and branch commit `4e3f1b758`,
   and flagged the missing primary source with a FABLE-CHECK.
3. **§1.2 helper attributes.** `ratio_max` is a `pure function` (`:3086`) with **no** FORCEINLINE
   directive; only `flux_elem`/`flux_elem_OBC` are `elemental subroutine`s carrying
   `!DIR$ ATTRIBUTES FORCEINLINE` (`:1086`, `:1149`). Corrected the "both elemental + FORCEINLINE"
   generalization.
4. **§3(i) mislabel.** The `:775-782` accumulation is not `visc_rem_max` — it is a genuine
   floating-point vertical **sum** `uh_tot_0/duhdu_tot_0 += ...(k)`. Corrected and used it as the
   stronger bitwise example (a real FP sum kept as a sequential `do k=1,nz` that tiling never splits).

### Enhancements

- Added §7 "The blessed k-blocking recipe" — an 11-step followable template (CS member → ifdef default
  → get_param → 0-resolution → scratch reshaping → outer loop → device kernel + index-mapping rule →
  helper sub-range → incremental-porting fallback → data mapping → team-count escalation), each step
  cited to a merged example; plus a §7.0 pre-flight qualify/disqualify checklist and a §7.2 checksum
  verification procedure.
- Added a branch-staleness caveat (kblock-hor-visc tip `9b69fc581` is ~15 commits past `28eb296f4`).

### FABLE-CHECK markers: 1

- §3(4): whether any primary source *directly* documents wrong numerical answers (vs. a slowdown) from
  missing inline of `flux_elem`/`ratio_max`.

### Confidence

**High** on all continuity and CoriolisAdv claims (verified verbatim against current tree and
pre-commit source). **High** on the `28eb296f4` correction (full diff inspected; loop fusion and
dimension-normalization are unambiguous in the hunk). **High** on bitwise-preservation for all three
cases, including the fused loop. The single residual uncertainty is the provenance of the
"inline-or-wrong-answers" claim (FABLE-CHECK), which affects wording, not the structural conclusion
that force-inlining is required.
