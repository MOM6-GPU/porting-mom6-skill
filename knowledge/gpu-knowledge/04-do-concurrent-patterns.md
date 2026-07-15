# `do concurrent` Usage Patterns on `dev/gpu`

> Companion to `00-architecture.md` §0(3), §5, §7.2. This document is about the *default parallel
> idiom* — Fortran `do concurrent` (DC) plus its F2018 locality specifiers — and the specific,
> catalogued cases where `!$omp target teams` is used instead. Repo: `dev/gpu`, baseline `dev-gfdl`.
> Source + git only; no build/run performed to produce this document.

---

## 1. `DO_LOCALITY` and configure-time feature detection

Not every Fortran compiler that MOM6 must build with supports F2018 locality specifiers on
`do concurrent` (`local`, `local_init`, `shared`, `default(none)`, `reduce`). To keep one source form
building everywhere, the port introduces a macro and an autoconf probe:

- **Macro** — `src/framework/do_concurrent_compat.h`:
  ```fortran
  #ifndef DO_CONCURRENT_COMPAT_H_
  #define DO_CONCURRENT_COMPAT_H_
  #ifdef HAVE_FC_DO_CONCURRENT_LOCAL
  #define DO_LOCALITY(X) X
  #else
  #define DO_LOCALITY(X) ;
  #endif
  #endif
  ```
  When the compiler supports locality clauses, `DO_LOCALITY(local(k))` expands to `local(k)` (attached
  to the `do concurrent` header). When it doesn't, it expands to `;` — a no-op statement separator —
  so the loop compiles as a bare, unlocalized `do concurrent` (correct as long as the compiler treats
  loop-body scalars conservatively/serially-consistent; on such compilers the code degrades toward
  CPU-safe but does not GPU-parallelize the flagged locals — this is a portability fallback, not a
  performance guarantee).

  > **FABLE-CHECK (reviewed 2026-07-14 — resolution or current status in KNOWLEDGE.md §8a/§8b):** The "degrades toward CPU-safe but does not GPU-parallelize the flagged locals"
  > semantics is an inference about compiler behaviour when a bare `do concurrent` carries implicit
  > (unspecified) locality — it is *not* verifiable from MOM6 source or git. The only load-bearing,
  > verified fact is that the fallback still *compiles* correctly (`;` is a valid empty statement after
  > the header). Confirm the actual codegen consequence against the F2018 standard / nvfortran docs
  > before relying on it.

- **Feature probe** — `ac/m4/mom6_fc_do_concurrent_local.m4` (`MOM6_FC_DO_CONCURRENT_LOCAL`):
  compiles a trivial `do concurrent(i=1:2) local(a,b)` program; if it compiles,
  `mom6_cv_fc_do_concurrent_local=yes` and `AC_DEFINE([HAVE_FC_DO_CONCURRENT_LOCAL], [1], ...)`.
  Invoked from `ac/configure.ac:173` (`MOM6_FC_DO_CONCURRENT_LOCAL`, under the `# Do concurrent
  configuration` comment at `:172`), right after the real-8 flag setup (`:168`) and right before the
  OpenMP configuration block (`:176`).

- **Known gap** (m4 comment, verbatim): *"Currently only LOCAL is tested, but this should also
  include LOCAL_INIT, SHARED, and DEFAULT(NONE)."* In practice the source already uses
  `local_init` and `reduce` unconditionally gated by the same single `HAVE_FC_DO_CONCURRENT_LOCAL`
  macro — i.e., the probe is a coarse yes/no gate for "any locality clause," not a per-clause matrix.
  A compiler that supports `local` but not `reduce` (or vice versa) is not distinguished; this is a
  latent risk flagged for future hardening, not yet hit in practice on the nvfortran target.

**How the same source compiles everywhere:** every locality clause in the whole tree is written as
`DO_LOCALITY(...)`, never as a bare Fortran clause. A compiler lacking support only needs
`HAVE_FC_DO_CONCURRENT_LOCAL` left undefined by `configure`; no source edits, no `#ifdef` scattered
through physics code — the single macro in one header is the entire compatibility shim.

---

## 2. Inventory: `do concurrent` and locality-clause usage across `src/`

```
grep -rn "do concurrent" src/ | wc -l      → 698
grep -rn "DO_LOCALITY"    src/ | wc -l     →  96   (94 real uses + 2 in the macro header itself:
                                                    the #define lines 7 and 9 of do_concurrent_compat.h)
```

### 2.1 Per-file `do concurrent` counts (all instances, with or without locality)

| File | `do concurrent` count | `DO_LOCALITY` clause-lines |
|---|---:|:---:|
| `src/core/MOM_barotropic.F90` | 242 | 5 |
| `src/tracer/MOM_tracer_advect.F90` | 77 | 8 |
| `src/parameterizations/lateral/MOM_hor_visc.F90` | 62 | 0 |
| `src/core/MOM_continuity_PPM.F90` | 56 | 11 |
| `src/core/MOM_CoriolisAdv.F90` | 56 | 42 |
| `src/tracer/MOM_tracer_hor_diff.F90` | 32 | 9 |
| `src/core/MOM_PressureForce_FV.F90` | 30 | 0 |
| `src/parameterizations/vertical/MOM_set_viscosity.F90` | 28 | 6 |
| `src/parameterizations/vertical/MOM_vert_friction.F90` | 24 | 6 |
| `src/core/MOM_dynamics_split_RK2.F90` | 23 | 0 |
| `src/equation_of_state/MOM_EOS_Roquet_rho.F90` | 11 | 0 |
| `src/core/MOM_PressureForce_Montgomery.F90` | 10 | 0 |
| `src/equation_of_state/MOM_EOS_Wright.F90` | 9 | 0 |
| `src/diagnostics/MOM_sum_output.F90` | 9 | 4 |
| `src/core/MOM_interface_heights.F90` | 9 | 0 |
| `src/parameterizations/vertical/MOM_diabatic_aux.F90` | 7 | 0 |
| `src/core/MOM_forcing_type.F90` | 5 | 0 |
| `src/core/MOM.F90` | 5 | 0 |
| `src/framework/MOM_coms.F90` | (in `increment_block_ints`) | 3 (see §4.1) |

(`DO_LOCALITY` clause-lines counts each `DO_LOCALITY(...)` occurrence textually — a single `do
concurrent` header split across a continuation line, e.g. `MOM_coms.F90:721-724`, contributes multiple
clause-lines for one loop; `MOM_CoriolisAdv.F90`'s 42 clause-lines cover 39 of its 56 `do concurrent`
loops, the remainder being continuation lines of an already-counted header.)

**Important asymmetry:** most `do concurrent` loops (≈86%) carry **no** `DO_LOCALITY` at all — e.g.
`MOM_hor_visc.F90` (62), `MOM_PressureForce_FV.F90` (30), `MOM_dynamics_split_RK2.F90` (23) have zero.
This is not an oversight: a locality clause is only needed when the loop body declares/derives a
**scalar temporary that must be private per iteration** or performs a **reduction**. A very large
fraction of MOM6's `do concurrent` loops are pure elementwise array assignment
(`h(i,j,k) = max(hin(i,j,k) - dt*..., h_min)`, `MOM_continuity_PPM.F90:430`) with no loop-body scalar
state at all, so nothing needs declaring local. Examples with zero locals, no clause needed:
```fortran
! MOM_continuity_PPM.F90:429-431 — pure elementwise, no locality needed
do concurrent (k=1:nz, j=jsh:jeh, i=ish:ieh)
  h(i,j,k) = max( hin(i,j,k) - dt * G%IareaT(i,j) * (uh(I,j,k) - uh(I-1,j,k)), h_min )
enddo
```
```fortran
! MOM_barotropic.F90:1087 — no locals, no clause
do concurrent (k=1:nz, j=js:je, I=is-1:ie)
  ...
enddo
```

### 2.2 Categorized locality specifiers (all 94 real `DO_LOCALITY` uses)

Counts below are **exact** textual `DO_LOCALITY(...)` occurrences (re-derived with per-clause greps,
e.g. `grep -rEn "DO_LOCALITY\(reduce\(\+" src/`); they sum to `67 + 2 + 5 + 11 + 9 = 94`, matching the
`96 − 2 header` total. The **mask** row is listed for contrast but is *not* a `DO_LOCALITY` clause, so
it does not count toward the 94.

| Specifier | Count | Purpose | Representative file:line |
|---|---:|---|---|
| `local(...)` | 67 | Private per-iteration scalar/small-array temporaries (the dominant case) | `MOM_CoriolisAdv.F90:383` `local(k)`; `MOM_continuity_PPM.F90:2747` `local(h_im1,h_ip1)` |
| `local_init(...)` | 2 | Private per-iteration scalar that must **start** with its pre-loop value (conditionally overwritten inside the loop, then read unconditionally) | `MOM_set_viscosity.F90:682` `local_init(cdrag_sqrt_H, cdrag_sqrt_H_RL)`; `:759` `local_init(cdrag_sqrt_H)` |
| `reduce(+: ...)` | 5 | Integer/exact-arithmetic accumulation (reproducing sums, truncation counters) | `MOM_coms.F90:723` `reduce(+: block_sum)`; `MOM_vert_friction.F90:3189,3213,3259,3283` `reduce(+: ntrunc)` |
| `reduce(max: ...)` (10) + `reduce(min: ...)` (1) | 11 | Running max/min scalar (magnitude tracking, CFL/dt limits, "any work remaining" flags encoded as 0/1 max) | `MOM_coms.F90:724` `reduce(max: block_max_pos, block_max_neg, inan, iovf)`; `MOM_tracer_hor_diff.F90:919,963,969` `reduce(max: PEmax_kRho/itmp/k_size)`; `MOM_tracer_advect.F90:299,302,336,339,359,362` `reduce(max: domore_k_tmp)`; `MOM_barotropic.F90:3903` `reduce(min: min_max_dt2)` (the sole `reduce(min:)`) |
| `reduce(.or.: ...)` | 9 | Boolean "did anything happen" flags across a horizontal sweep | `MOM_vert_friction.F90:3168,3238` `reduce(.or.: trunc_any, do_any_write)`; `MOM_barotropic.F90:2984,3032,3053` `reduce(.or.: eta_is_submerged)`; `MOM_continuity_PPM.F90:870,1973` `reduce(.or.: any_simple_OBC)`; `MOM_tracer_advect.F90:582,1003` `reduce(.or.: domore_u_jk / domore_v_jk)` |
| Header **mask** expression (*not* a `DO_LOCALITY` clause — an F2008 scalar-logical restriction on the DC index set) | (not counted) | Skip iterations without branching in the body; also used to gate a subsequent write-back loop | `MOM_vert_friction.F90:3183` `do concurrent (j=js:je, I=Isq:Ieq, dowrite(I,j))`; `MOM_set_viscosity.F90:672,681` `do concurrent (i=is:ie, do_i(i,j))`; `MOM_tracer_hor_diff.F90:929,937` `..., G%mask2dT(i,j) > 0.0)` |

**Confirmed: `shared(...)` and `default(none)` are never used on any `do concurrent` construct
anywhere in `src/`** (`grep -rn "do concurrent" src/ | grep -i "shared(\|default(none)"` → no matches).
Those clauses appear only on the **legacy CPU** directive `!$OMP parallel do default(shared)` /
`!$OMP parallel do default(none) shared(...)`, which persists in modules **not yet ported** to
`do concurrent` — `MOM_isopycnal_slopes.F90`, `MOM_dynamics_split_RK2b.F90` (the unsplit variant),
`MOM_kappa_shear.F90`, `MOM_set_diffusivity.F90`, `MOM_CVMix_KPP.F90`, `MOM_energetic_PBL.F90`,
`MOM_diabatic_driver.F90`, etc. — i.e. exactly the "untouched on `dev/gpu`" list in
`00-architecture.md` §4.3/§6.3. A few `!$OMP parallel do` survive even inside heavily-ported files
(e.g. `MOM_barotropic.F90:899-1060`, `MOM_set_viscosity.F90:2244-2548`) but only in cold/init-time or
alternate-configuration branches (e.g. the non-`linearized_BT_PV` else-branch at
`MOM_barotropic.F90:896-899`) that are not on the GPU-resident hot path — the two idioms coexist in
the same file without ever appearing on the same construct.

---

## 3. Why `!$omp target teams` replaces `do concurrent` — two distinct categories

`00-architecture.md` §0(3): *"`do concurrent` is the default parallel idiom. OpenMP `target teams`
directives are used only for reductions or where `do concurrent` misbehaves/underperforms."* The
`reduce()`-clause case is covered in §4 below (DC handles it directly once `HAVE_FC_DO_CONCURRENT_LOCAL`
is set — no `target teams` needed there). The two categories where the port falls back to explicit
OpenMP are:

### 3.1 Category A — manual team count because the OpenMP runtime under-launched

`MOM_continuity_PPM.F90:707` (`zonal_mass_flux`) and its meridional twin. The k-blocked kernel
processes one `niblock × njblock` tile per host-loop iteration; inside the tile, `!$omp target teams
num_teams(nteams)` wraps a serial `do k=1,nz` with `!$omp loop collapse(2) private(ii,jj)` per level,
calling `!$omp declare target` helpers `flux_elem` / `ratio_max`:

```fortran
! MOM_continuity_PPM.F90:696-736
do j_start=jsh,jeh,njblock ; do i_start=ish-1,ieh,niblock
  i_end = min(i_start+niblock-1,ieh)
  j_end = min(j_start+njblock-1,jeh)

  ! calculate number of teams
  !$ nteams = ceiling(real((j_end-j_start+1)*(i_end-i_start+1))/128.)
  ...
  !$omp target teams num_teams(nteams)
  do k=1,nz
    ...
    !$omp loop collapse(2) private(ii,jj)
    do j=j_start,j_end ; do i=i_start,i_end
      ii=i-i_start+1 ; jj=j-j_start+1
      call flux_elem(u(i,j,k),h_in(i,j,k),h_in(i+1,j,k),h_W(i,j,k),h_W(i+1,j,k),h_E(i,j,k),&
                     h_E(i+1,j,k),uh_t(ii,jj,k),duhdu(ii,jj,k),visc_rem(ii,jj,k),G%dy_Cu(i,j),&
                     G%IareaT(i,j),G%IareaT(i+1,j),G%IdxT(i,j),G%IdxT(i+1,j),dt,CS%vol_CFL,&
                     por_face_areaU(I,j,k))
    enddo ; enddo
    ...
  enddo
  !$omp end target teams
```

**Why:** commit `5b5f6b2b1` ("add teams spec to problematic target region") — direct quote:
> *"For some reason omp runtime was only starting a kernel with 17 blocks when the openacc version
> would start it with 238 or something like that. Manually calculating number of teams sped it up."*

I.e., nvfortran's default `omp target`/`omp target teams loop` team-count heuristic badly
under-subscribed the GPU for this particular tiled kernel shape (17 teams vs. the ~238 an OpenACC
`gang`-mapped equivalent got), so `nteams` is computed by hand
(`ceiling(real(tile_area)/128.)`) and pinned with `num_teams(nteams)` (occasionally paired with
`thread_limit(128)`, e.g. `MOM_set_viscosity.F90:803`
`!$omp target teams loop collapse(2) thread_limit(128)`). This whole family traces back to the port's
OpenACC→OpenMP migration, commit `3cb184edd` ("use openmp instead of openacc"): *"the translation
mapping from oacc to omp is an outer parallel region followed by multiple inner acc loops is
equivalent to an outer omp target followed by multiple inner omp loops... IMPORTANT: However for
OpenMP, inlining of ratio_max and flux_elem is MANDATORY... Otherwise results are incorrect."* Later,
`e8b0ecfbf` ("omp target teams loop -> do concurrent") reverted *other* regions of the same file back
to plain `do concurrent` once it was shown DC scheduled acceptably there — so `target teams` was not a
one-way migration; it is kept only where the manual-team-count fix is demonstrably needed
(`MOM_continuity_PPM.F90:707`), while nearby loops in the same file use bare
`do concurrent (k=1:nz, j=..., i=...)` (e.g. `:430`).

### 3.2 Category B — `teams loop collapse(2)` wrapping a serial tridiagonal column solve

`MOM_vert_friction.F90:737,938,1223,1255` (`teams loop collapse(2)`) and `:1443,1752`
(`teams distribute parallel do collapse(2)`); also `MOM_tracer_advect.F90:505,997,1169` and
`MOM_tracer_hor_diff.F90:1304,1465` (`teams loop` / `teams loop collapse(2)`). Representative kernel —
the u-momentum implicit vertical-friction tridiagonal solve:

```fortran
! MOM_vert_friction.F90:735-786 (abridged)
!$omp target teams loop collapse(2) &
!$omp   private(b1, c1, d1, Ray, b_denom_1)
do j=G%jsc,G%jec ; do I=Isq,Ieq ; if (G%mask2dCu(I,j) > 0.) then
  ...
  b1 = 1. / (b_denom_1 + dt * CS%a_u(I,j,2))
  d1 = b_denom_1 * b1
  u(I,j,1) = b1 * (CS%h_u(I,j,1) * u(I,j,1) + surface_stress(I,j))
  do k=2,nz                                    ! <-- serial recurrence, NOT do concurrent
    c1(k) = dt * CS%a_u(I,j,K) * b1
    b_denom_1 = CS%h_u(I,j,k) + dt * (Ray + CS%a_u(I,j,K) * d1)
    b1 = 1. / (b_denom_1 + dt * CS%a_u(I,j,K+1))
    d1 = b_denom_1 * b1
    u(I,j,k) = (CS%h_u(I,j,k) * u(I,j,k) + dt * CS%a_u(I,j,K) * u(I,j,k-1)) * b1
  enddo
  do k=nz-1,1,-1                               ! <-- back-substitution, also serial
    u(I,j,k) = u(I,j,k) + c1(k+1) * u(I,j,k+1)
  enddo
endif ; enddo ; enddo
```

**Why not `do concurrent`:** the forward sweep (`b1`, `d1`, `c1(k)` each depend on the previous `k`'s
`b1`/`d1` — a genuine sequential recurrence, the Schopf & Loughe 1995 stable tridiagonal form) and the
back-substitution sweep are both inherently **serial in k**, per water column. Only the *(I,j)*
horizontal dimension is embarrassingly parallel (each column is independent). `do concurrent` has no
notion of "parallelize the outer two dimensions, run the third serially with private per-thread
scratch (`b1,c1,d1,Ray,b_denom_1`)" as a single construct — `collapse(2)` explicitly says which 2 of
the enclosing loops are the parallel ones, and `private(...)` gives each (I,j) team/thread its own
scratch for the serial k-recurrence it runs internally. This is the exact idiom flagged in
`00-architecture.md` §4.3: *"Uses `!$omp target teams loop collapse(2)` with a serial inner
tridiagonal k-loop."* The same shape recurs in tracer advection/diffusion wherever a column-local
sequential dependency (limiter passes, vertical remap-like bookkeeping) sits inside an otherwise
horizontally-parallel loop.

### 3.3 The traffic is bidirectional, not a one-way port

Commits `0631c70bc`, `b4ae33d5c`, `e8b0ecfbf` are all literally titled **"omp target teams loop ->
do concurrent"** — i.e. `target teams loop` was tried first (straight OpenACC-style translation),
then *reverted to* `do concurrent` once it was shown to compile/perform acceptably as DC. Conversely
`5b5f6b2b1`, `dbc2521d1`, `0d2f4d7d7`, `38fd0ae32` are all titled **"add teams spec to problematic
target region"** — the opposite direction, applied surgically to the handful of regions where DC (or
plain `omp target`) demonstrably mis-scheduled. `b95083139` ("include k-loop in do concurrent") shows
a third motion entirely within the DC world: merging a separate per-k kernel launch into the DC header
itself (§5 below) for a 20%-ish speedup by cutting launch count — orthogonal to the DC-vs-teams
question, but part of the same tuning cycle.

---

## 4. Reduction idioms — why flags/reductions instead of writing CS/module globals from a loop

A `do concurrent`/GPU kernel body must not write a shared scalar (a `CS%` member, a module variable)
directly from every iteration — that is either a data race (undefined result under concurrent
execution) or, if serialized, defeats parallelism. The port's answer is always: **reduce into a local
scalar, then commit the scalar to persistent state once, after the loop, on the host/serial side.**

### 4.1 Exact-integer sums for reproducibility (`MOM_coms.F90`)

`increment_block_ints` (`MOM_coms.F90:695-772`), the core of the block-based reproducing sum
(`00-architecture.md` §7.2):

```fortran
! MOM_coms.F90:721-724, 741
do concurrent (j=jbs:jbe, i=ibs:ibe) &
    DO_LOCALITY(local(r, e, rmag, lnan, lovf)) &
    DO_LOCALITY(reduce(+: block_sum)) &
    DO_LOCALITY(reduce(max: block_max_pos, block_max_neg, inan, iovf))
  r = descale * array(i,j)
  call efp_decompose(r, e, rmag, lnan, lovf)
  inan = max(inan, lnan) ; iovf = max(iovf, lovf)
  if (r >= 0.) then ; if (rmag > block_max_pos) block_max_pos = rmag
  else               ; if (rmag > block_max_neg) block_max_neg = rmag ; endif
  block_sum(:) = block_sum(:) + e(:)     ! reduce(+: block_sum) — exact fixed-point carry array
enddo
```
`e(:)`/`block_sum(:)` are the fixed-point EFP carry-limbs (`00-architecture.md` §7.2); reducing over
*exact integers* (not floats) means the result is bit-identical regardless of thread/team scheduling
order — this is precisely what buys bitwise reproducibility on GPU. `inan`/`iovf` are `reduce(max:)`
"did we hit a NaN/overflow anywhere" flags, converted to `NaN_error`/`overflow_error` module state
*after* the loop (`:770-771`) — never written from inside the loop.

### 4.2 `ntrunc` truncation counters and `.or.` flags (`MOM_vert_friction.F90:3167-3283`)

```fortran
! MOM_vert_friction.F90:3162-3200 (u-component; v-component is the mirror at :3232-3271)
do concurrent (j=js:je, I=Isq:Ieq)
  dowrite(I,j) = .false. ; vel_report(I,j) = 3.0e8 * US%m_s_to_L_T
enddo

do concurrent (k=1:nz, j=js:je, I=Isq:Ieq) &
    DO_LOCALITY(reduce(.or.: trunc_any, do_any_write))
  ...
  if (CFL > CS%CFL_trunc) trunc_any = .true.
  if (CFL > CS%CFL_report) then
    dowrite(I,j) = .true. ; do_any_write = .true.
    vel_report(I,j) = min(vel_report(I,j), abs(u(I,j,k)))
  endif
enddo

do concurrent (j=js:je, I=Isq:Ieq, dowrite(I,j))   ! <-- mask specifier, not a locality clause
  u_old(I,j,:) = u(I,j,:)
enddo

if (trunc_any) then
  ntrunc = 0
  do concurrent (k=1:nz, j=js:je, I=Isq:Ieq) DO_LOCALITY(reduce(+: ntrunc))
    ... if (...) ntrunc = ntrunc + 1
  enddo
  CS%ntrunc = CS%ntrunc + ntrunc      ! <-- CS state updated once, outside/after the DC loop
endif
```
Two textbook reasons flags/reductions are used instead of touching `CS%` fields in-loop:
1. **`CS%ntrunc` is persistent cross-timestep state.** Incrementing it from inside a concurrent loop
   would race; instead every iteration increments a *local* `ntrunc`, and `CS%ntrunc = CS%ntrunc +
   ntrunc` happens exactly once, serially, after the reduction completes.
2. **Control flow decisions** (`if (trunc_any) then ...`, `if (do_any_write) then ... call
   write_u_accel(...)`) **must be made on the host/serial side** — you cannot conditionally branch
   per-GPU-iteration into a diagnostic I/O call (`write_u_accel`); `trunc_any`/`do_any_write` are
   `reduce(.or.:)` booleans precisely so the *aggregate* "did any point trip" question can be answered
   once, then used to gate a genuinely serial follow-up (`!$omp target update from(u_old,
   vel_report)` + a plain host `do j=... ; do I=... ; if (dowrite(I,j)) call write_u_accel(...)`,
   `:3204-3209`). The mask `dowrite(I,j)` at `:3183` is the vehicle that carries the per-point decision
   from the reduction pass to the later write-back pass without re-branching inside a device loop.

### 4.3 `domore_*` flags replacing early-exit search (`MOM_tracer_advect.F90`)

```fortran
! MOM_tracer_advect.F90:296-306
domore_k_tmp = 0
do concurrent (j=jsv:jev, domore_u(j,k)) DO_LOCALITY(reduce(max:domore_k_tmp))
  domore_k_tmp = 1
enddo
do concurrent (J=jsv+stencil-1:jev-stencil, domore_v(J,k)) DO_LOCALITY(reduce(max:domore_k_tmp))
  domore_k_tmp = 1
enddo
domore_k(k) = domore_k_tmp
```
and (`:582`) `do concurrent (I=is-1:ie) DO_LOCALITY(reduce(.or.:domore_u_jk))`. These replace a
sequential "scan until you find one true value, then stop" idiom (illegal to parallelize directly)
with a `reduce(max:)`/`reduce(.or.:)` over a 0/1 or logical flag set per iteration — equivalent to a
data-parallel logical-OR — used to synchronize the multi-pass mass-flux-limiting iteration
(`domore_k` gates whether level `k` needs another advection pass) without any per-iteration branch
into shared state. Note line 284-294 nests an ordinary serial `do i ... ; exit` search *inside* the
body of an outer `do concurrent (j=..., domore_k(k)>0)` — legal, because `exit` only terminates the
inner sequential `do`, never crosses a `do concurrent` boundary.

### 4.4 A genuine nvfortran limitation: reductions must target scalars, not array elements

`MOM_tracer_hor_diff.F90:959-971`, with an explicit in-source comment:
```fortran
! MOM_tracer_hor_diff.F90:959-971
do concurrent (j=js-1:je+1) DO_LOCALITY(local(itmp))
  itmp = 0
  ! nvfortran do concurrent cannot reduce array elements
  do concurrent (i=is-1:ie+1) DO_LOCALITY(reduce(max:itmp))
    itmp = max(itmp, num_srt(i,j))
  enddo
  max_srt(j) = itmp
enddo
k_size = 1
do concurrent (j=js-1:je+1) DO_LOCALITY(reduce(max:k_size))
  k_size = max(k_size, 2*max_srt(j))
enddo
```
The natural write would be `reduce(max: max_srt(j))` inside the inner loop, but nvfortran rejects (or
mishandles) reducing into an indexed array element — only a bare scalar can be a `reduce()` target. The
workaround stages the per-`j` maximum into a scalar `itmp` (privatized per outer `j` iteration via
`local(itmp)`), reduces into `itmp` over the inner `i` loop, then does a plain (non-reducing) scalar
assignment `max_srt(j) = itmp` in the outer iteration. This is the same reason `domore_k_tmp` (§4.3)
and `block_sum`/`block_max_pos` (§4.1, which *is* allowed to be a whole-array reduce target — arrays
are fine as long as they aren't *indexed elements* being reduced individually) exist as standalone
scalars/arrays rather than being reduced straight into `CS%`-member array elements.

> **FABLE-CHECK (reviewed 2026-07-14 — resolution or current status in KNOWLEDGE.md §8a/§8b):** The generalization "a *whole array* (`block_sum`) is a valid `reduce()` target but
> an *indexed element* (`max_srt(j)`) is not" is inferred from two data points — the in-source comment
> at `MOM_tracer_hor_diff.F90:962` ("nvfortran do concurrent cannot reduce array elements") plus the
> fact that `reduce(+: block_sum)` on the whole `block_sum(:)` array compiles and runs at
> `MOM_coms.F90:723`. The comment only asserts the *element* case fails; that whole-array reduction is
> *positively supported* (vs. merely happening to be written that way) is a reasonable but
> not-independently-confirmed reading. Sanity-check against nvfortran's actual `do concurrent reduce`
> support matrix before treating "whole-array reduce is fine" as a portable rule.

---

## 5. Loop-nest order and where k stays serial vs. joins the `do concurrent` header

### 5.1 Index-order census

Arity is counted by the number of `=` signs in each header (each `v=range` has exactly one; masks
carry none, and no header uses `>=`/`<=`/`==`, so `#(=)` is exactly the index count). Of the 698
`do concurrent`, 685 have a single-line header captured this way; the remaining ~13 span a
continuation line:

```
1-index headers: 126  — leading: i 58, j 26, I 14, J 14, m 9, k 2
                          (row sweeps at fixed j,k; (m=1:ntr) tracer loops; a few (k)/(i))
2-index headers: 390  — leading: j 218, J 129, k 27, jj 10, m 3
                          (dominated by (j,i)/(J,I) horizontal pairs)
3-index headers: 169  — leading: k 114, kk 54, j 1
                          (kk = block-local level index, k re-derived in the body — see §5.2)
```
**Convention (verified, 0 counterexamples):** whenever `k` (or `kk`) appears in a `do concurrent`
header at all, it is written **first** — `(k=1:nz, j=..., i=...)` / `(kk=1:kmax, J=..., I=...)`, never
last (`grep` for a header with `k=`/`kk=` in any non-leading position returns **nothing**). Among the
169 three-index headers, 168 lead with `k`/`kk` and only one leads with `j`. Horizontal-only 2D sweeps
(the plurality of all DC loops, 390) use `(j,i)`/`(J,I)` order, center-point before corner-point naming
following the grid convention in `00-architecture.md` §3.3. All 54 `kk`-led headers live in
`MOM_CoriolisAdv.F90` (its k-blocked kernels).

### 5.2 k inside the `do concurrent` header — the k-blocking connection

When a subroutine is k-blocked (`00-architecture.md` §5), the header index is a **block-local** `kk`
running `1:kmax` (where `kmax = k_end - k_start + 1`), and the **absolute** level `k` is a derived
scalar computed on the first line of the loop body and then used for all array indexing that must
reference the global level:

```fortran
! MOM_CoriolisAdv.F90:383-389 — kk drives the DC header, k is a body-local derived scalar
do concurrent (kk=1:kmax, J=Js_q:Je_q, I=Is_q:Ie_q) DO_LOCALITY(local(k))
  k = k_start + kk - 1
  dvSdx(I,J,kk) = (-Waves%us_y(i+1,J,k)*G%dyCv(i+1,J)) - (-Waves%us_y(i,J,k)*G%dyCv(i,J))
  duSdy(I,J,kk) = (-Waves%us_x(I,j+1,k)*G%dxCu(I,j+1)) - (-Waves%us_x(I,j,k)*G%dxCu(I,j))
enddo
```
`local(k)` is mandatory here: without it, `k` would be treated as a single shared variable written by
every concurrent iteration — a race, and wrong on every iteration but the "last" one under any
serialized interpretation. Declaring it `local` gives each `(kk,J,I)` iteration its own private copy,
which is exactly what makes the k-blocking transformation (whole-domain block on GPU, small cache
block on CPU) safe to express as a single `do concurrent` regardless of block size. This is far and
away the dominant reason `local(k)` appears (`MOM_CoriolisAdv.F90` alone: 39 of its 56 `do concurrent`
loops carry a `DO_LOCALITY(local(k...` clause).

### 5.3 k as a genuinely serial inner loop (not in the DC header at all)

Two distinct reasons k is pulled *out* of the `do concurrent` header and left as an ordinary serial
`do k=...` nested inside the parallel horizontal loop:

1. **Sequential recurrence** (§3.2) — tridiagonal forward/back-substitution in
   `MOM_vert_friction.F90:752-769,776-785`: `do concurrent`/`teams loop collapse(2)` over `(I,j)`
   (independent columns), ordinary serial `do k=2,nz` / `do k=nz-1,1,-1` inside each column body,
   because `b1`/`d1`/`u(...,k)` at level `k` depend on level `k-1`'s just-computed values.
2. **Historical/simplicity, later collapsed for performance** — `MOM_set_viscosity.F90:697-724`, a
   `do k=nz,1,-1 ; if (htot_vel>=CS%Hbbl) exit` bottom-boundary-layer accumulation with data-dependent
   early termination (`exit`) — inherently serial per column regardless of GPU target, nested inside
   an outer `do concurrent (i=is:ie, do_i(i,j)) DO_LOCALITY(local(k, cdrag_sqrt))
   DO_LOCALITY(local_init(cdrag_sqrt_H, cdrag_sqrt_H_RL))`. Commit `b95083139` ("include k-loop in do
   concurrent") documents the *opposite* move for a different case in the same file/module family —
   originally `MOM_vert_friction.F90`'s velocity-truncation logic issued **one kernel launch per k**
   (a host `do k=1,nz` wrapping single-level 2D `do concurrent` calls); the commit message: *"the
   velocity truncation is doing k launches of ij-kernels to truncate velocity. There's no real reason
   for the k loop to be separate from the ij loops, so merging them... for 500x500x100 grid, speeds up
   total vertvisc time by 20-ish%."* That merge produced exactly the `do concurrent (k=1:nz, j=js:je,
   I=Isq:Ieq) DO_LOCALITY(reduce(+: ntrunc))` form quoted in §4.2 — k joins the header whenever there
   is **no** cross-k dependency, purely to amortize kernel-launch overhead; it stays a serial inner
   loop only when correctness (recurrence, data-dependent early exit) requires it.

### 5.4 Rule of thumb

| Situation | Loop form |
|---|---|
| Elementwise/independent-per-level physics, no recurrence | `k` (or blocked `kk`) folded into the `do concurrent` header, `local(k)` if `k` is a derived scalar |
| Column has a top-to-bottom (or bottom-to-top) sequential dependency (tridiagonal solve, running accumulation with early exit) | `k` left as an ordinary serial `do`, nested inside a DC/`teams loop collapse(2)` over the horizontal indices only, with `private`/`local` scratch for the per-column recurrence state |
| A scalar must be aggregated across the whole iteration space (sum/max/min/or) | `reduce(...)` clause on whichever loop performs the aggregation; never accumulate directly into `CS%`/module state inside the loop |

### 5.5 Prescriptive decision tree — "which loop form do I write?"

Walk these branches **top to bottom** and take the first that matches. Every branch is grounded in a
verified in-tree example; copy that example's shape.

1. **Is the loop body a pure elementwise / independent-per-point assignment** — every RHS reads only
   its own `(i,j,k)` (and neighbours), every LHS is a distinct array element, no loop-body scalar is
   carried and nothing is aggregated?
   → **Bare `do concurrent`, no clause.** Fold `k`/`kk` into the header first.
   *Pattern:* `MOM_continuity_PPM.F90:430`
   `do concurrent (k=1:nz, j=jsh:jeh, i=ish:ieh)`; also `MOM_barotropic.F90:1087`. This is ≈86% of all
   DC loops — do **not** reach for a clause you don't need.

2. **Does the body compute one or more scalar (or tiny fixed-size) temporaries that must be private
   per iteration** — a re-derived index, a `sqrt`, a reused neighbour value — and each is *written
   before it is read* within the same iteration?
   → **`DO_LOCALITY(local(...))`.**
   *Pattern:* `MOM_CoriolisAdv.F90:383` `do concurrent (kk=1:kmax, J=…, I=…) DO_LOCALITY(local(k))`
   with `k = k_start + kk - 1` on the first body line (the k-blocking idiom, §5.2); or
   `MOM_continuity_PPM.F90:2747` `DO_LOCALITY(local(h_im1,h_ip1))`. Without `local`, the scalar is a
   shared write = race.

3. **Same as (2), but the private scalar must *start* each iteration holding its pre-loop value**
   because the loop only *conditionally* overwrites it and then reads it unconditionally?
   → **`DO_LOCALITY(local_init(...))`** (add a plain `local(...)` for the always-written temps in the
   same header).
   *Pattern:* `MOM_set_viscosity.F90:681-682`
   `do concurrent (i=is:ie, do_i(i,j)) DO_LOCALITY(local(k, cdrag_sqrt)) DO_LOCALITY(local_init(cdrag_sqrt_H, cdrag_sqrt_H_RL))`
   — `cdrag_sqrt_H` is set only inside `if (CS%bottomdragmap)`, so `local_init` carries the outer value
   for the else-path. Only **2** such loops exist tree-wide; use it only when the conditional-init
   pattern genuinely holds.

4. **Do you need to aggregate one value across the whole iteration space** (sum, max, min, logical
   or)?
   - **Target is a bare scalar (or a whole array reduced as a unit):** put a
     **`DO_LOCALITY(reduce(<op>: <scalar>))`** on the loop; commit it to `CS%`/module state **once,
     after** the loop.
     *Pattern:* `MOM_vert_friction.F90:3189` `… DO_LOCALITY(reduce(+: ntrunc))` then
     `CS%ntrunc = CS%ntrunc + ntrunc` (§4.2); exact-integer `reduce(+: block_sum)` +
     `reduce(max: …)` at `MOM_coms.F90:723-724` (§4.1); boolean gate
     `reduce(.or.: trunc_any, do_any_write)` at `:3168`.
   - **Target is an *indexed array element* `a(j)`:** nvfortran rejects it. **Stage a scalar** —
     `local(itmp)` on the outer loop, `reduce(<op>: itmp)` on the inner loop, then plain
     `a(j) = itmp`.
     *Pattern:* `MOM_tracer_hor_diff.F90:959-966` (see the in-source comment at `:962`, §4.4).
   - **A `reduce()` you expected to work is rejected / mis-scheduled, or the aggregation sits over
     independent columns you also want teamed:** fall back to explicit OpenMP (next branches).

5. **Is there a genuine sequential recurrence down a column** — `x(k)` depends on `x(k-1)` (tridiagonal
   forward/back-substitution, running accumulation)?
   → **`!$omp target teams loop collapse(2)` over the horizontal `(I,j)` only, with `private(...)` for
   the per-column scratch, and a plain serial `do k=…` inside.** `do concurrent` cannot express
   "parallelize 2 dims, run the 3rd serially with private scratch" as one construct.
   *Pattern:* `MOM_vert_friction.F90:737-786`
   `!$omp target teams loop collapse(2) private(b1, c1, d1, Ray, b_denom_1)` wrapping serial
   `do k=2,nz` / `do k=nz-1,1,-1` (§3.2). The same shape recurs at `:938,1223,1255` and in tracer
   advect/diff limiter passes (`MOM_tracer_advect.F90:505,997,1169`;
   `MOM_tracer_hor_diff.F90:1304,1465`).
   *Contrast:* a column loop with only a **data-dependent early `exit`** (not a recurrence) can stay a
   serial `do k=…` nested inside a *`do concurrent`* over the horizontal — `MOM_set_viscosity.F90:697`
   (`do k=nz,1,-1 ; if (htot_vel>=CS%Hbbl) exit`) — no OpenMP needed.

6. **You wrote a plain `do concurrent` / `omp target teams loop` and profiling shows the GPU is badly
   under-subscribed** (few teams launched for a large tile)?
   → **Compute the team count by hand and pin it: `!$omp target teams num_teams(nteams)`** (optionally
   `thread_limit(128)`), with the compute split into `!$omp loop collapse(2)` regions calling
   `!$omp declare target` helpers (which **must** inline — `-Minline` / `!NVF$ INLINE`).
   *Pattern:* `MOM_continuity_PPM.F90:696-736`,
   `nteams = ceiling(real(tile_area)/128.)` then `!$omp target teams num_teams(nteams)` at `:707`
   (commit `5b5f6b2b1`: 17 teams → 238); `thread_limit(128)` at `MOM_set_viscosity.F90:803`. Apply this
   **surgically** — it is the exception, not the default. Conversely, if a hand-placed
   `target teams loop` schedules fine as plain DC, revert it (commits `e8b0ecfbf`, `0631c70bc`,
   `b4ae33d5c`, all "omp target teams loop -> do concurrent").

**Default bias:** branches 1-4 (`do concurrent`, with a clause only when forced) cover the
overwhelming majority; branches 5-6 (explicit OpenMP) are the two catalogued, evidence-backed
exceptions. Reach for OpenMP only when a column recurrence (5) or a measured under-launch (6) makes DC
insufficient.

---

## 6. Cross-references

- `docs/gpu-knowledge/00-architecture.md` §0(3) guiding principle, §5 k-blocking, §6.1 directive
  totals (698/829/213/167/21), §7.2 EFP reproducing sums, §9 quick-reference recipe.
- Compiler-workaround commits referenced here: `5b5f6b2b1`, `3cb184edd`, `e8b0ecfbf`, `0631c70bc`,
  `b4ae33d5c`, `dbc2521d1`, `0d2f4d7d7`, `38fd0ae32`, `b95083139`.
- Infra: `src/framework/do_concurrent_compat.h`, `ac/m4/mom6_fc_do_concurrent_local.m4`,
  `ac/configure.ac:173`.

---

## Verification notes

Independent Opus verification pass (source + git only; no build/run). Every count re-derived with
fresh greps; every cited exemplar line re-read; every quoted commit body re-fetched.

**Confirmed (unchanged):**
- Totals `698 do concurrent`, `96 DO_LOCALITY`, and **all 19 per-file counts** in the §2.1 table
  (barotropic 242/5, tracer_advect 77/8, hor_visc 62/0, continuity 56/11, CoriolisAdv 56/42, … coms
  1/3) reproduce exactly.
- `do_concurrent_compat.h` macro and `mom6_fc_do_concurrent_local.m4` probe (incl. the "Currently only
  LOCAL is tested…" comment) verbatim as quoted. `shared(...)`/`default(none)` never appear on any DC
  construct (0 matches).
- `local_init` at `MOM_set_viscosity.F90:682,759` and its conditional-init rationale; the mask idiom
  and reductions at `MOM_vert_friction.F90:3162-3209` (`:3183` mask, `:3189` `reduce(+: ntrunc)`);
  the array-element-reduce comment at `MOM_tracer_hor_diff.F90:962`; the `num_teams` kernel at
  `MOM_continuity_PPM.F90:696-736` (`:707`); the tridiagonal `teams loop collapse(2)` at
  `MOM_vert_friction.F90:737-786`; the `kk`/`k` re-derivation at `MOM_CoriolisAdv.F90:383-384`
  (`local(k)` = 39 of its loops) — all confirmed.
- Commit subjects **and bodies** verbatim: `5b5f6b2b1` ("17 blocks … 238"), `3cb184edd`
  (OpenACC→OpenMP, inlining mandatory), `b95083139` (20%-ish), and the `e8b0ecfbf`/`0631c70bc`/
  `b4ae33d5c` reverts / `dbc2521d1`/`0d2f4d7d7`/`38fd0ae32` teams-spec additions.

**Corrected:**
- `ac/configure.ac` line: the `MOM6_FC_DO_CONCURRENT_LOCAL` invocation is at **`:173`** (comment at
  `:172`), not `:172` (two occurrences fixed). *(Note: `00-architecture.md` §6.1 still says `:172` and
  is out of this doc's edit scope.)*
- `DO_LOCALITY` breakdown: **94** real uses + **2** in the header (not "95 + 1"); the header defines
  the macro on two `#define` lines (7 and 9).
- §2.2 category counts made exact: `local` **67** (was ~70), `reduce(+)` **5** (was 4), `reduce(max)`
  **10** + `reduce(min)` **1** = **11** (was ~13), `reduce(.or.)` **9** (was ~7); these sum to
  `67+2+5+11+9 = 94`. `local_init` = 2 was already correct.
- §5.1 index-order census rewritten: it was internally inconsistent (a "3-index = 165 total" claim
  whose own leading-index rows summed to 217, having conflated 3-index leading with all-header
  leading). Corrected via `=`-count arity: **1-index 126, 2-index 390, 3-index 169**; three-index
  leading is **k 114, kk 54, j 1** (168 of 169 lead with k/kk; 0 headers put k/kk non-leading). The
  `kk`=54 figure survives and all 54 are in `MOM_CoriolisAdv.F90`.

**Enhancements:** added §5.5, a 6-branch prescriptive "which loop form" decision tree (bare DC →
`local` → `local_init` → `reduce`/staged-scalar/OpenMP → `teams loop collapse(2)` for k-recurrence →
manual `num_teams` for under-launch), each branch grounded in a verified file:line + commit.

**FABLE-CHECK markers:** 2 — (1) the `DO_LOCALITY→;` fallback codegen semantics in §1 (compiler
behaviour, unverifiable from source); (2) the whole-array-vs-indexed-element `reduce` capability
generalization in §4.4.

**Confidence:** High. Every numeric claim was recomputed and every exemplar/commit re-read against the
`dev/gpu` tree; the only residual uncertainty is the two explicitly flagged compiler-behaviour
inferences, which no amount of source reading can settle.
