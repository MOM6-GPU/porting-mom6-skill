# KNOWLEDGE.md — MOM6 GPU Porting: Agent Startup Knowledge Base

> **Who this is for.** An agent (or human) booting into the `dev/gpu` fork of MOM6 with zero prior
> context, tasked with continuing — or finishing — the GPU port. This file is the operational
> playbook: identity and ground rules, a compact architecture orientation, the end-to-end porting
> procedure, decision rules, a symptom→fix debugging index, the prioritized work queue, the
> proven-works and never-do inventories, and the open questions queued for strongest-model review.
> Depth lives in `docs/gpu-knowledge/00-…14-*.md` (index in §9); every load-bearing rule here is
> inlined with its citation so you can act without opening the deep docs, and every section links
> to them for the full evidence.
>
> **Repo state.** Branch `dev/gpu`; upstream baseline `dev-gfdl`. `git diff dev-gfdl...dev/gpu` =
> the totality of merged GPU work (320 commits, 52 files, +8310/−4615). Side branches
> (`kblock-hor-visc`, `bodner-naive-port`, `port/pressureforce-benchmark_ALE`,
> `diag_map_mediator_port`, `remotes/edoyango/*`, `remotes/origin/*`) carry the in-flight work.
> All file:line references are against `dev/gpu` HEAD unless a commit hash or branch is named.

---

## 1. Identity and non-negotiable ground rules

This is the **dev/gpu fork of MOM6** (Modular Ocean Model 6), being ported to **NVIDIA GPUs** with
**NVHPC nvfortran**, using **OpenMP target offload** for data management and **Fortran
`do concurrent`** for compute. The rules below are not preferences; every one has been enforced in
review and has commits behind it.

1. **Bitwise reproducibility is mandatory.** No refactor may change the order of floating-point
   operations. Two runs (CPU vs GPU, pre- vs post-refactor, 1 vs N ranks) must produce
   bit-identical fields, verified by `MOM_checksums` popcnt checksums (`MOM_checksums.F90:2680`,
   `bc_modulus=1e9` `:112`) and EFP reproducing-sum energy output (`MOM_coms.F90`). A single
   differing bit means the port is wrong — never accept a nonzero diff as "rounding"
   (doc 05 §7.2, doc 07 §5–6).
   - The escape hatch for restructuring: **extract arithmetic into `pure`/`elemental`
     procedures verbatim** — moving code, never reordering it (doc 08 §4; `efp_decompose`
     `MOM_coms.F90:778`, EOS `_loc` kernels).
   - Producer loops and the reductions that consume their outputs must stay **fused**: splitting
     them lets a compiler re-associate the accumulation (ifort did, even at `-O0` — commit
     `5f413739b`, doc 09 §3).
2. **Preserve CPU performance.** One source form must serve both targets. The accepted mechanism is
   **k-blocking / i-j tiling** with runtime block-size CS parameters whose defaults diverge at
   compile time: `#ifdef __NVCOMPILER_OPENMP_GPU` → `0` (whole extent) on GPU, real cache-tile
   sizes on CPU (`32/4/1` for continuity, `nkblock=1` for CorAdCalc/hor_visc)
   (`MOM_continuity_PPM.F90:3120-3129`; doc 05 §2). `0` is resolved to the full extent at the point
   of use, never stored back into the CS.
3. **`do concurrent` is the default parallel idiom.** OpenMP `target teams` compute constructs are
   used only for (a) serial-in-k column recurrences that need `collapse(2)` + `private` scratch,
   and (b) regions where nvfortran demonstrably mis-schedules DC (manual `num_teams`, commit
   `5b5f6b2b1`). The traffic has run both directions (`e8b0ecfbf` "omp target teams loop -> do
   concurrent") — escalate only on evidence (doc 04 §3, §5.5).
4. **No runtime polymorphism on device.** `class(...)`/type-bound dispatch inside a
   `do concurrent`/`target` region causes runtime errors or mishandled implicit copies of `this`
   on nvfortran. The EOS layer is the case study and the `_loc` free-function duplication is the
   fix (doc 06). ALE's `Recon1d` class dispatch is sidestepped via the non-polymorphic OM4
   select-case path (doc 14 §7).
5. **Expect real compiler bugs.** NVHPC 25.5 (A100 crash on conditional map of an optional dummy,
   `2108e0eba`), 25.9 (silently wrong `do concurrent` result, `MOM_tracer_hor_diff.F90:1464`), and
   25.11 (early-`exit` miscompile, `e23d6a7b1`) have all produced genuine, version-identifiable
   defects. Document every workaround with a comment and a commit; but **check your own diff for a
   missing map or misplaced accumulation before blaming nvfortran** (doc 13 row 24).
6. **Constraints for research/porting agents:** study source + git only unless explicitly told to
   build; never write outside the repo (temp files → `tmp_local_artifacts/` at repo root).

---

## 2. Architecture orientation (compact)

Full treatment: `docs/gpu-knowledge/00-architecture.md` (layout, call tree, inventory) and
`01-memory-control-structures.md` (memory model, CS graph).

### 2.1 Layout

`src/core/` — dynamical core (`MOM.F90` driver, `MOM_dynamics_split_RK2.F90`,
`MOM_continuity_PPM.F90`, `MOM_CoriolisAdv.F90`, `MOM_PressureForce_FV.F90`,
`MOM_barotropic.F90`, `MOM_variables.F90`, grids). `src/parameterizations/{lateral,vertical}/` —
physics (hor_visc, thickness_diffuse, MLE; vert_friction, set_viscosity, diabatic stack).
`src/ALE/` — vertical Lagrangian remap/regrid. `src/equation_of_state/` — EOS.
`src/tracer/` — registry + advection/diffusion. `src/framework/` — domains, comms
(`MOM_coms.F90` EFP sums), diag mediator, restart, checksums, `do_concurrent_compat.h`,
`MOM_memory_macros.h`. `config_src/infra/FMS2/` — FMS shims (incl. the GPU-aware
`do_group_pass`). `config_src/memory/` — `SYMMETRIC_MEMORY_` on/off (one `#define`).

### 2.2 The time-stepping hot path

`step_MOM` (`MOM.F90:522`) → `step_MOM_dynamics` → **`step_MOM_dyn_split_RK2`**
(`MOM_dynamics_split_RK2.F90:302`, predictor–corrector):
PressureForce (`:527/:909`) → CorAdCalc (`:589/:972`) → vertvisc_coef/remnant → btcalc →
continuity (`:695/:853/:1148`) → **btstep** (`:726/:1023`, inner barotropic sub-cycle,
`MOM_barotropic.F90:480`, wide-halo `CS%BT_Domain` "march inward", doc 09 §1.2) →
hor_visc (`:962`) → vertvisc — with ~10 grouped halo passes, all
`do_group_pass(..., omp_offload=.true.)`. Then `step_MOM_thermo` → `diabatic` (**host-only**) and
ALE remap (**host-only**, bracketed `update from(u,v,h)`/`to(u,v,h)` at `MOM.F90:1036/1038`), then
`step_MOM_tracer_dyn` → `advect_tracer`/`tracer_hordiff` (ported). The pure compute kernels contain
no halo updates and no reproducing sums by design — communication is hoisted into the drivers.

### 2.3 The control-structure (CS) pattern and memory

- Every module owns a `<name>_CS` derived type holding all persistent state; populated by
  `<name>_init`, torn down by `<name>_end`. Root: `MOM_control_struct` (`MOM.F90:204`, a **plain
  value** owned by the driver, `map(alloc: MOM_CSp)` in `MOM_driver.F90:282` *before*
  `initialize_MOM`). Dycore hub: `dyn_split_RK2_CSp` (pointer, allocated+mapped at
  `MOM.F90:3258-3259`), which embeds `hor_visc`/`continuity_CSp`/`CoriolisAdv`/`barotropic_CSp` by
  value and points to `vertvisc_CSp`/`set_visc_CSp` (doc 01 §3, §5 for the full tree with line
  numbers).
- Array members are **macro-allocatable** (`ALLOCABLE_`, `ALLOC_(x)` → `allocate(x)`;
  `MOM_memory_macros.h`). `pointer` members exist only for restart-registry targets and
  cross-module aliasing (`MOM_variables.F90:294` comment) — see doc 02 for the aliasing hazard map.
- Index conventions: computational `isc:iec/jsc:jec`, data (=comp+halo, `NIHALO_=2`)
  `isd:ied/jsd:jed`; B-grid (velocity/corner) `IsdB:IedB` with `IsdB = isd-1` in symmetric mode
  (`MOM_hor_index.F90:89-99`). Shapes: `h(isd:ied,jsd:jed)`, `u(IsdB:IedB,jsd:jed)`,
  `v(isd:ied,JsdB:JedB)`, `q(IsdB:IedB,JsdB:JedB)`. Dynamic-mode symmetric offsets live in the
  runtime `ALLOC_` bounds, not the type declaration — always take bounds from
  `G`/`HI` (`IsdB`, never hand-rolled `isd-1`) (doc 01 §1).
- Directive totals on `dev/gpu` `src/`: 698 `do concurrent`, 213 `target enter data`,
  168 `exit data`, 398 `target update`, 21 `declare target` (docs 03, 04).

### 2.4 What is ported, in flight, and untouched (quantified)

- **Ported (merged):** continuity (fully k/i/j-blocked), barotropic (most directive-dense, 242 DC),
  CoriolisAdv (k-blocked, `b8c471cfa`), vert_friction + set_viscosity (teams-loop tridiagonals),
  hor_visc (ported; full k-block on branch), tracer advection + hor_diff (incl. multi-GPU fixes),
  EOS Wright(buggy) + Roquet_rho 2D/3D direct paths, reproducing sums (`8593a732a`),
  PressureForce_FV, `find_uv_at_h` (`MOM_diabatic_aux.F90` — the diabatic-stack template),
  `cuberoot`/`nth_root` intrinsics, `find_eta`.
- **In flight (branches):** `kblock-hor-visc`, `port/pressureforce-benchmark_ALE` (PLM density
  integrals), `bodner-naive-port` (naive contrast case), `diag_map_mediator_port`,
  `edoyango/port-set_diffusivity` (pre-device groundwork), `edoyango/port/thickness_diffuse`
  (real DC directives, 17-file footprint), `origin/epbl-3d` (naive whole-column, "100x"),
  `origin/jorge/diagnostics_port` (ALE OM4-chain `declare target` + `NK_GPU_MAX=500`),
  `edoyango/acc-btstep` (OpenACC async experiment), `fix/nan_repro_sum`.
- **Untouched (0-diff vs dev-gfdl):** `MOM_set_diffusivity.F90`, `MOM_CVMix_KPP.F90`,
  `MOM_energetic_PBL.F90`, `MOM_mixed_layer_restrat.F90`, `MOM_regridding.F90`,
  `MOM_remapping.F90`, `MOM_diag_mediator.F90`, `MOM_restart.F90` — the whole diabatic stack,
  ALE remap machinery, and diagnostics/restart IO (docs 12, 14).

---

## 3. THE PORTING PROCEDURE — end-to-end recipe for one module

Follow the numbered steps in order. Each inlines its load-bearing rules and cites the deep doc for
the full pattern. This is the distillation of every merged port.

### Step 1 — Pick the target from the work queue (§6)

Prefer Tier-1 items with an existing precedent (e.g. a diabatic tridiagonal → copy `find_uv_at_h`).
Check the in-flight branch table first (§2.4, doc 14 §9): if a branch already has groundwork
(j-blocking, `declare target` prep), build on it rather than restarting.

### Step 2 — Pre-flight hazard audit (read the module before writing anything)

Grep the module and record:
- **Pointer members and `associated()` control flow** (doc 02): which CS/type members are
  `pointer` (restart targets? cross-module aliases like `tv%T => CS%T`?). Every pointer read on
  device needs `map(to:)` (never `alloc`) and an `if (associated(...))` guard; every allocatable
  needs `if (allocated(...))` — the intrinsics are not interchangeable (doc 02 §3).
- **Restart-registered fields** the module mutates: they need a `target update from` dominating
  every `save_restart` site (doc 12 §8.2).
- **EOS calls**: which `form_of_EOS` paths are exercised? Only buggy-Wright and Roquet_rho have
  GPU-safe direct 2D/3D kernels; the default `Wright_full` and 6 others are still polymorphic and
  device-fatal (doc 06 §3). If the module needs EOS on device for another form, port that form
  first with doc 06 §6.3's 6-step `_loc` recipe.
- **Recurrences**: identify every loop where `x(k)` depends on `x(k±1)` (tridiagonal solves,
  cumulative sums, early-exit searches) — these dictate the loop form (§4.1) and disqualify naive
  k-in-header parallelization (doc 05 §7.0 disqualifier list).
- **Halo calls**: `pass_var`/`pass_vector` (host-only, no `omp_offload` parameter exists) vs
  grouped passes — plan the promotion (Step 7).
- **Diagnostics** (`post_data`) and debug checksums: every one is a device→host transfer to plan
  (Step 8).
- **Float reductions**: any `+` over reals that parallelization would reorder → must stay serial,
  or route through the EFP reproducing sum (doc 07 §6.1). If a block boundary would fall inside a
  float sum, **stop and redesign**.

### Step 3 — Apply the blessed k-blocking template (doc 05 §7, 11 steps, summarized)

Qualify first (doc 05 §7.0): outer `do k=1,nz` with per-layer 2-D work and **no cross-layer
coupling**; vertical reductions either associative (max/or/int) or kept as an unsplit serial
`do k=1,nz` float sum. Then:
1. Add `integer :: nkblock` (and/or `niblock/njblock`) to the CS
   (`MOM_continuity_PPM.F90:76-78`).
2. `#ifdef __NVCOMPILER_OPENMP_GPU` defaults: `0` GPU / `1` (or `32/4`) CPU (`:3120-3129`).
3. `get_param("MYMOD_NKBLOCK", ..., layoutParam=.true.)`; negative → `MOM_error(FATAL,...)`.
4. Resolve `0`→whole extent **at point of use**: `nkblock = merge(GV%ke, CS%nkblock,
   CS%nkblock==0)` (`MOM_CoriolisAdv.F90:275`); expose an accessor if callers are external
   (`hor_visc_nkblock`, `d6fe494e6`).
5. Shrink full-column scratch to block extent: `dimension(...,max(1,nkblock))`
   (`MOM_continuity_PPM.F90:2706`) — never let a `0` sentinel reach an array bound. Beware: with
   many large stack arrays, **declaration order itself moves CPU perf** ("Move with caution!"
   comment; `28eb296f4`, doc 05 §4).
6. Outer host loop over block starts: `do k_start=1,nz,nkblock ; k_end=min(...) ;
   kmax=k_end-k_start+1`.
7. Device kernel with a block-local index; either convention is bitwise-equivalent:
   iterate `kk=1:kmax` with `k = k_start+kk-1` under `DO_LOCALITY(local(k))`
   (`MOM_CoriolisAdv.F90:383`), or iterate global `k` computing `kk` (continuity, hor_visc).
   **Index-mapping rule: scratch arrays use `kk`; full-size in/out and grid metrics use global
   `k`.** Whichever index is not the DC control variable must be `local(...)` — otherwise it is a
   shared-write race.
8. Thread the active sub-range into every helper (`PPM_limit_pos(...,ks,ke)` — a helper still
   iterating `1:nz` against block-sized scratch reads out of bounds).
9. Incremental porting is allowed: unported sub-blocks stay as serial
   `do k=k_start,k_end ! TODO: port` loops *inside* the installed block nest (CorAdCalc OBC/WENO,
   hor_visc Leithy) — the block structure and the body port are separable steps.
10. Data mapping per Step 5.
11. Manual `num_teams` only on evidence (§4.5).

Bitwise argument to re-verify for your case: blocking changes only which loop an operation nests in
and which buffer holds an intermediate; the per-point expression and every stencil read are copied
verbatim; `kk` is a bijection of `k` — so no FP op is reordered (doc 05 §3). Loop *fusion* is safe
only when the fused loops share an identical iteration space, write distinct arrays, and any
intra-set read is same-index within the same iteration (doc 05 §4, the `28eb296f4` fusion).

### Step 4 — Device-callable helper rules (doc 08 §7 checklist)

For every procedure called from inside a device region:
1. **Side-effect-free?** No module-state writes, no I/O. If not: extract the arithmetic into a new
   `pure`/`elemental` procedure returning error flags via `intent(out)` args (the `efp_decompose`
   model — `carry_overflow` couldn't be `pure` because it writes `overflow_error`;
   `fix/nan_repro_sum` commit `939d06704` "cant be pure" relearned this).
2. **No polymorphic dummy?** A `class(...) :: this` actual in a device loop → duplicate as a free
   `_loc` function with `this` deleted and the body **copied verbatim** (`52a1b3954`;
   doc 06 §6.3).
3. **Which idiom is the call inside?** Bare `do concurrent`: nvfortran's lowering generally handles
   visible `pure`/`elemental` calls. `!$omp target teams`/`loop`: the callee **must** be
   `!$omp declare target` or force-inlined — getting this wrong is **silently wrong numbers, not a
   build error** (`3cb184edd`: "inlining of ratio_max and flux_elem is MANDATORY … Otherwise
   results are incorrect").
4. **Directive state at HEAD:** force-inline is `!DIR$ ATTRIBUTES FORCEINLINE :: <name>`
   (`MOM_continuity_PPM.F90:1086,1149`; `93dbbd36e`). There are **zero** `!NVF$ INLINE` directives
   left in the tree (that form and the `-Minline=name:` flag list are historical). `ratio_max`
   currently has *no* directive — an open question (§8 item 11); do not imitate that gap in new
   code.
5. `declare target` placement: after all declarations, before the first executable
   (`MOM_set_viscosity.F90` even repeats it harmlessly). Same-module vs cross-module is irrelevant
   — the boundary is the lexical scope of the `target` construct.
6. **Verify** (Step 9) — there is no compile-time signal for a helper that silently failed to
   inline.

### Step 5 — Data-mapping lifecycle (doc 03)

There is **no central mapping utility** — every directive is hand-written next to its
`ALLOC_`/`DEALLOC_` (deliberate: kind varies per array, balance must be visually auditable;
doc 03 §4). The canonical lifecycle:
```fortran
ALLOC_(CS%x(IsdB:IedB,jsd:jed,nz)) ; CS%x(:,:,:) = 0.0
!$omp target enter data map(to: CS%x)          ! in *_init  (map(alloc:) if device writes first)
...
DEALLOC_(CS%x)
!$omp target exit data map(delete: CS%x)       ! in *_end, mirrored member-by-member
```
(`MOM_dynamics_split_RK2.F90:1350-1368` / `:2065-2083`.)
- **Shell before members, teardown in reverse.** Map the bare CS shell `map(alloc: CS%child)`
  *before* calling `child_init` (pointer children need host `allocate` first); nested sub-types
  one level at a time (`CS%pbv` then `CS%pbv%por_face_areaU`, `MOM.F90:3225-3231`). Scalar CS
  members need no separate map — they ride in the shell and are refreshed with whole-struct
  `!$omp target update to(CS)` after batches of host scalar writes (diag IDs, `dtbt`;
  `MOM_barotropic.F90:6576`).
- **Map the parent exactly once; never re-`enter data` a parent after members are attached** — a
  second whole-struct `map(to:)` clobbers member attachments and `associated()` state
  (`c82e1254a`: the `visc` "allocated twice" bug was a double *device mapping* at
  `MOM.F90:3278`+`:3709`, not a host double-allocate; doc 02 §5.2). Correct order: allocate host →
  `map(alloc:)` parent once → register restarts (binds pointers) → `update to(parent)` →
  `map(to: member) if (associated(member))`.
- **`map(to:)` vs `map(alloc:)`:** any struct or array whose host-set contents (including pointer
  descriptors read by `associated()`) are read on device must be `to`; `alloc` is only for pure
  workspace written on device before any read. `map(alloc: Reg, Reg%Tr(:))` was the multi-GPU
  answer-change bug (`a774eb331`; doc 02 §4b).
- **Scratch:** subroutine-scoped `enter data map(alloc:...)` at entry, `exit data
  map(delete/release:...)` at return; early-delete once last use passes is fine
  (`up,vp` at `:1253`). Any array touched inside a device region needs an explicit map *before*
  first touch — omission compiles fine and silently reintroduces per-statement traffic
  (`bc05a6a89`).
- **Balance discipline:** every edit to an enter-data list must grep the same routine for the
  paired exit-data list (`15ca2a25f` leaked `b_denom_1`). Neither `delete` nor `release` copies
  back — host-needed values require `update from`/`map(from:)` first.
- **`delete` vs `release` (load-bearing):** `delete` forces the refcount to zero — a per-call
  `delete` inside a callee destroys any outer persistent mapping of the same object (live example:
  `MOM_vert_friction.F90:1105` kills `MOM.F90:3190`'s ADp map; §8b finding B). Use `release` for
  scoped/per-call teardown; `delete` only in the owning `*_end`. And **a `map(to:)` on an
  already-present object does not refresh device contents** — refresh is `update to(...)` only
  (§8b finding C).
- **Flatten arrays-of-structs:** never map a derived-type array whose elements each hold an array,
  inside a loop — one attach/detach per element (`1865612de` halved GPU time by flattening
  `type(p2d), dimension(SZJ_)` to a flat 3-D array, +20% memory). The tolerated exception is the
  tracer registry `Reg%Tr(:)` per-tracer mapped loop (`MOM_tracer_hor_diff.F90:209-216`).
- **Don't pass a CS by `pointer` into leaf routines** — plain derived-type dummies avoid
  per-call descriptor "microtransfers" (`d75e4870e`).

### Step 6 — Loop-form selection

Apply the decision tree in §4.1 to every loop in the module. Reductions: scalar target only
(nvfortran can't reduce array elements — `MOM_tracer_hor_diff.F90:962`); commit to CS/module state
once, after the loop (`CS%ntrunc = CS%ntrunc + ntrunc`, doc 04 §4.2). No early `exit`/`return`
inside device-offloaded loop bodies — rewrite as an `if`-guard (`e23d6a7b1`, NVHPC 25.11 wrong
answers).

### Step 7 — Halo strategy (doc 11 §9)

- A module calling `pass_var`/`pass_vector` (no `omp_offload` parameter exists on those entry
  points) has two options:
  **A (cheap, cold paths):** keep the pass, bracket it `update from(field)` / `update to(field)`.
  **B (blessed, hot paths):** add a `type(group_pass_type)` CS member, register fields with
  `create_group_pass(CS%pass_x, field, G%Domain, halo=<stencil>)` (batch several fields on one
  handle = one MPI message), replace with `do_group_pass(CS%pass_x, G%Domain, omp_offload=.true.)`.
  Preconditions: fields already `enter data`-mapped; FMS2 build (the flag forwards to the external
  FMS `mpp_do_group_update` — §8 item 15).
- **Size `halo=` to the consuming stencil**, not reflexively `NIHALO_=2` (`pass_eta` uses
  `halo=1`); if your port widens a stencil, widen the pass (doc 11 §9.2).
- The nonblocking path (`start_group_pass`/`complete_group_pass`) has **no** offload awareness:
  it is the mutually-exclusive `else` branch and must be hand-staged. 14 of the 26
  `omp_offload=.true.` sites are gated behind `if (G%nonblocking_updates)`; 12 (incl. the
  barotropic inner sub-cycle `:2757` and all tracer passes) are unconditional (doc 11 §4).
- For a solver with many cheap sub-steps, consider the wide-halo clone + march-inward pattern
  (`clone_MOM_domain(..., min_halo=wd_halos)`, `MOM_barotropic.F90:6104`; `O(nstep)` →
  `O(nstep/num_cycles)` exchanges) — but wide-halo residency is all-or-nothing: one host-resident
  wide-bound array defeats the whole optimization (doc 09 §6.5.4).

### Step 8 — Diagnostics / transfer discipline (doc 12 §8)

- `post_data` and the whole diag mediator are **host-only** (zero directives, unchanged on
  dev/gpu). The transfer is the producer's job.
- **Decouple transfer from post:** one `!$omp target update from(<fields>)` covering all consumers,
  guarded `if (CS%debug .or. CS%id_a>0 .or. CS%id_b>0)` (or `any([...] > 0)`), then the individual
  `if (id>0) call post_data(...)` guards (`MOM_tracer_hor_diff.F90:719-733`,
  `MOM_diagnostics.F90:1825-1827`).
- At coarse sync points the codebase default is a **blanket** transfer
  (`MOM.F90:1091 update from(u,v,h,CS%uhtr,CS%vhtr)` before `calculate_diagnostic_fields`); match
  it unless profiling shows a stall — the guarded push-down (`feat/new-diag-manager` `b2a30750a`)
  is the direction of travel, not the current norm.
- Bracket unavoidable host-only detours: `from(...)` before when the host reads device data,
  `to(...)` after when the host modified data the device needs (ALE `MOM.F90:1036/1038`;
  `write_energy` `to(tv%S,tv%T)` `MOM_sum_output.F90:762`).
- **Every debug checksum on a mapped array needs its own `update from` immediately upstream** — a
  transfer for a *different* array does not cover it (`b29b27150`; doc 07 §5.2). Restart-registered
  device-mutated arrays need a dominating transfer before `save_restart` (`MOM_restart.F90` does
  none of its own; doc 12 §6, §8 item 17).
- The derived-type deep-copy trap: touching `CS%tv%T` through the container in an offload
  pass materializes many small implicit transfers — wrap in an explicit
  `map(to: CS%tv, CS%tv%T, CS%tv%S)` bracket or alias to a bare pointer first (`ff86497d5`).

### Step 9 — Bitwise verification (mandatory acceptance gate; doc 07 §6, doc 05 §7.2)

1. Build pre- and post-change at the same optimization level; run and compare `MOM_checksums`
   field checksums (`hchksum`/`uchksum`/`vchksum`) at matching steps — **every checksum
   bit-identical**, plus the EFP `write_energy` output.
2. Confirm block-size invariance: `nkblock=0`, `=1`, `=nz` must all agree.
3. Validate at **≥2 ranks/GPUs** and compare across GPU counts — single-GPU correctness does not
   prove a port (missing `reduce` and `alloc`-vs-`to` bugs are latent until multi-device;
   doc 11 §9.5).
4. On a checksum "mismatch", check the missing-`update from` stale-host case *first* (doc 07 §6.4),
   then your own diff (map balance, misplaced accumulation), then §5's symptom index.
5. Strip debug prints; run the naive-vs-blessed review rubric (doc 10, 8 gates) against your diff
   before proposing merge.

---

## 4. DECISION RULES

### 4.1 Loop-form decision tree (doc 04 §5.5 — take the first matching branch)

1. **Pure elementwise, no loop-carried scalar, nothing aggregated** → bare `do concurrent`,
   `k`/`kk` folded into the header. (~86% of all DC loops; `MOM_continuity_PPM.F90:430`.) Don't
   add clauses you don't need.
2. **Per-iteration scalar temporaries (written before read)** → `DO_LOCALITY(local(...))`
   (`MOM_CoriolisAdv.F90:383`, the k-blocking `local(k)` idiom).
3. **Private scalar must start with its pre-loop value** (conditionally overwritten, read
   unconditionally) → `DO_LOCALITY(local_init(...))` (only 2 uses tree-wide;
   `MOM_set_viscosity.F90:681-682`).
4. **Aggregate across the iteration space** →
   - scalar (or whole-array) target: `DO_LOCALITY(reduce(<op>: var))`; commit to persistent state
     once after the loop.
   - indexed array element target: **rejected by nvfortran** — stage a scalar
     (`local(itmp)` outer + `reduce` inner + `a(j)=itmp`; `MOM_tracer_hor_diff.F90:959-966`).
   - float `+` over reals: **not safe at all** — keep serial or use EFP (§4.4).
5. **Genuine serial-in-k recurrence** (tridiagonal, cumulative) →
   `!$omp target teams loop collapse(2) private(<column scratch>)` over `(I,j)` with a plain
   serial `do k` inside (`MOM_vert_friction.F90:737-786`; `find_uv_at_h`). Right-way nesting:
   `do concurrent(j) → serial do k → do concurrent(i)` — never a serial `do k` wrapping a full 2-D
   DC (the `2a99c9dd1` fix). A data-dependent early-`exit` column search (not a recurrence) may
   stay a serial `do k` inside a DC (`MOM_set_viscosity.F90:697`).
6. **Measured GPU under-subscription** on a hand-written teams region → pin
   `num_teams(ceiling(real(tile_iters)/128.))` (`5b5f6b2b1`, 17→~238 teams;
   `MOM_continuity_PPM.F90:701,707`), optionally `thread_limit(128)`
   (`MOM_set_viscosity.F90:803`). Surgical, evidence-only; revert teams→DC where DC schedules fine.

### 4.2 Map-kind selection (doc 03 §3.5)

| Kind | Use when |
|---|---|
| `map(to:)` | Host-initialized data read on device before first device write; **always** for structs whose scalars/pointer descriptors feed device control flow |
| `map(alloc:)` | Pure workspace, first access is a device write; CS shells |
| `map(from:)` | One-shot device→host copy-out before `post_data`/host math |
| `map(delete:)` | Mirrored CS-member teardown in `*_end` (forces refcount to 0; no copy-back) |
| `map(release:)` | Scratch teardown where conditional branches make the map count uncertain (decrement; no copy-back) |
| `update to/from` | Refresh an existing mapping across a host-only detour; `update to(CS)` for batch scalar refresh |

Guard intrinsics: `if (associated(x))` for pointers, `if (allocated(x))` for allocatables — never
mixed. Never `map(...) if (present(optional_arg))` inside a callee (A100/25.5 crash,
`2108e0eba`) — map optional buffers at the call site where presence is unambiguous.

### 4.3 Pointer vs allocatable (docs 01 §7.1, 02 §7)

- New persistent arrays: **macro-allocatable**, never pointer. Pointer only for restart-registry
  targets or genuine cross-module aliasing.
- New child CS: `allocatable` unless trivially small and always-present (embedded value only for
  the zero-array `continuity_PPM_CS` shape). Conditional allocation, independent device lifetime,
  or `associated()`-gated flow control → `allocatable` (the `81680c15d`/`c82e1254a` rule).
- Local shorthand aliases (`eta => CS%eta`) are removable noise; *reassigned* aliases
  (`p_surf`, `u_ptr` feeding btstep) are load-bearing and keep their per-call guarded maps.

### 4.4 Reduction-operator safety (doc 07 §6.1)

`+` over reals in a parallel loop → **never** (order-dependent). `max/min/.and./.or.` over anything,
and `+` over exact `int64` within a carry-safe block → safe. Reproducible real sums → only via
`reproducing_sum`/`reproducing_sum_EFP` (`MOM_coms.F90:80-90`); keep multi-call totals in
`EFP_type` until the last conversion. Never add a size-dependent host fallback branch to a device
reduction (the `fix/nan_repro_sum` NaN).

### 4.5 Guarded vs blanket transfers (doc 12 §8.1)

Blanket `update from` at sync points is the codebase default for wide fan-outs; guarded
(`if (id>0 .or. debug)`) per-field transfers for module-local diagnostics; push guards deeper only
on profile evidence (nvtx-on-clocks, `ae67665d3`, gives named `nsys` ranges for free).

---

## 5. SYMPTOM → FIX debugging index

Consolidated from doc 13 (§1/§1a) plus the mapping/multi-GPU failure modes of docs 02/03/11.
`SILENT` = wrong answer, no crash (the dangerous class). Check row 0 first, always.

| # | Symptom signature | Root cause | Fix | Anchor |
|---|---|---|---|---|
| 0 | Wrong GPU answers right after a refactor, "looks like a compiler bug" | Your own diff: missing map/copy, misplaced accumulation, unbalanced enter/exit | Audit the diff before blaming nvfortran | `6f3a42d53`, `b404caae2`, `799836a54` |
| 1 | Runtime error / illegal address when an EOS/type-bound method runs in a device loop | Polymorphic `this` v-table dispatch on device | Call a free `_loc` kernel (no `this`); thin host wrapper keeps the API | doc 06; `52a1b3954`, `7c7af5572` |
| 2 | `SILENT` divergence; a `class(...)` actual still appears in a device loop (dead or live) | Residual implicit copy of `this` | Port the remaining call to `_loc`; else accept the copy (open compiler issue) | `MOM_EOS_Wright.F90:1008,1048,1114,1147`; `Roquet_rho:817` |
| 3 | Build error / wrong reduction when target is an array element | nvfortran can't reduce array elements | Stage a named scalar, assign after the loop | `MOM_tracer_hor_diff.F90:962` |
| 4 | Unsupported intrinsic / wrong `modulo()` on device | `modulo()` not implemented on all targets | `sign()` + truncating division (`e - 3*(e/3)` style) | `MOM_intrinsic_functions.F90:232` |
| 5 | `SILENT` last-bit CPU↔GPU diff in `x**(1./n)` | `exp((1/n)*log x)` lowering differs host libm vs libdevice | `cuberoot`/`nth_root` (fixed-iteration Newton, `declare target`) | `MOM_intrinsic_functions.F90:120-132` |
| 6 | `SILENT` wrong result from a specific `do concurrent` (NVHPC 25.9, meridional tracer-flux loop) | Version-specific DC miscompile | Keep it `!$omp target teams loop collapse(2) private(...)` | `MOM_tracer_hor_diff.F90:1464` |
| 7 | `SILENT` wrong answers from early `exit` in a loop nested under DC (NVHPC 25.11) | DC lowering mis-compiles `exit` | Negate condition into an `if`-guarded body; never `exit`/`return`/`cycle` in device loops | `e23d6a7b1` |
| 8 | Correct but slow; kernel launches far fewer teams than expected (17 vs ~238) | Runtime team-count heuristic under-launches | Manual `num_teams(ceiling(iters/128.))` | `5b5f6b2b1`; `MOM_continuity_PPM.F90:707` |
| 9 | `SILENT` wrong numbers when a tiny helper is called from a `target teams`/`loop` region | Un-inlined cross-procedure device call miscompiled | `!DIR$ ATTRIBUTES FORCEINLINE :: name` (or `declare target`) | `3cb184edd`, `93dbbd36e`; `:1086,1149` |
| 10 | Segfault in an `!$omp target`+`parallel loop` region wrapping a k-recurrence | Compiler/runtime bug | Rewrite as `do concurrent` (or teams-loop with serial k) | `5274c3a8e`, `0f05b360f` |
| 11 | Crash on A100 + NVHPC 25.5 at a `map(to: x) if (present(x))` | Conditional map of optional dummy | Delete it; map at the caller | `2108e0eba` |
| 12 | Answers differ **by GPU count** (or run-to-run) — control flow | `map(alloc:)` on a struct whose host-set scalars/`associated()` are read on device (garbage device memory) | `map(to:)` the struct and its member array | `a774eb331`; doc 02 §4b |
| 13 | Answers differ by GPU count — dropped flag/accumulation | Shared-scalar write in DC without `reduce` | `DO_LOCALITY(reduce(<op>: scalar_tmp))`, assign after | `e182de310`; doc 11 §7.1 |
| 14 | Device "addressing error" after init; `associated()` misbehaves in kernels | Parent struct re-`enter data`'d after members were attached | Map parent once; refresh with `update to(parent)`; re-order per doc 02 §5.2 | `c82e1254a` |
| 15 | Checksum "mismatch" that isn't reproducible arithmetic | Stale host copy read by host-only checksum/diag | `update from(<exact array>)` immediately before the call | `b29b27150`; doc 07 §6.4 |
| 16 | GPU time dominated by attach/detach; ~2x slowdown mapping struct arrays | Array-of-structs each holding an array | Flatten to one array with the loop index as a dimension | `1865612de` |
| 17 | Per-call implicit micro-transfers around a derived-type member in an offload pass | Deep-copy trap (`CS%tv%T` through the container) | Explicit `map(to: CS%tv, CS%tv%T, ...)` bracket or bare-pointer alias | `ff86497d5` |
| 18 | Build error on `do concurrent(...) local(...)` on another compiler | Compiler lacks F2023 locality | Wrap every locality clause in `DO_LOCALITY(...)` (`HAVE_FC_DO_CONCURRENT_LOCAL`) | `do_concurrent_compat.h`; `ac/m4/mom6_fc_do_concurrent_local.m4` |
| 19 | Reproducing sum returns NaN on large domains | Size-dependent branch fell back to an unported host routine reading device-only data | Single carry-safe blocked code path for all sizes | `0ac71d482`; doc 07 §3.2 |
| 20 | Bit-repro regression after splitting a producer loop from its accumulation | Compiler re-associates the split reduction (even ifort, even -O0) | Re-fuse producer and consumer loops | `5f413739b`; doc 09 §3 |
| 21 | Device compile failure on automatic arrays sized from non-dummy expressions in `pure`/DC procedures | nvfortran limitation | Fixed-size replacement (`nk=75`, `NK_GPU_MAX=500`) — parameterize before merging | `05c74b56b`; `jorge/diagnostics_port` |
| 22 | Suspected races in a teams-loop kernel; nondeterministic wrong answers | Scalars written in the body missing from `private()`/`local()` | Privatize **every** body-written scalar | `origin/merge-omp-debug` (`7a51e5fb3`) |

⚠ Row 7's bug shape is only *partially* fixed at HEAD: six sibling early-`exit`-under-DC sites
remain (`MOM_tracer_hor_diff.F90:911,913`; `MOM_tracer_advect.F90:287,292`;
`MOM_vert_friction.F90:700,929`) — treat as latent until checksum-validated under ≥25.11
(§8b finding A).

NVHPC versions on record: **25.5** (A100 conditional-map crash), **25.9** (DC wrong result),
**25.11** (early-exit miscompile). `__NVCOMPILER_OPENMP_GPU` is the compile-time GPU-build switch
(block-size defaults; disables CPU-only early-exit convergence tests) — flagged in `93dbbd36e` as
"to be replaced at a later time".

---

## 6. PRIORITIZED WORK QUEUE (doc 14 §8, updated with cross-doc context)

Dependency ordering for Tier 1: **EOS `_loc` coverage → N²/density inputs (`find_N2`,
`isopycnal_slopes`) → `set_diffusivity` → KPP/EPBL → `kappa_shear`**; tridiagonal solves are
independent warm-ups.

### Tier 1 — unconditional per-step critical path

1. **`MOM_diabatic_aux.F90` tridiagonals** (`tracer_vertdiff`/`triDiagTS`[`_Eulerian`]).
   *Approach:* copy the in-file `find_uv_at_h` template verbatim (teams-loop over `j`,
   `do concurrent(i)`, serial `do k`, `map(alloc/release)` scratch). *Hazards:* `tv%T/S` are
   pointer members — map the targets; diag posts behind `id>0` updates. *Lowest effort, do first.*
2. **`MOM_set_diffusivity.F90`.** *Precedent:* `edoyango/port-set_diffusivity` has j-blocking +
   `(i,j,k)` reorder groundwork (zero device directives yet). *Dependencies:* EOS `_loc` chain and
   `MOM_isopycnal_slopes.F90` device coverage first — prefer merging the density-integral work from
   `port/thickness_diffuse`. *Hazards:* writes restart-target pointer fields `visc%Kd_*`.
3. **`MOM_energetic_PBL.F90`.** *Precedent:* `origin/epbl-3d` proves whole-column `pure` +
   `do concurrent(j,i)` works ("100x… ~2ms/step") but is GPU-only (no CPU-preserving story) and
   needed two workarounds to reuse deliberately: manual inlining of
   `get_Langmuir_Number`/`find_mstar`, and fixed-size column arrays (parameterize the hardcoded
   `nk=75` before merging). Decide naive-vs-blessed explicitly against principle 2.
4. **`MOM_CVMix_KPP.F90`.** Zero in-flight work; same column pattern; *hazard:* calls into external
   `pkg/CVMix-src` — the largest `declare target`/inlining surface of Tier 1.
5. **`MOM_kappa_shear.F90`.** Feeds `Kd_shear/Kv_shear`; invoked from the dynamics side
   (`set_viscous_ML`), so device data must be live across the dycore→diabatic boundary. Port after
   set_diffusivity.

### Tier 2 — critical in ALE configs / config-conditional

6. **ALE remap (`MOM_remapping.F90`, `Recon1d_*`)** — the hardest problem (ragged per-column
   sizing, `class(Recon1d)` dispatch, deep call chains). *Not a blank slate:*
   `origin/jorge/diagnostics_port` tags the whole non-polymorphic OM4 chain `!$omp declare target`
   and adds `NK_GPU_MAX=500` fixed sizing (`MOM_remapping.F90:47,1275-1277`) — routines are
   device-*callable*, not yet device-*driven*. *Next:* resolve §8 item 19, then add the driving
   `do concurrent(j,i)` at `MOM_ALE.F90:745`, then bitwise-validate. Treat as research-grade.
7. **`MOM_regridding.F90`** — integer select-case dispatch, no polymorphism; standard template
   should apply; zero in-flight work.
8. **`MOM_thickness_diffuse.F90`** — furthest along (`edoyango/port/thickness_diffuse`: real
   `do concurrent` + `DO_LOCALITY`, submodule split), but drags in the density-integral/EOS
   subsystem (17 files). Merge its EOS/density work early to unblock item 2.
9. **`MOM_mixed_layer_restrat.F90` (Bodner MLE)** — `bodner-naive-port` is the naive contrast case
   (doc 10): no block params, many small data regions, three commits to reach `do concurrent`.
   Redo blessed-style using doc 10's checklist; reuse its 3-D `calculate_density` call
   (`2271af66e`), which is bitwise-safe on the no-`rho_ref` path.
10. **EOS remaining forms** — 7 of 9 unported, **including the default `WRIGHT_FULL`**
    (`EOS_DEFAULT`, `MOM_EOS.F90:192`). Mechanical per-form recipe: doc 06 §6.3 (6 steps, one file
    per form; do not skip `density_anomaly`). Also `int_spec_vol_dp_wright` (non-Boussinesq path)
    is unported.

### Tier 3 — supporting/infra

11. **Diagnostics mediator** — continue `diag_map_mediator_port` (transfer elimination for
    mask/downsample/conversion paths; the FMS write path stays host). 12. **k-block completion** —
    land `kblock-hor-visc` (watch the declaration-order CPU-perf lesson) and the remaining
    CorAdCalc TODO bodies (OBC/WENO). 13. **btstep tuning** — evaluate the `acc-btstep` async
    hypothesis (§8 item 12) before investing. 14. **Restart path** — keep host-only (`noport`
    class); enforce the dominating-transfer rule instead. 15. **Port-coverage tooling** — adopt
    `edoyango/gpu-port-tracking` (`.testing/tools/track_gpu_port.py`, `!@start noport/toport`
    sentinels) as the objective progress metric.

---

## 7. PROVEN-WORKS inventory and NEVER-DO list

### 7.1 Proven to work (each with its merged evidence)

1. Mapping allocatable arrays inside derived types with co-located
   `enter/exit data` next to `ALLOC_`/`DEALLOC_` (`MOM_dynamics_split_RK2.F90:1350-1368`).
2. Whole-CS shells mapped `alloc` before child `_init`; nested sub-type shell-then-members
   (`MOM.F90:3225-3231`); whole-struct `update to(CS)` for batch scalar refresh
   (`MOM_barotropic.F90:6576`).
3. k-blocking with `#ifdef __NVCOMPILER_OPENMP_GPU` 0-vs-tile defaults, bitwise-verified —
   continuity (`93dbbd36e`), CorAdCalc (`b8c471cfa`); the hybrid tiled
   `target teams num_teams` + `loop collapse(2)` kernel (`MOM_continuity_PPM.F90:696-738`).
4. `do concurrent` + `DO_LOCALITY(local/local_init/reduce)` as the compute idiom, with the
   configure-time compatibility macro (698 uses; doc 04).
5. Teams-loop `collapse(2)` + serial-k tridiagonal columns with `declare target` column kernels
   (`MOM_vert_friction.F90:737`, `find_uv_at_h`).
6. Block-based EFP reproducing sums on GPU — exact-integer `reduce(+:block_sum)` partitioned into
   carry-safe blocks, bit-identical under any scheduling (`8593a732a`; `MOM_coms.F90:618-772`).
7. GPU-aware grouped halo exchange: `do_group_pass(..., omp_offload=.true.)` at 26 sites,
   replacing manual staging (`656e09013`); wide-halo BT_Domain march-inward amortization.
8. EOS `_loc` free-function + explicit `do concurrent` 2D/3D overrides (Wright, Roquet_rho;
   `7c7af5572`); host-resolved v-table, device `_loc` execution (`2271af66e` reuse in MLE).
9. `pure` + `!$omp declare target` helpers returning error flags via arguments (`efp_decompose`);
   `cuberoot`/`nth_root` bit-stable intrinsic replacements.
10. Guarded transfers (`update from(...) if (id>0 .or. debug)`, `if (allocated/associated)` maps);
    blanket state transfer at sync points (`MOM.F90:1091`).
11. Struct-of-arrays flattening for attach-cost (halved GPU time, `1865612de`).
12. nvtx-on-clocks profiling wrapper — every existing `cpu_clock` becomes a named `nsys` range with
    zero call-site changes (`ae67665d3`, branch).
13. Multi-GPU-correct tracer advection (scalar-temp reductions + `map(to:)` registry;
    `e182de310`/`a774eb331`).

### 7.2 NEVER-DO list

1. **Never reorder floating-point arithmetic** — no re-association, no split producer/reduction
   loops, no distributing parentheses (`5f413739b`; Fortran parens pin evaluation order).
2. **Never pass polymorphic `this`/`class(*)` into a device region**, or dispatch through a
   v-table on device (`MOM_EOS_Wright.F90:1008` comments; doc 06 §1.4).
3. **Never `map(alloc:)` a struct whose host-set contents are read on device** (`a774eb331`).
4. **Never re-`enter data` a parent struct after its members are attached** (`c82e1254a`).
5. **Never write a shared scalar/array element from a DC body without `reduce`** — and the reduce
   target must be a scalar (`e182de310`; `MOM_tracer_hor_diff.F90:962`).
6. **Never use early `exit`/`return`/`cycle` inside device-offloaded loop bodies** (`e23d6a7b1`).
7. **Never `map(...) if (present(optional_arg))` inside a callee** (`2108e0eba`).
8. **Never call a helper from a `target teams`/`loop` region without guaranteed inline or
   `declare target`** — silent wrong numbers (`3cb184edd`).
9. **Never use `modulo()` or `x**(1./n)` in device code where bits matter** (rows 4/5 of §5).
10. **Never allocate, do I/O, or call `post_data` inside a device loop.**
11. **Never mix guard intrinsics** (`associated` on an allocatable or vice versa) (doc 02 §3).
12. **Never map arrays-of-structs element-by-element in a loop** on a hot path (`1865612de`).
13. **Never leave an enter-data without its mirrored exit-data** (`15ca2a25f`), never expect
    `delete`/`release` to copy back, and **never `map(delete:)` an object your scope does not
    own** — it zeroes the refcount and destroys outer mappings (§8b finding B).
14. **Never ship a port without the bitwise checksum gate and a ≥2-GPU run** (§3 Step 9).
15. **Never hand-roll a reproducible float sum** — use `reproducing_sum`/EFP (doc 07 §6.1).

---

## 8. Open questions for Fable review

All inline `> **FABLE-CHECK:**` markers from docs 01–14, verbatim (lightly re-wrapped), with
locations; items 20–22 are added by the integrating architect. **The Fable review pass has run:
see §8a for per-item resolutions and §8b for new findings — read those before re-investigating
any item below.**

1. **[01 §2, `01-memory-control-structures.md:199`]** "Is the device copy of `GV%Rlay`/`GV%g_prime`
   (mapped at `MOM.F90:3650`) actually consumed by device kernels, or is it a vestigial/incomplete
   map? The disabled sibling at `:3057` ('This does not work. GV%RLay changes sometime later.')
   suggests GV-on-device has a fraught history; confirm which kernels read `GV%Rlay`/`GV%g_prime`
   on device before treating this map as load-bearing." *(low)*
2. **[01 §6.1, `:618`]** "The 'partial presence' mechanism described here (one member of `CS`
   present on device while a sibling needed by the same region is absent, which nvfortran cannot
   reconcile) is an inference; the commit message only says 'ambiguous partial presence errors with
   fields on the top-level CS in associated work.' Confirm this is the actual nvfortran failure
   mode (vs. e.g. a whole-struct-vs-member aliasing conflict) before relying on it as a general
   rule." *(medium)*
3. **[02 §5.2, `02-pointer-usage.md:305`]** "The precise OpenMP/nvfortran semantics behind this
   'second `map(to:)` clobbers member attachments' claim deserve the strongest-model check. Under a
   strict OpenMP 5.x reading, a `map(to:)` on a variable already present should only bump the
   reference count… Is the correct root-cause framing '(i) nvfortran does not honor present-check
   semantics for derived types with allocatable/pointer components and re-copies the descriptor',
   or '(ii) the two directives mapped different storage so refcounts/attach state genuinely
   diverged'? The distilled porting rule holds either way, but the *why* should be stated correctly
   for the compiler-workarounds doc." *(medium)*
4. **[02 §6, `:377`]** "`CS%ADp` is mapped with `map(alloc: CS%ADp)` at `MOM.F90:3190` (an
   all-pointer `accel_diag_ptrs`), while `associated(ADp%sal_u)` etc. are read for control flow in
   `MOM_PressureForce_FV.F90:913-931/:2044-2058`. This is the exact `map(alloc:)`-on-a-pointer-
   struct shape that `a774eb331` identified as a multi-GPU bug for `Reg%Tr(:)`. Is `CS%ADp` a
   latent version of the same bug, or safe — (a) because those reads execute host-side, or (b)
   because the parent's descriptors are never dereferenced on device? Should it be `map(to:)` for
   safety?" *(HIGH — potential live multi-GPU bug)*
5. **[03 §3.5, `03-openmp-mapping.md:475`]** "Is the `delete` vs `release` split genuinely
   load-bearing (reference-count correctness) or merely stylistic convention? … Worth confirming
   against one multiply-mapped case (e.g. `CS` itself, entered at `MOM_tracer_hor_diff.F90:209`
   and released at `:723`, but possibly also shell-mapped by a caller) before presenting the split
   as a hard rule." *(low)*
6. **[04 §1, `04-do-concurrent-patterns.md:34`]** "The 'degrades toward CPU-safe but does not
   GPU-parallelize the flagged locals' semantics is an inference about compiler behaviour when a
   bare `do concurrent` carries implicit locality — not verifiable from MOM6 source or git. …
   Confirm the actual codegen consequence against the F2018 standard / nvfortran docs before
   relying on it." *(low)*
7. **[04 §4.4, `:390`]** "The generalization 'a whole array (`block_sum`) is a valid `reduce()`
   target but an indexed element (`max_srt(j)`) is not' is inferred from two data points… That
   whole-array reduction is positively supported (vs. merely happening to be written that way) is
   a reasonable but not-independently-confirmed reading. Sanity-check against nvfortran's actual
   `do concurrent reduce` support matrix before treating 'whole-array reduce is fine' as a portable
   rule." *(medium)*
8. **[05 §3(4), `05-kblocking-tiling.md:438`]** "Is there a primary source (commit message, code
   comment, or issue) that *directly* states missing inline of `flux_elem`/`ratio_max` produced
   wrong numerical answers (as opposed to a slowdown)? `00-architecture.md` §7.5 asserts 'mandatory
   or wrong answers,' but I could not locate the originating evidence in git log/source — check the
   PR discussion for #165 and the history of `4e3f1b758`/the `-Minline` flag in the build config."
   *(medium — couples with item 11)*
9. **[06 §2.3, `06-eos-layer.md:265`]** "The Wright anomaly branch (`:1013, :1053`) is the one
   place a device `do concurrent` body still literally passes `this`. Confirm whether this is
   deliberate (a form the ported PressureForce path never exercises with `rho_ref` present) or an
   oversight. Roquet shows the `_loc` fix is trivially available, so if any GPU code path reaches
   `calculate_density_array_2d/3d_buggy_Wright` *with* `rho_ref`, this is a live correctness bug.
   Look at callers of `calculate_density(..., rho_ref=...)` for `EOS_WRIGHT` in
   `MOM_density_integrals.F90` / `MOM_PressureForce_FV.F90`." *(HIGH — potential live bug)*
10. **[07 §6.3, `07-reproducibility.md:568`]** "Does the port depend on nvfortran and the CPU
    reference compiler making the *same* FMA-contraction choices for CPU-vs-GPU bit-identity to
    hold? Check whether any build actually pins contraction (search build configs for
    `-ffp-contract`, `-Mnofma`, `-fma`, `Kieee`) — if not, CPU↔GPU checksum agreement may be
    relying on the two toolchains happening to contract identically, which is fragile. Ground truth
    is the flags the `dev/gpu` GPU build harness actually passes (not in this repo's `ac/`)."
    *(HIGH — foundational to the verification gate)*
11. **[08 §3, `08-cross-module-inlining.md:175`]** "`ratio_max` at HEAD has **no** inline directive
    and **no** `declare target`, yet it is still called from inside `!$omp target`/`loop`
    regions (`MOM_continuity_PPM.F90:768-769,…`). Per the core rule and `3cb184edd`'s 'MANDATORY …
    otherwise results are incorrect', how is correctness preserved now? Candidates: (a) nvfortran
    auto-inlines a tiny `pure` function at -O2; (b) the build still passes `-Minline=name:ratio_max`
    (not found in `.testing/`/`ac/` — check actual CI flags); (c) call sites restructured by
    `93dbbd36e` off the device path. Look at `93dbbd36e`'s full diff and the CI compile flags."
    *(HIGH — potential silent-wrongness regression at HEAD)*
12. **[09 §5, `09-barotropic-solver.md:396`]** "Does nvfortran actually launch consecutive
    `do concurrent` loops on a single in-order CUDA stream (making logically-independent loops
    serialize), such that OpenACC `async(N)` queues are the intended remedy? This is the
    load-bearing assumption of §5 and can only be settled by an NVHPC-runtime/`nsys` timeline.
    Look for profiling notes on `edoyango/acc-btstep` or `benchmark_ALE_nvtx_clocks` before
    repeating the serialization claim as fact." *(medium)*
13. **[10 §3, `10-inflight-ports.md:322`]** "Does the PLM density-integral hot path actually need
    continuity's manual `num_teams(ceiling(...))` workaround, or does the tile geometry here (a
    `5*TILE_SIZE_X` inner dimension) keep nvfortran's default team launch adequate? Look at whether
    any surviving `target teams loop` in `int_density_dz_generic_plm`/`PressureForce_FV_Bouss` on
    `port/pressureforce-benchmark_ALE` carries an explicit team count, and compare against the
    under-launch symptom that motivated `5b5f6b2b1`." *(low)*
14. **[10 §4, `:341`]** "Is `port/pressureforce-benchmark_ALE` genuinely the more merge-ready
    branch on the shared PLM code, or merely *different*? … pf's only edges are the `0x1` CPU
    default and the `desubmodule`. Whether `0x1` beats `32x4` on CPU, and whether desubmoduling is
    the intended end-state, needs a benchmarking/maintainer judgment this source-only study can't
    settle." *(low)*
15. **[11 §2.1, `11-halos-domains-multigpu.md:124`]** "The whole 'device-resident, no host
    round-trip' claim for `omp_offload=.true.` rests on what the *external* FMS
    `mpp_do_group_update(..., omp_offload)` overload actually does with the flag — this repo only
    shows the flag being forwarded. Confirm in the linked FMS version that the offload path
    genuinely issues CUDA-aware MPI on device pointers (and does not silently fall back to internal
    host staging on stacks without GPUDirect), since the MOM-side code deliberately removed its own
    staging on the assumption that it does." *(HIGH — correctness+perf of every halo exchange)*
16. **[12 §3, `12-diagnostics-io.md:243`]** "The rewritten `diag_remap_calc_hmask`/`downsample_*`
    routines now *assume* their array arguments are already device-resident … and do no transfer
    themselves. Confirm every caller on `origin/diag_map_mediator_port` actually establishes that
    residency — a caller that hands in a host-only array would read uninitialized device memory
    silently. … Check the `h` argument threaded into `diag_remap_calc_hmask` is likewise mapped at
    every call site, not just the mask." *(medium — gate before merging that branch)*
17. **[12 §6, `:419`]** "The 'would silently write stale host memory' risk is an *inference*, not
    an observed bug. Confirm it: locate the actual `save_restart` write call sites and check
    whether each is dominated by a preceding `target update from(...)` covering every registered
    device-resident field (not just `u,v,h,uhtr,vhtr`). If a restart-registered array that is
    *only* written on device (e.g. `Kd_shear`/`MLD`) has no transfer before the write, the risk is
    real *today*; if all restart writes happen at sync points already covered by blanket transfers,
    it is only a latent trap for future ports." *(HIGH — potential silent restart corruption)*
18. **[13 row 17, `13-compiler-workarounds.md:125`]** "Does the commit `f74525ae8` message 'We lose
    present()' refer to the **OpenACC `present()` data clause** (which asserts device residency
    and has no direct OpenMP-target equivalent) rather than the Fortran `present(optional_arg)`
    intrinsic? The diff drops several `!$acc … present(...)` clauses while converting `!$acc` →
    `!$omp`, which points at the data clause. If so, the entire #17-↔-#16 'foreshadowing' narrative
    is a conflation of two unrelated meanings of `present()` and should be deleted, not merely
    softened." *(low — historiography)*
19. **[14 §7.3, `14-vertical-physics-ale-status.md:523`]** "Is the `diagnostics_port` strategy (OM4
    select-case path + blanket `declare target` + `NK_GPU_MAX=500` fixed sizing) the right
    long-term direction, or a dead end? (1) `NK_GPU_MAX=500` over-allocates every column to 500
    layers of private stack per thread — check whether that blows the device stack/register budget
    for realistic `GV%ke` (~75), vs. sizing at `GV%ke`; (2) the OM4 path still `select case`s over
    ~9 reconstruction kinds per column — confirm nvfortran handles that branch divergence
    acceptably inside a `target teams loop`." *(HIGH — gates the ALE remap plan, Tier-2 item 6)*

**Added by the integrating architect:**

20. **FABLE-CHECK [architect]: `00-architecture.md` contains known-stale claims superseded by the
    verified docs.** Its §7.5 still lists "`!NVF$ INLINE` … mandatory or wrong answers" and "3
    `!$omp declare target` [inline] sites" (superseded by doc 08: zero `!NVF$ INLINE` at HEAD,
    FORCEINLINE on two routines, none on `ratio_max`), its §6.1 cites `configure.ac:172` (doc 04
    corrected to `:173`) and 167 exit-data (doc 03: 168), and its §7.3 says "~25 call sites"
    (doc 11: exactly 26, 14 gated/12 unconditional). Decide whether to patch 00 in place or leave
    it with a superseded-by banner — agents reading 00 first will otherwise re-propagate the stale
    directive story. *(medium — knowledge-base hygiene; this file already states the corrected
    facts.)*
21. **FABLE-CHECK [architect]: what is the intended runtime setting of `NONBLOCKING_UPDATES` for
    GPU production runs?** 14 of 26 GPU-aware halo sites silently revert to host-staged
    communication when it is enabled (doc 11 §4). If GPU configs are expected to run with it off,
    that should be documented (and possibly asserted at init when `__NVCOMPILER_OPENMP_GPU` builds
    detect it on); if on, the 14 gated sites are a standing performance trap. Needs a maintainer
    decision + a param-doc note. *(medium)*
22. **FABLE-CHECK [architect]: is there a strategy decision on the EOS endgame** — continue the
    per-form `_loc` boilerplate for the remaining 7 forms (including the default `Wright_full`),
    or adopt the polymorphism-free `select case (form_of_EOS)` dispatch sketched (as a proposal
    only) in doc 06 §6.4? The residual `this`-descriptor copy (item 2 above, doc 06 case-2) is
    structural under the current design; the `select case` route removes it once for all forms.
    Upstream PR #156 / `eos-3d` branches may already answer this — check before an agent invests
    in 7 more `_loc` conversions. *(medium — shapes Tier-2 item 10.)*

---

## 8a. Fable review — per-item resolutions (2026-07-14, source+git only; no builds/runs)

**RESOLVED** (finding stated; the item above is settled):

1. **GV device map is load-bearing.** `GV%Rlay` is read *inside* a device `do concurrent`
   (`MOM_tracer_hor_diff.F90:902-915`, the `Rml_max`-vs-`GV%Rlay` binary density search) — the
   `map(to: GV, GV%Rlay, GV%g_prime)` at `MOM.F90:3650` is consumed, not vestigial.
2. **"Partial presence" is the literal NVIDIA runtime diagnostic.** The NVHPC OpenMP/OpenACC
   runtime raises a FATAL "partially present" error when a mapping's address range partially
   overlaps an existing present-table entry — exactly the whole-struct-over-attached-member
   overlap the doc inferred. The inference is the actual mechanism; rely on it.
3. **`c82e1254a` root cause: framing (ii).** The host re-allocated `CS%visc`, so the second
   `map(to:)` targeted *different storage* than the first mapping — orphaning member attachments.
   Under a strict OpenMP reading, re-mapping the *same* storage is only a refcount bump with no
   copy (which is itself a trap: a later `map(to:)` on an already-present struct **does not
   refresh device contents** — see §8b finding B). The "map the parent exactly once" rule guards
   against both readings.
4. **`CS%ADp` is safe today, but the mapping economy is incoherent** — see §8b finding B. The
   `associated(ADp%sal_u/sal_v)` reads at `MOM_PressureForce_FV.F90:913/2044` are in plain host
   loops (not device regions), so the `a774eb331` bug shape does not apply there. The only
   *device*-evaluated `associated()` on ADp members (`MOM_vert_friction.F90:699,928`) is on
   members explicitly attach-mapped at `:672-673`. Do NOT `map(to:)` the shell at `MOM.F90:3190`
   as proposed — fix per finding B instead.
5. **`delete` vs `release` is load-bearing — in the dangerous direction.** `exit data
   map(delete:)` forces the refcount to zero, so a per-call `delete` inside a callee destroys any
   *outer, persistent* mapping of the same object. This is not hypothetical: vert_friction's
   `map(delete: ADp)` (`:1105`) kills MOM.F90:3190's init-time `map(alloc: CS%ADp)` on the first
   call (finding B). Rule: `release` for scoped/per-call teardown; `delete` only in the owning
   `*_end` teardown that mirrors the owning `enter data`.
7. **Array-element `reduce` is an F2023 language rule, not an nvfortran quirk.** A
   locality-spec/reduce list takes *variable names*; `max_srt(j)` is an array element, not a
   variable, so rejecting it is conforming. Whole-array `reduce(+: block_sum)` is conforming
   Fortran and positively supported. Reframe the doc's caution accordingly (the staged-scalar
   workaround stays correct).
8. **The "inline or wrong answers" claim has a primary source: `3cb184edd`'s own commit body** —
   "for OpenMP, inlining of ratio_max and flux_elem is MANDATORY. do so with
   `-Minline=name:ratio_max,name:flux_elem`. Otherwise results are incorrect." (Doc 05's verifier
   searched the wrong commit.) The claim is era-specific evidence (the OpenACC→OpenMP
   translation, pre-`num_teams`-fix kernel), not a timeless law — but treat it as binding for new
   code.
9. **Wright anomaly `this` branch: mainline-safe, live hazard on the pf branch.** On `dev/gpu`
   the generic 2D/3D+`rho_ref` dispatch is only reached from host paths. But on
   `port/pressureforce-benchmark_ALE`, the k-blocked `int_density_dz_generic_plm` calls 3-D
   `calculate_density(..., rho_ref=rho_ref)` with `use_rho_ref = .true.` **by default**
   (branch `MOM_density_integrals.F90:221,682`) → dispatches into the `present(rho_ref)` branch
   passing polymorphic `this` inside `do concurrent`
   (`MOM_EOS_Wright.F90:1011-1014,1052-1055`). **Merge gate for that branch:** add
   `density_anomaly_elem_buggy_Wright_loc` (trivial, Roquet proves the pattern) first.
10. **FMA contraction IS pinned in the canonical NVHPC toolchain.** `mkmf/templates/ncrc5-nvhpc.mk:93`
    (and `ncrc-nvhpc.mk`) put `-Mnofma` (plus `-Mdaz`) in the **base** `FFLAGS`, all build modes.
    Action: ensure the site GPU build harness (external to this repo) inherits `-Mnofma`; if it
    does, CPU↔GPU bit-identity does not depend on matching contraction choices.
11. **`ratio_max`'s missing directive is a deliberate removal, resting on implicit device
    codegen.** `93dbbd36e` removed `!NVF$ INLINE` from `ratio_max` *without* replacement (while
    giving `flux_elem`/`flux_elem_OBC` `FORCEINLINE`); no `-Minline` exists in any in-repo or
    mkmf-template build config. `ratio_max` is still called from `!$omp target`+`loop` regions
    (`MOM_continuity_PPM.F90:768-769` etc.). Correctness at HEAD therefore rests on nvfortran
    implicitly compiling/inlining a small same-file `pure` function for the device — empirically
    fine on the tested toolchain (the commit is merged and checksum-gated), but fragile.
    **Recommendation:** add `!DIR$ ATTRIBUTES FORCEINLINE :: ratio_max` for parity; never imitate
    the gap in new code.
15. **FMS `omp_offload` is a genuine device path with NO fallback.** In the sibling FMS checkout:
    `mpp_group_update.fh:422-531` device-packs halos (`target teams distribute … if(use_device_ptr)`
    into a device buffer) and `mpp_transmit_mpi.fh:90-96,145+` posts `MPI_ISEND`/`IRECV` inside
    `!$omp target data use_device_ptr(...)` — real CUDA-aware MPI on device pointers. There is no
    capability check: a non-GPUDirect MPI stack means crash/corruption, not graceful host staging.
    The nonblocking variants hardcode `use_device_ptr = .false. ! placeholder`
    (`mpp_group_update.fh:662,769`) — confirming doc 11's gated/unconditional analysis from the
    FMS side.
17. **Restart staleness is latent, not live.** `save_MOM_restart` (`MOM.F90:4687`) does no
    transfer of its own, but the sync-point blanket `update from(u, v, h, CS%uhtr, CS%vhtr)` at
    `:1091` runs whenever `MOM_state_is_synchronized(CS)` — the same condition under which the
    driver writes restarts — and thermo/mixing fields are host-authoritative (diabatic is
    host-only, `:1827` brackets it). Standing rule confirmed: any *newly device-resident*
    restart-registered field must be added to a dominating `update from` before `save_restart`.
18. **"We lose present()" = the Fortran intrinsic, contra the doc-13 verifier's suspicion.**
    `f74525ae8`'s diff contains **no** OpenACC `present()` data clauses anywhere, but it does
    contain commented-out OpenMP maps of the form `!!!$omp target enter data if(present(pbce))
    map(to: pbce)` — the author tried intrinsic-`present()` conditional maps and disabled them.
    The foreshadowing link to the `2108e0eba` A100 crash (same construct) **stands**; restore the
    stronger narrative in doc 13 row 17.
20. **Done — `00-architecture.md` patched in place** (stale `!NVF$ INLINE` story, exit-data count,
    call-site count, configure.ac line) with a pointer to the verified docs; see the file's
    correction note.

**ADVANCED but needing a run/profile to close** (do not re-derive from source; go straight to the
stated experiment):

6. **DC unspecified-locality semantics:** F2018 makes locality *unspecified* by default; nvfortran
   documents privatizing scalars whose first access in the construct is a write. The doc's caution
   stands; the decisive check is `-Minfo=accel` output on one kernel, not more source reading.
12. **DC single-stream serialization:** NVHPC's documented model launches `do concurrent` kernels
    on the default CUDA stream per host thread — serialization of independent kernels is
    *expected*, which is exactly what `acc-btstep`'s `async(1..3)` queues attack. Confidence:
    high (documented behavior), but quantify with one `nsys` timeline of btstep before investing
    (Tier-3 item 13).
19. **`NK_GPU_MAX=500` sizing:** at `GV%ke≈75`, 500-deep per-thread private column arrays
    over-allocate device local memory ~6.7×; multiple such arrays per thread will spill and crush
    occupancy. Prefer sizing from the dummy argument (`size(h,3)`) or a blocked redesign;
    the underlying constraint (nvfortran rejecting non-dummy-sized automatics in device `pure`
    procs, `05c74b56b`) is real, so a `parameter` sized to a *realistic* max (e.g. 128) with an
    init-time `FATAL` guard is the pragmatic middle. Needs an occupancy measurement to settle.

**OPEN — genuinely needs a maintainer decision or hardware run** (unchanged): 13, 14 (benchmark
judgments), 16 (branch-merge gate audit), 21 (NONBLOCKING_UPDATES policy), 22 (EOS endgame; note
item 9's finding makes the `_loc` boilerplate route costlier than doc 06 estimated — the anomaly
kernels must be duplicated too).

## 8b. Fable review — new findings (not in any FABLE-CHECK)

**A. Six early-`exit`-under-`do concurrent` sites remain at HEAD — the `e23d6a7b1` bug shape was
fixed at only one site.** NVHPC 25.11 produced wrong answers from an early `exit` in a loop nested
inside a DC (never-do #6), yet the identical shape survives at:
`MOM_tracer_hor_diff.F90:911,913` (binary-search `exit`s, *same subroutine* as the fixed
insert-sort), `MOM_tracer_advect.F90:287,292` (`domore` search loops), and
`MOM_vert_friction.F90:700,929` (`direct_stress` column loops, also containing device-side
`associated(ADp%…)` reads). The knowledge base previously contradicted itself here (doc 04 §5.5
branch 5 cites `MOM_set_viscosity.F90:697` as an acceptable serial-k early-exit inside DC).
Resolution of the contradiction: the ban is *empirical per NVHPC version*, not structural — but
until each site is checksum-validated under ≥25.11, treat all six as latent wrong-answer bugs and
apply the `e23d6a7b1` if-guard rewrite opportunistically. (`direct_stress` and
`tracer_epipycnal_ML_diff` are non-default code paths, which is likely why nothing has tripped.)

**B. The `ADp` mapping lifecycle is internally inconsistent (resolves items 4/5).** Sequence at
HEAD: `MOM.F90:3190` maps `CS%ADp` persistently (`enter data map(alloc:)`, refcount 1, device
copy = garbage shell). First `vertvisc` call: `enter data map(to: ADp)`
(`MOM_vert_friction.F90:670`) — already present ⇒ refcount 2 and **the `to` copy is skipped**
(present semantics), so the device shell stays garbage; only the explicitly attach-mapped
`du_dt_str`/`dv_dt_str` (`:672-673`) get valid device descriptors — which is the only reason the
device-side `associated()` reads at `:699/:928` are safe. Then `exit data map(delete: ADp)`
(`:1105`) **forces refcount to 0**, destroying the init-time mapping; every subsequent `vertvisc`
call re-creates the shell fresh (now with a real `to` copy). Net: the `MOM.F90:3190` map is dead
weight that (a) suppresses the first call's shell refresh and (b) is silently destroyed. Fix
options (maintainer choice): drop the init-time map and let vertvisc own the per-call lifecycle
with `release`, or make the init-time map authoritative (`map(to:)` + per-call `update to(ADp)`
and no per-call delete). Either way, add the general rule: **never `map(delete:)` an object your
scope does not own** (now reflected in §3 Step 5 and never-do #13).

**C. Generalized rule from item 3 + finding B:** a `map(to:)` on an already-present object does
not refresh device contents. If a struct's host scalars/descriptors changed since its first map,
the *only* refresh is `target update to(...)`. Several existing patterns rely on this implicitly;
new code must not "re-map to refresh".

---

## 9. Index of `docs/gpu-knowledge/`

| Doc | One line |
|---|---|
| `00-architecture.md` | Anchor: layout, CS pattern, step_MOM/RK2 call tree, memory model, merged-work inventory (⚠ see §8 item 20 for its known-stale spots) |
| `01-memory-control-structures.md` | Memory macros, symmetric-memory mechanics, full CS type/allocation graph, the 3 shaping commits (`81680c15d`/`c82e1254a`/`1865612de`) |
| `02-pointer-usage.md` | Pointer taxonomy, `associated()` map guards, the `Reg%Tr(:)` and `visc` mapping bugs, pointer-hazard table |
| `03-openmp-mapping.md` | Proven mapping patterns: lifecycle, shells, conditional maps, map-kind table, declare-target catalogue, update-from taxonomy |
| `04-do-concurrent-patterns.md` | DC census (698), `DO_LOCALITY` machinery, locality-specifier catalogue, DC-vs-teams evidence, the loop-form decision tree |
| `05-kblocking-tiling.md` | The blessed transform: before/after diffs (continuity/CorAdCalc/hor_visc), block-size machinery, bitwise argument, 11-step recipe |
| `06-eos-layer.md` | Old polymorphic dispatch, `_loc` rewrite, per-form port-status table, 6-step port-a-form recipe, endgame proposal |
| `07-reproducibility.md` | EFP algorithm + GPU blocking, purity rules, checksum tooling, order-of-operations rulebook |
| `08-cross-module-inlining.md` | Device-callable helper catalogue, inline-directive history (3 forms), refused constructs, device-call checklist |
| `09-barotropic-solver.md` | btstep structure, wide-halo march-inward, repro/crash fixes, acc-async experiment, transferable lessons |
| `10-inflight-ports.md` | Naive (bodner) vs blessed (pf-ALE) contrast, branch forensics, 8-gate merge review rubric |
| `11-halos-domains-multigpu.md` | Group-pass machinery, `omp_offload` end-to-end, 26-site table, gated/unconditional split, multi-GPU bug anatomy, halo porting rules |
| `12-diagnostics-io.md` | Host-only mediator/restart, transfer audit (guarded vs blanket), diag_map_mediator_port scope, nvtx profiling, restart rules |
| `13-compiler-workarounds.md` | The 24-row bug/workaround catalogue, symptom-signature index, NVHPC version table, directive/flag reference |
| `14-vertical-physics-ale-status.md` | Diabatic/ALE status tables, column-kernel anatomy, in-flight branch survey, tiered remaining-work plan |
