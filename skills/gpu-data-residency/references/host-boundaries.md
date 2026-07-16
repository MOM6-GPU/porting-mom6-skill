# The host-boundary catalogue

Every entry is a **host-only consumer**: if it touches an array that is `DEV_FRESH`, a
`!$omp target update from(<that exact array>)` must dominate it. If it *writes* an array the device
later reads, an `update to(...)` must follow it.

Verified against `dev/gpu` HEAD on 2026-07-14. Each row carries the command that re-establishes it —
re-run them if the tree has moved, rather than trusting this table.

## Host-only by module (zero device directives in the whole file)

| Consumer | Why host-only | Re-verify |
|---|---|---|
| `post_data`, `register_diag_field`, the entire diag mediator | 0 `omp target` directives | `grep -c "omp target" src/framework/MOM_diag_mediator.F90` → 0 |
| `hchksum`, `uchksum`, `vchksum`, `uvchksum`, `Bchksum`, `chksum`, `MOM_tracer_chksum` | 0 `omp target` directives | `grep -c "omp target" src/framework/MOM_checksums.F90` → 0 |
| `save_restart`, `save_MOM_restart`, `register_restart_field` | 0 `omp target` directives | `grep -c "omp target" src/framework/MOM_restart.F90` → 0 |

`MOM_restart.F90` does **no transfer of its own**. Restart staleness is currently latent, not live,
only because the sync-point blanket `update from(u, v, h, CS%uhtr, CS%vhtr)` at `MOM.F90:1091` runs
under the same condition the driver writes restarts under (`knowledge/KNOWLEDGE.md` §8, "restart staleness is latent, not live"). Any
*newly* device-resident restart-registered field you add must be added to a dominating
`update from` before `save_restart`.

## Halo exchange — the one place the verdict flips

| Call | Verdict | Why |
|---|---|---|
| `do_group_pass(group, dom, omp_offload=.true.)` | **DEVICE** — no transfer needed | forwards to FMS `mpp_do_group_update`, which device-packs halos and posts `MPI_ISEND`/`IRECV` under `!$omp target data use_device_ptr(...)` — real CUDA-aware MPI on device pointers, with **no host-staging fallback** (§8, "FMS `omp_offload` is a genuine device path") |
| `do_group_pass(...)` **without** the flag | HOST | bracket it `update from` / `update to` |
| `pass_var`, `pass_vector` | **HOST** — always | no `omp_offload` argument exists on these entry points |
| `start_group_pass` / `complete_group_pass` | **HOST** — always | the nonblocking path hardcodes `use_device_ptr = .false. ! placeholder` in FMS |

Re-verify the `pass_var`/`pass_vector` claim (the argument lists must contain no `omp_offload`):

```bash
sed -n '173,175p;662,664p' config_src/infra/FMS2/MOM_domain_infra.F90
grep -n "omp_offload" config_src/infra/FMS2/MOM_domain_infra.F90
```

⚠ 14 of the 26 `omp_offload=.true.` sites are gated behind `if (G%nonblocking_updates)` and revert
to the host-staged branch when it is on — so the *same call site* is a host boundary or not
depending on a runtime parameter (`knowledge/KNOWLEDGE.md` §9, "`NONBLOCKING_UPDATES` policy", open). If your analysis depends on a
gated site, say which branch you assumed.

## Host-only by call type

| Consumer | Note |
|---|---|
| `max_across_PEs`, `min_across_PEs`, `sum_across_PEs` | host MPI |
| `reproducing_sum`, `reproducing_sum_EFP` | GPU-aware internally (`8593a732a`), but confirm the *inputs* it reads are resident; never hand-roll a float sum |
| `write_energy` | brackets its own `to(tv%S, tv%T)` at `MOM_sum_output.F90:762` |
| `MOM_error`, any I/O, any `print`/`write` | host |
| `get_param` / `param_file` | host, init-time |

## Host-only by module status (the untouched list)

Calling into any of these is a host boundary — they have **0 diff vs `dev-gfdl`** and no device
awareness (`knowledge/KNOWLEDGE.md` §2.4):

`MOM_set_diffusivity.F90`, `MOM_CVMix_KPP.F90`, `MOM_energetic_PBL.F90`,
`MOM_mixed_layer_restrat.F90`, `MOM_regridding.F90`, `MOM_remapping.F90`,
`MOM_diag_mediator.F90`, `MOM_restart.F90` — i.e. the whole **diabatic stack**, **ALE remap
machinery**, and **diagnostics/restart IO**.

The two structural brackets that exist today because of this:

- ALE remap: `update from(u,v,h)` at `MOM.F90:1036` → host-only remap → `update to(u,v,h)` at `:1038`
- diabatic: host-only throughout; `:1827` brackets it

Re-verify the untouched list before relying on it (branches land):

```bash
git diff --stat dev-gfdl...dev/gpu -- src/parameterizations/vertical/MOM_set_diffusivity.F90
```

## The classification traps

1. **`!$OMP parallel do` is HOST.** CPU threads, not device. It looks like an OpenMP offload
   directive and is not. This is the most common misread in this codebase, and it is *everywhere*
   in the same files as real `target` directives (`MOM_tracer_hor_diff.F90:297-382` has eight of
   them interleaved with device regions).
2. **`do concurrent` is DEVICE** on the GPU build — the default compute idiom (698 uses).
3. **A commented-out directive is not a directive.** `!!$omp target update from(Shear_mag)` at
   `MOM_hor_visc.F90:1656-1658` and `!**!$omp target update to(...)` at `MOM.F90:867` are disabled.
   Do not count them as transfers.
4. **A transfer of a *different* array does not cover yours** (`b29b27150`).
5. **A transfer under a different predicate than the read does not cover it.** This is the
   highest-yield bug shape — see `worked-example.md`.
