# Polymorphic types and device code

MOM6 selects some algorithms at run time through polymorphic (`class`) types: the equation of
state (`EOS_type%type`, `class(EOS_base)`) and the ALE reconstructions (`class(Recon1d)`).
Findings are for nvfortran 26.3 unless marked. Tags as in `loop-constructs.md`. Reproducer:
nvfortran-mres repo, `polymorphism/`.

Contents: 1 what fails · 2 what works · 3 the MOM_EOS pattern · 4 ALE `Recon1d` · 5 before
porting a loop that calls one of these

## 1. What fails

**Any call that needs the object's dynamic type inside a kernel** `[run-verified]`:

| Inside the loop | `do concurrent` | `!$omp target teams distribute parallel do` |
|---|---|---|
| A call through a binding on a `class` variable: `obj%f(x(i))`, `EOS%type%density_elem(...)` | builds; the loop **runs on the host** and device data is never updated. Only sign: `-Minfo` says `Accelerator restriction: Indirect function/procedure calls are not supported` | builds silently; at run time `Failed to find device function 'nvkernel_...'! ... Rebuild this file with -gpu=cc80` (the advice is wrong) |
| `this` passed to another procedure: `density_anomaly_elem_buggy_Wright(this, T(i,j), ...)` | works at `-O0`; at `-O2` fails at run time with `variable in data clause is partially present` on a compiler temporary | not tested |

Search build logs for `Indirect function` after touching code near a polymorphic type.

## 2. What works

- **Dispatch once on the host, then run plain kernels inside the concrete routine.** A
  type-bound array routine may have a `class(<concrete>), intent(in) :: this` dummy; its loops
  must call only free (non-type-bound) procedures with ordinary arguments `[run-verified]`.
- Reading a scalar component of `this` in the loop works (`-Minfo`:
  `implicit copyin(this)`). Copying it to a local first avoids the copy. The in-source comments
  "There is an implicit copy of `this` which cannot yet be prevented" refer to this small,
  harmless copy.
- `select type` on the host, then pass the object as `type(<concrete>)`: works.
- `type(EOS_type)` itself can be mapped and passed to a device routine; `tv%eqn_of_state`
  needs `map(to: tv, tv%eqn_of_state)` (`data-mapping.md` section 4). `calculate_TFreeze`
  selects its formula with an integer `select case`, not a binding, so it runs on the device.

## 3. The MOM_EOS pattern

`EOS_type` wraps `class(EOS_base), allocatable :: type`; the generic front door in `MOM_EOS.F90`
(`calculate_density`, `calculate_density_derivs`, ...) calls through it once per call, on the
host. The base type's generic array fallbacks apply `this%..._elem` to whole arrays, which
runs on the host (`device-calls.md` section 1), so with those forms the inputs and outputs must
be current on the host. A form needs its own array overrides to run on the device.

Ported forms: `buggy_Wright_EOS` (`MOM_EOS_Wright.F90`; density and first derivatives) and
`Roquet_rho_EOS` (`MOM_EOS_Roquet_rho.F90`; also second derivatives, the reference
implementation). All other forms, including the default `WRIGHT_FULL`, still use the fallbacks.

To port another form, in `MOM_EOS_<Form>.F90` only:
1. For each elemental kernel the arrays need (`density_elem`, `density_anomaly_elem`,
   `calculate_density_derivs_elem`, ...), add a free `elemental` `<kernel>_<form>_loc` with the
   body copied verbatim and the `this` dummy removed. Do not reorder the arithmetic.
2. Reduce the type-bound original to a one-line call to the `_loc` version.
3. Override `calculate_density_array_2d/3d`, `calculate_density_derivs_2d/3d` (and
   `calculate_density_second_derivs_2d` if needed) on the concrete type: copy the signature of
   the `a_*` fallback in `MOM_EOS_base_type.F90`, and loop with `do concurrent` over the `dom`
   bounds calling only `_loc` kernels. Cover both branches of `rho_ref`.

Known gap: Wright has no `density_anomaly_elem_buggy_Wright_loc`, so the `present(rho_ref)`
branch of its 2-D/3-D density overrides calls `density_anomaly_elem_buggy_Wright(this, ...)` in
the loop, the `-O2` failure of section 1. On `dev/gpu` that branch is reached only from host
paths; add the `_loc` kernel before any device code calls `calculate_density` with `rho_ref`.

## 4. ALE remapping and `Recon1d`

`remapping_core_h` (`MOM_remapping.F90`) has two paths `[source-only]`:
- **`REMAPPING_VIA_CLASS`**: calls through `class(Recon1d), pointer :: reconstruction`
  (`CS%reconstruction%reconstruct(h0, u0)`, `remap_to_sub_grid`). Every call goes through a
  binding (section 1), and the single object also holds the column's working state (`u_mean` and
  the scheme's arrays), so columns could not run in parallel even with the dispatch removed.
- **Otherwise, the OM4-era functions** (`build_reconstructions_1d`,
  `remap_src_to_sub_grid_om4`, `remap_sub_to_tgt_grid_om4`), selected by an integer
  `select case` on the scheme. No polymorphism, so this is the path to port.

Either way, remapping is called per column from host loops in `MOM_ALE.F90`, with column-sized
automatics (`n0`, `n1`) and deep call chains, so a port is a substantial refactor. Ask the user
before starting one.

## 5. Before porting a loop that calls one of these

1. Check whether the body calls a MOM_EOS or remapping routine.
2. If the call can be lifted out of the loop, do it: compute the EOS quantity once for the
   whole 2-D or 3-D block into a temporary through the array interface, then port the loop. A
   ported EOS form keeps that call on the device.
3. If lifting it out needs a real refactor, stop. Explain why the call blocks the port, sketch
   one or two approaches, and ask the user before writing offload code.
