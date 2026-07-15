# nvfortran: automatic arrays in device-called procedures

**One line:** nvfortran cannot allocate an **automatic array on the device stack** when its size
comes from a runtime expression that is **not a dummy argument**, inside a procedure **called from**
a device region.

## Symptom

Device compile failure (or a forced fixed-size workaround) on a local array like:

```fortran
real, dimension(SZK_(GV)+1) :: dz_col     ! SZK_(GV) -> GV%ke : a derived-type component
```

in a routine invoked from inside a `!$omp target`/`do concurrent` region. The compiler needs a
per-thread stack size it cannot know: `GV%ke` is a runtime value reached through a derived type,
and there is no device-side dynamic stack to fall back on.

## What it is *not*

It is **not** "automatic arrays can't be privatized", and it is **not** specific to
`do concurrent`. This is merged on `dev/gpu` and bitwise-gated, and it privatizes an automatic
array just fine:

```fortran
subroutine vertvisc                        ! MOM_vert_friction.F90
  real :: c1(SZK_(GV))                     ! :575  automatic, sized GV%ke

  !$omp target teams loop collapse(2) &    ! :737  WORKS
  !$omp   private(b1, c1, d1, Ray, b_denom_1)
```

## The actual distinction

| Shape | Works? | Why |
|---|---|---|
| Automatic array in the directive's **own routine**, listed in `private(...)` | **Yes** | the host sizes it at routine entry; the compiler emits N per-thread copies |
| Automatic array **local to a callee** invoked from a device region, sized from non-dummy data | **No** | would need device-side dynamic stack allocation |

## Evidence

`05c74b56b` — Marshall Ward, 2026-06-03, *"ePBL: Replace auto array size (nk=75)"*, on the
epbl-3d branch (`MOM_energetic_PBL_smod.F90`):

```fortran
!  in ePBL_column_3d -- CALLED from the device loop at :528
-  real, dimension(SZK_(GV)+1) :: ...
+  real, dimension(75+1)       :: ...      ! hardcoded to escape the bug
```

The failing construct there was `!$omp target loop private(SpV_dt)` in a `module subroutine` —
**neither `pure` nor `do concurrent`**.

## Workarounds, best to worst

1. **Size the automatic from a dummy argument** (`size(h,3)`, or an `nz` dummy). Then nvfortran can
   size it. *Untested — this is the cheap experiment to run before doing anything below.*
2. **Hoist the array to the caller and `private()` it** on the directive — the `vert_friction:737`
   pattern above. Known to work.
3. **Compile-time `parameter`** (`nk=75`, `NK_GPU_MAX=500`). Works, but over-allocates every column
   to the max: ~6.7x waste at a realistic `GV%ke≈75`, which will spill and crush occupancy.
   Parameterize before merging anything that relies on it.
4. **Static memory mode** sidesteps it entirely — `MOM_memory_macros.h:86` makes `SZK_(G)` expand to
   `NK_`, a compile-time constant, versus `:172`'s `G%ke` in dynamic mode. **So this bug only bites
   dynamic-memory builds.** Check which one you're on before chasing it.

## Status

- **Verified from source:** the `vert_friction:737` counter-example (merged + checksum-gated) and
  the `05c74b56b` diff. No build/run was done.
- **Inferred, untested:** that sizing from a dummy argument fixes it (workaround 1).
- **Unclear:** which nvfortran versions are affected. Our evidence is 25.x; the
  `porting-mom6-skill` skill claims 26.5 but may have inherited rather than re-tested the claim.

## Note for the two existing write-ups

Both are misleading and should be corrected against the table above:

- **`porting-mom6-skill/SKILL.md`** says `do concurrent` cannot privatize an automatic array and to
  *"fall back to `!$omp target teams loop`"*. But the failing construct was `target loop`, not DC —
  and the suggested fallback is exactly what already works (`vert_friction:737`). It would send an
  agent to swap constructs, which either succeeds for the wrong reason or hits the same wall.
- **`KNOWLEDGE.md` §5 row 21** says *"in `pure`/DC procedures"* — but the evidence commit is neither.
  (§8a item 19 gets it right: "non-dummy-sized automatics", and already recommends workaround 1.)
