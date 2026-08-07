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

0. **Have the caller pass the workspace in as a full 3-D array**, indexed `(i,j,k)` by the callee
   along with the `i,j` it is working on. This removes the automatic entirely rather than trying to
   make it sizeable, and it composes with the blocking plan (a 1x1 i-j block degenerates to a
   column call on CPU). Used in the `wave_speed` port: `tdma6` sizes `beta`/`I_beta`/`yy` from its
   dummy `n`, and `tdma6_3d` drops all three by taking `I_beta`/`yy` as caller-supplied 3-D
   workspace. **Declare those dummies explicit-shape and give the input scalars `VALUE`**, or the
   calling loop loses its collapse — see `SKILL.md`, "Calling a procedure from inside a kernel".
1. **Size the automatic from a dummy argument** (`size(h,3)`, or an `nz` dummy). Then nvfortran can
   size it. *Still untested as of 2026-08-03.* The `wave_speed` port passed through this case but
   went to workaround 0 instead, and its two configurations never set `calc_modal_structure`, so
   the dummy-sized automatics in `tdma6` were never exercised on device.
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
- **Unclear:** which nvfortran versions are affected. Our evidence is 25.x. The 26.5 figure that was
  in `SKILL.md` was never re-tested and has been removed; the toolchain actually in use for the
  2026-08 work is **26.3** (`/scratch/cimes/uw4770/envs/linux-nvidia_2026_3.env`).

## A separate, easily-confused failure: automatics are *not* stack-allocated

Do not reach for automatics as a way to avoid the cost of `allocate`. nvfortran heap-allocates them
too — verified 2026-07-31 by compiling with `-S`: automatic → `pgf90_auto_alloc04_i8`, allocatable →
`pgf90_alloc04_chka_i8`, and `pgf90_auto_alloc04_i8` in `libnvf.so` calls `malloc`. Only
`-Mstack_arrays` moves them to the stack. At MOM6 scratch sizes a microbenchmark measured automatics
slightly *slower* than allocatables (16.2 vs 14.6 ms/call). This is a different issue from the
device-side bug above, but the two get conflated because both are about "automatic arrays".

## Note for the two existing write-ups

- **`porting-mom6-skill/SKILL.md`** — **corrected 2026-08-03.** It used to say `do concurrent`
  cannot privatize an automatic array and to *"fall back to `!$omp target teams loop`"*, which was
  wrong twice over: the failing construct was `target loop`, not DC, and the suggested fallback is
  exactly what already works (`vert_friction:737`). It now states the callee/caller distinction and
  points here. Its construct-fallback ordering was also rewritten, for an unrelated reason: on
  nvfortran 26.3 `target teams loop` containing a call to a `declare target` routine is a hard ICE,
  so `target teams distribute parallel do` is now listed first.
- **`KNOWLEDGE.md` §5 row 21** says *"in `pure`/DC procedures"* — but the evidence commit is neither.
  Still uncorrected. (§9, "`NK_GPU_MAX=500` sizing", gets it right: "non-dummy-sized automatics", and
  already recommends workaround 1.)
