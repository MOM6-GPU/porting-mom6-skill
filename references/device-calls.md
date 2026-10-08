# Calling procedures from device code

How to write and call a procedure that runs inside a kernel. Findings are for nvfortran 26.3
unless marked. Tags as in `loop-constructs.md`. Reproducers: uwagura/nvfortran-mres repo,
`device_calls/`, plus the numbered issues in its `README.MD`. Which loop construct to put the
call in is in `loop-constructs.md` section 1; polymorphic (`class`) arguments and type-bound
calls are in `polymorphism.md`.

Contents: 1 making a procedure device-callable · 2 inlining · 3 passing arrays ·
4 local arrays in a callee

## 1. Making a procedure device-callable

- **Called from `do concurrent`, it must be `pure`** (`NVFORTRAN-S-0488 ... not PURE`).
  `elemental` implies `pure`. A `call MOM_error(FATAL, ...)` makes a routine impure: return an
  error flag through an argument and report it on the host after the loop, as
  `efp_decompose` (`MOM_coms.F90`) does.
- **Mark it `!$omp declare target`**, placed after the dummy declarations. Same-file callees get
  an implicit one (`-Minfo`: `Generating implicit omp declare target routine`), but a callee in
  another file without it fails to link: `nvlink error: Undefined reference to '<module>_<name>_'`
  `[run-verified]`. Mark it explicitly in either case, so the routine says it runs on the device.
- **Check that a kernel exists.** Calling an `elemental` routine on whole arrays outside a
  `do concurrent` (`call elem_sub(T3d, S3d, out3d)`) runs on the host. There is no `-Minfo` line,
  no kernel launch under `NVCOMPILER_ACC_NOTIFY=1`, and the answers are right, so only the
  timing shows it `[run-verified]`. Call the elemental in scalar form inside a `do concurrent`.
- No allocation, I/O, `post_data` or MPI inside a device-called routine.

An older rule said a callee without `declare target` or forced inlining gives *silently wrong
answers* in an `!$omp target` region (`3cb184edd`, from the OpenACC-to-OpenMP translation of
`MOM_continuity_PPM`). That is `[unverified]` on 26.3: the simple cases above either worked or
failed to link.

## 2. Inlining

Inlining at the Fortran level needs the build flag `-Minline=name:<routine>`, plus `reshape`
when the callee has dummies with explicit lower bounds or receives a section of a different
rank (otherwise `-Minfo=inline` says `subprogram not inlined -- array reshaping not enabled`).
Confirm with `-Minfo=inline`: `<routine> inlined, size=...`.

- **`!DIR$ ATTRIBUTES FORCEINLINE :: <name>` does nothing in nvfortran** `[run-verified]`. The
  front end still emits a call and `-Minfo=inline` is silent. At `-O2` the device back end inlines
  small same-file helpers on its own; calls into other files stay real calls.
- **Inlining can fail silently.** If an argument's type contains a `pointer` to a type from
  another module, the call is not inlined and nothing is printed. `ocean_grid_type` is such a type
  (`type(MOM_domain_type), pointer :: Domain`), so a routine taking `G` is never inlined
  (nvfortran-mres Issue 3). This issue should be resolved in nvfortran 26.9 or later.
- **Inlining can change answers.** An inlined loop that accumulates a value invariant in that loop
  can lose its reduction when the loop lands on GPU threads: wrong answers, no message
  (Issue 5). Inlining can also change whether an expression is contracted into an FMA. This is
  hidden while we build with `-Mnofma`; consider it only when an answer change is otherwise
  unexplained.
- `-Minline` is also the workaround for the code-generation bug in `loop-constructs.md`
  section 1 (Issue 4).
- On nvfortran 26.9, a module-scope `use omp_lib` (even with `only:`) in the callee's module, or a
  module it uses, makes inlined callees unoffloadable. The region silently runs on the host;
  `-Minfo` says `Accelerator restriction: datatype not supported: _in_NNN` (Issue 6). Put
  `use omp_lib` inside the procedure that needs it.

## 3. Passing arrays

- **Never pass a non-contiguous section built inside the kernel**, such as `Igu(i,j,:)` to a
  `dimension(:)` dummy. nvfortran copies it into a per-thread temporary on the device heap
  (~8 MB by default). With enough columns the heap runs out and the kernel fails with a bare
  `CUDA_ERROR_ILLEGAL_ADDRESS`; at smaller sizes it is ~400x slower `[run-verified]`.
  `compute-sanitizer --tool memcheck` shows `Device-side malloc failed`. Pass the whole 3-D
  array plus `i, j`, and declare the dummies `(is:ie, js:je, nk)` (last item below).
- A contiguous section taken once on the host, outside any kernel, is fine: `tv%T(:,:,1)` passed
  to a 2-D interface that runs its own `do concurrent` showed no slowdown and identical answers.
  Write a real 2-D interface instead of adding a fake extent-1 dimension to reuse a 3-D one.
- **Declare array dummies explicit-shape** (`real, intent(in) :: a(is:ie, js:je, nk)`, not
  `a(:,:,:)`). It is a speed-up, not a correctness or collapse requirement: in a
  `do concurrent` reproducer explicit-shape was worth ~30% and kept the collapse
  `[run-verified]` (`device_calls/repro_collapse.F90`).

## 4. Local arrays in a callee

An automatic array local to a device-called routine (`real :: w(nk)`, sized from a dummy or a
type component such as `GV%ke`) compiles and gives correct answers `[run-verified]`. `-Minfo`
says `CUDA global memory used for w`: it is a device `malloc` per call. This is unlike
automatics in a `do concurrent` `local()` clause, which fail (`loop-constructs.md` section 4).

It is slow. Ranked from fastest, measured on the `wave_speed` clock:
1. The caller supplies the workspace: a 3-D `(i,j,k)` array passed in, or the caller's
   `private(...)` scratch (baseline).
2. A fixed size such as `dimension(GPU_nk_max)`: +6%, lost to occupancy when `nk` is much
   smaller than the bound.
3. Automatics: +30%.

Automatics are heap-allocated on the host too (`pgf90_auto_alloc04_i8` calls `malloc`), so
they are no way to avoid allocation cost. With many columns and large automatics the device heap
can run out; nvfortran checks user automatics and prints a message naming
`NV_ACC_CUDA_HEAPSIZE`, unlike the silent section-repack failure in section 3.

An earlier claim that nvfortran cannot size a callee automatic from non-dummy data such as
`SZK_(GV)` (`05c74b56b`, nvfortran 25.x) did not reproduce on 26.3.
