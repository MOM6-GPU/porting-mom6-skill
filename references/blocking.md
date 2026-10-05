# Blocking: one source for CPU and GPU performance

Blocking splits a domain loop into an outer loop over blocks and inner loops within a block. The
block size is a runtime parameter: the default reproduces the CPU-oriented loop structure and
work-array sizes of the unported code, while a GPU build uses one block spanning the whole local
domain, so each kernel gets the most parallel work. ("Block" is used to avoid confusion with MPI
domains and FMS tiles.) Modules already blocked this way: `MOM_continuity_PPM`,
`MOM_CoriolisAdv`, `MOM_hor_visc` and `MOM_diabatic_aux`. Copy their shape.

Contents: 1 when to block, and along which axis · 2 parameters · 3 the wrapper ·
4 the block loop · 5 work arrays · 6 answers

## 1. When to block, and along which axis

Blocking is usually the last step of a port, after the GPU version reproduces answers, and it
is what brings the CPU clock back to within about 1-2% of the unported code. A routine needs it
when promoting work arrays to full 2-D/3-D for the GPU changed the CPU loop structure, for
example a routine that worked one layer or one row at a time with small scratch arrays.

- **k-blocking** suits a nest whose per-point work at `(i,j,k)` reads only layer `k`
  (horizontal stencils are fine): `CorAdCalc`, `horizontal_viscosity`, the PPM reconstruction in
  continuity. Do not k-block across a vertical dependence: a tridiagonal solve, a running
  integral down the column, remapping, or a float sum over `k` that a block boundary would split.
- **i/j-blocking** suits column physics with such vertical dependences: `make_frazil` and
  `triDiagTS` in `MOM_diabatic_aux` block in i/j only and keep `k` whole in every block.

## 2. Parameters

One integer per blocked direction in the module's CS, read in `<module>_init`:

```fortran
integer :: niblock  !< The i block size used in array calculations [nondim].
integer :: njblock  !< The j block size used in array calculations [nondim].
...
call get_param(param_file, mdl, "DIABATIC_AUX_NJBLOCK", CS%njblock, &
               "The j-direction block size used in the auxiliary diabatic calculations. "//&
               "The default 0 setting is dynamic and fits the "//&
               "full computational j-domain length.", default=default_njblock, layoutParam=.true.)
if (CS%njblock < 0) &
  call MOM_error(FATAL, "DIABATIC_AUX_NJBLOCK must be nonnegative; "//&
                        "use 0 to select the default block size.")
```

- Names `<MODULE>_NIBLOCK`, `_NJBLOCK`, `_NKBLOCK`; `layoutParam=.true.`; a negative value is
  FATAL.
- `0` means the full local extent in that direction.
- The default reproduces the unported loop structure on the CPU: `1` for a direction the old
  code walked one row or layer at a time, `0` for one it already did whole. A GPU build wants `0`
  everywhere. On the GPU development branch that default is chosen at compile time:

  ```fortran
  integer, parameter :: default_niblock = 0
  #ifdef __NVCOMPILER_OPENMP_GPU
  integer, parameter :: default_njblock = 0
  #else
  integer, parameter :: default_njblock = 1  ! one row at a time, as before the port
  #endif
  ```

## 3. The wrapper

Keep the public routine's name and interface. It resolves the block sizes and calls a private
`<name>_block` that does the work and takes the resolved sizes as arguments:

```fortran
subroutine CorAdCalc(u, v, h, uh, vh, CAu, CAv, OBC, AD, G, GV, US, CS, pbv, Waves)
  ...
  integer :: nkk  ! The resolved k block size
  nkk = CS%nkblock
  if (nkk == 0) nkk = GV%ke
  call CorAdCalc_block(u, v, h, uh, vh, CAu, CAv, OBC, AD, G, GV, US, CS, nkk, pbv, Waves)
end subroutine CorAdCalc
```

- Resolved sizes are `nii`, `njj`, `nkk`. A block of face points spans one more point than a
  block of tracer points, so its size gets the staggered name: `nIIB = G%iec - G%isc + 2` beside
  `nii = G%iec - G%isc + 1`, and likewise `nJJB`.
- Resolve against the extent the routine actually loops over: the computational domain, or the
  halo-extended bounds when the routine takes a `halo` argument (`make_frazil`).
- If the routine has no CS, the wrapper gains one. That is the only interface change.
- Do not add accessor functions that make every caller resolve the block size.

## 4. The block loop

```fortran
do ksb=1,nz,nkk
  keb = min(ksb + nkk - 1, nz)
  kke = keb - ksb + 1
  do concurrent (kk=1:kke, j=js:je, i=is:ie) DO_LOCALITY(local(k))
    k = ksb + kk - 1
    work(i,j,kk) = ...   ! block-local index into block-sized scratch, global k into model arrays
  enddo
  ...
enddo
```

```fortran
do jsb=jsh,jeh,njj ; do IsbB=ish-1,ieh,nIIB
  IebB = min(IsbB + nIIB - 1, ieh)
  jeb = min(jsb + njj - 1, jeh)
  IIe = IebB - IsbB + 1
  jje = jeb - jsb + 1
  do concurrent (k=1:nz, jj=1:jje, II=1:IIe) DO_LOCALITY(local(I, j))
    I = IsbB + II - 1 ; j = jsb + jj - 1
    uh_b(II,jj,k) = ...
  enddo
enddo ; enddo
```

Naming:
- block start/end `isb`/`ieb`, `jsb`/`jeb`, `ksb`/`keb`; staggered `IsbB`/`IebB`, `JsbB`/`JebB`;
- block-local indices `ii`, `jj`, `kk` (`II`, `JJ` on faces) and their extents `iie`, `jje`,
  `kke`;
- global indices `i`, `j`, `k`, derived in the body and listed in `local()`.

Either index can drive the loop. `MOM_diabatic_aux` loops over global `i, j` and derives
`ii = i - isb + 1, jj = j - jsb + 1`. Choose the one that keeps the body simplest, and keep the
`do concurrent (j) / do k / do concurrent (i)` structure of `loop-constructs.md` section 6
inside a block where it applies.

- Pass the block's bounds to helpers (`ksb, keb`, or the block extent) instead of `1:nz`, so a
  helper never walks past the scratch it was given.
- A block with nothing to do can be skipped with a `reduce(.or.:)` flag and `cycle`, as
  `make_frazil` does for blocks with no freezing columns.

## 5. Work arrays

Dimension scratch by the block, and full model arrays by the grid:

```fortran
real, dimension(SZIB_(G),SZJB_(G),nkk) :: q          ! k-blocked
real, dimension(nIIB,njj,SZK_(GV))     :: uh_b       ! i/j-blocked, face points in i
```

Map block-sized scratch once around the block loop (`enter data map(alloc:)` before, `exit data`
after), not inside it. With the GPU default of one block, each array is mapped once per call.

## 6. Answers

Blocking must leave answers bitwise identical at every block size. It changes where an
intermediate is stored, not what is computed from what, provided that:
- every expression is copied unchanged, with only the scratch subscript changed;
- no float sum over the blocked direction is split across blocks. Keep vertical sums as a serial
  `do k=1,nz` inside an i/j block;
- every block-sized array is fully written before it is read in each block.

