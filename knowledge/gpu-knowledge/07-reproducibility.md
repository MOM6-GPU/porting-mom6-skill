# Bitwise Reproducibility on `dev/gpu`

> Drills into §0.2 and §7.2 of `00-architecture.md`. Covers the Extended Fixed Point (EFP)
> reproducing-sum machinery in `src/framework/MOM_coms.F90`, the GPU block-based restructuring
> (commit `8593a732a`), the `fix/nan_repro_sum` bugfix branch and why it could not stay `pure`,
> where reproducing sums are actually invoked in the timestep, and how `MOM_checksums.F90` verifies
> a port bit-for-bit. Source + git only; nothing here was built or run.

---

## 1. The EFP fixed-point algorithm

**Why floating-point sums aren't reproducible at all.** IEEE addition is not associative:
`(a+b)+c /= a+(b+c)` in general due to rounding. A naive parallel/distributed sum's result therefore
depends on the order values are added, which depends on domain decomposition (PE count, tile shape)
and, on GPU, thread/warp scheduling. MOM6's global diagnostics (total mass, KE, PE, heat/salt
budgets) must be identical across PE counts and across CPU/GPU builds for restart and regression
testing to mean anything, so MOM6 replaces the FP sum with an **exact integer sum**, described in
Hallberg & Adcroft 2014 (*Parallel Computing* 40(5-6), doi:10.1016/j.parco.2014.04.007; cited at
`MOM_coms.F90:101`).

**Parameters** (`src/framework/MOM_coms.F90:31-67`):

```f90
integer, parameter :: accum_width = digits(1_int64)   ! :31  -- 63 usable bits (excl. sign) of an int64
integer, parameter :: prec_width = 46                 ! :33  -- bits of precision per EFP "digit"
integer, parameter :: guard_width = accum_width - prec_width   ! :35 -- 17 guard/carry bits
! A sum of N points does N - 1 additions, which at most adds N - 1 carry bits.
! For G guard bits, the maximum value is 2**G - 1.  A summation of N values
! therefore requires that N - 1 <= 2**G - 1, or simply N <= 2**G.
integer, parameter :: max_summands = 2**guard_width   ! :42  -- 2**17 = 131072
integer(kind=int64), parameter :: prec = (2_int64)**prec_width   ! :46 -- 2**46, the EFP "digit base"
integer, parameter :: efp_digits = 6                  ! :54  -- number of base-2**46 words
```

**Decomposition.** Every `real` is decomposed into `efp_digits = 6` signed `int64` words, each
representing a base-`2^46` "digit" (`EFP_type`, `:103`):

```f90
type, public :: EFP_type ; private
  integer(kind=int64), dimension(efp_digits) :: v !< The value in this type
end type EFP_type
```

The decomposition (`efp_decompose`, `:778-821`) peels off successive base-`prec` digits, most
significant first, exactly like writing a number in a mixed-radix positional system:

```f90
do n=1,efp_digits
  ival = int(rs * I_pr(n), kind=int64)
  rs = rs - ival * pr(n)
  e(n) = sgn * ival
enddo
```

where `pr = [r_prec**2, r_prec, 1., r_prec**(-1), r_prec**(-2), r_prec**(-3)]` (`:56-57`, with
`r_prec = 2.**prec_width` the *real* value of `prec`, `:49`) — i.e. digit 3 holds the
"integer part" scale, digits 1-2 hold larger magnitudes (up to `max_efp_float = pr(1)*huge(1_int64)`,
`:64`), and digits 4-6 hold successively finer fractional remainders. Because `int()` truncation and
subtraction are exact for values representable at that scale, **the decomposition of one `real` into
6 `int64`s is exact** (no rounding is introduced beyond the fixed truncation to `prec_width` bits per
digit, which is the deliberate finite precision of the scheme, not an order-dependent error).

**Guard bits / max_summands carry budget.** Each digit is stored in a 63-bit-capacity `int64`, but
only the low 46 bits (`prec_width`) are "normalized" content — the high 17 bits (`guard_width`) are
headroom for carry accumulation. Summing `N` per-element digit values can overflow a single digit by
at most `N-1` (one carry unit per addition), so as long as `N <= 2**guard_width = max_summands`
(131072), the accumulated carry cannot overflow the `int64` container before it is redistributed
downward into the next-more-significant digit by `carry_overflow` (`:825-845`). This is the exact
comment at `MOM_coms.F90:38-40`, quoted above.

**Exact integer summation.** Once every real is an array of 6 `int64`s, summing many reals reduces to
6 independent columns of **exact integer addition**. Integer addition on a fixed-width machine word is
associative and commutative up to overflow (`a+b+c` gives the same bit pattern in any grouping/order,
provided no intermediate overflows) — unlike IEEE float addition, which is neither associative nor
order-invariant due to rounding. That is the entire trick: convert an inexact, order-sensitive
floating sum into an exact, order-insensitive integer sum, then convert back.

**Reconstruction** (`ints_to_real`, `:570-579`):

```f90
function ints_to_real(ints) result(r)
  integer(kind=int64), dimension(efp_digits), intent(in) :: ints
  real :: r
  integer :: i
  r = 0.0
  do i=1,efp_digits ; r = r + pr(i)*ints(i) ; enddo
end function ints_to_real
```

`EFP_to_real` (`:937-943`) is a thin wrapper, but note it first calls `regularize_ints(EFP1%v)`
(`:941`) — which carries overflow *and* forces every digit to the sign of the overall value
(`regularize_ints`, `:849-887`) — **before** `ints_to_real(EFP1%v)` (this is why `EFP1` is
`intent(inout)`: it is normalized in place). Reconstruction sums only 6
terms of geometrically separated magnitude (`pr(i)` spans `prec**2` down to `prec**-3`), so this final
FP summation step is itself insensitive to evaluation order in practice (fixed 6-term unrolled loop —
same order every time, on every platform, by construction) and is *not* where reproducibility would be
at risk even if it were reordered, because the loop is always a straight-line 6-iteration unroll,
never parallelized or reduced.

**Why this is order/PE-count independent.** The full pipeline is: decompose each element to 6 exact
integers -> sum the integer columns (associative, exact) -> redistribute carries (exact, deterministic
given the summed magnitude only, not the summation order) -> reconstruct one `real` from the 6 final
digits (fixed unrolled order). Nowhere in this pipeline does the *order* in which array elements were
visited affect the final bit pattern — only the *total* per-digit integer sum matters, and integer
addition of a fixed set of addends yields a fixed total regardless of association order (so long as no
digit's running sum exceeds the guard-bit budget, which is what `max_summands`/block partitioning
guarantees, see §2). This makes the sum simultaneously: independent of loop iteration order,
independent of PE count / domain decomposition (`sum_across_PEs(ints_sum, efp_digits)`, `:226`, is
itself an exact integer all-reduce), and independent of GPU thread/team scheduling.

---

## 2. GPU block restructuring: `increment_block_ints` (`MOM_coms.F90:618-772`)

Commit `8593a732a` ("MOM_coms: GPU port of block-based repro sum", by Marshall Ward) rewrote the
inner summation loop to run as a `do concurrent` GPU reduction while still respecting the
`max_summands` carry-overflow bound, for domains of **any** size. The previous (pre-GPU) code chose
between three CPU code paths based on array size (`increment_ints_2d`, `increment_ints_faster`,
scalar `increment_ints`+`real_to_ints`) — see §3 for why one of those paths was the source of a
reproducibility bug on branch `fix/nan_repro_sum`.

### 2.1 Partitioning math (`:665-704`)

The compute domain (`ni x nj` elements) is split into rectangular blocks small enough that **no
single block's do-concurrent reduction can overflow the carry budget**, accounting for the two
"cumulant" additions (`block_sum -> array_sum`, `array_sum -> ints_sum`) that also consume carry
headroom:

```f90
max_sum_count = max_summands - 2                      ! :671, reserve headroom for the 2 cumulant adds

ni = ie - is + 1 ; nj = je - js + 1                   ! :674-675, compute-domain size

! Partition in i so that the widest i-slice fits within max_sum_count.
niblocks = (ni + max_sum_count - 1) / max_sum_count   ! :678  = ceil(ni / max_sum_count)

isize_max = (ni + niblocks - 1) / niblocks            ! :686  = ceil(ni / niblocks)

! Set jsize so that the widest i-slice times the number of j-rows does not exceed max_sum_count.
jsize = max_sum_count / isize_max                     ! :691  = floor(max_sum_count / isize_max)

njblocks = (nj + jsize - 1) / jsize                    ! :695  = ceil(nj / jsize)

nblocks = niblocks * njblocks                          ! :698

if (nblocks > max_sum_count) call MOM_error(FATAL, &
    "reproducing sum: Number of blocks exceeds summmation carry limit.")  ! :702-704
```

For the default `guard_width=17` (`max_summands = 131072`), `niblocks` is "typically one" (comment
at `:681`) since 131072 columns vastly exceeds typical tile widths; the blocking logic only engages
for very large domains. The carry budget is per `increment_block_ints` *call*, and each call sums a
**single 2-D `(i,j)` slice** — `ni x nj` elements only. The vertical dimension does **not** add to a
call's carry budget: `reproducing_sum_3d` (`:353-523`) calls `increment_block_ints` once per k-layer
(`:434`) into a **separate per-layer accumulator** `ints_sums(:,k)` (`:386, 432, 435`), *not* a shared
`ints_sum`, so k never enters the `max_sum_count` arithmetic. (Correcting a natural misreading: many
k-layers do not force blocking — only a horizontally huge slice does.) The final
`if (nblocks > max_sum_count)` guard (`:702`) is the hard ceiling — "over 17 billion points per PE"
for default settings, per the code comment (`niblocks*njblocks <= max_sum_count`, i.e. a slice with
more than `(2^17-2)^2 ~= 1.7e10` points).

### 2.2 The kernel — quoted in full context (`:706-772`)

```f90
array_sum(:) = 0

do jb=1,njblocks ; do ib=1,niblocks
  ! Use evenly distributed blocks, either floor(n / nblocks) or ceil(n / nblocks).
  jbs = js + ((jb - 1) * nj) / njblocks
  jbe = js + (jb * nj) / njblocks - 1
  ibs = is + ((ib - 1) * ni) / niblocks
  ibe = is + (ib * ni) / niblocks - 1

  block_sum(:) = 0
  block_max_pos = 0. ; block_max_neg = 0.

  ! Compute the sum of each block
  do concurrent (j=jbs:jbe, i=ibs:ibe) &
      DO_LOCALITY(local(r, e, rmag, lnan, lovf)) &
      DO_LOCALITY(reduce(+: block_sum)) &
      DO_LOCALITY(reduce(max: block_max_pos, block_max_neg, inan, iovf))

    ! Convert array(i,j) to EFP form
    r = descale * array(i,j)
    call efp_decompose(r, e, rmag, lnan, lovf)

    inan = max(inan, lnan)
    iovf = max(iovf, lovf)

    if (r >= 0.) then
      if (rmag > block_max_pos) block_max_pos = rmag
    else
      if (rmag > block_max_neg) block_max_neg = rmag
    endif

    ! Add the EFP result (including potential carry bits)
    block_sum(:) = block_sum(:) + e(:)
  enddo ; enddo

  array_sum(:) = array_sum(:) + block_sum(:)

  ! Redistribute carry bits across bins
  ! For the final pass (or single pass) this is handled by ints_sum.
  b = (jb - 1) * niblocks + ib
  if (b < nblocks) call carry_overflow(array_sum, prec_error)

  max_pos = max(max_pos, block_max_pos)
  max_neg = max(max_neg, block_max_neg)
enddo

ints_sum(:) = ints_sum(:) + array_sum(:)
call carry_overflow(ints_sum, prec_error)
```

### 2.3 Why `reduce(+:block_sum)` over exact integers is bit-identical regardless of scheduling

`block_sum` is an `integer(kind=int64), dimension(efp_digits)` accumulator, and `e(:)` (one element's
EFP decomposition) is likewise exact int64s. The `do concurrent ... DO_LOCALITY(reduce(+: block_sum))`
directive (`DO_LOCALITY(X)` expands to `X` when `HAVE_FC_DO_CONCURRENT_LOCAL` is defined, else to a
bare `;` no-op — `do_concurrent_compat.h:6-10`; `reduce`/`local` are **Fortran 2023 `do concurrent`
locality specifiers**, part of the base language, *not* OpenMP clauses) tells the compiler it may
split the
reduction across threads/teams/warps in any grouping and combine partial sums in any order — which is
exactly the freedom **integer addition** tolerates without changing the result, given the block was
sized so no intermediate partial sum can exceed the guard-bit budget (§2.1). Contrast this with the
`reduce(max: block_max_pos, block_max_neg, inan, iovf)` reduction on the same line: `max` is also
associative/commutative and exact for reals (no rounding in comparing magnitudes), so it is equally
safe to parallelize — it is used only for overflow/NaN bookkeeping and the diagnostic "largest term"
message, never for the sum itself. **No floating-point `+` reduction ever appears in this kernel** —
every quantity that GPU threads reduce into (`block_sum`, `max`/`min` trackers) is either an exact
integer or an exact-comparison real max, which is why arbitrary thread/team scheduling cannot perturb
a single bit of the final answer.

The one place order still matters is the *serial* host loop over blocks (`do jb=1,njblocks; do
ib=1,niblocks`) and the `carry_overflow` calls between blocks (`:749`, `:760`) — but that loop always
executes in the same fixed order (row-major block index `b`) on every run, independent of PE count or
GPU scheduling, so it introduces no non-determinism; it exists purely to keep `array_sum`/`ints_sum`
from overflowing between blocks, not to control summation order for correctness.

**Latent build caveat.** The `HAVE_FC_DO_CONCURRENT_LOCAL` feature test
(`ac/m4/mom6_fc_do_concurrent_local.m4:18`) probes only a `local(a,b)` specifier — it does **not**
compile-test the `reduce(...)` specifier this kernel actually relies on (the m4 comment at `:4-5`
notes `LOCAL_INIT`, `SHARED`, `DEFAULT(NONE)` are also untested). So on a hypothetical compiler that
accepts `local` but not `reduce`, the macro would be defined and the `reduce(+:block_sum)` /
`reduce(max:...)` clauses would be emitted and fail to compile. This is not a reproducibility bug, but
it is the load-bearing assumption behind the whole GPU path: the port presumes a compiler (nvfortran)
where `local` support implies `reduce` support.

---

## 3. `efp_decompose`: `pure` + `declare target`, and the `fix/nan_repro_sum` lesson

### 3.1 Why `pure` + `!$omp declare target`, and flags instead of module globals

```f90
!> Decompose one real into its 6 signed EFP bin contributions.  NaNs and
!! overflows are reported by flags, rather than the module-level error
!! logicals, so that the routine is free of side effects.
pure subroutine efp_decompose(r, e, rmag, is_nan, is_ovf)
  !$omp declare target
  real, intent(in)  :: r
  integer(kind=int64), intent(out) :: e(efp_digits)
  real, intent(out) :: rmag
  integer, intent(out) :: is_nan
  integer, intent(out) :: is_ovf
  ...
end subroutine efp_decompose
```
(`MOM_coms.F90:775-821`)

The module also carries two **module-level** `logical` flags used elsewhere in the file:
`overflow_error` and `NaN_error` (`:71-74`). Fortran's `pure` attribute forbids a procedure from
modifying any entity outside its own dummy-argument list (no writes to module variables, no I/O, no
`stop`), and a `pure` procedure additionally cannot call an `impure` one. `efp_decompose` therefore
cannot set `NaN_error`/`overflow_error` directly — instead it returns `is_nan`/`is_ovf` as ordinary
`intent(out)` integer flags (`1` if a NaN/Inf or an unrepresentable magnitude was seen, else `0`),
which the caller (`increment_block_ints`) folds into thread-local `inan`/`iovf` accumulators via
`DO_LOCALITY(reduce(max: ..., inan, iovf))` (`:724`) and only *after* the parallel region converts them
to the module flags (`:769-771`):

```f90
if (inan /= 0) NaN_error = .true.
if (iovf /= 0) overflow_error = .true.
```

This buys two things simultaneously: (1) `efp_decompose` qualifies as `pure`, which is required for it
to be callable inside a `do concurrent` reduction region and legally `!$omp declare target`-able (a
device-resident routine must not perform host-only side effects like setting a host module variable);
and (2) the NaN/overflow signal is carried out of the parallel region via an *exact-max* reduction
(`inan`, `iovf` are `0`/`1` integers — associative, no rounding), so detecting a NaN anywhere in the
domain is itself scheduling-independent, consistent with the rest of the reproducibility design.

### 3.2 What `fix/nan_repro_sum` changed, and why it couldn't stay `pure`

Two commits on the (unmerged, stale) branch `fix/nan_repro_sum`, branched from `c82e1254a` — an
ancestor *older than* the `8593a732a` block-based rewrite, i.e. this branch still has the
three-way-dispatch pre-GPU-block version of the summation code, not the version described in §2:

- `0ac71d482` "fix nan in repro sum for large domain sizes" — the pre-`8593a732a` code chose between
  `increment_ints_2d` (small tile, on-device `do concurrent`), `increment_ints_faster` (medium tile,
  scalar accumulate), and a fully scalar `increment_ints`+`real_to_ints` loop (large tile) based on
  `(je+1-js)*(ie+1-is)` vs. `max_count_prec`. The large-tile branches called `increment_ints_faster`/
  `real_to_ints`, which had never been ported to run `!$omp declare target` and so implicitly read the
  (device-resident) `array` through host memory — returning NaN whenever `array` lived only on the
  GPU. The commit message: *"Larger domains triggered increment_ints_faster which was not ported.
  Folded the routine into a single one and labelled it pure to reduce bloat."* The fix collapsed all
  three paths into one `increment_ints_2d` that internally chunks the flattened `(i,j)` window into
  `csize = max_count_prec - 1`-sized pieces, each summed by a `do concurrent` reduction into a fresh
  chunk accumulator, carried, and folded into the running total — structurally the same "partition
  into carry-safe chunks, do-concurrent-reduce each chunk over exact integers" idea as
  `increment_block_ints` in §2, arrived at independently and by a different author. The routine was
  marked `pure` in this commit.
- `939d06704` "cant be pure" — one commit later, `pure` was **removed** from both `increment_ints_2d`
  and `carry_overflow`:
  ```f90
  -pure subroutine increment_ints_2d(array, is, ie, js, je, descale, ints_sum, max_mag_term, prec_error)
  +subroutine increment_ints_2d(array, is, ie, js, je, descale, ints_sum, max_mag_term, prec_error)
  ...
  -pure subroutine carry_overflow(int_sum, prec_error)
  +subroutine carry_overflow(int_sum, prec_error)
  ```
  **Why it couldn't stay `pure`:** `carry_overflow` sets the module-level `overflow_error = .true.`
  when a carried sum exceeds `prec_error` (`MOM_coms.F90:841-843` in the current tree; same logic
  existed on the branch) — a write to non-local (module) state, which the Fortran standard forbids
  inside a `pure` procedure. `increment_ints_2d` calls `carry_overflow` once per chunk, so it too
  cannot be `pure` (a `pure` procedure may only call other `pure` procedures). This is exactly the
  discipline `efp_decompose` was designed around in §3.1 — return flags through the argument list
  instead of writing a module global — but the `fix/nan_repro_sum` branch's `carry_overflow` was never
  refactored that way, so the compiler (correctly) rejected `pure` on the caller chain. The lesson
  generalizes: **a routine can only be `pure`/`declare target` if every side effect, including
  warning/error flags, is threaded through `intent(out)` dummy arguments — not module variables** —
  which is precisely what `efp_decompose`'s doc comment (`:776-777`, "reported by flags... so that the
  routine is free of side effects") states as a design rule, and what this branch had to relearn the
  hard way for `carry_overflow`.

  Because `fix/nan_repro_sum` predates the `8593a732a` rewrite, the block-based `increment_block_ints`
  in the current `dev/gpu` tree independently avoids the same bug class: it never has a
  size-dependent CPU/host fallback branch, so there is no code path in the current tree that silently
  reads a device-resident array through host memory. The branch is best read as a documented case study
  in the purity constraint, not as an outstanding patch that still needs to land.

---

## 4. Where reproducing sums are invoked in the timestep

Per `00-architecture.md` §4.2, the pure dycore compute kernels (`continuity_PPM`, `CorAdCalc`,
`PressureForce_FV`, `hor_visc`) contain **no reproducing sums** — communication and diagnostics are
hoisted out of the hot compute path into the driver and the diagnostics layer. Confirmed call sites:

- **`src/diagnostics/MOM_sum_output.F90`, `write_energy`** — the periodic (not every-timestep) global
  energy/mass/heat/salt diagnostic:
  - `:566` — `mass_tot = reproducing_sum(tmp1, ..., sums=mass_lay, EFP_sum=mass_EFP, unscale=...)`
    (total ocean mass + per-layer masses)
  - `:576` — `vol_tot = reproducing_sum(tmp1, ..., sums=vol_lay, unscale=...)` (non-Boussinesq volume)
  - `:746` — `PE_tot = reproducing_sum(PE_pt, ..., sums=PE, unscale=RZL4_T2_to_J)` (potential energy)
  - `:758` — `KE_tot = reproducing_sum(tmp1, ..., sums=KE, unscale=RZL4_T2_to_J)` (kinetic energy)
  - `:770-773` — `salt_EFP = reproducing_sum_EFP(Salt_int, ...)`, `heat_EFP = reproducing_sum_EFP(Temp_int, ...)`
    (returned as `EFP_type` so they can be exactly accumulated across calls before conversion to real)
  - `:776-781` — the salt/heat EFP values plus three running-total `CS%*_EFP` fields are packed into a
    5-element `EFP_type` array and reduced across PEs in one call, `EFP_sum_across_PEs(EFP_list, 5)`
    (`:778`) — "Combining the sums avoids multiple blocking all-PE updates" (comment at `:775`).
- **`src/core/MOM_forcing_type.F90`, forcing/flux diagnostics** — every "total_*" (area-integrated) and
  "*_ga" (area-averaged) diagnostic funnels through `MOM_spatial_means`, which itself calls
  `reproducing_sum` once per invocation:
  - `global_area_integral` calls sites at `:2848, 2875, 2908, 2919, 2933, 2945, 2957, 2969, 2981, 2989,
    2997, 3005, 3026, 3034, 3041, 3047, 3054, 3061, 3068, 3075, 3082` (net P-E, net mass in/out, evap,
    precip, runoff, and every heat_content_* term) — each ultimately reaches
    `reproducing_sum(tmpForSumming, unscale=temp_scale)` at `src/diagnostics/MOM_spatial_means.F90:234`.
  - `global_area_mean` calls at `:2852, 2923, 2937, 2949, 2961, 2973` -> `reproducing_sum(...) *
    G%IareaT_global` at `MOM_spatial_means.F90:84`.
- **`src/diagnostics/MOM_checksums.F90`** also calls `reproducing_sum` itself, for the `aMean` statistic
  reported alongside every bitcount checksum (`subStats`, `MOM_checksums.F90:550`:
  `aMean = reproducing_sum(array(HI%isc:HI%iec,HI%jsc:HI%jec))`).

**Net picture:** reproducing sums appear only in (a) the low-frequency `write_energy` global-budget
diagnostic and (b) forcing/flux area-integral diagnostics, both of which run far less often than the
per-timestep dycore kernels and both of which are *diagnostic outputs*, not part of the prognostic
state update — consistent with §4.2 of `00-architecture.md` ("the pure compute kernels ... contain no
... reproducing sums by design").

---

## 5. `MOM_checksums.F90` — verifying a port bit-for-bit

### 5.1 The bitcount checksum (`popcnt` mod 10^9)

```f90
integer, parameter :: bc_modulus = 1000000000 !< Modulus of checksum bitcount    ! :112

!> Does a bitcount of a number by first casting to an integer and then using BTEST
!! to check bit by bit
integer function bitcount(x)
  real, intent(in) :: x
  integer, parameter :: xk = kind(x)
  ! NOTE: Assumes that reals and integers of kind=xk are the same size
  bitcount = popcnt(transfer(x, 1_xk))
end function bitcount
```
(`MOM_checksums.F90:2678-2687`)

`popcnt` (population count, i.e. number of set bits) applied to the raw bit pattern of a `real`
(reinterpreted via `transfer` as an integer of the same storage size) is an **exact fingerprint of the
IEEE bit pattern** — it is not a numeric function of the value, it's a function of the bits. Any
change to even the last mantissa bit (a genuine bitwise-non-reproducibility bug — rounding difference,
reordered FP op, different fused-multiply-add contraction, etc.) changes `popcnt` and thus the
checksum; conversely, if two runs (e.g., CPU vs. GPU, or before/after a k-blocking refactor) print
identical checksums for every field, the fields are bit-identical, not merely "close." The per-element
`bitcount` results are summed with plain **exact integer addition** (`subchk = subchk + bc`,
`:527` in `chksum_h_2d`'s internal `subchk` function) and then reduced across PEs with
`sum_across_PEs(subchk)` (`:529`, also an exact integer reduction) before being folded into a fixed
range with `mod(subchk, bc_modulus)` (`:530`) purely so the printed number stays a manageable ~9-digit
value — the same exact-integer-reduction principle as the EFP reproducing sum (§1-2), applied here to
a diagnostic fingerprint instead of a physical total. This pattern (`subchk`/`bitcount`/`bc_modulus`)
recurs in every stagger-specific checksum routine: `chksum_h_2d` (`:389-557`), `chksum_B_2d`
(`:690-878`), `chksum_u_2d` (`:1007-1208`), `chksum_v_2d` (`:1211-1412`), and their 3-D and
pair-checksum (`chksum_pair_h_2d`, `chksum_uv_2d`, etc.) counterparts, exposed through the generic
interfaces `hchksum`/`Bchksum`/`uchksum`/`vchksum`/`qchksum`/`chksum` (`:22-23, 60-77`). Each of these
routines also calls `is_NaN` (generic over 0d/1d/2d/3d, `:85-87`) before checksumming, so a NaN
introduced by a bad port is caught with a `FATAL` error naming the field (`chksum_error(FATAL, 'NaN
detected: '//trim(mesg))`, e.g. `:435-436`) rather than silently corrupting the checksum.

### 5.2 Device -> host checksum transfers (`b29b27150`)

Checksum routines are host-only code (they call `MOM_error`, do character formatting, and write to
`error_unit`) and are never `!$omp declare target`. Any array that is device-resident (mapped via
`enter data`/`omp target` and updated only by device kernels) must be explicitly synced back to the
host with `!$omp target update from(...)` **immediately before** it is passed into a checksum call, or
the checksum silently reads stale/uninitialized host memory instead of the current device values —
which looks like a reproducibility failure but is actually a missing transfer. Commit `b29b27150`
("btstep: Update GPU checksum transfers", `src/core/MOM_barotropic.F90`, +8 lines) added exactly these
missing update directives ahead of debug checksums inside `btstep`:

```f90
!$omp target update from(CS%q_D)
call Bchksum(CS%q_D, "BT PV (q_D)", CS%debug_BT_HI, ...)
...
!$omp target update from(q)
call Bchksum(q, "BT PV (q)", CS%debug_BT_HI, ...)
!$omp target update from(DCor_u, DCor_v)
call uvchksum("BT DCor_[uv]", DCor_u, DCor_v, G%HI, ...)
!$omp target update from(Cor_ref_u, Cor_ref_v)
call uvchksum("BT Cor_ref_[uv]", Cor_ref_u, Cor_ref_v, CS%debug_BT_HI, ...)
!$omp target update from(uhbt0, vhbt0)
call uvchksum("BT [uv]hbt0", uhbt0, vhbt0, CS%debug_BT_HI, ...)
...
!$omp target update from(visc_rem_u, visc_rem_v)
call uvchksum("BT visc_rem_[uv]", visc_rem_u, visc_rem_v, G%HI, ...)
!$omp target update from(bc_accel_u, bc_accel_v)
call uvchksum("BT bc_accel_[uv]", bc_accel_u, bc_accel_v, G%HI, ...)
!$omp target update from(CS%IDatu, CS%IDatv)
call uvchksum("BT IDat[uv]", CS%IDatu, CS%IDatv, G%HI, ...)
```

**Porting rule:** every debug/diagnostic checksum call on a mapped array needs its own `target update
from` immediately upstream (an existing transfer for a *different* array, e.g. the pre-existing
`!$omp target update from(CS%frhatu, CS%frhatv)` a few lines above in the same routine, does **not**
cover a different array). This is a distinct failure mode from a genuine reproducibility bug and
should be the first thing checked when a checksum "mismatch" appears after porting a new kernel.

### 5.3 Rotated-grid checksums

`MOM_checksums` supports comparing a field computed on a quarter-turned ("rotated") test grid against
the same field on the canonical grid, a MOM6 technique for catching orientation-dependent bugs (stencil
asymmetries, sign errors in vector components, etc.) — a different axis of "reproducibility" from
GPU-vs-CPU but implemented with the same machinery. Each stagger-specific checksum routine takes the
input array on the *model's* (possibly rotated) index space and un-rotates it before checksumming, e.g.
`chksum_h_2d` (`MOM_checksums.F90:389-432`):

```f90
turns = HI_m%turns
if (modulo(turns, 4) /= 0) then
  allocate(HI)
  call rotate_hor_index(HI_m, -turns, HI)
  allocate(array(HI%isd:HI%ied, HI%jsd:HI%jed))
  call rotate_array(array_m, -turns, array)
else
  HI => HI_m
  array => array_m
endif
```

`rotate_array`/`rotate_array_pair`/`rotate_vector` (imported from `MOM_array_transform`,
`MOM_checksums.F90:8-9`) implement the four index-map rotations (`+90`: transpose + row-reverse;
`180`: row+column reversal; `-90`: row-reverse + transpose, per the module doc comment in
`src/framework/MOM_array_transform.F90:6-13`) so that a field computed on a rotated test grid and
checksummed through `chksum_h_2d`/`chksum_B_2d`/`chksum_u_2d`/`chksum_v_2d` is transformed back onto
the canonical index space first — meaning the printed checksum for a "rotated" run and an "unrotated"
run of a correctly-ported, order-preserving kernel should be bit-identical, and any mismatch flags
either a genuine order-of-operations bug or a stencil that implicitly assumes a fixed index direction
(e.g. hard-coded `i+1` where a rotation-safe kernel should use a metric-relative offset).

---

## 6. Order-of-operations rules for a porter (prescriptive)

These are the rules an agent must apply, in order, when porting any module that touches a sum,
average, reduction, or restart-critical total. Each is a decision an agent can execute mechanically;
the grounding for each is cited so it can be re-verified.

### 6.1 Deciding whether a loop is safe to parallelize

1. **Classify the reduction operator before touching the loop.**
   - **`+` over reals → NOT safe. STOP.** A `do concurrent`/`omp target` loop that sums or averages
     real values changes FP operation order the instant it is parallelized (thread/team/warp order is
     not sequential loop order), and IEEE `+` is non-associative. A raw `reduce(+: real_var)` in a
     device loop is a reproducibility bug even if the CPU serial version is "correct".
   - **`max`/`min`/`.and.`/`.or.` over reals or integers → SAFE.** Comparison and logical fold are
     exact (no rounding) and associative, so any grouping gives the same result. The kernel's
     `reduce(max: block_max_pos, block_max_neg, inan, iovf)` (`:724`) is the canonical safe use — it
     carries the largest-magnitude term and the NaN/overflow flags out of the parallel region without
     threatening reproducibility (§2.3, §3.1).
   - **`+` over exact `int64` (or `int64` arrays) → SAFE, *if* the block is carry-bounded.** This is
     the whole EFP trick: `reduce(+: block_sum)` over `integer(int64)` digits is order-invariant
     because integer `+` is associative up to overflow, and the block was sized so no partial sum
     overflows (§2.1). `MOM_checksums`'s `subchk = subchk + bc` over `bitcount` integers (`:527`) is
     the same pattern for a fingerprint.

2. **If you need a reproducible real sum, route it through the EFP path — never hand-roll.** Use
   `reproducing_sum` / `reproducing_sum_EFP` (public interfaces, `MOM_coms.F90:80-90`). Return an
   `EFP_type` (via `reproducing_sum_EFP` or the `EFP_sum=` argument) when the total must be
   accumulated across multiple calls or PEs *before* conversion to real — converting to real early
   and re-summing reintroduces FP non-associativity. `write_energy` packs 5 running EFP totals into
   one `EFP_sum_across_PEs(EFP_list, 5)` for exactly this reason (§4, `MOM_sum_output.F90:775-778`).

3. **When a slice is "too large" for one reduction pass, partition into carry-safe blocks and reduce
   each block exactly** (§2.1's `max_sum_count = max_summands - 2`, `niblocks`/`isize_max`/`jsize`/
   `njblocks`). Do **not** add a size-dependent scalar/host fallback branch: that pattern is exactly
   what produced the NaN on `fix/nan_repro_sum` (§3.2), where the large-domain branch called an
   unported routine that read a device-resident array through host memory. In the current tree
   `increment_block_ints` has a single code path for all sizes, by design.

### 6.2 Restructuring without perturbing arithmetic

4. **k-blocking / tiling may change loop *structure* only, never the arithmetic order within a
   column or stencil.** Rewrites must move already-computed scalar operations around; they must not
   re-associate a running accumulation (`00-architecture.md` §5). `increment_block_ints` embodies
   this: it partitions into whole rectangular blocks summed with a fixed, deterministic per-block
   carry step (`carry_overflow` between blocks, `:749`, `:760`) — not an arbitrarily scheduled global
   reduction. The serial host block-loop runs in fixed row-major order every time (§2.3).

5. **Extract shared math into `pure` (host) or `pure` + `!$omp declare target` (device) helpers**
   so host and device evaluate bit-identically from one source (`efp_decompose`, §3.1). This is the
   preferred restructuring tool (`00-architecture.md` §0.2).

6. **A `pure`/`declare target` routine may not write module or host-global state — thread every
   error/warning/overflow flag out through `intent(out)` dummy arguments.** `efp_decompose` returns
   `is_nan`/`is_ovf` integer flags instead of setting the module `NaN_error`/`overflow_error` globals;
   the caller folds them in *after* the parallel region (`:770-771`). The cautionary counter-example
   is `fix/nan_repro_sum`'s `939d06704` ("cant be pure"): `carry_overflow` sets `overflow_error`
   (`:841-843`), so neither it nor any caller that invokes it can be `pure` (§3.2).

### 6.3 Compiler / language hazards that silently break bit-identity

7. **Preserve parentheses; never let a rewrite re-associate an expression.** The Fortran standard
   *forbids* a processor from breaking parenthesized sub-expressions (`(a+b)+c` must not become
   `a+(b+c)`), so parentheses are the porter's tool for pinning evaluation order in the compute
   kernels themselves — do not "simplify" `(a*b) + (c*d)` into a rearranged form when refactoring, and
   do not distribute/factor terms in a reconstruction loop. The EFP reconstruction `r = r + pr(i)*ints(i)`
   (`ints_to_real`, `:578`) is a fixed 6-term unrolled loop precisely so its order is invariant.

8. **FMA contraction is a bitwise hazard across CPU vs. GPU.** Fusing `a*b + c` into a single
   fused-multiply-add changes the rounding (one rounding instead of two), so a kernel that contracts
   on GPU but not on CPU (or vice-versa) yields a different last mantissa bit and a `MOM_checksums`
   mismatch that is *not* an algorithm bug. The CPU reference build enables the FMA instruction
   (`FCFLAGS_OPT = -g -O3 -mavx -mfma`, `.testing/README.rst:147`); no explicit `-ffp-contract` /
   `-Mnofma` pin was found anywhere in `ac/`, the test `Makefile`s, or `.testing/`.
   > **Resolved (2026-07-14):** Contraction *is* pinned in the canonical NVHPC toolchain:
   > `mkmf/templates/ncrc5-nvhpc.mk` (and `ncrc-nvhpc.mk`) put `-Mnofma` — plus `-Mdaz` — in the
   > **base** `FFLAGS`, for all build modes. So bit-identity does not rest on nvfortran and the CPU
   > reference compiler happening to contract identically. The one action item that remains: the site
   > GPU build harness is external to this repo, so confirm it inherits `-Mnofma`. If it does, this
   > hazard is closed; if it does not, the fragility above is live.

9. **The GPU reduction path assumes a compiler where `do concurrent local` implies `reduce`
   support.** The `HAVE_FC_DO_CONCURRENT_LOCAL` autoconf probe tests only `local(a,b)`, not the
   `reduce(...)` specifier the repro kernel emits (§2.3). This is a build-portability, not a
   reproducibility, constraint — but a porter adding a new `DO_LOCALITY(reduce(...))` kernel inherits
   the same assumption.

### 6.4 Verification (always, non-negotiable)

10. **Before any checksum on a device-resident array, emit `!$omp target update from(...)` for that
    exact array.** A missing transfer looks identical to a reproducibility failure but is really a
    stale-host-memory read; an existing transfer for a *different* array does not cover it (§5.2,
    commit `b29b27150`). Check this *first* when a checksum "mismatch" appears after a new port.

11. **Verify every port with `MOM_checksums`** (`hchksum`/`Bchksum`/`uchksum`/`vchksum`), comparing
    CPU vs. GPU and, where the config supports it, rotated vs. unrotated runs (§5.3). The `popcnt`
    bitcount (mod `bc_modulus = 10^9`, §5.1) is an exact bit-pattern fingerprint: it passes only on
    bit-identical fields, never on "acceptably close" ones. A rotated-vs-unrotated mismatch
    additionally flags a stencil that hard-codes an index direction (e.g. literal `i+1`) instead of a
    metric-relative offset.

---

## References

- `src/framework/MOM_coms.F90` — EFP type/params (`:31-113`), `reproducing_EFP_sum_2d` (`:121-232`),
  `reproducing_sum_2d` (`:239-347`), `reproducing_sum_3d` (`:353-523`), `real_to_ints`/`ints_to_real`
  (`:526-579`), `increment_ints`/`increment_block_ints` (`:583-772`), `efp_decompose` (`:778-821`),
  `carry_overflow` (`:825-845`), `regularize_ints` (`:849-887`), `EFP_to_real` (`:937-943`).
- `src/framework/do_concurrent_compat.h` — `DO_LOCALITY(X)` macro (expands to `X` if
  `HAVE_FC_DO_CONCURRENT_LOCAL`, else a no-op semicolon).
- `src/diagnostics/MOM_sum_output.F90` — `write_energy` reproducing-sum call sites (`:566, 576, 746,
  758, 770-781`).
- `src/diagnostics/MOM_spatial_means.F90` — `global_area_mean` (`:40-86`), `global_area_mean_v/_u`
  (`:89-160`), `global_area_integral` (`:166-236`); all wrap `reproducing_sum`.
- `src/core/MOM_forcing_type.F90` — forcing/flux diagnostic call sites (`:2848-3082` and beyond) into
  `global_area_integral`/`global_area_mean`.
- `src/framework/MOM_checksums.F90` — module header/interfaces (`:1-121`), `bc_modulus` (`:112`),
  `chksum_h_2d`/`subchk`/`subStats` (`:389-557`), `bitcount` (`:2678-2687`).
- `src/framework/MOM_array_transform.F90` — `rotate_array`/`rotate_array_pair`/`rotate_vector` module
  header describing the four rotation cases (`:1-60`).
- `config_src/drivers/unit_tests/test_reproducing_sum.F90` — reference unit test: standard vs.
  reproducing vs. fast-reproducing sum agreement, exact analytic sum check, and order-invariance under
  random element swaps (`randomly_swap_elements`, whole file).
- Commits: `8593a732a` (block-based GPU repro-sum rewrite), `be42560d9` and `813dc1f5b` (earlier
  reproducing-sum/`write_energy` porting steps that `8593a732a` supersedes), `b29b27150` (btstep GPU
  checksum transfer fixes). Branch `fix/nan_repro_sum` (`0ac71d482`, `939d06704`) — unmerged, based on
  an ancestor (`c82e1254a`) that predates `8593a732a`; documents the NaN/"cant be pure" lesson rather
  than an outstanding patch.

---

## Verification notes

Verified by an Opus verification agent against `dev/gpu` source + git only (no build/run), 2026-07.

**Confirmed (checked line-by-line against source/git):**
- All EFP parameters and line numbers: `accum_width = digits(1_int64)` (=63) `:31`, `prec_width = 46`
  `:33`, `guard_width = 17` `:35`, `max_summands = 2**17 = 131072` `:42`, `prec = 2**46` `:46`,
  `efp_digits = 6` `:54`, `EFP_type` `:103`. Comment `:38-40` quoted correctly.
- **Carry-budget arithmetic re-derived independently and confirmed:** each EFP digit `e(n>=1)` holds
  `< prec = 2^46`; summing `N` of them keeps `|block_sum| < N*2^46`, which stays inside the signed
  `int64` range (`< 2^63`) iff `N <= 2^17 = max_summands`. `max_sum_count = max_summands - 2` `:671`
  correctly reserves headroom for the two cumulant adds (`block_sum→array_sum`, `array_sum→ints_sum`).
  The `nblocks > max_sum_count` FATAL guard `:702` and the "over 17 billion points per PE"
  (`≈ (2^17-2)^2`) claim check out.
- `increment_block_ints` kernel `:706-772`, `efp_decompose` `pure`+`declare target` `:778-821`,
  `carry_overflow` `:825-845` (sets module `overflow_error` `:841-843`), flag-folding `:770-771`,
  `DO_LOCALITY` macro `do_concurrent_compat.h:6-10`, `sum_across_PEs(ints_sum, efp_digits)` `:226`.
- Git: `8593a732a` (M. Ward, "MOM_coms: GPU port of block-based repro sum"); `b29b27150` (M. Ward,
  "btstep: Update GPU checksum transfers", **+8 lines**, all quoted directives verified against the
  diff, incl. the pre-existing `frhatu/frhatv` transfer that does *not* cover neighbors);
  `fix/nan_repro_sum` = `939d06704` "cant be pure" + `0ac71d482` (author **Jorge Galvez Vallejo**, not
  M. Ward — supports the doc's "different author" claim), parented on `c82e1254a`, confirmed a genuine
  ancestor of `8593a732a`. `0ac71d482`'s message quoted accurately.
- `MOM_checksums`: `bc_modulus = 10^9` `:112`, `bitcount = popcnt(transfer(...))` `:2680-2687`,
  `subchk`/`sum_across_PEs`/`mod` `:527/529/530`, `subStats` `aMean = reproducing_sum(...)` `:550`,
  rotate block `:422-432`; `MOM_array_transform.F90:8-13` rotation cases; call sites in
  `MOM_sum_output.F90` (`:566,576,746,758,770,772,778`) and `MOM_spatial_means.F90` (`:84,234`) all
  confirmed at the stated lines.

**Corrected:**
1. §1 — `EFP_to_real` was described as "a thin wrapper (`ints_to_real`)"; it actually calls
   `regularize_ints` *first* (`:941`), which is why `EFP1` is `intent(inout)`. Clarified.
2. §2.1 — **Substantive:** the claim that 3-D calls engage blocking via "many k-layers accumulating
   into the same `ints_sum`" is wrong. `reproducing_sum_3d` sums each layer into a *separate*
   `ints_sums(:,k)` accumulator (`:386,432,434-435`); k never enters the carry budget. Each
   `increment_block_ints` call sums exactly one `ni×nj` 2-D slice — only horizontal extent can force
   blocking. Rewritten.
3. §2.3 — `reduce`/`local` were called "OpenMP-style"; they are **Fortran 2023 `do concurrent`
   locality specifiers** (base language). Corrected, and added the feature-detection caveat: the
   autoconf probe (`mom6_fc_do_concurrent_local.m4:18`) tests only `local(a,b)`, not `reduce(...)`.
4. §1 — `pr` array was written with `prec`; it uses the *real* `r_prec` (`:56-57`). Fixed. Reference
   line for `regularize_ints` tightened to `:849-887`.

**Enhanced:**
- §6 restructured into prescriptive, agent-executable rules (6.1 classify-the-operator decision;
  6.2 restructuring; 6.3 compiler/language hazards; 6.4 verification). Added: **FMA-contraction**
  hazard (grounded in `-mfma` at `.testing/README.rst:147`; no `-ffp-contract`/`-Mnofma` pin found in
  `ac/`/`.testing/`), **parentheses-preservation** as a Fortran-standard anti-reassociation tool, and
  the `reduce`-clause build-portability assumption.

**Confidence.** High on all EFP/checksum mechanics, the carry math, and git provenance (directly
verified). The FMA hazard is a real and correctly-described class of bug, but it is not currently live:
contraction is pinned by `-Mnofma` in the base `FFLAGS` of the NVHPC mkmf templates (§6.3), so it bites
only if the site GPU build harness fails to inherit that flag.
