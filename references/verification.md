# Verifying a port: answers and performance

A port is done when it gives bitwise-identical answers and the CPU build is no slower (within
about 1-2%). Findings are for nvfortran 26.3 unless marked. Tags as in `loop-constructs.md`.

Contents: 1 build flags · 2 answers · 3 keeping arithmetic order fixed · 4 math intrinsics ·
5 performance · 6 intermittent failures ·
7 review before a pull request

## 1. Build flags

Our GPU build: `-mp=gpu -stdpar=gpu -gpu=mem:separate -Mnofma -r8`, at `-O3` (or
`-O0` for debugging). Other references give flags only where they differ.

`-Mnofma` is required for bitwise comparison: without it, host and device contract `a*b + c`
differently and answers differ in the last bit `[run-verified]`. Check that your build's flags
include it. An existing build directory does not pick up changed flags, so delete the affected
`.o` files after changing them.

## 2. Answers

- **`ocean.stats` must match bitwise**, both against the same configuration before the change
  (`dev/gpu`) and between the CPU and GPU builds.
- **`ocean.stats` does not cover passive tracers** such as the ideal age tracer. Restart files
  carry a checksum per field: `ncdump -h RESTART/MOM.res.nc | grep checksum` gives a bitwise check
  of the full end state.
- **To find where answers diverge**, run with `DEBUG = True` and compare the checksum lines
  (`hchksum`, `uvchksum`, ...) between the two runs; the first differing field points at the
  kernel. Checksums run on the host, so each array checksummed after a device write needs its own
  `!$omp target update from(...)` immediately before the call. An update of a different array
  does not cover it. Check this first when a checksum mismatch appears right after a port.
- Some mapping and reduction bugs (`map(alloc:)` on a struct read on the device, a missing
  `reduce`) only show up on more than one GPU. Multi-GPU runs are not part of routine
  verification; suggest one when answers differ only with the decomposition, or when the user
  asks.
- The repository test suite (`.testing`) runs in CI. Run it locally only when the user asks.
- A port can leave `ocean.stats` bitwise identical and still be wrong, if a stale host value only
  reaches control flow (`data-mapping.md` section 5).

## 3. Keeping arithmetic order fixed

- Never reduce reals in parallel (`reduce(+:)`) or split a float sum across blocks
  (`loop-constructs.md` section 5, `blocking.md` section 6). For a reproducible global sum, use
  `reproducing_sum` / `reproducing_sum_EFP` (`MOM_coms.F90`), keeping multi-part totals as
  `EFP_type` until the end.
- Keep a sum in the same loop as the values it sums. Splitting the producer and the
  accumulation into separate loops let ifort re-associate the sum and broke reproducibility
  (`5f413739b`, barotropic `hat[uv]tot`) `[source-only]`.
- Copy expressions unchanged when restructuring. Keep every parenthesis: the compiler may not
  re-associate a parenthesized expression, so parentheses pin the order.
- A helper shared by host and device code must be `pure`, with error and overflow flags returned
  through arguments rather than module variables (`efp_decompose` in `MOM_coms.F90`). A routine
  that writes module state cannot be `pure`.
- Do not add a size- or configuration-dependent branch that falls back to unported host code
  for some inputs: it reads device-resident data through stale host memory.

## 4. Math intrinsics differ between host and device

Even with `-Mnofma`, several intrinsics round differently on the GPU than on the CPU
`[run-verified]` (1e6 points, nvfortran-mres repo `device_calls/intrinsic_bits.F90`):

| Function | Points with different bits |
|---|---|
| `log` | 33% |
| `sin` | 16% |
| `x**(1./3.)` | 0.7% |
| `exp` | 0.5% |
| `tanh` | 2 points |
| `sqrt`, `x**2.5`, integer powers | none |

A kernel that calls one of these cannot match the CPU build bitwise. Where a reproducible
replacement exists in `MOM_intrinsic_functions.F90`, use it in ported code: `exp_repro(x)` for
`exp`, `cuberoot(x)` for `x**(1./3.)`, `nth_root(x, n)` for `x**(1./n)`. These are
`!$omp declare target` and give the same bits on both platforms.

**If a kernel that runs in a target configuration must call an intrinsic with no reproducible
replacement** (`log`, `sin`, `tanh`, ...), warn the user before porting it and ask how to proceed.
Reproducible versions will be written by the team; do not write one yourself.

## 5. Performance

**CPU.** Use the model's `cpu_clock` timers:
1. Find the clock around the routine you are porting: a `cpu_clock_begin(id_clock_X)` /
   `cpu_clock_end` pair at its call site, and a matching name in the end-of-run timing summary.
2. If none exists, add one: an integer id (module variable or CS field, like its neighbours),
   registered in `_init` with `cpu_clock_id(...)`, wrapped around the call site together with the
   call's `target update` directives.
3. Compare the CPU build's clock before and after the port. A slowdown beyond about 1-2% blocks
   the pull request (`blocking.md`).

Clocks at benchmark size scatter by up to about ±25% between runs of the same binary. Take
several runs of each variant, and alternate the variants within one job on the same node, before
reading a difference smaller than that.

**GPU.** Before reading a timing, confirm the code actually runs on the device: `-Minfo=accel`
should report `Generating NVIDIA GPU code` for the loop, and `NVCOMPILER_ACC_NOTIFY=1` should show
`launch CUDA kernel` lines. `NVCOMPILER_ACC_NOTIFY=2` lists every host/device transfer, which
finds implicit copies. A loop that silently runs on the host still gives the right answers
(`device-calls.md` section 1, `polymorphism.md` section 1).

## 6. Intermittent failures

GPU kernels are deterministic, so a failure in some runs and not others almost always means
uninitialized memory is read: a `map(alloc:)` array read before any device write, or a host
array the device wrote but never copied back. Measure a rate instead of reasoning from one run:
alternate candidate and control within one job on one device, keep CPU and GPU runs separate, run
one GPU job per node at a time, and report counts (for example 11/24 vs 0/8).

A run that looks hung often is not: if `nvidia-smi` shows the GPU busy, or the
`NVCOMPILER_ACC_NOTIFY` trace keeps growing, it is still executing, perhaps a loop whose
iteration count comes from garbage data.

## 7. Review before a pull request

- Answers bitwise identical (section 2); CPU clock within about 1-2% (section 5), with block-size
  defaults chosen by measurement, not copied from another module (`blocking.md`).
- Data mapping in as few regions as possible: one `enter data` / `exit data` pair around the
  routine's device work, not one per loop. Consider persistent residency for types whose users
  are now mostly on the device (`data-mapping.md` section 1).
- Every EOS or other polymorphic call on the ported path reaches a device-safe implementation
  for the configurations you target (`polymorphism.md`).
- No leftover debugging: `print *`, `write(0,*)` instrumentation, extra `NVCOMPILER_ACC_NOTIFY`
  scripts, or commented-out experiments.
