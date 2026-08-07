# Calibration: what a portable routine looks like in this tree

Measured on `dev/gpu` HEAD, 2026-07-14. **Regenerate rather than trust** — branches land:

```bash
scripts/routine-map.sh <file.F90>
scripts/callees.sh <file.F90> <routine>
```

## The shape spectrum

| Routine | File | LINES | MAXLOOP | Loop-interior callees | Status |
|---|---|---|---|---|---|
| `find_uv_at_h` | `MOM_diabatic_aux.F90:495` | 123 | 65 | **none** (all MAXDEP 0) | **ported — the template** |
| `set_diffusivity` | `MOM_set_diffusivity.F90:244` | 624 | 269 | 13+, incl. 27× `hchksum` @1, `thickness_to_dz` generic @2 | untouched, Tier-1 #2 |
| `KPP_compute_BLD` | `MOM_CVMix_KPP.F90:996` | 522 | 369 | 6, incl. 3× external `cvmix_kpp_*` @2–3, `calculate_density` generic @2 | untouched, Tier-1 #4 |
| `applyBoundaryFluxesInOut` | `MOM_diabatic_aux.F90:685` | 664 | 452 | 5, incl. `MOM_error` @4, `forcing_SinglePointPrint` @4, `post_data` @1, 2 generics @2 | untouched |
| `ePBL_column` | `MOM_energetic_PBL.F90:896` | 1061 | 701 | 8, **all in-file, no host sinks, no polymorphism** | naive port on `origin/epbl-3d` |

The two ends of the spectrum are the whole lesson:

- **`find_uv_at_h`** — 123 lines, MAXLOOP 65, and its *entire* call list (`cpu_clock_begin`,
  `MOM_error`, `cpu_clock_end`) sits at MAXDEP 0, outside the loops. The loop bodies are
  self-contained arithmetic. That is why it ported cleanly as `target teams loop collapse(2)` +
  serial-k, and why `knowledge/KNOWLEDGE.md` §6 Tier-1 #1 says to copy it verbatim for the diabatic
  tridiagonals.
- **`applyBoundaryFluxesInOut`** — 664 lines, one `do j` at `:871` spanning to `:1324` (452-line
  body), with `MOM_error(FATAL)` and `forcing_SinglePointPrint` **four loops deep** at `:1167`/`:1174`.
  Those are never-do #10 outright. No amount of directive placement fixes it; the arithmetic has to
  come out into a `pure` column kernel with error flags first.

## Why the call list beats the shape metric

`ePBL_column` is the counter-example that stops MAXLOOP from becoming a dumb threshold. It is the
**biggest** loop body in the Tier-1 set (701) — and it is the one with a *clean* interior call
list: 8 callees, all in-file or a single cross-module helper, no host sinks, no generic interfaces.
That is exactly why `origin/epbl-3d`'s naive whole-column `pure` + `do concurrent(j,i)` worked
("100x… ~2ms/step").

`callees.sh` flags its interior callees at MAXDEP 2 as `get_langmuir_number` (cross-module,
`MOM_wave_interface.F90`) and `find_mstar` (in-file `:3522`). `knowledge/KNOWLEDGE.md` §6 Tier-1 #3
independently records that epbl-3d "needed two workarounds to reuse deliberately: **manual inlining
of `get_Langmuir_Number`/`find_mstar`**, and fixed-size column arrays". The tool predicts the
documented workarounds from the call graph alone — which is the evidence that the loop-interior
call list is the right triage input.

Same check on `KPP_compute_BLD`: the tool flags 3 external `cvmix_kpp_*` calls at depth 2–3;
§6 Tier-1 #4 independently says its hazard is that it "calls into external `pkg/CVMix-src` — the
largest `declare target`/inlining surface of Tier-1". Agreement, derived two different ways.

So: **big MAXLOOP means "read the call list before judging", not "refuse".** But note that
epbl-3d's success is only half a result — it is GPU-only, with no CPU-preserving story, so it
fails ground rule 2. A clean call graph tells you the port is *mechanically* possible; ground
rule 2 still decides whether that port is *acceptable*.

## Reading `routine-map.sh` output

```
 LINES  MAXLOOP  DEPTH  CALLS  PURE  START  NAME
   664      452      3     23  -       685  applyboundaryfluxesinout
   123       65      3      3  -       495  find_uv_at_h
```

- **MAXLOOP / LINES ratio** is the monolith tell: 452/664 = 68% of the routine inside one loop.
- **DEPTH** 3–4 with a large MAXLOOP means the extraction target is probably the *innermost*
  full-column body, not the outer `do j`.
- **PURE** already `pure`/`elemental` → cheap to make device-callable; just needs `declare target`.
  Most of `MOM_EOS_Wright.F90`'s elementals show `pure` with MAXLOOP 0 — the shape you are
  refactoring *toward*.
- Both scripts strip comments and string literals before counting, so `MOM_error("... do ...")`
  cannot fake a loop. They are heuristic reading aids, not parsers — verify the boundaries of any
  loop you are about to act on (`sed -n '871p;1324p' <file>`).

## Known blind spots

1. **Function references are invisible to `callees.sh`.** Only `call` statements are caught. EOS
   elementals, `ratio_max`, `cuberoot` are function calls and carry the identical device-callable
   constraint. Read the loop bodies.
2. **`? (external/pkg, or generic interface)`** means the definition isn't under `src/` — usually
   `pkg/` (CVMix) or FMS. Not benign: it is the largest inlining surface there is.
3. **`GENERIC-INTERFACE`** means the call dispatches through a generic. On device that is the
   doc 06 polymorphic hazard — resolve which specific it binds to before designing around it.
4. Neither script understands `#ifdef`. A routine's shape can differ between the CPU and
   `__NVCOMPILER_OPENMP_GPU` builds.
