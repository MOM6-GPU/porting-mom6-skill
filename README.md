A work-in-progress Claude skill for porting MOM6 to NVIDIA GPUs with nvfortran.

**Install.** Copy this directory, as a directory, to `.claude/skills/porting-mom6-skill/` in your
MOM6 checkout, or to `~/.claude/skills/porting-mom6-skill/` to make it available everywhere.
Claude Code finds `SKILL.md` one level below `skills/`. This README is not read by the skill.

**Layout.**
- `SKILL.md`: the workflow and the hard rules, loaded when the skill triggers.
- `references/`: one file per topic, read on demand.
- `scripts/`: intentionally empty. Add your own scripts for building, running the target
  configurations, timing, and comparing answers on your system, and mention them in `SKILL.md`.

**Evidence.** Claims are tagged `[run-verified]`, `[source-only]` or `[unverified]`, and are
for nvfortran 26.3 unless marked. Compiler findings point to reproducers in the nvfortran-mres
repository. When you confirm or overturn a claim, update its tag, and keep entries short: the
rule, one line of why, and the evidence.
