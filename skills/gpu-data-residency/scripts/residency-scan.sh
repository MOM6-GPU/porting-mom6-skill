#!/usr/bin/env bash
# residency-scan.sh <file.F90> <array-name> [more-array-names...]
#
# Prints, in line order, every touch of the named array(s) interleaved with every marker that
# opens a device region, a host-threaded region, a map/transfer directive, or a host-only sink.
# That interleaving is the raw material for the residency ledger (SKILL.md Step 3).
#
# Tags:
#   MAP           !$omp target enter/exit data
#   XFER          !$omp target update to/from
#   DEV-REGION    do concurrent, !$omp target teams/loop  -> touches inside are DEVICE
#   DEV-HALO      do_group_pass(..., omp_offload=.true.)  -> device-resident, no transfer needed
#   HOST-THREADS  !$OMP parallel do                       -> HOST CPU threads, NOT device
#   HOST-SINK     a host-only consumer (see references/host-boundaries.md)
#   TOUCH         a mention of the array
#
# This is a reading aid, not an oracle: it tags the lines that OPEN regions. You still have to read
# the code to see which touches fall inside which region, and Fortran is case-insensitive so name
# matching is too.

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $(basename "$0") <file.F90> <array-name> [more...]" >&2
  exit 2
fi

file=$1; shift
if [ ! -r "$file" ]; then echo "cannot read: $file" >&2; exit 2; fi

names=$(printf '%s|' "$@" | sed 's/|$//')

awk -v names="$names" '
BEGIN { names = tolower(names) }   # Fortran is case-insensitive; source lines are lowercased below
{
  line = tolower($0)
  tag = ""

  if (line ~ /!\$omp[ \t]+target[ \t]+(enter|exit)[ \t]+data/)      tag = "MAP"
  else if (line ~ /!\$omp[ \t]+target[ \t]+update/)                 tag = "XFER"
  else if (line ~ /!\$omp[ \t]+target/)                             tag = "DEV-REGION"
  else if (line ~ /do[ \t]+concurrent/)                             tag = "DEV-REGION"
  else if (line ~ /!\$omp[ \t]+(parallel|do[ \t])/)                 tag = "HOST-THREADS"
  else if (line ~ /call[ \t]+do_group_pass/)
    tag = (line ~ /omp_offload[ \t]*=[ \t]*\.true\./) ? "DEV-HALO" : "HOST-SINK"
  else if (line ~ /call[ \t]+(post_data|pass_var|pass_vector|start_group_pass|complete_group_pass|hchksum|uchksum|vchksum|uvchksum|bchksum|chksum|mom_tracer_chksum|save_restart|save_mom_restart|register_restart_field|max_across_pes|min_across_pes|sum_across_pes|global_area_mean|global_area_integral|reproducing_sum|write_energy|mom_error)/)
    tag = "HOST-SINK"

  # A commented-out directive is not a directive. A live one has "!$omp" as the first
  # non-blank text on the line; anything else in front of it (!!$omp, !**!$omp) disables it.
  if (tag ~ /^(MAP|XFER|DEV-REGION|HOST-THREADS)$/) {
    trimmed = line; sub(/^[ \t]+/, "", trimmed)
    if (trimmed ~ /\$omp/ && trimmed !~ /^!\$omp/) tag = "DISABLED"
  }

  touch = 0
  if (names != "" && line ~ ("(^|[^a-z0-9_%])(" names ")([^a-z0-9_]|$)")) touch = 1

  if (tag == "" && !touch) next
  if (touch) tag = (tag == "") ? "TOUCH" : tag "+TOUCH"
  printf "%-18s %6d  %s\n", tag, NR, $0
}' "$file"
