#!/usr/bin/env bash
# callees.sh <file.F90> <routine-name>
#
# Every `call` made by one routine, with the LOOP DEPTH at the call site and where the callee is
# defined. Depth is the design-critical column:
#
#   DEPTH 0   host orchestration, outside any loop -- free, needs no device work
#   DEPTH >0  inside the loop nest -- to port that loop, this callee MUST become device-callable
#             (pure/elemental + declare target, or force-inlined), or be hoisted out first.
#
# WHERE resolves the definition: in-file (with pure/elemental noted), or the module that defines
# it, or HOST-SINK for known host-only consumers.
#
# LIMITATION: only `call` statements. Fortran function references are indistinguishable from array
# indexing without a symbol table, so device-relevant *functions* (EOS elementals, ratio_max, ...)
# will NOT appear here -- read the loop bodies for those.

set -euo pipefail
[ "$#" -ge 2 ] || { echo "usage: $(basename "$0") <file.F90> <routine-name>" >&2; exit 2; }
file=$1; want=$2
[ -r "$file" ] || { echo "cannot read: $file" >&2; exit 2; }

# Resolve callee definitions against the repo's src/ tree, not the caller's cwd -- otherwise every
# WHERE silently degrades to "?", which reads as "external" and is a false signal.
ROOT=$(cd "$(dirname "$file")" && git rev-parse --show-toplevel 2>/dev/null || true)
if [ -z "${ROOT:-}" ] || [ ! -d "$ROOT/src" ]; then ROOT=.; fi
[ -d "$ROOT/src" ] || echo "warning: no src/ tree found from $file -- WHERE will be unresolved" >&2

HOST_SINKS='post_data|pass_var|pass_vector|start_group_pass|complete_group_pass|hchksum|uchksum|vchksum|uvchksum|bchksum|chksum|save_restart|register_restart_field|register_diag_field|max_across_pes|min_across_pes|sum_across_pes|global_area_mean|global_area_integral|mom_error|forcing_singlepointprint|get_param|cpu_clock_begin|cpu_clock_end|calltree_enter|calltree_leave|calltree_waypoint'

awk -v want="$(printf '%s' "$2" | tr 'A-Z' 'a-z')" -v sinks="$HOST_SINKS" '
{
  raw = $0; line = tolower($0)
  sub(/!.*/, "", line); gsub(/"[^"]*"/, "", line); gsub(/'"'"'[^'"'"']*'"'"'/, "", line)

  if (line ~ /^[ \t]*end[ \t]*(subroutine|function)/) { if (inr) exit; next }
  if (line !~ /^[ \t]*end[ \t]/ && line ~ ("(^|[ \t])(subroutine|function)[ \t]+" want "([ \t]*\\(|[ \t]*$)")) {
    inr = 1; sp = 0; next
  }
  if (!inr) next

  nclose = gsub(/(^|[;[:space:]])end[[:space:]]*do([;[:space:]]|$)/, " ", line)
  nopen_line = line
  nopen  = gsub(/(^|[;[:space:]])do([[:space:]]|$)/, " ", line)
  for (n = 0; n < nclose; n++) if (sp > 0) sp--

  tmp = line
  while (match(tmp, /(^|[;[:space:]])call[[:space:]]+[a-z_][a-z_0-9]*/)) {
    s = substr(tmp, RSTART, RLENGTH); sub(/^.*call[[:space:]]+/, "", s)
    key = s
    if (!(key in cnt)) { order[++nord] = key; firstline[key] = NR; maxd[key] = sp }
    if (sp > maxd[key]) maxd[key] = sp          # deepest call site is what constrains the port
    cnt[key]++
    tmp = substr(tmp, RSTART + RLENGTH)
  }
  for (n = 0; n < nopen; n++) sp++
}
END {
  printf "%-40s %5s %7s  %s\n", "CALLEE", "N", "MAXDEP", "FIRST"
  for (i = 1; i <= nord; i++) {
    k = order[i]
    printf "%-40s %5d %7d  %d\n", k, cnt[k], maxd[k], firstline[k]
  }
}
' "$file" | while IFS= read -r row; do
  name=$(printf '%s' "$row" | awk '{print $1}')
  case "$name" in
    CALLEE) printf '%s  %s\n' "$row" "WHERE"; continue ;;
  esac
  low=$(printf '%s' "$name" | tr 'A-Z' 'a-z')
  if printf '%s' "$low" | grep -qE "^($HOST_SINKS)$"; then
    where="HOST-SINK"
  elif def=$(grep -niE "^ *([a-z_ ()]* )?(subroutine|function) +$name *\(" "$file" | head -1); then
    if printf '%s' "$def" | grep -qiE "(pure|elemental)"; then where="in-file:$(printf '%s' "$def" | cut -d: -f1) (pure)"
    else where="in-file:$(printf '%s' "$def" | cut -d: -f1)"; fi
  else
    hit=$(grep -rliE "^ *([a-z_ ()]* )?(subroutine|function) +$name *\(" "$ROOT/src" 2>/dev/null | head -1 || true)
    if [ -n "$hit" ]; then
      where=${hit#$ROOT/}
    elif grep -rqiE "^ *interface +$name *$" "$ROOT/src" 2>/dev/null; then
      # No concrete definition, but a generic interface exists: the call dispatches through it.
      # On the device that is the doc 06 polymorphic hazard -- resolve which specific it binds to.
      where="GENERIC-INTERFACE (polymorphic hazard, doc 06)"
    else
      where="? (external/pkg, or generic interface)"
    fi
  fi
  printf '%s  %s\n' "$row" "$where"
done
