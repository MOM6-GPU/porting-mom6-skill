#!/usr/bin/env bash
# routine-map.sh <file.F90>
#
# Triage table: one row per procedure, sorted by MAXLOOP (the longest single loop body).
# MAXLOOP is the shape metric that matters -- a routine that is 600 lines of small loops ports
# fine; a routine with one 200-line loop body is the "1500 lines of a single do loop" problem
# and wants a verbatim extraction before any directive goes near it. See SKILL.md Step 3.
#
#   LINES    total lines in the procedure
#   MAXLOOP  longest loop body (lines between a do and its matching enddo)
#   DEPTH    deepest loop nesting
#   CALLS    number of `call` statements
#   PURE     procedure is already pure/elemental (cheap to make device-callable)
#
# Heuristic reading aid, not a parser. Fortran is case-insensitive; comments and string
# literals are stripped before counting so `MOM_error("... do ...")` cannot fake a loop.

set -euo pipefail
[ "$#" -ge 1 ] || { echo "usage: $(basename "$0") <file.F90>" >&2; exit 2; }
[ -r "$1" ] || { echo "cannot read: $1" >&2; exit 2; }

awk '
function flush_routine() {
  if (name != "") printf "%6d %8d %6d %6d  %-4s %6d  %s\n", NR-start, maxloop, maxdepth, ncalls, (ispure?"pure":"-"), start, name
  name = ""; maxloop = 0; maxdepth = 0; ncalls = 0; ispure = 0; sp = 0
}
{
  line = tolower($0)
  sub(/!.*/, "", line)                       # strip trailing comment
  gsub(/"[^"]*"/, "", line)                  # strip string literals
  gsub(/'"'"'[^'"'"']*'"'"'/, "", line)

  if (line ~ /^[ \t]*end[ \t]*(subroutine|function)/) { flush_routine(); next }

  if (line !~ /^[ \t]*end[ \t]/ && line ~ /(^|[ \t])(subroutine|function)[ \t]+[a-z_][a-z_0-9]*/ \
      && line !~ /(^|[ \t])(module|abstract|procedure)[ \t]/) {
    if (name != "") flush_routine()
    match(line, /(subroutine|function)[ \t]+[a-z_][a-z_0-9]*/)
    nm = substr(line, RSTART, RLENGTH); sub(/^(subroutine|function)[ \t]+/, "", nm)
    name = nm; start = NR; maxloop = 0; maxdepth = 0; ncalls = 0; sp = 0
    ispure = (line ~ /(^|[ \t])(pure|elemental)[ \t]/)
    next
  }
  if (name == "") next

  ncalls += gsub(/(^|[;[:space:]])call[[:space:]]+[a-z_]/, " ", line)

  nclose = gsub(/(^|[;[:space:]])end[[:space:]]*do([;[:space:]]|$)/, " ", line)
  nopen  = gsub(/(^|[;[:space:]])do([[:space:]]|$)/, " ", line)

  for (n = 0; n < nopen; n++)  { stack[sp++] = NR; if (sp > maxdepth) maxdepth = sp }
  for (n = 0; n < nclose; n++) {
    if (sp > 0) { body = NR - stack[--sp] - 1; if (body > maxloop) maxloop = body }
  }
}
END { flush_routine() }
' "$1" | sort -k2,2rn | awk 'BEGIN{printf "%6s %8s %6s %6s  %-4s %6s  %s\n","LINES","MAXLOOP","DEPTH","CALLS","PURE","START","NAME"} {print}'
