#!/bin/bash
# Is the shipped feature layout a bug, and are the shipped prompts entangled with it?
# Per-point argmax only: no aggregation, so this isolates features x prompts.
set -euo pipefail
C=${CLASSES:-all}
for lay in repo tokens; do for pr in repo simple; do
  python part_run.py --classchoice $C --baseline point --layout $lay --prompt $pr \
    --out out/layout_${lay}_${pr}.json
done; done
