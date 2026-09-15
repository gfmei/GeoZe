#!/bin/bash
# VCCS partition on ShapeNetPart: end task and ceiling, against the shipped default.
set -euo pipefail
C=${CLASSES:-all}
R="python part_run.py --classchoice $C"
for np in 32 64; do
  $R --baseline meanpool --part vccs --n_sp $np --tag "vccs/$np" --out out/f_vccs_$np.json
done
# NOTE refine=3 is already the default in best_param_v2, so every row above includes it.
# `--refine 0` is the way to see the partition without boundary refinement.
$R --baseline meanpool --part vccs --n_sp 32 --refine 0 --tag "vccs-noref/32" --out out/f_vccsr_32.json
$R --baseline oracle   --part vccs --n_sp 64 --tag "orc-vccs/64" --out out/f_orcv_64.json
