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

# the same algorithm on the point kNN graph, batched on the GPU
for np in 32 64; do
  $R --baseline meanpool --part vccs_gpu --n_sp $np --tag "vccs_gpu/$np" --out out/f_vg_$np.json
done
$R --baseline meanpool --part vccs_gpu --n_sp 64 --vccs_boundary 0.5 --tag "vccs_gpu+bnd/64" \
  --out out/f_vgb_64.json
$R --baseline oracle   --part vccs_gpu --n_sp 64 --tag "orc-vccs_gpu/64" --out out/f_orcvg_64.json
