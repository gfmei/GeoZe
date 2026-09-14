#!/bin/bash
# The settled comparison.  One code state, every row, so the table is internally consistent.
set -euo pipefail
C=${CLASSES:-all}
R="python part_run.py --classchoice $C"
$R --baseline point                                    --out out/f_point.json
$R --model geoze                                       --out out/f_geoze.json
for np in 16 32 64 128; do
  $R --baseline meanpool --part fps      --n_sp $np --tag "fps/$np"  --out out/f_pool_fps_$np.json
  $R --baseline meanpool --n_sp $np                 --tag "v2/$np"   --out out/f_pool_v2_$np.json
  $R --baseline oracle   --n_sp $np                 --tag "orc/$np"  --out out/f_oracle_v2_$np.json
done
