#!/bin/bash
# End task for the two partitions worth shipping, at their good resolutions.
set -euo pipefail
C=${CLASSES:-all}
R="python part_run.py --classchoice $C --baseline meanpool"
for np in 32 48 64; do
  $R --part spectral --embed sparse --n_sp $np --tag "sparse/$np" --out out/f_sp_$np.json
done
$R --n_sp 96  --tag "km+ref/96"  --out out/f_km_96.json
python part_run.py --classchoice $C --baseline oracle --part spectral --embed sparse --n_sp 48 \
  --tag "orc-sparse/48" --out out/f_orcs_48.json
