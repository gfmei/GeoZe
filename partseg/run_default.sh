#!/bin/bash
# The shipped default partition, at the resolutions that matter, plus its oracle.
set -euo pipefail
C=${CLASSES:-all}
for np in 32 64 128; do
  python part_run.py --classchoice $C --baseline meanpool --n_sp $np --tag "km+ref/$np" \
    --out out/f_km_$np.json
done
python part_run.py --classchoice $C --baseline oracle --n_sp 64 --tag "orc-km/64" \
  --out out/f_orcm_64.json
python part_run.py --classchoice $C --model v2 --n_sp 64 --tag "v2-final" --out out/f_v2.json
