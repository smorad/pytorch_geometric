#!/usr/bin/env bash

CONFIG=graphconv_genagg
GRID=genagg
REPEAT=1
MAX_JOBS=4
SLEEP=1
MAIN=main

# generate configs (after controlling computational budget)
# please remove --config_budget, if don't control computational budget
python3 configs_gen.py --config configs/pyg/${CONFIG}.yaml \
  --grid grids/pyg/${GRID}.txt \
  --out_dir configs
#python configs_gen.py --config configs/ChemKG/${CONFIG}.yaml --config_budget configs/ChemKG/${CONFIG}.yaml --grid grids/ChemKG/${GRID}.txt --out_dir configs
# run batch of configs
# Args: config_dir, num of repeats, max jobs running, sleep time
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN
# rerun missed / stopped experiments
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN
# rerun missed / stopped experiments
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN

# aggregate results for the batch
python3 agg_batch.py --dir results/${CONFIG}_grid_${GRID}
