#!/bin/bash

# utility for spawning a whole bunch of relayers
N_WORKERS=$1

echo "[*] spawning many top candidate arbitrage seekers";

if [[ "$PREFIX" -ne '' ]];
then
    PREFIX="$PREFIX-";
fi;

seq 0 $(($N_WORKERS - 1)) | parallel --halt now,fail=1 --nice -10 -j $N_WORKERS --ungroup python3 -m backtest.top_of_block --worker-name "${PREFIX}top_arbs-{}" fill-top-arbs --id {} --n-workers  $N_WORKERS
