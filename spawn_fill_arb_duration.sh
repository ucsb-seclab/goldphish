#!/bin/bash

# utility for spawning a whole bunch of duration finders....... or w/e
N_WORKERS=$1

echo "[*] spawning $N_WORKERS to find arbitrage duration";

if [[ "$PREFIX" -ne '' ]];
then
    PREFIX="$PREFIX-";
fi;

seq 0 "$(($N_WORKERS - 1))" | parallel --halt now,fail=1 --nice -10 -j $N_WORKERS --ungroup python3 -m backtest.top_of_block do-arb-duration  --id={} --n-workers=$N_WORKERS
