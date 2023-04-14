#!/bin/bash

N_WORKERS=$1

echo "[*] spawning modification scanners...";

if [[ "$PREFIX" -ne '' ]];
then
    PREFIX="$PREFIX-";
fi;

seq 0 "$(($N_WORKERS - 1))" | parallel --halt now,fail=1 --nice -10 -j $N_WORKERS --ungroup python3 -m backtest.top_of_block --worker-name "${PREFIX}relayer-{}" do-arb-duration --fill-modified --id {} --n-workers $N_WORKERS
