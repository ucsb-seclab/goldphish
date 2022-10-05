#!/bin/bash

# utility for spawning a whole bunch of sample gatherers

N_WORKERS=$1

echo "[*] spawning $N_WORKERS eth price fill";

PREFIX=$3

if [[ "$PREFIX" -ne '' ]];
then
    PREFIX="$PREFIX-";
fi;

seq 0 "$(($N_WORKERS - 1))" | parallel --halt now,fail=1 --nice -10 -j $N_WORKERS --ungroup python3 -m backtest.gather_samples.fill_eth_price --id {} --n-workers  $N_WORKERS
