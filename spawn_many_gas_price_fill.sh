#!/bin/bash

# utility for spawning a whole bunch of relayers
N_WORKERS=$1

echo "[*] spawning $N_WORKERS gas price oracle fillers";

if [[ "$PREFIX" -ne '' ]];
then
    PREFIX="$PREFIX-";
fi;

seq 0 "$(($N_WORKERS - 1))" | parallel --halt now,fail=1 --nice -10 -j $N_WORKERS --ungroup python3 -m backtest.gather_samples.fill_naive_gas_price --id {} --n-workers $N_WORKERS
