#!/bin/bash

N_WORKERS=$1

echo "[*] spawning $N_WORKERS coinbase transfer fillers";

if [[ "$PREFIX" -ne '' ]];
then
    PREFIX="$PREFIX-";
fi;

seq 0 "$(($N_WORKERS - 1))" | parallel --halt now,fail=1 --nice -10 -j $N_WORKERS --ungroup python3 -m backtest.gather_samples.fill_coinbase_xfers --id={} --n-workers=$N_WORKERS
