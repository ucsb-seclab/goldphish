#!/bin/bash

# utility for spawning a whole bunch of relayers
N_WORKERS=$1

echo "[*] spawning $N_WORKERS closure fillers";

if [[ "$PREFIX" -ne '' ]];
then
    PREFIX="$PREFIX-";
fi;

seq 0 "$(($N_WORKERS - 1))" | parallel --halt now,fail=1 --nice -10 -j $N_WORKERS --ungroup python3 -m backtest.top_of_block --worker-name "${PREFIX}closure-{}" fill-closure
