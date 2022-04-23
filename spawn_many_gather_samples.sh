#!/bin/bash

# utility for spawning a whole bunch of sample gatherers

N_WORKERS=$(($2 - $1 + 1))

echo "[*] spawning $N_WORKERS gatherers";

PREFIX=$3

if [[ "$PREFIX" -ne '' ]];
then
    PREFIX="$PREFIX-";
fi;

seq $1 $2 | parallel -j $N_WORKERS --ungroup taskset -c '{}' python3 -m backtest.gather_samples --worker-name "${PREFIX}gather-samples{}"
