#!/bin/bash

# utility for spawning a whole bunch of searchers
echo "[*] spawning $(($2 - $1 + 1)) searchers";

N_WORKERS=$(($2 - $1 + 1))

echo "[*] spawning $N_WORKERS searchers";

PREFIX=$3

if [[ "$PREFIX" -ne '' ]];
then
    PREFIX="$PREFIX-";
fi;

seq $1 $2 | parallel --halt now,fail=1 --nice -10 -j $N_WORKERS --ungroup python3 -m backtest.top_of_block --worker-name "${PREFIX}searcher-{}"
