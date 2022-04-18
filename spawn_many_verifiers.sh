#!/bin/bash

# utility for spawning a whole bunch of verifiers

N_VERIFIERS=$(($2 - $1 + 1))

echo "[*] spawning $N_VERIFIERS verifiers";

PREFIX=$3

if [[ "$PREFIX" -ne '' ]];
then
    PREFIX="$PREFIX-";
fi;

seq $1 $2 | parallel -j $N_VERIFIERS --ungroup taskset -c '{}' python3 -m backtest.top_of_block --mode verify --worker-name "$PREFIXverify{}"
