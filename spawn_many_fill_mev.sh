#!/bin/bash

START=$1
END=$2

echo "[*] spawning duration scanners for priority $1 to $2";

if [[ "$PREFIX" -ne '' ]];
then
    PREFIX="$PREFIX-";
fi;

seq $1 $2 | parallel --halt now,fail=1 --nice -10 -j "$(($2 - $1 + 1))" --ungroup python3 -m backtest.top_of_block --worker-name "${PREFIX}mev-finder-{}" find-mev --priority {}
