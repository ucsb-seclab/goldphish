#!/bin/bash

# utility for spawning a whole bunch of searchers
echo "[*] spawning $(($2 - $1 + 1)) searchers";

for i in `seq $1 $2`; do
    docker run \
        --name "arbitrage-search$i" \
        --network ethereum-measurement-net \
        -v/data/robert/ethereum-arb/storage:/mnt/goldphish \
        -it \
        --init \
        --cpuset-cpus "$i" \
        -d \
        --rm \
        ethereum-arb python3 -m backtest.top_of_block \
            --worker-name "search$i";
done

