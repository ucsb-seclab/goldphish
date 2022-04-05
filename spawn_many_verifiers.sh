#!/bin/bash

# utility for spawning a whole bunch of verifiers
echo "[*] spawning $(($2 - $1 + 1)) verifiers";

for i in `seq $1 $2`; do
    docker run \
        --name "arbitrage-verify$i" \
        --network ethereum-measurement-net \
        -v/home/robert/Source/goldphish:/mnt/goldphish \
        -it \
        --init \
        --cpuset-cpus "$i" \
        -d \
        --rm \
        ethereum-arb python3 -m backtest.top_of_block \
            --mode verify \
            --worker-name "verify$i";
done

