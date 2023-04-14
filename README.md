# Goldphish - historical ethereum arbitrage analysis

Goldphish is an arbitrage analyzer for the ethereum blockchain.


# Overview

The entrypoints into this project are largely broken into two parts:

1. Historical scrape of performed arbitrages -- found in `backtest/gather_samples`
2. Arbitrage seeking through history -- found in `backtest/top_of_block`

Models for the exchanges are in `pricers/`.

The transaction relayer smart contract is in `contracts/`.

The optimization procedure is in `find_circuit/find.py`.

# Building

This system is dockerized.
To build, run `docker build -t goldphish .`

# Setup

This system requires access to a postgresql database and a go-ethereum (geth) archive node with a websocket JSON-rpc server.
At the time of writing, an archive node consumes around 12 terabytes of space -- be warned!


## Postgresql

We run postgres dockerized. For convenience, we also run it on a separate docker network (so we can get dns-resolution).

To create the network:

```
docker network create ethereum-measurement-net
```

Pull the docker image,

```
docker pull postgres:14
```

Then spawn postgres. **NOTE: we are running with a weak password and open port (BAD!!!)**

Replace `YOUR_DATA_DIR_HERE` with the path to a directory that has at least 2T storage space, this is where your postgresql database files will live.

```
docker run \
    -d \
    --name ethereum-measurement-pg \
    -e POSTGRES_PASSWORD=password \
    -e POSTGRES_USER=measure \
    -e POSTGRES_DB=eth_measure_db \
    -p 0.0.0.0:5410:5432 \
    -v YOUR_DATA_DIR_HERE:/var/lib/postgresql/data \
    --network ethereum-measurement-net \
    postgres:14
```

Let's configure it to allow more connections. Edit `YOUR_DATA_DIR_HERE/postgresql.conf` and change:

```diff
- max_connections = 100
+ max_connections = 1000
...
- shared_buffers = 128MB
+ shared_buffers = 512MB
```

And then restart the container:

```
docker container restart ethereum-measurement-pg
```



# Getting Started

## Storage dir setup

We need a directory to persist some information (and logs), which we will call `STORAGE_DIR`. It should have this structure:

```
.
logs/
tmp/
```

## Setup block samples

We parallelize work by chunks of blocks about 1 day long. Generate this table:

```bash
docker run \
    --rm \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    python3 -m backtest.top_of_block \
    generate-sample
```

You should see logged `generated sample`.

## Scrape exchanges

You will need to scrape the list of Uniswap, Sushiswap, Shibaswap, and Balancer exchanges.

Get started by setting up the database:

```bash
docker run \
    --rm \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    goldphish \
    python3 -m backtest.gather_samples \
    --setup-db
```

And then also here:

```bash
docker run \
    --rm \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    goldphish \
    python3 -m backtest.gather_samples.fill_known_exchanges \
    --setup-db
```

And finally, run the scrape:

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    python3 -m backtest.gather_samples.fill_known_exchanges
```

You should see an ETA printed.

## Scrape ethereum price

This scrapes the USD price of ETH using either the the Chainlink oracle, or if an early block, the MakerDAO price oracle.

First, setup the db:

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    python3 -m backtest.gather_samples.fill_eth_price \
    --setup-db
```

Then, run the scrape. Here we use N_WORKERS to represent the number of worker-processes you would like to use. We set it to 50.

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    ./spawn_many_fill_eth_price.sh \
    $N_WORKERS
```

Finalize the work (abt 1min):

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    python3 -m backtest.gather_samples.fill_eth_price \
    --finalize
```

## Scrape arbitrages.

This takes a while! We chose 50 workers. Be sure that you increased your postgresql connections.

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    ./spawn_many_gather_samples.sh \
    $N_WORKERS
```

You can watch the ETA here, in another bash session. This takes about 1-2 days.
If you would like to perform this on multiple machines, that is okay!
Get the docker container on the second (third, fourth...) machine and launch it similar to the command below.
Except, you'll want to set the environment variables PSQL_HOST and PSQL_PORT appropriately
(ie, your database machine's ip and port). Set these with `-e PSQL_HOST=xxx.xxx.xxx.xxx` etc.

Scrape work is processed in random order, so the ETA should be somewhat reliable.

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    goldphish \
    python3 tmp_eta_gather.py
```

## Load flashbots transactions

This scrapes flashbots transaction from their server.

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    python3 -m backtest.gather_samples.load_flashbots
```

We scrape linearly in time. Since activity is greater as time goes on, expect the ETA to grow as blocks per second slows. This is a common issue with linear scans.


It should take about 5 hours. This could probably be made faster by varying the http request batch sizes a bit smarter.

## Attribute exchanges to 0x

We need to figure out which exchanges were, in fact, 0x exchanges. This needs to be run twice, first for v3, then for v4.

First, v3 (should take about 30min - 1 hour):

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    python3 -m backtest.gather_samples.fill_zerox --v3
```

Then, v4 (NOTE: This is parallelized, so a bit faster. we picked 50 workers.):

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    ./spawn_many_fill_zerox.sh \
    $N_WORKERS
```

## Scrape direct-to-miner coinbase transfers

NOTE: This will take a LONG time!

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    ./spawn_fill_coinbase_xfer.sh \
    $N_WORKERS
```

## Fill back-runners

Do transaction re-ordering to determine who was backrunning. First, setup db and fill the work queue:

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    python3 -m backtest.gather_samples.fill_backrunners --setup-db
```

Then, do the reordering. This is parallelized -- will take a while! You might want to spawn more workers on more machines. If you do that, be sure to specify the postgresql host/port as an environment variable (see section "Scrape Arbitrages").

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    ./spawn_many_backrunner_detect.sh \
    $N_WORKERS
```

You can watch the ETA here:

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    goldphish \
    python3 tmp_eta_fill_backrunner.py
```

## Compute the table of false-positives

Setup some tables:

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    python3 -m backtest.gather_samples.fill_odd_token_xfers --setup-db
```

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    python3 -m backtest.gather_samples.fill_odd_token_xfers --setup-db
```

Fill odd token transfer table (mistaken NFTs)

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    ./spawn_many_fill_odd_tokens.sh \
    $N_WORKERS
```

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    python3 -m backtest.gather_samples.fill_false_positive
```

## Compute naive gas-price oracle

First, set-up the database

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    python3 -m backtest.gather_samples.fill_naive_gas_price --setup-db
```

Then, run the job (NOTE: this is parallelized).

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    ./spawn_many_gas_price_fill.sh \
    $N_WORKERS
```

## Find arbitrages used in sandwich-attacks

First, set-up the database

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    python3 -m backtest.gather_samples.fill_arb_sandwich --setup-db
```

Run the scrape:

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    ./spawn_fill_arb_sandwich.sh \
    $N_WORKERS
```

You can view the ETA here:

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    goldphish \
    python3 tmp_eta_fill_arb_sandwich.py
```

## Scrape for candidate arbitrages

THIS TAKES A LONG TIME!

Run the historical arbitrage opportunity search algorithm. You will likely need to have several hundred workers across several machines. We find that each worker will peg a CPU core -- ie, this is CPU-bound, so adding many more workers than CPU cores is not recommended.

About 500 workers should finish the job in about under 1 month.

Set up the job:

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    python3 -m backtest.top_of_block seek-candidates --setup-db
```

And run the job. Here we show explicitly how to set the postgresql host IP and port number. This is not necessary if you are on the same docker network as the database (then defaults work fine).
One can request cancellation of work by sending SIGHUP to the worker python processes, each worker should cleanly stop its current work unit and break off any remaining work into a new unit.

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    -e PSQL_PORT=5041 \
    -e PSQL_HOST=XXX.XXX.XXX.XXX \
    goldphish \
    ./spawn_many_searchers.sh \
    $N_WORKERS
```

You can view the ETA here. Note that some warm-up time is required before the ETA computation works.
During the start, the ETA will slowly rise, as the easier (early-history) blocks are processed.
ETA will also under-estimate time remaining toward the end of the run, as the parallelism decreases because no work-units remain.

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    goldphish \
    python3 tmp_eta.py
```

## Fill table with top candidate arbitrages

Set up database:

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    python3 -m backtest.top_of_block fill-top-arbs --setup-db
```

Run the fill:

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    ./spawn_many_fill_top_arbitrages.sh \
    $N_WORKERS
```

## Execute arbitrages

This executes the candidate arbitrages, to check profitability. Takes quite a while!

This has two modes: 'all' and 'top arbitrages'. The 'all' mode runs arbitrages in order of decreasing priority. Priority was determined when generating block samples (a few steps back), and is a shuffle of the blockchain broken into day-long segments. This facilitates random sampling, presuming (and I do) that you do not have time to execute the entire thing. 'Top arbitrages' mode will relay all of the _large_ arbitrages.


Setup DB

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    python3 -m backtest.top_of_block do-relay --setup-db
```

Run 'all'

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    ./spawn_many_relayers.sh \
    $N_WORKERS
```

Watch the ETA at:

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    goldphish \
    python3 tmp_eta_relay.py
```

Run 'top arbitrages'. First, we need to fill this record of exchange modification history.

To do that, first set-up the db:

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    python3 -m backtest.top_of_block do-arb-duration --setup-db
```

Then do the run:

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    ./spawn_many_fill_modifications.sh \
    $N_WORKERS
```

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    -e WEB3_HOST=ws://$GETH_NODE \
    goldphish \
    ./spawn_many_relay_top_arbs.sh \
    $N_WORKERS
```

Watch the ETA at:

```bash
docker run \
    --rm -t \
    --network ethereum-measurement-net \
    -v $STORAGE_DIR:/mnt/goldphish \
    goldphish \
    python3 tmp_eta_relay_top_arbs.py
```
