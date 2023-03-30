# Goldphish - historical ethereum arbitrage analysis

Goldphish is an arbitrage analyzer for the ethereum blockchain.


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
+ max_connections = 100
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

Scrape arbitrages. This takes a while! We chose 50 workers. Be sure that you increased your postgresql connections.

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
