import logging
import sys
import time
import typing
import psycopg2.extensions
import web3
from backtest.gather_samples.models import Arbitrage
from backtest.gather_samples.tokens import get_token
import cachetools

l = logging.getLogger(__name__)

def setup_db(curr: psycopg2.extensions.cursor):
    curr.execute(
        '''
        CREATE TABLE IF NOT EXISTS gather_sample_arbitrages_reservations (
            id SERIAL PRIMARY KEY NOT NULL,
            from_block INTEGER NOT NULL,
            to_block_exclusive INTEGER NOT NULL,
            started_on TIMESTAMP WITHOUT TIME ZONE,
            finished_on TIMESTAMP WITHOUT TIME ZONE
        );

        CREATE TABLE IF NOT EXISTS tokens (
            id SERIAL PRIMARY KEY NOT NULL,
            address BYTEA NOT NULL,
            name TEXT,
            symbol TEXT,
            verified BOOLEAN DEFAULT FALSE,
            usd_stablecoin BOOLEAN DEFAULT FALSE
        );
        
        CREATE INDEX IF NOT EXISTS idx_token_address ON tokens USING hash (address);
        CREATE INDEX IF NOT EXISTS idx_token_name ON tokens (name);

        CREATE TABLE IF NOT EXISTS sample_arbitrage_exchanges (
            id SERIAL NOT NULL PRIMARY KEY,
            address BYTEA NOT NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_sample_arbitrage_exchanges_address ON sample_arbitrage_exchanges (address);

        CREATE TABLE IF NOT EXISTS sample_arbitrages (
            id            SERIAL PRIMARY KEY NOT NULL,
            txn_hash      BYTEA NOT NULL,
            block_number  INTEGER NOT NULL,
            n_cycles      INTEGER NOT NULL,
            gas_used      NUMERIC(78, 0) NOT NULL,
            gas_price     NUMERIC(78, 0) NOT NULL,
            shooter       BYTEA DEFAULT NULL,
            coinbase_xfer NUMERIC(78, 0) DEFAULT NULL,
            miner         BYTEA DEFAULT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_sample_arbitrages_block_number ON sample_arbitrages (block_number);

        CREATE TABLE IF NOT EXISTS sample_arbitrage_cycles (
            id SERIAL PRIMARY KEY NOT NULL,
            sample_arbitrage_id INTEGER NOT NULL REFERENCES sample_arbitrages(id) ON DELETE CASCADE,
            profit_token  INTEGER NOT NULL REFERENCES tokens(id),
            profit_amount NUMERIC(78, 0) NOT NULL,
            profit_taker  BYTEA DEFAULT NULL
        );

        CREATE TABLE IF NOT EXISTS sample_arbitrage_cycle_exchanges (
            id          SERIAL PRIMARY KEY NOT NULL,
            cycle_id    INTEGER NOT NULL REFERENCES sample_arbitrage_cycles(id) ON DELETE CASCADE,
            token_in    INTEGER NOT NULL REFERENCES tokens(id),
            token_out   INTEGER NOT NULL REFERENCES tokens(id)
        );

        CREATE INDEX IF NOT EXISTS idx_sample_arbitrage_cycle_exchanges_cycle_id ON sample_arbitrage_cycle_exchanges (cycle_id);

        CREATE TABLE IF NOT EXISTS sample_arbitrage_cycle_exchange_items (
            id                SERIAL PRIMARY KEY NOT NULL,
            cycle_exchange_id INTEGER NOT NULL REFERENCES sample_arbitrage_cycle_exchanges(id) ON DELETE CASCADE,
            exchange_id       INTEGER NOT NULL REFERENCES sample_arbitrage_exchanges (id),
            amount_in         NUMERIC(78, 0) NOT NULL,
            amount_out        NUMERIC(78, 0) NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_sample_arbitrage_cycle_exchange_items_cycle_exchange_id ON sample_arbitrage_cycle_exchange_items (cycle_exchange_id);
        '''
    )
    curr.connection.commit()
    l.debug('setup database')


def xor(var, key, byteorder=sys.byteorder):
    key, var = key[:len(var)], var[:len(key)]
    int_var = int.from_bytes(var, byteorder)
    int_key = int.from_bytes(key, byteorder)
    int_enc = int_var ^ int_key
    return int_enc.to_bytes(len(var), byteorder)

def stable_hash_addr(baddr: bytes) -> int:
    """
    stable hash of the given address bytes for use in advisory locks
    """
    assert len(baddr) == 20, 'must be 20 bytes'
    ret = 0
    for i in range(20 // 4):
        ret = ret ^ int.from_bytes(baddr[i * 4 : (i + 1) * 4], byteorder=sys.byteorder, signed=False)
    return abs(ret) & 0x7fffffff


LOCK_MASK = 0xffff

_exchange_cache = cachetools.LRUCache(maxsize=10_000)
def get_exchange_ids(curr: psycopg2.extensions.cursor, addresses: typing.List[str]) -> typing.List[int]:
    ret = [None] * len(addresses)

    needs_lookup: typing.List[typing.Tuple[int, str, bytes]] = []
    for i, address in enumerate(addresses):
        if address in _exchange_cache:
            ret[i] = _exchange_cache[address]
        else:
            baddress = bytes.fromhex(address[2:])
            needs_lookup.append((i, address, baddress))

    # look up the ids
    needs_insert: typing.List[typing.Tuple[int, str, bytes]] = []
    for i, address, baddress in needs_lookup:
        curr.execute('SELECT id FROM sample_arbitrage_exchanges WHERE address = %s', (baddress,))
        if curr.rowcount == 0:
            # this isn't in the db yet
            needs_insert.append((i, address, baddress))
        else:
            (id_,) = curr.fetchone()
            ret[i] = id_
            _exchange_cache[address] = id_

    locks_needed = set()
    for _, _, baddress in needs_insert:
        locks_needed.add(stable_hash_addr(baddress) & LOCK_MASK)

    if len(needs_insert) > 0:
        l.debug(f'locking {len(locks_needed)} locks for exchange inserts')
        start = time.time()
        for lock in sorted(locks_needed):
            curr.execute('SELECT pg_advisory_xact_lock(3333::integer, %s::integer)', (lock,))
        elapsed = time.time() - start
        l.debug(f'spent {elapsed:.3f} seconds waiting for exchange insert lock(s)')

        # we have exclusive access -- see if we won the races
        for i, address, baddress in needs_insert:
            curr.execute('SELECT id FROM sample_arbitrage_exchanges WHERE address = %s', (baddress,))
            if curr.rowcount == 0:
                # won this race, insert + get id
                curr.execute('INSERT INTO sample_arbitrage_exchanges (address) VALUES (%s) returning id', (baddress,))
                assert curr.rowcount == 1
            (id_,) = curr.fetchone()
            assert id_ is not None
            ret[i] = id_

    return ret


def insert_arbs(w3: web3.Web3, curr: psycopg2.extensions.cursor, arbs: typing.List[Arbitrage]):
    if len(arbs) == 0:
        return

    all_exchanges = set()
    for arb in arbs:
        if arb.only_cycle is not None:
            for exc in arb.only_cycle.cycle:
                for exc_item in exc.items:
                    all_exchanges.add(exc_item.address)

    all_exchanges = sorted(list(all_exchanges))
    exchange_ids = get_exchange_ids(curr, all_exchanges)
    exchange_to_ids = {k: v for k, v in zip(all_exchanges, exchange_ids)}
    assert len(exchange_to_ids) == len(all_exchanges)

    already_inserted = set()
    for arb in arbs:
        # sanity check
        assert arb.txn_hash not in already_inserted
        already_inserted.add(arb.txn_hash)

        _insert_arb(w3, curr, arb, exchange_to_ids)

    l.debug(f'inserted arbitrages')


def _insert_arb(w3: web3.Web3, curr: psycopg2.extensions.cursor, arb: Arbitrage, exchange_to_ids: typing.Dict[str, int]) -> int:
    """
    Insert the given arbitrage ... takes a little bit
    """
    curr.execute(
        '''
        INSERT INTO sample_arbitrages (
            txn_hash, block_number, n_cycles, gas_used, gas_price, shooter
        ) VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
        ''',
        (
            arb.txn_hash,
            arb.block_number,
            arb.n_cycles,
            arb.gas_used,
            arb.gas_price,
            bytes.fromhex(arb.shooter[2:]) if arb.shooter is not None else None,
        ),
    )
    assert curr.rowcount == 1
    (arbitrage_id,) = curr.fetchone()

    if arb.only_cycle is not None:
        assert arb.only_cycle.profit_token is not None
        profit_token = get_token(w3, curr, arb.only_cycle.profit_token, arb.block_number)


        curr.execute(
            '''
            INSERT INTO sample_arbitrage_cycles (sample_arbitrage_id, profit_token, profit_amount, profit_taker)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            ''',
            (arbitrage_id, profit_token.id, arb.only_cycle.profit_amount, arb.only_cycle.profit_taker),
        )
        assert curr.rowcount == 1
        (cycle_id,) = curr.fetchone()

        for exchange in arb.only_cycle.cycle:
            token_in = get_token(w3, curr, exchange.token_in, arb.block_number)
            token_out = get_token(w3, curr, exchange.token_out, arb.block_number)
            curr.execute(
                '''
                INSERT INTO sample_arbitrage_cycle_exchanges (cycle_id, token_in, token_out)
                VALUES (%s, %s, %s)
                RETURNING id
                ''',
                (
                    cycle_id,
                    token_in.id,
                    token_out.id,
                )
            )
            (cycle_exchange_id,) = curr.fetchone()
            for item in exchange.items:
                exc_id = exchange_to_ids[item.address]

                curr.execute(
                    '''
                    INSERT INTO sample_arbitrage_cycle_exchange_items (
                        cycle_exchange_id, exchange_id, amount_in, amount_out
                    ) VALUES (%s, %s, %s, %s)
                    ''',
                    (cycle_exchange_id, exc_id, item.amount_in, item.amount_out),
                )
    return arbitrage_id

