"""
Attempts to re-find arbitrages that actually occurred on the blockchain.
"""

import argparse
import decimal
import math
import random
import time
import typing
import numpy as np
import web3
import web3.types
import web3._utils.filters
import psycopg2
import psycopg2.extensions
import logging
import pricers
import pricers.balancer_v2.common

from backtest.utils import connect_db
from pricers.balancer import BalancerPricer
from pricers.balancer_v2.liquidity_bootstrapping_pool import BalancerV2LiquidityBootstrappingPoolPricer
from pricers.balancer_v2.weighted_pool import BalancerV2WeightedPoolPricer
from pricers.base import BaseExchangePricer, NotEnoughLiquidityException
from pricers.uniswap_v2 import UniswapV2Pricer
from pricers.uniswap_v3 import UniswapV3Pricer
from find_circuit import PricingCircuit
from find_circuit.find import detect_arbitrages_bisection
from utils import BALANCER_VAULT_ADDRESS, get_abi

l = logging.getLogger(__name__)

def add_args(subparser: argparse._SubParsersAction) -> typing.Tuple[str, typing.Callable[[web3.Web3, argparse.Namespace], None]]:
    parser_name = 'replicate'
    parser: argparse.ArgumentParser = subparser.add_parser(parser_name)

    parser.add_argument('--setup-db', action='store_true', help='Setup the database (run before mass scan)')

    return parser_name, replicate


def replicate(w3: web3.Web3, args: argparse.Namespace):
    db = connect_db()
    curr = db.cursor()

    if args.setup_db:
        setup_db(curr)
        return

    l.info('Starting replication')
    time.sleep(4)

    while True:
        candidate = get_candidate(curr)

        if candidate is None:
            # out of work
            break

        try_replicate(w3, curr, candidate)

        # some jitter for the parallel workers
        if random.choice((True, False)):
            time.sleep(random.expovariate(1 / 0.005))

    l.info('done')


def setup_db(curr: psycopg2.extensions.cursor):
    l.info('setting up database')
    curr.execute(
        '''
        CREATE TABLE IF NOT EXISTS sample_arbitrage_replications (
            sample_arbitrage_id   INTEGER NOT NULL REFERENCES sample_arbitrages (id) ON DELETE CASCADE,
            verification_started  BOOLEAN NOT NULL DEFAULT FALSE,
            verification_finished BOOLEAN,
            supported             BOOLEAN,
            replicated            BOOLEAN,
            our_profit            NUMERIC(78, 0),
            percent_change        DOUBLE PRECISION
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_sample_arbitrage_replication_id ON sample_arbitrage_replications (sample_arbitrage_id);
        CREATE INDEX IF NOT EXISTS idx_sample_arbitrage_replication_started ON sample_arbitrage_replications (verification_started);
        '''
    )
    curr.execute(
        '''
        INSERT INTO sample_arbitrage_replications (sample_arbitrage_id)
        SELECT sample_arbitrage_id
        FROM sample_arbitrage_cycles sac
        WHERE
            NOT EXISTS(SELECT 1 FROM sample_arbitrage_replications sar WHERE sar.sample_arbitrage_id = sac.sample_arbitrage_id) AND
            EXISTS(SELECT 1 FROM sample_arbitrages sa WHERE sa.id = sac.sample_arbitrage_id)
        '''
    )
    l.info(f'inserted {curr.rowcount:,} replication rows')

    curr.connection.commit()
    l.info('done setting up database')
    pass


def get_candidate(curr: psycopg2.extensions.cursor):
    """
    Get the next candidate ID from the queue
    """
    curr.execute(
        '''
        SELECT sample_arbitrage_id
        FROM sample_arbitrage_replications sar
        WHERE verification_started = false
        LIMIT 1
        FOR UPDATE SKIP LOCKED
        '''
    )
    if curr.rowcount == 0:
        curr.connection.rollback()
        l.info('No more work')
        return None

    assert curr.rowcount == 1

    (id_,) = curr.fetchone()
    curr.execute('UPDATE sample_arbitrage_replications SET verification_started = true WHERE sample_arbitrage_id = %s', (id_,))
    assert curr.rowcount == 1
    curr.connection.commit()

    l.debug(f'Processing id_={id_}')

    return id_

def try_replicate(w3: web3.Web3, curr: psycopg2.extensions.cursor, candidate: int):
    # if we could not replicate it exactly via backrun-detector, then ignore
    curr.execute(
        '''
        SELECT rerun_exactly FROM sample_arbitrage_backrun_detections
        WHERE sample_arbitrage_id = %s
        ''',
        (candidate,)
    )
    assert curr.rowcount == 1

    (rerun_exactly,) = curr.fetchone()

    if not rerun_exactly:
        l.debug(f'Did not rerun exactly, cannot support replicating {candidate}')
        curr.execute(
            '''
            UPDATE sample_arbitrage_replications
            SET verification_finished = true, supported = false
            WHERE sample_arbitrage_id = %s
            ''',
            (candidate,)
        )
        assert curr.rowcount == 1
        curr.connection.commit()
        return

    curr.execute(
        '''
        SELECT sac.id, sac.profit_amount, sa.txn_hash, sa.block_number, t.address
        FROM sample_arbitrages sa
        JOIN sample_arbitrage_cycles sac ON sa.id = sac.sample_arbitrage_id
        JOIN tokens t ON t.id = sac.profit_token
        WHERE sa.id = %s
        ''',
        (candidate,)
    )

    assert curr.rowcount == 1, f'Expected rowcount of 1 for {candidate} but got {curr.rowcount}'

    (sac_id, original_profit_amount, txn_hash, block_number, pivot_token) = curr.fetchone()
    original_profit_amount = int(original_profit_amount)
    txn_hash = txn_hash.tobytes()
    pivot_token = web3.Web3.toChecksumAddress(pivot_token.tobytes())

    l.debug('getting directions...')

    # get exchange directions
    curr.execute(
        '''
        SELECT sace.id, tin.address, tout.address
        FROM sample_arbitrage_cycle_exchanges sace
        JOIN tokens tin  ON token_in  = tin.id
        JOIN tokens tout ON token_out = tout.id
        WHERE cycle_id = %s
        order by sace.id asc
        ''',
        (sac_id,)
    )
    assert curr.rowcount > 1, f'Cannot form a cycle with only {curr.rowcount} on {candidate}'

    directions = []
    sace_ids = []
    for sace_id, tin_baddr, tout_baddr in curr:
        tin_address = web3.Web3.toChecksumAddress(tin_baddr.tobytes())
        tout_address = web3.Web3.toChecksumAddress(tout_baddr.tobytes())
        d = (tin_address, tout_address)

        assert d not in directions, f'cannot have duplicate directions on {candidate} (saw duplicate {d})'

        directions.append((tin_address, tout_address))
        sace_ids.append(sace_id)

    # ensure directions point end-to-end
    for (_, t2), (t3, _) in zip(directions, directions[1:] + [directions[0]]):
        assert t2 == t3, f'directions did not align for {candidate}'

    # ensure pivot is somewhere in directions
    assert any((pivot_token in x) for x in directions), f'could not find pivot token in directions for {candidate}'

    l.debug('getting exchanges...')

    # get the exchanges
    curr.execute(
        '''
        SELECT sace.id, sae.address
        FROM sample_arbitrage_cycles sac
        JOIN sample_arbitrage_cycle_exchanges sace ON sace.cycle_id = sac.id
        JOIN sample_arbitrage_cycle_exchange_items sacei ON sace.id = sacei.cycle_exchange_id
        JOIN sample_arbitrage_exchanges sae ON sae.id = sacei.exchange_id
        WHERE sac.sample_arbitrage_id = %s
        ORDER BY sace.id ASC
        ''',
        (candidate,)
    )
    assert curr.rowcount >= len(sace_ids), f'wrong number of rows ({curr.rowcount}) while getting exchange addresses on {candidate}'

    if curr.rowcount > len(sace_ids):
        # one of the exchange items is split across several exchanges
        # we don't support this, so give up
        l.debug(f'Exchange flow was split, cannot support {candidate}, skipping')
        curr.execute(
            '''
            UPDATE sample_arbitrage_replications
            SET verification_finished = true, supported = false
            WHERE sample_arbitrage_id = %s
            ''',
            (candidate,)
        )
        assert curr.rowcount == 1
        curr.connection.commit()
        return

    circuit: typing.List[BaseExchangePricer] = []
    # attempt to load each of the exchanges
    for expected_id, (actual_id, baddr) in zip(sace_ids, list(curr)):
        baddr = baddr.tobytes()
        assert expected_id == actual_id, f'expected {expected_id} == {actual_id} on {candidate}'

        maybe_pricer = load_pricer_for(w3, curr, baddr, txn_hash)

        if maybe_pricer is not None:
            circuit.append(maybe_pricer)
        else:
            # pricer not found
            l.debug(f'Could not find pricer for {web3.Web3.toChecksumAddress(baddr)} on {candidate}')
            curr.execute(
                '''
                UPDATE sample_arbitrage_replications
                SET verification_finished = true, supported = false
                WHERE sample_arbitrage_id = %s
                ''',
                (candidate,)
            )
            assert curr.rowcount == 1
            curr.connection.commit()
            return

    assert len(circuit) == len(directions) # this should not be possible to voilate, but just to be sure

    # rotate so pivot is in place
    pc = PricingCircuit(circuit, directions)
    while pc.pivot_token != pivot_token:
        pc.rotate()

    maybe_arbs = detect_arbitrages_bisection(pc, block_number - 1, try_all_directions=False)

    assert len(maybe_arbs) <= 1, f'Got too many arbitrages out from detection on {candidate}'

    if len(maybe_arbs) == 0:
        l.error(f'could not replicate id={candidate} https://etherscan.io/tx/0x{txn_hash.hex()}')
        curr.execute(
            '''
            UPDATE sample_arbitrage_replications
            SET verification_finished = true, supported = true, replicated = false
            WHERE sample_arbitrage_id = %s
            ''',
            (candidate,)
        )
        assert curr.rowcount == 1
        curr.connection.commit()
        return

    # get our profit and compare it to theirs
    arb = maybe_arbs[0]
    
    profit_percent_changed = None
    if original_profit_amount > 0:
        original_profit_amount_dec = decimal.Decimal(original_profit_amount)
        diff = decimal.Decimal(arb.profit) - original_profit_amount_dec
        profit_percent_changed = diff / original_profit_amount_dec * 100

    curr.execute(
        '''
        UPDATE sample_arbitrage_replications
        SET
            verification_finished = true,
            supported = true,
            replicated = true,
            our_profit = %s,
            percent_change = %s
        WHERE sample_arbitrage_id = %s
        ''',
        (arb.profit, profit_percent_changed, candidate)
    )
    assert curr.rowcount == 1
    curr.connection.commit()
    return


def load_pricer_for(w3: web3.Web3, curr: psycopg2.extensions.cursor, exchange: bytes, txn_hash: bytes) -> typing.Optional[pricers.BaseExchangePricer]:
    """
    Attempt to get a Pricer for this exchange.

    NOTE: if the 'exchange' is Balancer Vault, then we need to find the pool_id from the txn_hash (TODO fix this)
    """
    assert isinstance(txn_hash, bytes)

    exchange_address = w3.toChecksumAddress(exchange)

    if exchange_address == BALANCER_VAULT_ADDRESS:
        receipt = w3.eth.get_transaction_receipt(txn_hash)

        # find pool_id
        found = False
        for log in receipt['logs']:
            if log['address'] == BALANCER_VAULT_ADDRESS and log['topics'][0] == pricers.balancer_v2.common.SWAP_TOPIC:
                assert found != True, f'Could not identify single Balancer pool in txn_hash {txn_hash.hex()}'
                found = True
                pool_id = log['topics'][1]

        assert found == True, f'Could not find Balancer pool in txn https://etherscan.io/tx/0x{txn_hash.hex()}'

        curr.execute(
            '''
            SELECT address, pool_type
            FROM balancer_v2_exchanges
            WHERE pool_id = %s
            ''',
            (pool_id,)
        )
        assert curr.rowcount == 1, f'Did not find Balancer pool with id {pool_id.hex()}'
        (pool_address, pool_type) = curr.fetchone()
        pool_address = web3.Web3.toChecksumAddress(pool_address.tobytes())

        vault = w3.eth.contract(
            address = BALANCER_VAULT_ADDRESS,
            abi = get_abi('balancer_v2/Vault.json'),
        )

        if pool_type in ['WeightedPool', 'WeightedPool2Tokens']:
            return BalancerV2WeightedPoolPricer(w3, vault, pool_address, pool_id)
        elif pool_type in ['LiquidityBootstrappingPool', 'NoProtocolFeeLiquidityBootstrappingPool']:
            return BalancerV2LiquidityBootstrappingPoolPricer(w3, vault, pool_address, pool_id)
        
        # we don't know about this pool_type
        l.debug(f'Cannot handle pool_type = {pool_type} for balancer pool_id={pool_id.hex()}')
        return None

    curr.execute(
        '''
        SELECT t0.address, t1.address
        FROM uniswap_v2_exchanges uv2
        JOIN tokens t0 ON uv2.token0_id = t0.id
        JOIN tokens t1 ON uv2.token1_id = t1.id
        WHERE uv2.address = %s
        ''',
        (exchange,)
    )
    if curr.rowcount > 0:
        assert curr.rowcount == 1

        token0, token1 = curr.fetchone()
        token0 = w3.toChecksumAddress(token0.tobytes())
        token1 = w3.toChecksumAddress(token1.tobytes())
        p = UniswapV2Pricer(w3, exchange_address, token0, token1)
        return p

    curr.execute(
        '''
        SELECT t0.address, t1.address
        FROM sushiv2_swap_exchanges sv2
        JOIN tokens t0 ON sv2.token0_id = t0.id
        JOIN tokens t1 ON sv2.token1_id = t1.id
        WHERE sv2.address = %s
        ''',
        (exchange,)
    )
    if curr.rowcount > 0:
        assert curr.rowcount == 1

        token0, token1 = curr.fetchone()
        token0 = w3.toChecksumAddress(token0.tobytes())
        token1 = w3.toChecksumAddress(token1.tobytes())
        p = UniswapV2Pricer(w3, exchange_address, token0, token1)
        return p

    curr.execute(
        '''
        SELECT t0.address, t1.address
        FROM shibaswap_exchanges ss
        JOIN tokens t0 ON ss.token0_id = t0.id
        JOIN tokens t1 ON ss.token1_id = t1.id
        WHERE ss.address = %s
        ''',
        (exchange,)
    )
    if curr.rowcount > 0:
        assert curr.rowcount == 1

        token0, token1 = curr.fetchone()
        token0 = w3.toChecksumAddress(token0.tobytes())
        token1 = w3.toChecksumAddress(token1.tobytes())
        p = UniswapV2Pricer(w3, exchange_address, token0, token1)
        return p

    curr.execute(
        '''
        SELECT t0.address, t1.address, originalfee
        FROM uniswap_v3_exchanges uv3
        JOIN tokens t0 ON uv3.token0_id = t0.id
        JOIN tokens t1 ON uv3.token1_id = t1.id
        WHERE uv3.address = %s            
        ''',
        (exchange,)
    )
    if curr.rowcount > 0:
        assert curr.rowcount == 1
        token0, token1, fee = curr.fetchone()
        token0 = w3.toChecksumAddress(token0.tobytes())
        token1 = w3.toChecksumAddress(token1.tobytes())
        p = UniswapV3Pricer(w3, exchange_address, token0, token1, fee)
        return p

    curr.execute(
        '''
        SELECT EXISTS(SELECT 1 FROM balancer_exchanges WHERE address = %s)
        ''',
        (exchange,)
    )
    (is_balancerv1,) = curr.fetchone()
    if is_balancerv1:
        p = BalancerPricer(w3, exchange_address)
        return p

    return None
