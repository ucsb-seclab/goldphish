import random
import backoff
import psycopg2.extensions
import psycopg2.errors
import psycopg2
import hashlib
import itertools
import logging
import time
import typing
import web3
from backtest.top_of_block.common import TraceMode, WrappedFoundArbitrage, connect_db, load_exchanges, shoot

from backtest.utils import CancellationToken, parse_logs_for_net_profit
import find_circuit
import pricers
from utils import WETH_ADDRESS

l = logging.getLogger(__name__)


def do_verify(w3: web3.Web3, job_name: str, worker_name: str):
    l.info('Starting verification of profitability')
    db = connect_db()
    curr = db.cursor()
    setup_db(curr)
    cancellation_token = CancellationToken(job_name, worker_name, connect_db())

    for block_number, candidates in get_candidate_arbitrages(w3, curr):
        check_candidates(w3, curr, block_number, candidates, cancellation_token)
        if cancellation_token.cancel_requested():
            l.debug(f'Quitting because cancel requested')
            curr.execute(
                'UPDATE candidate_arbitrage_blocks_to_verify SET verify_started = null WHERE block_number = %s',
                (block_number,)
            )
            break
        l.debug(f'finished block_number={block_number:,}')
        curr.execute('UPDATE candidate_arbitrage_blocks_to_verify SET verify_finished = now()::timestamp WHERE block_number = %s', (block_number,))
        curr.connection.commit()


def setup_db(curr: psycopg2.extensions.cursor):
    curr.execute(
        """
        CREATE TABLE IF NOT EXISTS verified_arbitrages (
            id                     SERIAL PRIMARY KEY NOT NULL,
            candidate_id           INTEGER NOT NULL REFERENCES candidate_arbitrages (id) ON DELETE CASCADE,
            gas_used               NUMERIC(78, 0) NOT NULL,
            gas_price              NUMERIC(78, 0) NOT NULL,
            measured_profit_no_fee NUMERIC(78, 0) NOT NULL,
            net_profit             NUMERIC(78, 0) NOT NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_verified_arbitrages_candidate_id ON verified_arbitrages(candidate_id);

        CREATE TABLE IF NOT EXISTS failed_arbitrages (
            id             SERIAL PRIMARY KEY NOT NULL,
            candidate_id   INTEGER NOT NULL REFERENCES candidate_arbitrages (id) ON DELETE CASCADE,
            diagnosis      TEXT DEFAULT NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_failed_arbitrages_candidate_id ON failed_arbitrages(candidate_id);
        """
    )
    curr.connection.commit()


class ShootResult(typing.NamedTuple):
    measured_profit_no_fee: int
    gas_used: int


def check_candidates(
        w3: web3.Web3,
        curr: psycopg2.extensions.cursor,
        block_number: int,
        candidates: typing.List[WrappedFoundArbitrage],
        cancellation_token: CancellationToken,
    ):
    processed_ids: typing.Set[int] = set()

    shoots = check_shoot(w3, candidates, block_number, cancellation_token)

    if cancellation_token.cancel_requested():
        return

    #
    # Find the gasprice to use
    for i in itertools.count(1):
        l.debug(f'checking for positive gas prices in block {block_number + i:,}')
        block = w3.eth.get_block(block_number + i)
        for txn in reversed(block['transactions']):
            gasprice = w3.eth.get_transaction_receipt(txn)['effectiveGasPrice']
            if gasprice > 1:
                break
        else:
            continue
        break


    if cancellation_token.cancel_requested():
        return

    #
    # insert each result
    for fa, maybe_result in shoots:
        # ensure no dupes
        assert fa.id not in processed_ids
        processed_ids.add(fa.id)

        if maybe_result is not None:
            total_fee = gasprice * maybe_result.gas_used
            net_profit = maybe_result.measured_profit_no_fee - total_fee
            # mark as succeeded
            curr.execute(
                """
                INSERT INTO verified_arbitrages (candidate_id, gas_used, gas_price, measured_profit_no_fee, net_profit)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (fa.id, maybe_result.gas_used, gasprice, maybe_result.measured_profit_no_fee, net_profit)
            )
        else:
            # mark as failed
            curr.execute(
                """
                INSERT INTO failed_arbitrages (candidate_id) VALUES (%s)
                """,
                (fa.id,)
            )


@backoff.on_exception(backoff.expo, max_tries=10, exception=psycopg2.errors.QueryCanceled, logger=__name__)
def get_next_block(curr: psycopg2.extensions.cursor) -> typing.Optional[int]:
    curr.execute('BEGIN TRANSACTION')
    curr.execute("SET LOCAL statement_timeout = '10s'")

    try:
        curr.execute(
            """
            SELECT block_number
            FROM candidate_arbitrage_blocks_to_verify
            WHERE verify_started IS NULL
            ORDER BY max_profit_no_fee DESC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
            """
        )
    except:
        curr.connection.rollback()
        raise

    maybe_bn = curr.fetchall()
    if len(maybe_bn) == 0:
        curr.connection.rollback() # release lock
        return None

    block_number = maybe_bn[0][0]

    curr.execute(
        """
        UPDATE candidate_arbitrage_blocks_to_verify SET verify_started = now()::timestamp WHERE block_number = %s
        """,
        (block_number,)
    )
    assert curr.rowcount == 1
    curr.connection.commit() # release lock

    return block_number


def get_candidate_arbitrages(w3: web3.Web3, curr: psycopg2.extensions.cursor) -> typing.Iterator[typing.Tuple[int, typing.List[WrappedFoundArbitrage]]]:
    """
    Continuously polls for candidate arbitrages in a sleep-loop.
    """
    uniswap_v2_exchanges, uniswap_v3_exchanges = load_exchanges()

    while True:
        start = time.time()
        l.debug(f'getting more work to do')
        block_number = get_next_block(curr)

        if block_number is None:
            l.debug('Nothing to do, sleeping for a bit...')
            curr.connection.commit() # release transaction
            time.sleep(30)
            continue

        l.debug(f'took {time.time() - start:.3f} seconds to get new candidate block to analyze')

        # gather found arbitrages to return
        ret = []
        curr.execute(
            '''
            SELECT id, exchanges, directions, amount_in, profit_no_fee
            FROM candidate_arbitrages
            WHERE block_number = %s
            ''',
            (block_number,)
        )
        for id_, exchanges, directions, amount_in, profit in curr:

            assert len(exchanges) == len(directions)

            exchanges = [web3.Web3.toChecksumAddress(x.tobytes()) for x in exchanges]

            # reconstruct found arbitrage
            amount_in = int(amount_in)
            profit = int(profit)
            circuit: typing.List[pricers.BaseExchangePricer] = []
            for address in exchanges:            
                if address in uniswap_v2_exchanges:
                    token0, token1 = uniswap_v2_exchanges[address]
                    pricer = pricers.UniswapV2Pricer(w3, address, token0, token1)
                else:
                    assert address in uniswap_v3_exchanges
                    token0, token1, fee = uniswap_v3_exchanges[address]
                    pricer = pricers.UniswapV3Pricer(w3, address, token0, token1, fee)
                circuit.append(pricer)

            if directions[0] == True:
                assert circuit[0].token0 == WETH_ADDRESS
            else:
                assert directions[0] == False
                assert circuit[0].token1 == WETH_ADDRESS

            fa = find_circuit.FoundArbitrage(
                amount_in = amount_in,
                circuit = circuit,
                directions = directions,
                pivot_token = WETH_ADDRESS,
                profit = profit
            )
            ret.append(WrappedFoundArbitrage(fa, id_))

        l.debug(f'verifying {len(ret)} candidates in block number {block_number:,}')

        assert len(ret) == len(set(x.id for x in ret)), 'sanity check; expect no duplicate IDs'

        yield block_number, ret


def check_shoot(
        w3: web3.Web3,
        fas: typing.List[WrappedFoundArbitrage],
        block_number: int,
        cancellation_token: CancellationToken,
    ) -> typing.List[
        typing.Tuple[WrappedFoundArbitrage,
        typing.Optional[ShootResult]]
    ]:
    """
    Attempts re-shoot and returns gas usage on success. On failure, returns None.
    """
    ret = []
    r = shoot(w3, fas, block_number, do_trace = TraceMode.NEVER, cancellation_token = cancellation_token)

    if cancellation_token.cancel_requested():
        return

    shooter_address, results = r

    for result in results:
        if result.encodable == False or result.receipt['status'] != 1:
            ret.append((result.arbitrage, None))
        else:
            movements = parse_logs_for_net_profit(result.receipt['logs'])
            measured_profit_no_fee = movements[WETH_ADDRESS][shooter_address]
            if measured_profit_no_fee - result.arbitrage.profit > -10 and measured_profit_no_fee > 0: # a little tolerance for rounding
                ret.append((
                    result.arbitrage,
                    ShootResult(
                        measured_profit_no_fee,
                        result.receipt['gasUsed'],
                    ),
                ))
            else:
                ret.append((result.arbitrage, None))
    assert set(x.id for (x, _) in ret) == set(x.id for x in fas)
    return ret
