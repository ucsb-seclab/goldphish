import psycopg2.extensions
import psycopg2
import hashlib
import itertools
import logging
import time
import typing
import web3
from backtest.top_of_block.common import TraceMode, connect_db, load_exchanges, shoot

from backtest.utils import parse_logs_for_net_profit
import find_circuit
import pricers
from utils import WETH_ADDRESS

l = logging.getLogger(__name__)

def do_verify(w3: web3.Web3):
    l.info('Starting verification of profitability')
    db = connect_db()
    curr = db.cursor()
    setup_db(curr)

    for block_number, candidate_id, fa in get_candidate_arbitrages(w3, curr):
        # time to reshoot
        maybe_gas = check_reshoot(w3, fa, block_number)
        if maybe_gas is not None:
            for i in itertools.count(1):
                block = w3.eth.get_block(block_number + i)
                if len(block['transactions']) > 0:
                    gasprice = w3.eth.get_transaction_receipt(block['transactions'][-1])['effectiveGasPrice']
                    break
            total_fee = gasprice * maybe_gas
            net_profit = fa.profit - total_fee
            # mark as succeeded
            l.debug(f'Succeeded verification')
            curr.execute(
                """
                INSERT INTO verified_arbitrages (candidate_id, gas_used, gas_price, net_profit)
                VALUES (%s, %s, %s, %s)
                """,
                (candidate_id, maybe_gas, gasprice, net_profit)
            )
        else:
            # mark as failed
            l.debug(f'Failed verification')
            curr.execute(
                """
                INSERT INTO failed_arbitrages (candidate_id) VALUES (%s)
                """,
                (candidate_id,)
            )


def setup_db(curr: psycopg2.extensions.cursor):
    curr.execute(
        """
        CREATE TABLE IF NOT EXISTS verified_arbitrages (
            id             SERIAL PRIMARY KEY NOT NULL,
            candidate_id   INTEGER NOT NULL REFERENCES candidate_arbitrages (id) ON DELETE CASCADE,
            gas_used       NUMERIC(78, 0) NOT NULL,
            gas_price      NUMERIC(78, 0) NOT NULL,
            net_profit     NUMERIC(78, 0) NOT NULL
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


def get_candidate_arbitrages(w3: web3.Web3, curr: psycopg2.extensions.cursor) -> typing.Iterator[typing.Tuple[int, int, find_circuit.FoundArbitrage]]:
    """
    Continuously polls for candidate arbitrages in a sleep-loop.
    """
    uniswap_v2_exchanges, uniswap_v3_exchanges = load_exchanges()

    while True:
        curr.execute(
            """
            SELECT id
            FROM candidate_arbitrages
            WHERE verify_run = false AND verify_started IS NULL
            ORDER BY profit_no_fee DESC
            LIMIT 1
            FOR UPDATE
            """
        )
        maybe_id = curr.fetchall()
        if len(maybe_id) == 0:
            l.debug('Nothing to do, sleeping for a bit...')
            curr.connection.commit()
            time.sleep(30)
            continue

        id_ = maybe_id[0][0]

        curr.execute(
            """
            UPDATE candidate_arbitrages ca
            SET verify_started = NOW()::timestamp
            WHERE id = %s
            RETURNING ca.block_number, ca.exchanges, ca.directions, ca.amount_in, ca.profit_no_fee
            """,
            (id_,)
        )
        block_number, exchanges, directions, amount_in, profit = curr.fetchone()
        curr.connection.commit() # release FOR UPDATE lock

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

        l.debug(f'Verifying candidate arbitrage id={id_:,} block_number={block_number:,} expected_profit={w3.fromWei(profit, "ether"):.6f} ETH')
        for t in fa.tokens:
            l.debug(f'uses token {t}')

        yield block_number, id_, fa

        l.debug(f'marking id={id_:,} as examined')
        curr.execute(
            """
            UPDATE candidate_arbitrages ca
            SET verify_finished = NOW()::timestamp, verify_run = True
            WHERE id = %s
            """,
            (id_,)
        )
        curr.connection.commit()


def check_reshoot(w3: web3.Web3, fa: find_circuit.FoundArbitrage, block_number: int) -> typing.Optional[int]:
    """
    Attempts re-shoot and returns gas usage on success. On failure, returns None.
    """
    shooter_address, receipt, _ = shoot(w3, fa, block_number, do_trace = TraceMode.NEVER)

    if receipt['status'] != 1:
        # trace, txn = maybe_trace
        # print('----------------------trace---------------------------')
        # decoded = decode_trace_calls(trace, txn, receipt)
        # pretty_print_trace(decoded, txn, receipt)
        # print('------------------------------------------------------')
        return None

    movements = parse_logs_for_net_profit(receipt['logs'])
    if movements[WETH_ADDRESS][shooter_address] == fa.profit:
        return receipt['gasUsed']
    return None

