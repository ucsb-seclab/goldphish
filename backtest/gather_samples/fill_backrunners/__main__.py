"""
Fills information about which arbitrages must have been backrunning.

Detects backrunning by re-ordering arbitrages to top-of-block
and checking against our model for the same or different profit before fees.
"""

import argparse
import itertools
import os
import random
import typing
import psycopg2
import psycopg2.extensions
import pricers
import web3
import web3.contract
import web3.types
from web3.logs import DISCARD
import web3._utils.filters
import logging
import find_circuit
import networkx as nx

from backtest.utils import connect_db
from utils import get_abi, setup_logging
from utils import uv2, uv3

l = logging.getLogger(__name__)

uv2_factory: web3.contract.Contract = web3.Web3().eth.contract(address=b'\x00'*20, abi=get_abi('uniswap_v2/IUniswapV2Factory.json'))
uv3_factory: web3.contract.Contract = web3.Web3().eth.contract(address=b'\x00'*20, abi=get_abi('uniswap_v3/IUniswapV3Factory.json'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose')
    parser.add_argument('--worker-name', type=str, default=None, help='worker name for log, must be POSIX path-safe')
    args = parser.parse_args()

    setup_logging('fill_backrunners', worker_name=args.worker_name, stdout_level=logging.DEBUG if args.verbose else logging.INFO)

    db = connect_db()
    curr = db.cursor()


    web3_host = os.getenv('WEB3_HOST', 'ws://172.17.0.1:8546')
    w3 = web3.Web3(web3.WebsocketProvider(
        web3_host,
        websocket_timeout=60 * 5,
        websocket_kwargs={
            'max_size': 1024 * 1024 * 1024, # 1 Gb max payload
        },
    ))
    
    if not w3.isConnected():
        l.error(f'Could not connect to web3')
        exit(1)

    l.debug(f'Connected to web3, chainId={w3.eth.chain_id}')

    try:
        setup_db(curr)
        compute_only_uniswap_queue(curr)
        do_reorder(w3, curr)
    except:
        l.exception('top-level exception')
        raise


def setup_db(curr: psycopg2.extensions.cursor):
    curr.execute(
        '''
        CREATE TABLE IF NOT EXISTS sample_arbitrages_backrun_detection (
            sample_arbitrage_id INTEGER NOT NULL PRIMARY KEY,
            backran_creation BOOLEAN DEFAULT FALSE,
            rerun_optimal_profit NUMERIC(78, 0),
            reorder_optimal_profit NUMERIC(78, 0),
            any_exchanges_touched BOOLEAN CHECK(any_exchanges_touched is not null OR backran_creation = true)
        );
        '''
            # reorder_rerun_difference NUMERIC(78, 0) NOT NULL,
            # reorder_difference NUMERIC(78, 0),
            # rerun_difference NUMERIC(78, 0),
    )

def compute_only_uniswap_queue(curr: psycopg2.extensions.cursor):
    """
    Finds arbitrages that only use uniswap exchanges and puts them in
    the temporary table tmp_samples_only_uniswap
    """
    curr.connection.commit()

    curr.execute(
        '''
        LOCK TABLE sample_arbitrages; -- to avoid multiple clients doing this computation

        SELECT NOT EXISTS (
            SELECT FROM 
                pg_tables
            WHERE 
                schemaname = 'public' AND 
                tablename  = 'fill_backrunners_queue'
        );
        '''
    )
    (needs_fill,) = curr.fetchone()
    
    if needs_fill:
        l.info(f'Loading uniswap-only arbitrages')
        curr.execute(
            '''
            CREATE TEMP TABLE tmp_uniswap_exchange_ids AS
            SELECT id
            FROM sample_arbitrage_exchanges sae
            WHERE EXISTS(SELECT 1 FROM uniswap_v1_exchanges uv1 WHERE uv1.address = sae.address) OR
                EXISTS(SELECT 1 FROM uniswap_v2_exchanges uv2 WHERE uv2.address = sae.address) OR
                EXISTS(SELECT 1 FROM uniswap_v3_exchanges uv3 WHERE uv3.address = sae.address);

            CREATE TEMP TABLE tmp_sample_arbitrages_only_uniswap AS
            SELECT sample_arbitrage_id, bool_and(EXISTS(SELECT 1 FROM tmp_uniswap_exchange_ids ues WHERE ues.id = sacei.exchange_id)) is_all_uniswap
            FROM sample_arbitrage_cycles sac
            JOIN sample_arbitrage_cycle_exchanges sace ON sace.cycle_id = sac.id
            JOIN sample_arbitrage_cycle_exchange_items sacei ON sacei.cycle_exchange_id = sace.id
            GROUP BY sample_arbitrage_id;

            CREATE TABLE fill_backrunners_queue AS
            SELECT sample_arbitrage_id, false as backrun_detect_started, null as backrun_detect_finished
            FROM tmp_sample_arbitrages_only_uniswap
            WHERE is_all_uniswap = true;
            '''
        )
        l.info(f'done fill')
        curr.connection.commit()
    else:
        l.debug(f'no fill needed')
        curr.connection.commit()


def do_reorder(w3: web3.Web3, curr: psycopg2.extensions.cursor):
    for i in itertools.count():
        if i % 50 == 0 and random.randint(0, 6) == 0:
            # count + report progress
            curr.execute('SELECT COUNT(*) FROM fill_backrunners_queue')
            (queue_size,) = curr.fetchone()
            curr.execute('SELECT COUNT(*) FROM fill_backrunners_queue WHERE backrun_detect_started = true')
            (n_remaining,) = curr.fetchone()
            l.info(f'Progress: {n_remaining}/{queue_size} ({n_remaining / queue_size * 100:.2f}%)')

        curr.connection.commit()
        curr.execute(
            '''
            SELECT sample_arbitrage_id
            FROM fill_backrunners_queue
            WHERE sample_arbitrage_id = 25756297 -- backrun_detect_started = false
            LIMIT 1
            FOR UPDATE SKIP LOCKED
            '''
        )
        assert curr.rowcount <= 1
        if curr.rowcount == 0:
            curr.connection.rollback()
            l.info('Done processing')
            break
        
        (id_,) = curr.fetchone()
        assert isinstance(id_, int)
        l.debug(f'processing id={id_}')
        curr.execute(
            '''
            UPDATE fill_backrunners_queue
            SET backrun_detect_started = true
            WHERE sample_arbitrage_id = %s
            ''',
            (id_,)
        )
        curr.connection.commit()

        try:
            test_a_reorder(w3, curr, id_)
            raise Exception('himom; done')
        except:
            l.critical(f'Failed while processing id={id_}')
            raise

        curr.execute(
            '''
            UPDATE fill_backrunners_queue
            SET backrun_detect_finished = true
            WHERE sample_arbitrage_id = %s
            ''',
            (id_,)
        )



def test_a_reorder(w3: web3.Web3, curr: psycopg2.extensions.cursor, id_: int):
    curr.execute('SELECT txn_hash FROM sample_arbitrages WHERE id = %s', (id_,))
    assert curr.rowcount == 1
    (txn_hash,) = curr.fetchone()
    txn_hash = txn_hash.tobytes()

    curr.execute(
        '''
        SELECT t.address, sac.profit_amount
        FROM sample_arbitrage_cycles sac
        JOIN tokens t ON sac.profit_token = t.id
        WHERE sac.sample_arbitrage_id = %s
        ''',
        (id_,)
    )
    assert curr.rowcount == 1
    profit_token_addr, profit_amount = curr.fetchone()
    profit_token_addr = w3.toChecksumAddress(profit_token_addr.tobytes())
    profit_amount = int(profit_amount)

    l.debug(f'Processing transaction id={id_} - 0x{txn_hash.hex()} (profit token {profit_token_addr}, profit amount {profit_amount})')
    receipt = w3.eth.get_transaction_receipt(txn_hash)
    swaps_uv2 = uv2.events.Swap().processReceipt(receipt, errors=DISCARD)
    swaps_uv3 = uv3.events.Swap().processReceipt(receipt, errors=DISCARD)
    all_swaps = [(True, x) for x in swaps_uv2] + [(False, x) for x in swaps_uv3]
    all_swaps = sorted(all_swaps, key = lambda x: x[1]['logIndex'])

    # get inferred exchanges
    curr.execute(
        '''
        SELECT sae.address, token_in.address, token_out.address
        FROM sample_arbitrage_cycles sac
        JOIN sample_arbitrage_cycle_exchanges sace ON sace.cycle_id = sac.id
        JOIN tokens token_in ON sace.token_in = token_in.id
        JOIN tokens token_out ON sace.token_out = token_out.id
        JOIN sample_arbitrage_cycle_exchange_items sacei ON sacei.cycle_exchange_id = sace.id
        JOIN sample_arbitrage_exchanges sae ON sae.id = sacei.exchange_id
        WHERE sac.sample_arbitrage_id = %s
        ''',
        (id_,)
    )
    assert 2 <= curr.rowcount <= 50

    exchanges_to_inferred_token_movements = {
        web3.Web3.toChecksumAddress(x.tobytes()): (
            web3.Web3.toChecksumAddress(ti.tobytes()),
            web3.Web3.toChecksumAddress(to.tobytes())
        )
        for (x, ti, to)
        in curr
    }

    l.debug(f'expecting {len(exchanges_to_inferred_token_movements)} exchanges')

    #
    # collect logs in the block
    #
    block = w3.eth.get_block(receipt['blockHash'])
    logs = []
    creates_uv2 = []
    creates_uv3 = []
    for txn_hash in block['transactions']:
        if txn_hash == receipt['transactionHash']:
            # reached the end
            break
        this_receipt = w3.eth.get_transaction_receipt(txn_hash)
        logs += this_receipt['logs']
        creates_uv2 += uv2_factory.events.PairCreated().processReceipt(this_receipt, errors=DISCARD)
        creates_uv3 += uv3_factory.events.PoolCreated().processReceipt(this_receipt, errors=DISCARD)
    else:
        raise Exception('failed to re-find transaction')


    #
    # see if we're backrunning the creation of an exchange
    #
    backrunning_creation = False
    for c in creates_uv2:
        if c['args']['pair'] in exchanges_to_inferred_token_movements and c['transactionIndex'] < receipt['transactionIndex']:
            backrunning_creation = True
    for c in creates_uv3:
        if c['args']['pool'] in exchanges_to_inferred_token_movements and c['transactionIndex'] < receipt['transactionIndex']:
            backrunning_creation = True

    if backrunning_creation:
        # early-exit
        l.debug(f'Transaction {receipt["transactionHash"].hex()} was backrunning the creation of an exchange')
        curr.execute(
            '''
            INSERT INTO sample_arbitrages_backrun_detection (sample_arbitrage_id, backran_creation)
            VALUES (%s, %s)
            ''',
            (id_, True),
        )
        return


    #
    # reconstruct cycle graph
    #
    exchange_to_details: typing.Dict[str, typing.Tuple[pricers.BaseExchangePricer, web3.types.LogReceipt]] = {k: None for k in exchanges_to_inferred_token_movements.keys()}

    g = nx.DiGraph()
    for is_v2, swap in all_swaps:
        address = swap['address']
        if address in exchanges_to_inferred_token_movements:
            if exchange_to_details[address] is not None:
                l.debug(f'address already present: {address}')
                # we must already have a cycle .. this is probably converting profit to desired token
                nx.find_cycle(g, source = profit_token_addr) # throws when no cycle is found

            # silly trick, both v2/v3 have same token0 / token1 abi
            tmp_contract = w3.eth.contract(address=address, abi=uv2.abi)
            token0 = tmp_contract.functions.token0().call()
            token1 = tmp_contract.functions.token1().call()
            del tmp_contract

            assert set([token0, token1]) == set(exchanges_to_inferred_token_movements[address])

            if is_v2:
                pricer = pricers.UniswapV2Pricer(w3, address, token0, token1)
            else:
                tmp_contract = w3.eth.contract(address=address, abi=uv3.abi)
                fee = tmp_contract.functions.fee().call()
                pricer = pricers.UniswapV3Pricer(w3, address, token0, token1, fee)

            exchange_to_details[address] = (pricer, swap)
            token_in, token_out = exchanges_to_inferred_token_movements[address]
            g.add_edge(token_in, token_out, pricer = pricer)


    assert all(v is not None for v in exchange_to_details.values()), 'expected to find all swap logs'

    # sanity check -- verify cycle exists
    cycle = nx.find_cycle(g, source = profit_token_addr)

    # construct the circuit pricer
    directions = []
    circuit = []
    for token_in, token_out in cycle:
        assert token_in != token_out
        # zero for one happens when token_in is lower than token_out
        zero_for_one = bytes.fromhex(token_in[2:]) < bytes.fromhex(token_out[2:])
        pricer = g[token_in][token_out]['pricer']

        directions.append(zero_for_one)
        circuit.append(pricer)

    # discover how much was put in
    first_exchange_addr = g[cycle[0][0]][cycle[0][1]]['pricer'].address
    pricer, log = exchange_to_details[first_exchange_addr]
    if isinstance(pricer, pricers.UniswapV2Pricer):
        if directions[0] == True:
            # zero for one
            amount_in = log['args']['amount0In']
        else:
            amount_in = log['args']['amount1In']
    else:
        assert isinstance(pricer, pricers.UniswapV3Pricer)
        if directions[0] == True:
            amount_in = log['args']['amount0']
        else:
            amount_in = log['args']['amount1']
    assert amount_in > 0, f'expected amount_in to be positive but got {amount_in}'

    #
    # price the top-of-block shot
    #
    pc = find_circuit.PricingCircuit(circuit, directions)
    amount_in, amount_out = pc.optimize(receipt['blockNumber'] - 1, lower_bound=1, upper_bound=((1 << 256) - 1))
    reorder_profit = amount_out - amount_in

    #
    # price the in-place shot
    #

    # expose all pricers to logs that occurred before the arbitrage
    any_touched = False
    for pricer in pc.circuit:
        my_logs = [x for x in logs if x['address'] == pricer.address]
        if len(my_logs) > 0:
            pricer.observe_block(my_logs)
            any_touched = True

    l.debug(f'Was any exchange touched in an earlier transaction? ({any_touched})')

    pc = find_circuit.PricingCircuit(circuit, directions)
    amount_in, amount_out = pc.optimize(receipt['blockNumber'] - 1, lower_bound=1, upper_bound=((1 << 256) - 1))
    rerun_profit = amount_out - amount_in
    l.debug(f'rerun profit: {rerun_profit}')

    # sanity
    if any_touched == False:
        assert reorder_profit == rerun_profit

    # record in database
    curr.execute(
        '''
        INSERT INTO sample_arbitrages_backrun_detection (sample_arbitrage_id, rerun_optimal_profit, reorder_optimal_profit, any_exchanges_touched)
        VALUES (%s, %s, %s, %s)
        ''',
        (id_, reorder_profit, rerun_profit, any_touched)
    )


if __name__ == '__main__':
    main()

