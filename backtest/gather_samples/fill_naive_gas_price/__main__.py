import argparse
import collections
import datetime
import itertools
import logging
import os
import socket
import time
import typing
from eth_utils import event_abi_to_log_topic
import psycopg2
import psycopg2.extensions
import psycopg2.extras
import numpy as np


import web3
import web3.contract
import web3._utils.filters
from backtest.utils import connect_db

from utils import BALANCER_VAULT_ADDRESS, connect_web3, get_abi, setup_logging

ROLLING_WINDOW_SIZE_BLOCKS = 60 * 60 // 13 # about 1 hour

l = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose')
    parser.add_argument('--setup-db', action='store_true', dest='setup_db')
    parser.add_argument('--n-workers', type=int)
    parser.add_argument('--id', type=int)

    args = parser.parse_args()

    setup_logging('fill_naive_gas_price', stdout_level=logging.DEBUG if args.verbose else logging.INFO)

    db = connect_db()
    curr = db.cursor()

    if args.setup_db:
        setup_db(curr)
        db.commit()
        l.info('setup db')
        return

    assert args.n_workers > args.id
    assert args.id >= 0

    l.info('Starting naive gas price estimation...')

    gen_false_positives(curr)
    estimate_gas(curr, args.id, args.n_workers)

    
def setup_db(curr: psycopg2.extensions.cursor):
    curr.execute(
        '''
        CREATE TABLE IF NOT EXISTS naive_gas_price_estimate (
            block_number     INTEGER NOT NULL,
            niche            TEXT NOT NULL,
            gas_price_min    NUMERIC(78, 0) NOT NULL,
            gas_price_25th   NUMERIC(78, 0) NOT NULL,
            gas_price_median NUMERIC(78, 0) NOT NULL,
            gas_price_75th   NUMERIC(78, 0) NOT NULL,
            gas_price_max    NUMERIC(78, 0) NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_naive_gas_price_estimate_niche ON naive_gas_price_estimate (niche);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_naive_gas_price_estimate_niche_bn ON naive_gas_price_estimate (block_number, niche);
        '''
    )


def gen_false_positives(curr: psycopg2.extensions.cursor):
    """
    Build temp table 'tmp_false_positives'
    """
    t_start = time.time()

    curr.execute(
        '''
        CREATE TEMP TABLE tmp_false_positives (
            sample_arbitrage_id INTEGER NOT NULL,
            reason TEXT NOT NULL
        );

        CREATE INDEX idx_tmp_false_positives_id ON tmp_false_positives (sample_arbitrage_id);
        '''
    )

    # remove tokenlon
    curr.execute(
        '''
        INSERT INTO tmp_false_positives (sample_arbitrage_id, reason)
        SELECT id, 'tokenlon'
        FROM sample_arbitrages
        WHERE encode(shooter, 'hex') = '03f34be1bf910116595db1b11e9d1b2ca5d59659'
        '''
    )
    assert curr.rowcount > 0
    print(f'Labeled {curr.rowcount:,} samples as false-positive due to tokenlon')

    # remove CoW Swap
    curr.execute(
        '''
        INSERT INTO tmp_false_positives (sample_arbitrage_id, reason)
        SELECT id, 'CoW Swap'
        FROM sample_arbitrages
        WHERE encode(shooter, 'hex') = '9008d19f58aabd9ed0d60971565aa8510560ab41' OR encode(shooter, 'hex') = '3328f5f2cecaf00a2443082b657cedeaf70bfaef'
        '''
    )
    assert curr.rowcount > 0
    print(f'Labeled {curr.rowcount:,} samples as false-positive due to CoW Swap')

    # remove null address

    # find null exchange id
    curr.execute('SELECT id FROM sample_arbitrage_exchanges WHERE address = %s', (b'\x00' * 20,))
    assert curr.rowcount <= 1
    if curr.rowcount == 1:
        (zero_addr_exchange_id,) = curr.fetchone()
        print(f'Removing null exchange (id={zero_addr_exchange_id})')

        curr.execute(
            '''
            INSERT INTO tmp_false_positives (sample_arbitrage_id, reason)
            SELECT sa.id, 'null address'
            FROM sample_arbitrages sa
            WHERE EXISTS(
                SELECT 1
                FROM sample_arbitrage_cycles sac
                JOIN sample_arbitrage_cycle_exchanges sace ON sace.cycle_id = sac.id
                JOIN sample_arbitrage_cycle_exchange_items sacei ON sacei.cycle_exchange_id = sace.id
                WHERE sac.sample_arbitrage_id = sa.id AND sacei.exchange_id = %s
            )
            ''',
            (zero_addr_exchange_id,)
        )
        print(f'Labeled {curr.rowcount:,} arbitrages as false-positive due to null address')
    else:
        print('did not find zero address in exchanges')


    curr.execute('SELECT COUNT(distinct sample_arbitrage_id) FROM tmp_false_positives')
    (n_fp,) = curr.fetchone()

    curr.execute('SELECT COUNT(*) FROM sample_arbitrages')
    (tot_arbs,) = curr.fetchone()

    print(f'Have {n_fp:,} sample arbitrages labeled as false-positive ({n_fp / tot_arbs * 100:.2f}%)')

    # generate tables with false-positives removed
    curr.execute(
        '''
        CREATE TEMP TABLE sample_arbitrages_no_fp
        AS SELECT *
        FROM sample_arbitrages sa
        WHERE NOT EXISTS(SELECT 1 FROM tmp_false_positives WHERE sa.id = sample_arbitrage_id);
        '''
    )
    assert curr.rowcount == tot_arbs - n_fp
    curr.execute('CREATE INDEX idx_sample_arbitrages_no_fp_id ON sample_arbitrages_no_fp (id);')
    curr.execute('CREATE INDEX idx_sample_arbitrages_no_fp_bn ON sample_arbitrages_no_fp (block_number);')

    print(f'Took {time.time() - t_start:.1f} seconds to generate false-positive report')
    print()

def estimate_gas(curr: psycopg2.extensions.cursor, id_: int, n_workers: int):
    w3 = connect_web3()
    curr.execute('SELECT MIN(block_number), MAX(block_number) FROM sample_arbitrages')
    curr2 = curr.connection.cursor()
    min_block, max_block = curr.fetchone()

    slice_width = (max_block - min_block) // n_workers
    slice_start = min_block + slice_width * id_
    slice_end_exclusive = min(max_block + 1, min_block + slice_width * (id_ + 1))

    l.debug(f'min_block={min_block:,} max_block={max_block:,}')

    curr.execute(
        '''
        SELECT *, EXISTS(SELECT 1 FROM flashbots_transactions ft WHERE ft.transaction_hash = c.txn_hash) is_flashbots
            FROM (
            SELECT
                b.id,
                b.block_number,
                b.gas_price,
                b.coinbase_xfer,
                b.txn_hash,
                bool_or(is_uniswap_v2) has_uniswap_v2,
                bool_or(is_uniswap_v3) has_uniswap_v3,
                bool_or(is_sushiswap) has_sushiswap,
                bool_or(is_shibaswap) has_shibaswap,
                bool_or(is_balancer_v1) has_balancer_v1,
                bool_or(is_balancer_v2) has_balancer_v2,
                bool_and(is_known) all_known,
                count(*) n_exchanges
            FROM (
                SELECT *, is_uniswap_v2 or is_uniswap_v3 or is_sushiswap or is_shibaswap or is_balancer_v1 or is_balancer_v2 is_known
                FROM (
                    SELECT
                        sa.id,
                        sa.block_number,
                        sa.gas_price,
                        sa.coinbase_xfer,
                        sa.txn_hash,
                        EXISTS(SELECT 1 FROM uniswap_v2_exchanges e WHERE e.address = sae.address) is_uniswap_v2,
                        EXISTS(SELECT 1 FROM uniswap_v3_exchanges e WHERE e.address = sae.address) is_uniswap_v3,
                        EXISTS(SELECT 1 FROM sushiv2_swap_exchanges e WHERE e.address = sae.address) is_sushiswap,
                        EXISTS(SELECT 1 FROM shibaswap_exchanges e WHERE e.address = sae.address) is_shibaswap,
                        EXISTS(SELECT 1 FROM balancer_exchanges e WHERE e.address = sae.address) is_balancer_v1,
                        sae.address = '\\xBA12222222228d8Ba445958a75a0704d566BF2C8'::bytea is_balancer_v2
                    FROM (SELECT * FROM sample_arbitrages_no_fp where %s <= block_number and block_number < %s) sa
                    JOIN sample_arbitrage_backrun_detections bd ON bd.sample_arbitrage_id = sa.id
                    JOIN sample_arbitrage_cycles sac ON sac.sample_arbitrage_id = sa.id
                    JOIN sample_arbitrage_cycle_exchanges sace ON sace.cycle_id = sac.id
                    JOIN sample_arbitrage_cycle_exchange_items sacei ON sacei.cycle_exchange_id = sace.id
                    JOIN sample_arbitrage_exchanges sae ON sae.id = sacei.exchange_id
                    WHERE bd.rerun_exactly = true
                ) a
            ) b
            GROUP BY b.id, b.block_number, b.gas_price, b.coinbase_xfer, b.txn_hash
        ) c
        WHERE all_known = true AND n_exchanges <= 3
        ORDER BY block_number ASC
        ''',
        (slice_start - (slice_width + 2), slice_end_exclusive),
    )

    niches_updated = set()
    last_block_number = min_block

    t_start = time.time()
    t_last_update = time.time()
    rolling_windows = collections.defaultdict(lambda: collections.deque())

    for id_, block_number, gas_price, coinbase_xfer, txn_hash, has_uniswap_v2, has_uniswap_v3, has_sushiswap, has_shibaswap, has_balancer_v1, has_balancer_v2, all_known, n_exchanges, is_flashbots in list(curr):

        if block_number > last_block_number:
            # push update
            for niche in niches_updated:
                window = rolling_windows[niche]
                # trim rolling window
                while len(window) > 0 and window[0][0] <= last_block_number - ROLLING_WINDOW_SIZE_BLOCKS:
                    rolling_windows[niche].popleft()

                if slice_start <= block_number < slice_end_exclusive:
                    niche_min, qt1, med, qt3, niche_max = np.percentile([x for _, x in window], [0, 25, 50, 75, 100], method='closest_observation')

                    curr2.execute(
                        '''
                        INSERT INTO naive_gas_price_estimate
                        (block_number, niche, gas_price_min, gas_price_25th, gas_price_median, gas_price_75th, gas_price_max)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ''',
                        (last_block_number, niche, int(niche_min), int(qt1), int(med), int(qt3), int(niche_max))
                    )
            niches_updated.clear()
            last_block_number = block_number

        assert all_known
        
        if time.time() > t_last_update + 20:
            t_last_update = time.time()

            blocks_processed = block_number - slice_start
            elapsed = time.time() - t_start
            nps = blocks_processed / elapsed
            remaining = slice_end_exclusive - block_number
            eta_s = remaining / nps
            eta = datetime.timedelta(seconds = eta_s)

            l.info(f'Processed {blocks_processed:,} blocks ({blocks_processed / (slice_end_exclusive - slice_start)*100:.2f}%) ETA={eta}')

        if coinbase_xfer > 0:
            txn_hash = txn_hash.tobytes()
            coinbase_xfer = int(coinbase_xfer)
            # must get transaction to compute effective gas price
            l.debug(f'getting transaction for {id_} {txn_hash.hex()} to re-compute gas price')
            receipt = w3.eth.get_transaction_receipt(txn_hash)
            gas_used = receipt['gasUsed']

            gas_price = (gas_used * receipt['effectiveGasPrice'] + coinbase_xfer) // gas_used

        # construct niche string
        niche_sz = ''

        if is_flashbots:
            niche_sz += 'fb|'
        else:
            niche_sz += 'nfb|'
        
        niche_sz += str(n_exchanges) + '|'
        
        if has_uniswap_v2:
            niche_sz += 'uv2|'
        if has_uniswap_v3:
            niche_sz += 'uv3|'
        if has_sushiswap:
            niche_sz += 'sushi|'
        if has_shibaswap:
            niche_sz += 'shiba|'
        if has_balancer_v1:
            niche_sz += 'balv1|'
        if has_balancer_v2:
            niche_sz += 'balv2|'

        niches_updated.add(niche_sz)
        rolling_windows[niche_sz].append((block_number, gas_price))

    l.info('Done')
    curr.connection.commit()

if __name__ == '__main__':
    main()
