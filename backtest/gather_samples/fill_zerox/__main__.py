import argparse
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

import web3
import web3.contract
import web3._utils.filters
from backtest.utils import connect_db

from utils import get_abi, setup_logging


l = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose')
    parser.add_argument('--worker-name', type=str, default=None, help='worker name for log, must be POSIX path-safe')

    args = parser.parse_args()
    if args.worker_name is None:
        args.worker_name = socket.gethostname()


    setup_logging('fill_zerox', worker_name=args.worker_name, stdout_level=logging.DEBUG if args.verbose else logging.INFO)

    l.info('Starting zerox fill')

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

    setup_db(curr)

    start_block, end_block = scrape_range(curr)
    l.info(f'scraping from {start_block:,} to {end_block:,}')
    
    # scrape_v3(w3, curr, start_block, end_block)
    scrape_v4(w3, curr, start_block, end_block)


def setup_db(curr: psycopg2.extensions.cursor):
    curr.execute(
        '''
        CREATE TABLE IF NOT EXISTS sample_arbitrage_cycle_exchange_item_is_zerox (
            sample_arbitrage_cycle_exchange_item_id INTEGER NOT NULL REFERENCES sample_arbitrage_cycle_exchange_items (id) ON DELETE CASCADE,
            is_zerox BOOLEAN NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_sample_arbitrage_cycle_exchange_item_is_zerox_id
            ON sample_arbitrage_cycle_exchange_item_is_zerox (sample_arbitrage_cycle_exchange_item_id);
        '''
    )


def scrape_range(curr: psycopg2.extensions.cursor) -> typing.Tuple[int, int]:
    l.debug('computing work range')
    curr.execute(
        '''
        SELECT max(block_number)
        FROM sample_arbitrages sa
        JOIN sample_arbitrage_cycles sac ON sac.sample_arbitrage_id = sa.id
        JOIN sample_arbitrage_cycle_exchanges sace ON sace.cycle_id = sac.id
        JOIN sample_arbitrage_cycle_exchange_items sacei ON sacei.cycle_exchange_id = sace.id
        WHERE EXISTS(
            SELECT 1
            FROM sample_arbitrage_cycle_exchange_item_is_zerox is_zerox
            WHERE is_zerox.sample_arbitrage_cycle_exchange_item_id = sacei.id
        )
        '''
    )
    (last_end,) = curr.fetchone()
    if last_end is not None: # TODO put this back
        start_block = last_end + 1
    else:
        curr.execute('SELECT MIN(block_number) FROM sample_arbitrages')
        (start_block,) = curr.fetchone()
        # start_block = 14319428
    
    curr.execute('SELECT MAX(block_number) FROM sample_arbitrages')
    (end_block,) = curr.fetchone()
    return start_block, end_block


def scrape_v4(w3: web3.Web3, curr: psycopg2.extensions.cursor, start_block: int, end_block: int):
    start_block = 14324573

    l.info('scraping v4')
    zerox_proxy: web3.contract.Contract = w3.eth.contract(
        address = '0xDef1C0ded9bec7F1a1670819833240f027b25EfF',
        abi = get_abi('0x/IZeroEx.json')['compilerOutput']['abi'],
    )

    # find all transaction ids that we should query
    curr.execute(
        '''
        SELECT distinct sa.id
        FROM sample_arbitrages sa
        JOIN sample_arbitrage_cycles sac ON sac.sample_arbitrage_id = sa.id
        JOIN sample_arbitrage_cycle_exchanges sace ON sace.cycle_id = sac.id
        JOIN sample_arbitrage_cycle_exchange_items sacei ON sacei.cycle_exchange_id = sace.id
        JOIN sample_arbitrage_exchanges sae ON sacei.exchange_id = sae.id
        WHERE NOT EXISTS(SELECT 1 FROM uniswap_v2_exchanges e WHERE e.address = sae.address) AND
              NOT EXISTS(SELECT 1 FROM uniswap_v3_exchanges e WHERE e.address = sae.address) AND
              NOT EXISTS(SELECT 1 FROM sushiv2_swap_exchanges e WHERE e.address = sae.address) AND
              %s <= sa.block_number AND sa.block_number <= %s
        ''',
        (start_block, end_block),
    )
    l.debug(f'have {curr.rowcount} transactions to look through for zeroex v4')
    all_arb_ids: typing.List[int] = [x for (x,) in curr]

    start_time = time.time()

    for i, arb_id in enumerate(all_arb_ids):
        if i % 100 == 1:
            # status update
            elapsed = time.time() - start_time
            nps = i / elapsed
            n_remaining = len(all_arb_ids) - i
            remaining_sec = n_remaining / nps
            td_remaining = datetime.timedelta(seconds=remaining_sec)
            l.info(f'{i}/{len(all_arb_ids)} ({i / len(all_arb_ids)*100:.1f}%) -- ETA {td_remaining}')
            curr.connection.commit()


        curr.execute('SELECT txn_hash FROM sample_arbitrages WHERE id = %s', (arb_id,))
        (txn_hash,) = curr.fetchone()
        txn_hash = txn_hash.tobytes()
        l.debug(f'processing https://etherscan.io/tx/0x{txn_hash.hex()}')

        zerox_exchanges = set()

        receipt = w3.eth.get_transaction_receipt(txn_hash)
        for log in receipt['logs']:
            if log['address'] == zerox_proxy.address:
                
                # attempt to figure out what log this was
                for event in zerox_proxy.events._events:
                    topic = event_abi_to_log_topic(event)
                    if log['topics'][0] == topic:
                        event_obj = getattr(zerox_proxy.events, event['name'])()
                        parsed = event_obj.processLog(log)
                        break
                else:
                    print('Could not find topic!!!')
                    print(log)
                    raise Exception('Could not find topic')

                if 'maker' in parsed['args']:
                    zerox_exchanges.add(parsed['args']['maker'])

        if len(zerox_exchanges) > 0:
            l.debug(f'Found {len(zerox_exchanges)} exchanges')
            curr.execute(
                '''
                SELECT sacei.id, sae.address
                FROM (SELECT * FROM sample_arbitrages WHERE id = %s) sa
                JOIN sample_arbitrage_cycles sac ON sac.sample_arbitrage_id = sa.id
                JOIN sample_arbitrage_cycle_exchanges sace ON sace.cycle_id = sac.id
                JOIN sample_arbitrage_cycle_exchange_items sacei ON sacei.cycle_exchange_id = sace.id
                JOIN sample_arbitrage_exchanges sae ON sacei.exchange_id = sae.id
                ''',
                (arb_id,)
            )

            exchanges = set()
            for exc_item_id, baddr in list(curr):
                baddr = baddr.tobytes()
                address = w3.toChecksumAddress(baddr)
                assert address not in exchanges
                exchanges.add(address)
                if address in zerox_exchanges:
                    # insert this item!
                    curr.execute(
                        '''
                        INSERT INTO sample_arbitrage_cycle_exchange_item_is_zerox (sample_arbitrage_cycle_exchange_item_id, is_zerox)
                        VALUES (%s, %s)
                        ''',
                        (exc_item_id, True)
                    )
                    assert curr.rowcount == 1
                    l.info(f'Marked {address} as zerox in 0x{txn_hash.hex()}')



def scrape_v3(w3: web3.Web3, curr: psycopg2.extensions.cursor, start_block: int, end_block: int):
    zerox: web3.contract.Contract = w3.eth.contract(
        address = '0x61935CbDd02287B511119DDb11Aeb42F1593b7Ef',
        abi = get_abi('0x/exchange_proxy.abi.json'),
    )

    # get all known exchange addresses
    curr.execute('SELECT id, address FROM sample_arbitrage_exchanges')
    exchanges = {}
    for id_, baddr in curr:
        exchanges[web3.Web3.toChecksumAddress(baddr.tobytes())] = id_

    l.debug(f'Know about {len(exchanges):,} exchanges')

    exchanges_extended = []
    for addr in exchanges:
        exchange_extended = '0x' + addr[2:].rjust(64, '0')
        exchanges_extended.append(exchange_extended)

    exchanges_extended = set(bytes.fromhex(x[2:]) for x in exchange_extended)

    curr.execute(
        '''
        CREATE TEMP TABLE tmp_zeroxs (exchange_id INTEGER NOT NULL, txn_hash BYTEA NOT NULL);
        '''
    )


    # break into manageable batches
    batch_size = 2_000
    for i in itertools.count():
        batch_start = start_block + i * batch_size
        batch_end_exclusive = min(batch_start + batch_size, end_block + 1)
        l.info(f'Querying {batch_start:,} to {batch_end_exclusive - 1:,}')

        if batch_start > end_block:
            l.info('done.')
            break

        f: web3._utils.filters.Filter = w3.eth.filter({
            'address': '0x61935CbDd02287B511119DDb11Aeb42F1593b7Ef',
            'topics': ['0x6869791f0a34781b29882982cc39e882768cf2c96995c2a110c577c53bc932d5'],
            'fromBlock': batch_start,
            'toBlock': batch_end_exclusive,
        })
        logs = f.get_all_entries()

        l.debug(f'Have {len(logs):,} logs for this batch')

        n_marked = 0
        for log in logs:
            fill = zerox.events.Fill().processLog(log)
            if fill['args']['makerAddress'] not in exchanges:
                continue

            curr.execute(
                'INSERT INTO tmp_zeroxs (exchange_id, txn_hash) VALUES (%s, %s)',
                (exchanges[fill['args']['makerAddress']], log['transactionHash']),
            )

        curr.execute(
            '''
            INSERT INTO sample_arbitrage_cycle_exchange_item_is_zerox (sample_arbitrage_cycle_exchange_item_id, is_zerox)
            SELECT sacei.id, true
            FROM sample_arbitrages sa
            JOIN sample_arbitrage_cycles sac ON sac.sample_arbitrage_id = sa.id
            JOIN sample_arbitrage_cycle_exchanges sace ON sace.cycle_id = sac.id
            JOIN sample_arbitrage_cycle_exchange_items sacei ON sacei.cycle_exchange_id = sace.id
            JOIN tmp_zeroxs tz ON tz.txn_hash = sa.txn_hash AND tz.exchange_id = sacei.exchange_id
            '''
        )

        l.debug(f'Marked {curr.rowcount:,} as zerox')

        curr.execute('TRUNCATE TABLE tmp_zeroxs;')
        curr.connection.commit()


if __name__ == '__main__':
    main()
