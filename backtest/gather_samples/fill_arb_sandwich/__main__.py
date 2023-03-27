import argparse
import collections
import logging
import os
import socket
import time
import web3
import web3.logs
import web3.types
import web3.exceptions
import typing
import psycopg2
import psycopg2.extensions
from backtest.gather_samples.analyses import get_addr_to_movements, get_arbitrage_from_receipt_if_exists, get_potential_exchanges
from backtest.utils import ERC20_TRANSFER_TOPIC, connect_db
from backtest.gather_samples.models import *

from utils import setup_logging, erc20

l = logging.getLogger(__name__)

DEBUG = False
DEBUG_TXN = None # bytes.fromhex('d7633d66ae6ae684a4350baa87256ab83ca051689cfe450cc259a2a400582643')
DEBUG_SAMPLE = 26511363

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setup-db', action='store_true')
    parser.add_argument('--fixup-db', action='store_true')
    parser.add_argument('--reset-db', action='store_true')

    parser.add_argument('--worker-name', type=str, default=None, help='worker name for log, must be POSIX path-safe')

    args = parser.parse_args()

    if args.worker_name is None:
        args.worker_name = socket.gethostname()
    job_name = 'fill_sandwich'
    setup_logging(job_name, worker_name = args.worker_name, root_dir='/data/robert/ethereum-arb/storage')

    db = connect_db()
    curr = db.cursor()

    if args.setup_db:
        setup_db(curr)
        return
    if args.fixup_db:
        fixup_db(curr)
        return
    if args.reset_db:
        reset_db(curr)
        return

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

    if DEBUG and DEBUG_TXN:
        curr.execute('SELECT block_number FROM sample_arbitrages WHERE txn_hash = %s', (DEBUG_TXN,))
        (debug_block,) = curr.fetchone()
        process_reservation(w3, curr, debug_block, debug_block, -1)
    elif DEBUG and DEBUG_SAMPLE:
        curr.execute('SELECT block_number FROM sample_arbitrages WHERE id = %s', (DEBUG_SAMPLE,))
        (debug_block,) = curr.fetchone()
        process_reservation(w3, curr, debug_block, debug_block, -1)
    else:
        while True:
            if not DEBUG:
                curr.connection.commit()
            curr.execute(
                '''
                SELECT id, start_block, end_block
                FROM arb_sandwich_reservations
                WHERE claimed_on IS NULL
                FOR UPDATE SKIP LOCKED
                '''
            )
            if curr.rowcount < 1:
                l.info('Done')
                break

            id_, start_block, end_block = curr.fetchone()
            curr.execute('UPDATE arb_sandwich_reservations SET claimed_on = now()::timestamp WHERE id = %s', (id_,))
            assert curr.rowcount == 1
            if not DEBUG:
                curr.connection.commit()

            l.info(f'Processing reservation {id_} from {start_block:,} to {end_block:,}')

            try:
                process_reservation(w3, curr, start_block, end_block, id_)
            except:
                l.exception(f'Reservation id={id_} failed')
                curr.connection.rollback()
                raise

            curr.execute('UPDATE arb_sandwich_reservations SET completed_on = now()::timestamp, progress = end_block WHERE id = %s', (id_,))
            assert curr.rowcount == 1


def setup_db(curr: psycopg2.extensions.cursor):
    curr.execute(
        '''
        CREATE TABLE IF NOT EXISTS arb_sandwich_reservations (
            id           SERIAL PRIMARY KEY NOT NULL,
            start_block  INTEGER NOT NULL,
            end_block    INTEGER NOT NULL,
            worker       TEXT,
            progress     INTEGER,
            claimed_on   TIMESTAMP WITHOUT TIME ZONE,
            heartbeat    TIMESTAMP WITHOUT TIME ZONE,
            completed_on TIMESTAMP WITHOUT TIME ZONE
        );

        CREATE INDEX IF NOT EXISTS idx_jliqr_claimed_on ON arb_sandwich_reservations (claimed_on);

        CREATE TABLE IF NOT EXISTS arb_sandwich_detections (
            id                            SERIAL NOT NULL PRIMARY KEY,
            reservation_id                INTEGER REFERENCES arb_sandwich_reservations (id) ON DELETE SET NULL,
            relayer                       BYTEA NOT NULL,
            block_number                  INTEGER NOT NULL,
            front_arbitrage_txn_hash      BYTEA NOT NULL,
            front_arbitrage_index         INTEGER NOT NULL,
            front_arbitrage_gas_price     BIGINT NOT NULL,
            front_arbitrage_gas_used      INTEGER NOT NULL,
            front_arbitrage_coinbase_xfer NUMERIC(78, 0),
            front_arbitrage_loss          NUMERIC(78, 0) NOT NULL,
            rear_arbitrage_txn_hash       BYTEA NOT NULL CHECK(front_arbitrage_txn_hash <> rear_arbitrage_txn_hash),
            rear_arbitrage_index          INTEGER NOT NULL,
            rear_arbitrage_gas_price      BIGINT NOT NULL,
            rear_arbitrage_gas_used       INTEGER NOT NULL,
            rear_arbitrage_coinbase_xfer  NUMERIC(78, 0) NOT NULL,
            rear_arbitrage_gain           NUMERIC(78, 0) NOT NULL,
            sample_arbitrage_id           INTEGER NOT NULL REFERENCES sample_arbitrages (id)
        );

        CREATE TABLE IF NOT EXISTS arb_sandwich_unknowns (
            id SERIAL               NOT NULL PRIMARY KEY,
            reservation_id          INTEGER REFERENCES arb_sandwich_reservations (id) ON DELETE SET NULL,
            relayer                 BYTEA NOT NULL,
            block_number            INTEGER NOT NULL,
            sample_arbitrage_id     INTEGER NOT NULL REFERENCES sample_arbitrages (id),
            possible_front_arb_idxs INTEGER[] NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_jit_ld_relayer ON arb_sandwich_detections USING HASH (relayer);
        CREATE INDEX IF NOT EXISTS idx_jit_said ON arb_sandwich_detections (sample_arbitrage_id);
        '''
    )
    curr.connection.commit()
    curr.execute('SELECT COUNT(*) FROM arb_sandwich_reservations')
    (n_ress,) = curr.fetchone()
    if n_ress == 0:
        l.info(f'Filling reservations')
        curr.execute('SELECT start_block, end_block FROM block_samples')
        for start_block, end_block in curr.fetchall():
            midpoint = (start_block + end_block) // 2
            curr.execute(
                'INSERT INTO arb_sandwich_reservations (start_block, end_block) VALUES (%s, %s)',
                (start_block, midpoint)
            )
            assert curr.rowcount == 1
            curr.execute(
                'INSERT INTO arb_sandwich_reservations (start_block, end_block) VALUES (%s, %s)',
                (midpoint + 1, end_block)
            )
            assert curr.rowcount == 1
        curr.connection.commit()


def fixup_db(curr: psycopg2.extensions.cursor):
    # re-set database
    curr.execute(
        '''
        DELETE FROM arb_sandwich_detections asd
        WHERE EXISTS(SELECT FROM arb_sandwich_reservations asr WHERE asr.id = asd.reservation_id AND asr.claimed_on is not null and asr.completed_on is null)
        '''
    )
    l.info(f'Deleted {curr.rowcount} resutls')
    curr.execute(
        '''
        DELETE FROM arb_sandwich_unknowns asd
        WHERE EXISTS(SELECT FROM arb_sandwich_reservations asr WHERE asr.id = asd.reservation_id AND asr.claimed_on is not null and asr.completed_on is null)
        '''
    )
    l.info(f'Deleted {curr.rowcount} unknowns')

    curr.execute(
        '''
        UPDATE arb_sandwich_reservations
        SET claimed_on = null, progress = null, worker = null
        WHERE claimed_on is not null and completed_on is null
        '''
    )
    l.info(f'Reset {curr.rowcount} reservations')
    input('ENTER to continue')
    curr.connection.commit()


def reset_db(curr: psycopg2.extensions.cursor):
    raise NotImplementedError('didnt get around to this')


def process_reservation(w3: web3.Web3, curr: psycopg2.extensions.cursor, start_block: int, end_block: int, reservation_id: int):
    if DEBUG and DEBUG_TXN:
        curr.execute('SELECT id, txn_hash, block_number FROM sample_arbitrages_no_fp WHERE txn_hash = %s', (DEBUG_TXN,))
    elif DEBUG and DEBUG_SAMPLE:
        curr.execute('SELECT id, txn_hash, block_number FROM sample_arbitrages_no_fp WHERE id = %s', (DEBUG_SAMPLE,))
    else:
        curr.execute(
            '''
            SELECT id, txn_hash, block_number
            FROM sample_arbitrages_no_fp sa
            WHERE %s <= block_number AND block_number <= %s AND n_cycles = 1
            ''',
            (start_block, end_block)
        )
    block_to_samples = collections.defaultdict(lambda: [])
    for id_, txn_hash, bn in curr:
        block_to_samples[bn].append((id_, txn_hash.tobytes()))

    l.info(f'Have {len(block_to_samples):,} blocks to examine from {start_block:,} to {end_block:,}')

    for block_number in sorted(block_to_samples.keys()):
        l.debug(f'Processing block {block_number:,}')
        # if block_number < 12802592:
        #     continue

        block = w3.eth.get_block(block_number, full_transactions=True)
        txn_to_index = {bytes(txn['hash']): txn['transactionIndex'] for txn in block['transactions']}

        for id_, txn_hash in block_to_samples[block_number]:
            l.debug(f'Detecting arb sandwiches provision for sample id={id_}')
            idx = txn_to_index[txn_hash]
            if idx < 2:
                l.debug(f'Sample {id_} too high in block')
                continue
            
            relayer = block['transactions'][idx]['to']
            # gather all transactions at least two prior to our index that also go to this relayer
            same_relayer_txns = [x for x in block['transactions'][:idx - 1] if x['to'] == relayer]
            if len(same_relayer_txns) == 0:
                l.debug(f'Sample {id_} has no other txns before it to the same relayer')
                continue

            # get receipts for all prior transactions
            same_relayer_txns_receipts: typing.List[web3.types.TxReceipt] = []
            for t in same_relayer_txns:
                same_relayer_txns_receipts.append(
                    w3.eth.get_transaction_receipt(t['hash'])
                )

            # remove all that don't have enough ERC-20 movement (and parse the erc-20 txns, too)
            same_relayer_txns_receipts_with_erc20: typing.List[typing.Tuple[web3.types.TxReceipt, typing.List]] = []
            for txn in same_relayer_txns_receipts:
                erc20s = []
                for log in txn['logs']:
                    if len(log['topics']) > 0 and log['topics'][0] == ERC20_TRANSFER_TOPIC:
                        try:
                            erc20s.append(
                                erc20.events.Transfer().processLog(log)
                            )
                        except web3.exceptions.LogTopicError:
                            l.warning(f'Could not parse erc20 log from {log["address"]}')
                            pass
                if len(erc20s) >= 3:
                    same_relayer_txns_receipts_with_erc20.append((txn, erc20s))
            
            if len(same_relayer_txns_receipts_with_erc20) == 0:
                l.debug(f'Sample {id_} has no other txns before it with enough erc20 transfers to be an arbitrage')

            # get original profit token
            curr.execute(
                '''
                SELECT t.address
                FROM sample_arbitrage_cycles sac
                JOIN tokens t ON t.id = sac.profit_token 
                WHERE sac.sample_arbitrage_id = %s
                ''',
                (id_,)
            )
            (original_profit_token,) = curr.fetchone()
            original_profit_token = w3.toChecksumAddress(original_profit_token.tobytes())

            same_relayer_txns_with_arb_analysis: typing.List[typing.Tuple[web3.types.TxReceipt, Arbitrage]] = []
            for txn, erc20s in same_relayer_txns_receipts_with_erc20:
                arb = get_arbitrage_from_receipt_if_exists(
                    txn,
                    erc20s,
                    least_profitable=True
                )
                if arb is not None and arb.n_cycles == 1 and arb.only_cycle.profit_token == original_profit_token:
                    same_relayer_txns_with_arb_analysis.append((txn, arb))

            if len(same_relayer_txns_with_arb_analysis) == 0:
                l.debug(f'Sample {id_} has no negative-profit arbitrages before it')
                continue

            # get original exchanges used
            curr.execute(
                '''
                SELECT DISTINCT sae.address
                FROM (SELECT * FROM sample_arbitrage_cycles WHERE sample_arbitrage_id = %s) sac
                JOIN sample_arbitrage_cycle_exchanges sace ON sace.cycle_id = sac.id
                JOIN sample_arbitrage_cycle_exchange_items sacei ON sacei.cycle_exchange_id = sace.id
                JOIN sample_arbitrage_exchanges sae ON sae.id = sacei.exchange_id
                ''',
                (id_,)
            )
            original_exchanges = set(w3.toChecksumAddress(x.tobytes()) for (x,) in curr)
            assert len(original_exchanges) >= 2

            same_relayer_txns_same_exchanges: typing.List[typing.Tuple[web3.types.TxReceipt, Arbitrage]] = []
            for txn, arb in same_relayer_txns_with_arb_analysis:
                arb_excs = set()
                for exc in arb.only_cycle.cycle:
                    for item in exc.items:
                        arb_excs.add(item.address)
                if arb_excs == original_exchanges:
                    same_relayer_txns_same_exchanges.append((txn, arb))


            if len(same_relayer_txns_same_exchanges) == 0:
                continue
            elif len(same_relayer_txns_same_exchanges) > 1:
                l.critical(f'Not sure how to handle id={id_}, inserting into unknown table')
                unknown_idxs = [x['transactionIndex'] for x, _ in same_relayer_txns_same_exchanges]
                curr.execute(
                    '''
                    INSERT INTO arb_sandwich_unknowns (reservation_id, relayer, block_number, sample_arbitrage_id, possible_front_arb_idxs)
                    VALUES (%(reservation_id)s, %(relayer)s, %(block_number)s, %(sample_arbitrage_id)s, %(possible_front_arb_idxs)s)
                    ''',
                    {
                        'reservation_id': reservation_id,
                        'relayer': bytes.fromhex(relayer[2:]),
                        'block_number': block_number,
                        'sample_arbitrage_id': id_,
                        'possible_front_arb_idxs': unknown_idxs,
                    }
                )
                assert curr.rowcount == 1
                l.debug(f'Sample id={id_} has no arbitrages that use the same exchanges before it')
            else:
                assert len(same_relayer_txns_same_exchanges) == 1
                # get the sandwiched transaction(s) and ensure they use exchanges we expect
                with_shared_exchanges: typing.List[typing.Tuple[web3.types.TxReceipt, typing.Set[str], Arbitrage]] = []
                for txn, arb in same_relayer_txns_same_exchanges:
                    prior_idx = txn['transactionIndex']
                    had_shared_exchange_txn = False
                    for middle_txn in block['transactions'][prior_idx + 1 : idx]:
                        middle_txn_receipt = w3.eth.get_transaction_receipt(middle_txn['hash'])
                        erc20s = erc20.events.Transfer().processReceipt(middle_txn_receipt, errors=web3.logs.DISCARD)
                        potential_exchanges = get_potential_exchanges(middle_txn_receipt, get_addr_to_movements(erc20s))
                        isxn = original_exchanges.intersection(potential_exchanges)
                        if len(isxn) > 0:
                            had_shared_exchange_txn = True
                            break
                    if had_shared_exchange_txn:
                        with_shared_exchanges.append((txn, isxn, arb))

                original_reciept = w3.eth.get_transaction_receipt(txn_hash)

                ((front_txn, front_arb),) = same_relayer_txns_same_exchanges
                l.debug(f'Sample {id_} was sandwiching, front transaction: {front_txn["transactionHash"].hex()}')
                l.debug(f'Sample {id_} in block {block_number:,} sandwiched: {front_txn["transactionIndex"]} <-> {original_reciept["transactionIndex"]}')

                # assert front_txn['transactionIndex'] + 2 == original_reciept['transactionIndex']

                curr.execute(
                    '''
                    SELECT coinbase_xfer, profit_amount 
                    FROM (select * from sample_arbitrages where id = %s) sa
                    JOIN sample_arbitrage_cycles sac ON sa.id = sac.sample_arbitrage_id
                    ''',
                    (id_,)
                )
                assert curr.rowcount == 1
                (coinbase_xfer, profit_amount) = curr.fetchone()

                curr.execute(
                    '''
                    INSERT INTO arb_sandwich_detections
                        (
                            reservation_id,
                            relayer, block_number,
                            front_arbitrage_txn_hash, front_arbitrage_index, front_arbitrage_gas_price, front_arbitrage_gas_used, front_arbitrage_loss,
                            rear_arbitrage_txn_hash, rear_arbitrage_index, rear_arbitrage_gas_price, rear_arbitrage_gas_used, rear_arbitrage_coinbase_xfer, rear_arbitrage_gain,
                            sample_arbitrage_id
                        )
                    VALUES (
                        %(reservation_id)s,
                        %(relayer)s, %(block_number)s,
                        %(front_arbitrage_txn_hash)s, %(front_arbitrage_index)s, %(front_arbitrage_gas_price)s, %(front_arbitrage_gas_used)s, %(front_arbitrage_loss)s,
                        %(rear_arbitrage_txn_hash)s, %(rear_arbitrage_index)s, %(rear_arbitrage_gas_price)s, %(rear_arbitrage_gas_used)s, %(rear_arbitrage_coinbase_xfer)s, %(rear_arbitrage_gain)s,
                        %(sample_arbitrage_id)s
                    )
                    ''',
                    {
                        'reservation_id': reservation_id,
                        'relayer': bytes.fromhex(relayer[2:]),
                        'block_number': block_number,
                        'front_arbitrage_txn_hash': bytes(front_txn['transactionHash']),
                        'front_arbitrage_index': front_txn['transactionIndex'],
                        'front_arbitrage_gas_price': front_txn['effectiveGasPrice'],
                        'front_arbitrage_gas_used': front_txn['gasUsed'],
                        'front_arbitrage_loss': front_arb.only_cycle.profit_amount,
                        'rear_arbitrage_txn_hash': bytes(txn_hash),
                        'rear_arbitrage_index': original_reciept['transactionIndex'],
                        'rear_arbitrage_gas_price': original_reciept['effectiveGasPrice'],
                        'rear_arbitrage_gas_used': original_reciept['gasUsed'],
                        'rear_arbitrage_coinbase_xfer': coinbase_xfer,
                        'rear_arbitrage_gain': profit_amount,
                        'sample_arbitrage_id': id_,
                    }
                )

            # if True:
            #     # some debugging
            #     for txn in same_relayer_txns:
            #         print(txn['hash'].hex())

            # import pdb; pdb.set_trace()
        if not DEBUG:
            curr.execute('UPDATE arb_sandwich_reservations SET progress = %s WHERE id = %s', (block_number, reservation_id))
            curr.connection.commit()
    if not DEBUG:
        curr.connection.commit()

if __name__ == '__main__':
    main()
