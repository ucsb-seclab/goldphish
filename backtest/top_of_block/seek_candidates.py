import collections
import gzip
import logging
import os
import time
import typing
import web3
import web3.types
import web3._utils.filters
import psycopg2
import psycopg2.extensions

from backtest.top_of_block.common import connect_db, load_pool
from backtest.top_of_block.constants import FNAME_EXCHANGES_WITH_BALANCES, IMPORTANT_TOPICS_HEX, MIN_PROFIT_PREFILTER, THRESHOLDS, univ2_fname, univ3_fname
from backtest.utils import CancellationToken
import pricers
import find_circuit
from pricers.uniswap_v2 import UniswapV2Pricer
from pricers.uniswap_v3 import UniswapV3Pricer
from utils import ProgressReporter, get_abi
import utils.profiling


l = logging.getLogger(__name__)

LOG_BATCH_SIZE = 200
RESERVATION_SIZE = 1 * 24 * 60 * 60 // 13 # about 1 days' worth


DEBUG = False


def seek_candidates(w3: web3.Web3, job_name: str, worker_name: str):
    l.info('Starting candidate searching')
    db = connect_db()
    curr = db.cursor()
    time.sleep(4)

    setup_db(curr)
    load_exchange_balances(w3)
    cancellation_token = CancellationToken(job_name, worker_name, connect_db())

    #
    # Load all candidate profitable arbitrages
    start_block = 12_369_621
    end_block = 14_324_572 # w3.eth.get_block('latest')['number']
    l.info(f'Doing analysis from block {start_block:,} to block {end_block:,}')
    progress_reporter = ProgressReporter(l, end_block, start_block)
    batch_size_blocks = 200 # batch size for getting logs
    last_processed_block = None
    while not cancellation_token.cancel_requested():
        maybe_rez = get_reservation(curr, start_block, end_block)
        if maybe_rez is None:
            # we're at the end
            break

        reservation_id, reservation_start, reservation_end = maybe_rez
        pricer = load_pool(w3)
        if last_processed_block is not None:
            assert last_processed_block < reservation_start
            progress_reporter.observe(n_items=((reservation_start - last_processed_block) - 1))

        curr_block = reservation_start
        while curr_block <= reservation_end:
            this_end_block = min(curr_block + batch_size_blocks - 1, reservation_end)
            for block_number, logs in get_relevant_logs(w3, curr_block, this_end_block):
                updated_exchanges = pricer.observe_block(logs)
                utils.profiling.maybe_log()
                while True:
                    try:
                        process_candidates(w3, pricer, block_number, updated_exchanges, curr)
                        if not DEBUG:
                            db.commit()
                        break
                    except Exception as e:
                        db.rollback()
                        if 'execution aborted (timeout = 5s)' in str(e):
                            l.exception('Encountered timeout, trying again in a little bit')
                            time.sleep(30)
                        else:
                            raise e

            progress_reporter.observe(n_items = (this_end_block - curr_block)  + 1)
            curr_block = this_end_block + 1
            last_processed_block = this_end_block

            if cancellation_token.cancel_requested():
                l.info('exiting due to cancellation requested')
                break

        # mark reservation as completed
        if not DEBUG:
            if not cancellation_token.cancel_requested():
                assert this_end_block == reservation_end
                l.debug(f'Completed reservation id={reservation_id:,}')
                curr.execute(
                    'UPDATE candidate_arbitrage_reservations SET completed_on = NOW()::timestamp WHERE id = %s',
                    (reservation_id,)
                )
                db.commit()
            else:
                # cancellation was requested
                if curr_block <= reservation_end:
                    l.info('Splitting off unifinished reservation into new one')
                    curr.execute(
                        '''
                        UPDATE candidate_arbitrage_reservations SET block_number_end = %s, completed_on = NOW()::timestamp WHERE id = %s
                        ''',
                        (this_end_block, reservation_id),
                    )
                    curr.execute(
                        '''
                        INSERT INTO candidate_arbitrage_reservations (block_number_start, block_number_end)
                        VALUES (%s, %s)
                        RETURNING id
                        ''',
                        (curr_block, reservation_end)
                    )
                    (new_id,) = curr.fetchone()
                    l.debug(f'Created new reservation id={new_id:,} {curr_block:,} -> {reservation_end:,}')
                    db.commit()


def setup_db(curr: psycopg2.extensions.cursor):
    curr.execute(
        """
        CREATE TABLE IF NOT EXISTS candidate_arbitrages (
            id              SERIAL PRIMARY KEY NOT NULL,
            block_number    INTEGER NOT NULL,
            exchanges       bytea[] NOT NULL,
            directions      boolean[] NOT NULL,
            amount_in       NUMERIC(78, 0) NOT NULL,
            profit_no_fee   NUMERIC(78, 0) NOT NULL,
            verify_started  TIMESTAMP WITHOUT TIME ZONE,
            verify_finished TIMESTAMP WITHOUT TIME ZONE,
            verify_run      BOOLEAN DEFAULT FALSE
        );

        CREATE INDEX IF NOT EXISTS idx_candidate_arbitrages_block_number ON candidate_arbitrages (block_number);
        CREATE INDEX IF NOT EXISTS idx_candidate_arbitrages_verify_run ON candidate_arbitrages (verify_run);

        CREATE TABLE IF NOT EXISTS candidate_arbitrage_reservations (
            id                 SERIAL PRIMARY KEY NOT NULL,
            block_number_start INTEGER NOT NULL,
            block_number_end   INTEGER NOT NULL,
            claimed_on         TIMESTAMP WITHOUT TIME ZONE,
            completed_on       TIMESTAMP WITHOUT TIME ZONE 
        );

        CREATE TABLE IF NOT EXISTS candidate_arbitrage_blocks_to_verify (
            block_number INTEGER PRIMARY KEY NOT NULL,
            max_profit_no_fee NUMERIC(78, 0) NOT NULL,
            verify_started  TIMESTAMP WITHOUT TIME ZONE,
            verify_finished TIMESTAMP WITHOUT TIME ZONE
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_blocks_to_verify_block_number ON candidate_arbitrage_blocks_to_verify(block_number);
        CREATE INDEX IF NOT EXISTS idx_candidate_arbitrage_blocks_to_verify_max_profit_no_fee ON candidate_arbitrage_blocks_to_verify (max_profit_no_fee);
        """
    )
    curr.connection.commit()


def get_reservation(curr: psycopg2.extensions.cursor, start_block: int, end_block: int) -> typing.Optional[typing.Tuple[int, int, int]]:
    curr.execute('BEGIN TRANSACTION')
    curr.execute('LOCK TABLE candidate_arbitrage_reservations') # for safety
    query_do_reservation = '''
        UPDATE candidate_arbitrage_reservations car
        SET claimed_on = NOW()::timestamp
        FROM (
            SELECT id
            FROM candidate_arbitrage_reservations
            WHERE claimed_on IS NULL AND completed_on IS NULL
            ORDER BY block_number_start ASC
            LIMIT 1
        ) x
        WHERE car.id = x.id
        RETURNING car.id, car.block_number_start, car.block_number_end
    '''
    curr.execute(query_do_reservation)
    maybe_row = curr.fetchall()

    if len(maybe_row) == 0:
        # no reservations left -- are we at the end, or just need to insert some?
        curr.execute('SELECT MAX(block_number_end) FROM candidate_arbitrage_reservations')

        maybe_last_reservation = curr.fetchone()[0]
        if maybe_last_reservation:
            assert maybe_last_reservation > start_block

            # if we're at the end return None to indicate quit
            if maybe_last_reservation >= end_block:
                l.info(f'Reached end of reservations.')
                curr.execute('ROLLBACK')
                return None

            start_block = maybe_last_reservation + 1

        l.info(f'Inserting more reservations into the database...')
        for i in range(10):
            batch_start = start_block + i * RESERVATION_SIZE
            batch_end = min(end_block, start_block + (i + 1) * RESERVATION_SIZE - 1)
            if batch_start >= batch_end:
                # we reached the end
                break
            curr.execute(
                'INSERT INTO candidate_arbitrage_reservations (block_number_start, block_number_end) VALUES (%s, %s)',
                (batch_start, batch_end)
            )

        # insert done, select our reservation
        curr.execute(query_do_reservation)
        maybe_row = curr.fetchall()

    assert len(maybe_row) == 1
    id_, start, end = maybe_row[0]
    assert start >= start_block
    assert end <= end_block
    assert start <= end

    l.info(f'Processing reservation id={id_:,} from={start:,} to end={end:,} ({end - start:,} blocks)')

    if not DEBUG:
        curr.execute('COMMIT')

    return id_, start, end


def load_exchange_balances(w3: web3.Web3):
    if os.path.exists(FNAME_EXCHANGES_WITH_BALANCES):
        l.debug(f'already prefetched balances (\'{FNAME_EXCHANGES_WITH_BALANCES}\' exists), no need to redo')
        return

    tip = w3.eth.get_block('latest')
    block_number = tip['number'] - 10
    l.debug(f'Prefiltering based off balances in block {block_number:,}')

    with open(FNAME_EXCHANGES_WITH_BALANCES + '.tmp', mode='w') as fout:
        with gzip.open(univ2_fname, mode='rt') as fin:
            for i, line in enumerate(fin):
                if i % 100 == 0:
                    l.debug(f'Processed {i:,} uniswap v2 exchanges for prefilter')
                    fout.flush()
                address, origin_block, token0, token1 = line.strip().split(',')
                address = w3.toChecksumAddress(address)
                origin_block = int(origin_block)
                token0 = w3.toChecksumAddress(token0)
                token1 = w3.toChecksumAddress(token1)

                # record liquidity balance of both token0 and token1
                try:
                    bal_token0 = w3.eth.contract(
                        address=token0,
                        abi=get_abi('erc20.abi.json'),
                    ).functions.balanceOf(address).call()
                    bal_token1 = w3.eth.contract(
                        address=token1,
                        abi=get_abi('erc20.abi.json'),
                    ).functions.balanceOf(address).call()
                    fout.write(f'2,{address},{origin_block},{token0},{token1},{bal_token0},{bal_token1}\n')
                except:
                    l.exception('could not get balance, ignoring')


        l.debug('loaded all uniswap v2 exchanges')

        with gzip.open(univ3_fname, mode='rt') as fin:
            for i, line in enumerate(fin):
                if i % 100 == 0:
                    l.debug(f'Processed {i:,} uniswap v3 exchanges for prefilter')
                    fout.flush()
                address, origin_block, token0, token1, fee = line.strip().split(',')
                address = w3.toChecksumAddress(address)
                origin_block = int(origin_block)
                token0 = w3.toChecksumAddress(token0)
                token1 = w3.toChecksumAddress(token1)
                fee = int(fee)

                # record liquidity balance of both token0 and token1
                try:
                    bal_token0 = w3.eth.contract(
                        address=token0,
                        abi=get_abi('erc20.abi.json'),
                    ).functions.balanceOf(address).call()
                    bal_token1 = w3.eth.contract(
                        address=token1,
                        abi=get_abi('erc20.abi.json'),
                    ).functions.balanceOf(address).call()
                    fout.write(f'3,{address},{origin_block},{token0},{token1},{fee},{bal_token0},{bal_token1}\n')
                except:
                    l.exception('could not get balance, ignoring')

        l.debug('finished load of exchange graph')
    os.rename(FNAME_EXCHANGES_WITH_BALANCES + '.tmp', FNAME_EXCHANGES_WITH_BALANCES)


def get_relevant_logs(w3: web3.Web3, start_block: int, end_block: int) -> typing.Iterator[typing.Tuple[int, typing.List[web3.types.LogReceipt]]]:
    assert start_block <= end_block
    f: web3._utils.filters.Filter = w3.eth.filter({
        'topics': [IMPORTANT_TOPICS_HEX],
        'fromBlock': start_block,
        'toBlock': end_block,
    })
    logs = f.get_all_entries()
    gather = collections.defaultdict(lambda: [])
    for log in logs:
        gather[log['blockNumber']].append(log)

    for i in range(start_block, end_block + 1):
        yield (i, gather[i])


def process_candidates(w3: web3.Web3, pool: pricers.PricerPool, block_number: int, updated_exchanges: typing.Set[str], curr: psycopg2.extensions.cursor):
    # update pricer
    l.debug(f'{len(updated_exchanges)} exchanges updated in block {block_number:,}')

    n_ignored = 0
    n_found = 0
    max_profit_no_fee = -1
    for p in find_circuit.profitable_circuits(updated_exchanges, pool, block_number, only_weth_pivot=True):
        if p.profit < MIN_PROFIT_PREFILTER:
            n_ignored += 1
            continue

        if DEBUG:
            # some debugging
            exchange_outs = {}
            amount = p.amount_in
            for exc, dxn in zip(p.circuit, p.directions):
                if dxn == True:
                    amount_out = exc.exact_token0_to_token1(amount, block_identifier=block_number)
                    token_out = exc.token1
                else:
                    amount_out = exc.exact_token1_to_token0(amount, block_identifier=block_number)
                    token_out = exc.token0
                amount_out = pricers.token_transfer.out_from_transfer(token_out, amount_out)
                l.debug(f'{exc.address} out={amount_out}')
                amount = amount_out
                exchange_outs[exc.address] = amount_out
            if amount != p.amount_in + p.profit:
                l.warning(f'did not match up with profit! computed amount_out={amount_out} with profit={p.profit}')
            # run it again using fresh pricers
            amount = p.amount_in
            for exc, dxn in zip(p.circuit, p.directions):
                if isinstance(exc, UniswapV2Pricer):
                    pricer = UniswapV2Pricer(w3, exc.address, exc.token0, exc.token1)
                else:
                    assert isinstance(exc, UniswapV3Pricer)
                    pricer = UniswapV3Pricer(w3, exc.address, exc.token0, exc.token1, exc.fee)
                if dxn == True:
                    amount_out = pricer.exact_token0_to_token1(amount, block_identifier=block_number)
                    token_out = pricer.token1
                else:
                    amount_out = pricer.exact_token1_to_token0(amount, block_identifier=block_number)
                    token_out = pricer.token0
                amount_out = pricers.token_transfer.out_from_transfer(token_out, amount_out)
                l.debug(f'fresh {exc.address} out={amount_out}')
                if exchange_outs[exc.address] != amount_out:
                    l.debug(f'Amount_out {exc.address} changed from {exchange_outs[exc.address]} to {amount_out}')
                amount = amount_out
            if amount != p.amount_in + p.profit:
                l.debug('This did not match after fresh pricer!!!!!!!! weird!!!!')
                l.debug('----------------------------')
                l.debug(f'Found arbitrage')
                l.debug(f'block_number ..... {block_number}')
                l.debug(f'amount_in ........ {p.amount_in}')
                l.debug(f'expected profit .. {p.profit}')
                for p, dxn in zip(p.circuit, p.directions):
                    l.debug(f'    address={p.address} zeroForOne={dxn}')
                l.debug('----------------------------')
                raise Exception('what is this')


        # this is potentially profitable, log as a candidate
        curr.execute(
            """
            INSERT INTO candidate_arbitrages (block_number, exchanges, directions, amount_in, profit_no_fee)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING (id)
            """,
            (block_number, [bytes.fromhex(x.address[2:]) for x in p.circuit], p.directions, p.amount_in, p.profit)
        )
        (inserted_id,) = curr.fetchone()
        # l.debug(f'inserted candidate id={inserted_id}')
        n_found += 1
        
        max_profit_no_fee = max(max_profit_no_fee, p.profit)

    if max_profit_no_fee > 0:
        curr.execute(
            """
            INSERT INTO candidate_arbitrage_blocks_to_verify (block_number, max_profit_no_fee) VALUES (%s, %s)
            """,
            (block_number, max_profit_no_fee)
        )

    if n_ignored > 0:
        l.debug(f'Ignored {n_ignored} arbitrages due to not meeting profit threshold in block {block_number:,}')
    if n_found > 0:
        l.debug(f'Found {n_found} candidate arbitrages in block {block_number:,}')
