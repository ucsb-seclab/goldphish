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

from backtest.top_of_block.common import connect_db
from backtest.top_of_block.constants import FNAME_EXCHANGES_WITH_BALANCES, IMPORTANT_TOPICS_HEX, MIN_PROFIT_PREFILTER, THRESHOLDS, univ2_fname, univ3_fname
import pricers
import find_circuit
from pricers.uniswap_v2 import UniswapV2Pricer
from pricers.uniswap_v3 import UniswapV3Pricer
from utils import ProgressReporter, get_abi
import utils.profiling


l = logging.getLogger(__name__)

def seek_candidates(w3: web3.Web3):
    l.info('Starting candidate searching')
    db = connect_db()
    curr = db.cursor()
    time.sleep(4)

    try:
        setup_db(curr)
        load_exchange_balances(w3)
        pricer = load_pool(w3)

        #
        # Load all candidate profitable arbitrages
        start_block = get_resume_point(curr)
        l.info(f'Starting analysis from block {start_block:,}')
        end_block = w3.eth.get_block('latest')['number']
        progress_reporter = ProgressReporter(l, end_block, start_block)
        batch_size_blocks = 200
        curr_block = start_block
        while curr_block < end_block:
            this_end_block = min(curr_block + batch_size_blocks - 1, end_block)
            n_logs = 0
            for block_number, logs in get_relevant_logs(w3, curr_block, this_end_block):
                updated_exchanges = pricer.observe_block(logs)
                utils.profiling.maybe_log()
                while True:
                    try:
                        process_candidates(w3, pricer, block_number, updated_exchanges, curr)
                        db.commit()
                        break
                    except Exception as e:
                        db.rollback()
                        if 'execution aborted (timeout = 5s)' in str(e):
                            l.exception('Encountered timeout, trying again in a little bit')
                            time.sleep(30)
                        else:
                            raise e
                progress_reporter.observe(n_items=1)
                n_logs += len(logs)
            l.debug(f'Found {n_logs} logs this batch')
            curr_block = this_end_block + 1

    except Exception as e:
        l.exception('top-level exception')
        raise e


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
        """
    )
    curr.connection.commit()


def get_resume_point(curr: psycopg2.extensions.cursor):
    curr.execute('SELECT MAX(block_number) FROM candidate_arbitrages')
    (resume_point,) = curr.fetchone()
    if resume_point is None:
        return 12_369_621
    return resume_point + 1


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


def load_pool(w3: web3.Web3) -> pricers.PricerPool:
    l.debug('starting load of exchange graph')
    t_start = time.time()
    univ2_fname = os.path.abspath(os.path.dirname(__file__) + '/../univ2_excs.csv.gz')
    assert os.path.isfile(univ2_fname), f'should have file {univ2_fname}'
    univ3_fname = os.path.abspath(os.path.dirname(__file__) + '/../univ3_excs.csv.gz')
    assert os.path.isfile(univ3_fname)

    ret = pricers.PricerPool(w3)

    n_ignored = 0
    with open(FNAME_EXCHANGES_WITH_BALANCES) as fin:
        for line in fin:
            if line.startswith('2'):
                _, address, origin_block, token0, token1, bal0, bal1 = line.strip().split(',')

                origin_block = int(origin_block)
                bal0 = int(bal0)
                bal1 = int(bal1)
                if token0 in THRESHOLDS:
                    if bal0 < THRESHOLDS[token0]:
                        n_ignored += 1
                        continue
                if token1 in THRESHOLDS:
                    if bal1 < THRESHOLDS[token1]:
                        n_ignored += 1
                        continue
                ret.add_uniswap_v2(address, token0, token1, origin_block)
            else:
                assert line.startswith('3')

                _, address, origin_block, token0, token1, fee, bal0, bal1 = line.strip().split(',')
                fee = int(fee)

                origin_block = int(origin_block)
                bal0 = int(bal0)
                bal1 = int(bal1)
                if token0 in THRESHOLDS:
                    if bal0 < THRESHOLDS[token0]:
                        n_ignored += 1
                        continue
                if token1 in THRESHOLDS:
                    if bal1 < THRESHOLDS[token1]:
                        n_ignored += 1
                        continue
                ret.add_uniswap_v3(address, token0, token1, fee, origin_block)

    l.debug(f'Kept {ret.exchange_count:,} and ignored {n_ignored:,} exchanges below threshold ({n_ignored / (n_ignored + ret.exchange_count) * 100:.2f}%)')

    t_end = time.time()
    l.debug(f'Took {t_end - t_start:.2f} seconds to load into pricing pool')
    return ret


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
    for p in find_circuit.profitable_circuits(updated_exchanges, pool, block_number, only_weth_pivot=True):
        if p.profit < MIN_PROFIT_PREFILTER:
            n_ignored += 1
            continue

        if False:
            # some debugging
            exchange_outs = {}
            amount = p.amount_in
            for exc, dxn in zip(p.circuit, p.directions):
                if dxn == True:
                    amount_out = exc.exact_token0_to_token1(amount, block_identifier=block_number)
                else:
                    amount_out = exc.exact_token1_to_token0(amount, block_identifier=block_number)
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
                else:
                    amount_out = pricer.exact_token1_to_token0(amount, block_identifier=block_number)
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

    if n_ignored > 0:
        l.debug(f'Ignored {n_ignored} arbitrages due to not meeting profit threshold in block {block_number:,}')
    if n_found > 0:
        l.debug(f'Found {n_found} candidate arbitrages in block {block_number:,}')
