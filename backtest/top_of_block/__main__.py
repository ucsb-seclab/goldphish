import collections
import io
import os
import typing
import networkx as nx
import sys
import web3
import web3.types
import web3.exceptions
import web3.contract
import web3._utils.events
import web3._utils.filters
import time
import logging
import gzip
import pricers
import find_circuit
import logging.handlers
from eth_utils import event_abi_to_log_topic

from pricers.uniswap_v2 import UniswapV2Pricer
from ..utils import mine_block
from utils import TETHER_ADDRESS, UNI_ADDRESS, USDC_ADDRESS, WBTC_ADDRESS, WETH_ADDRESS, ProgressReporter, get_abi, decode_trace_calls, get_block_logs, pretty_print_trace

l = logging.getLogger(__name__)

THRESHOLDS = {
    WETH_ADDRESS: web3.Web3.toWei('0.01', 'ether'),
    USDC_ADDRESS: 10 * (10 ** 6),
    TETHER_ADDRESS: 10 * (10 ** 6),
    UNI_ADDRESS: web3.Web3.toWei('0.25', 'ether'), # UNI also uses 18 decimals and had a max price of 40, so this is about $400, optimistically
    WBTC_ADDRESS: 1 * (10 ** 8) // 10_000, # like $1-4 ish?
}

IMPORTANT_TOPICS = []

# Set up the important log topics we'll need to listen
univ2: web3.contract.Contract = web3.Web3().eth.contract(
    address=b'\x00' * 20,
    abi=get_abi('uniswap_v2/IUniswapV2Pair.json')['abi'],
)
IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ2.events.Mint().abi))
IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ2.events.Burn().abi))
IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ2.events.Swap().abi))
IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ2.events.Sync().abi))

univ3: web3.contract.Contract = web3.Web3().eth.contract(
    address=None,
    abi=get_abi('uniswap_v3/IUniswapV3Pool.json')['abi'],
)
IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ3.events.Mint().abi))
IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ3.events.Burn().abi))
IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ3.events.Initialize().abi))
IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ3.events.SetFeeProtocol().abi)) # I'm not sure this is ever used?
IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ3.events.CollectProtocol().abi)) # I'm not sure this is ever used?
IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ3.events.Swap().abi))

IMPORTANT_TOPICS_HEX = ['0x' + x.hex() for x in IMPORTANT_TOPICS]


FNAME_EXCHANGES_WITH_BALANCES = '/mnt/goldphish/tmp/exchanges_prefilter.csv'
def load_exchange_balances(w3: web3.Web3):
    if os.path.exists(FNAME_EXCHANGES_WITH_BALANCES):
        l.debug(f'already prefetched balances (\'{FNAME_EXCHANGES_WITH_BALANCES}\' exists), no need to redo')
        return

    univ2_fname = os.path.abspath(os.path.dirname(__file__) + '/../univ2_excs.csv.gz')
    assert os.path.isfile(univ2_fname), f'should have file {univ2_fname}'
    univ3_fname = os.path.abspath(os.path.dirname(__file__) + '/../univ3_excs.csv.gz')
    assert os.path.isfile(univ3_fname)

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


# profit must be enough to pay for 100k gas @ 10 gwei (both overly optimistic)
MIN_PROFIT_PREFILTER = (100_000) * (10 * (10 ** 9))

def process_candidates(w3: web3.Web3, pool: pricers.PricerPool, block_number: int, block_logs: typing.List[web3.types.LogReceipt], fout: io.TextIOWrapper, fout_errors: io.TextIOWrapper):
    # update pricer
    updated_exchanges = pool.observe_block(block_logs)
    l.debug(f'{len(updated_exchanges)} exchanges updated in block {block_number:,}')

    n_ignored = 0
    n_found = 0
    for p in find_circuit.profitable_circuits(updated_exchanges, pool, block_number):
        if p.profit < MIN_PROFIT_PREFILTER:
            n_ignored += 1
            continue

        # this is potentially profitable, log as a candidate
        exchanges = [f'{x.address}|{isinstance(x, UniswapV2Pricer)}' for x in p.circuit]
        exchanges_str = '#'.join(exchanges)
        fout.write(f'{block_number},{exchanges_str},{p.profit},{w3.fromWei(p.profit, "ether")}\n')
        n_found += 1

    fout.flush()

    if n_ignored > 0:
        l.debug(f'Ignored {n_ignored} arbitrages due to not meeting profit threshold in block {block_number:,}')
    if n_found > 0:
        l.debug(f'Found {n_found} candidate arbitrages in block {block_number:,}')


def main():
    #
    # Set up logging
    #

    root_logger = logging.getLogger()
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(fmt)
    fh = logging.handlers.WatchedFileHandler(
        '/mnt/goldphish/tmp/reshoot_top_block_log.txt'
    )
    fh.setFormatter(fmt)
    root_logger.addHandler(sh)
    root_logger.addHandler(fh)
    root_logger.setLevel(logging.DEBUG)

    # silence some annoying logs from subsystems
    for lname in ['websockets.protocol', 'web3.providers.WebsocketProvider',
                  'web3.RequestManager', 'websockets.server', 'asyncio']:
        logging.getLogger(lname).setLevel(logging.WARNING)

    l.info('Booting up...')

    #
    # Connect to web3
    #

    w3 = web3.Web3(web3.WebsocketProvider(
        'ws://172.17.0.1:8546',
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
        load_exchange_balances(w3)
        pricer = load_pool(w3)

        #
        # Load all candidate profitable arbitrages
        with open('/mnt/goldphish/tmp/top_block_candidate_arbitrages.csv', mode='a') as fout, open('/mnt/goldphish/tmp/reshoot_log_top_block_errors.csv', mode='w') as fout_errors:
            start_block = 12371099 # 12_369_621
            end_block = w3.eth.get_block('latest')['number']
            progress_reporter = ProgressReporter(l, end_block, start_block)
            batch_size_blocks = 200
            curr_block = start_block
            while curr_block < end_block:
                this_end_block = min(curr_block + batch_size_blocks - 1, end_block)
                n_logs = 0
                for block_number, logs in get_relevant_logs(w3, curr_block, this_end_block):
                    process_candidates(w3, pricer, block_number, logs, fout, fout_errors)
                    progress_reporter.observe(n_items=1)
                    n_logs += len(logs)
                l.debug(f'Found {n_logs} logs this batch')
                curr_block = this_end_block + 1

    except Exception as e:
        l.exception('top-level exception')
        raise e

if __name__ == '__main__':
    main()
