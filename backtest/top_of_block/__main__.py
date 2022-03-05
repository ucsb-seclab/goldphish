import os
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
import logging.handlers
from eth_utils import event_abi_to_log_topic
from ..utils import mine_block
from utils import ProgressReporter, get_abi, decode_trace_calls, pretty_print_trace

l = logging.getLogger(__name__)



def load_exchange_graph(w3: web3.Web3) -> nx.MultiDiGraph:
    t_start = time.time()
    univ2_fname = os.path.abspath(os.path.dirname(__file__) + '/../univ2_excs.csv.gz')
    assert os.path.isfile(univ2_fname), f'should have file {univ2_fname}'
    univ3_fname = os.path.abspath(os.path.dirname(__file__) + '/../univ3_excs.csv.gz')
    assert os.path.isfile(univ3_fname)

    with gzip.open(univ2_fname, mode='rt') as fin:
        for line in fin:
            print(line)
            exit(1)

    t_end = time.time()
    l.debug(f'Took {t_end - t_start:.2f} seconds to load exchange graph')


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

    exchange_graph = load_exchange_graph(w3)

    try:
        # with open('/mnt/goldphish/tmp/reshoot_log_top_block.csv', mode='w') as fout, open('/mnt/goldphish/tmp/reshoot_log_top_block_errors.csv', mode='w') as fout_errors:
        #     for a in get_arbitrages_from_sample(w3, fout_rejects):
        #         reshoot_arbitrage(w3, a, fout, fout_rejects)
        pass
    except Exception as e:
        l.exception('top-level exception')
        raise e

if __name__ == '__main__':
    main()
