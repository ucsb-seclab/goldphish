import argparse
import web3
import web3.types
import web3.exceptions
import web3.contract
import web3._utils.events
import web3._utils.filters
import logging
import logging.handlers

from backtest.top_of_block.one_off_trace import print_trace
from backtest.top_of_block.diagnose_failures import do_diagnose
from backtest.top_of_block.seek_candidates import seek_candidates
from backtest.top_of_block.verify import do_verify
from utils import setup_logging

l = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['verify', 'diagnose', 'trace'], help='verification mode', default=None)
    parser.add_argument('--job-name', type=str, default=None, help='job name for log, must be POSIX path-safe')
    parser.add_argument('--id', type=int, default=None, help='id to trace')
    parser.add_argument('--verify', action='store_true', help='verification mode', default=False)

    args = parser.parse_args()

    if args.mode == 'verify':
        setup_logging('top_block_verify', suppress=['shooter.deploy'], job_name = args.job_name)
    elif args.mode == 'diagnose':
        setup_logging('top_block_diagnose', suppress=['shooter.deploy'], job_name = args.job_name)
    elif args.mode == 'trace':
        setup_logging('top_block_trace_candidate', suppress=['shooter.deploy'], job_name = args.job_name)
    else:
        setup_logging('top_block_candidates', suppress=['shooter.deploy'], job_name = args.job_name)

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
        if args.mode == 'verify':
            l.info(f'Verifying candidate arbitrages')
            do_verify(w3)
        elif args.mode == 'diagnose':
            l.info(f'Diagnosing failures')
            do_diagnose(w3)
        elif args.mode == 'trace':
            assert args.id is not None
            print_trace(w3, args.id)
        else:
            seek_candidates(w3)
    except:
        l.exception('fatal exception')
        raise

if __name__ == '__main__':
    main()

