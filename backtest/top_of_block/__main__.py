import argparse
import os
import socket
import time
import typing
import web3
import web3.types
import web3.exceptions
import web3.contract
import web3._utils.events
import web3._utils.filters
import logging
import logging.handlers
from backtest.top_of_block.cleanup import do_cleanup
from backtest.top_of_block.find_arb_termination import do_find_arb_termination

from backtest.top_of_block.one_off_trace import print_trace
from backtest.top_of_block.diagnose_failures import do_diagnose
from backtest.top_of_block.seek_candidates import seek_candidates
from backtest.top_of_block.verify import do_verify

import backtest.top_of_block.measure_tvl
import backtest.top_of_block.seek_candidates

from utils import setup_logging

l = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker-name', type=str, default=None, help='worker name for log, must be POSIX path-safe')

    subparser = parser.add_subparsers(help='subcommand', dest='subcommand')
    
    handlers: typing.Dict[str, typing.Callable[[web3.Web3, argparse.Namespace], None]] = {}

    cmd, handler = backtest.top_of_block.seek_candidates.add_args(subparser)
    handlers[cmd] = handler

    cmd, handler = backtest.top_of_block.measure_tvl.add_args(subparser)
    handlers[cmd] = handler

    args = parser.parse_args()

    if args.worker_name is None:
        args.worker_name = socket.gethostname()

    job_name = 'top_block_' + args.subcommand
    setup_logging(job_name, suppress=['shooter.deploy'], worker_name = args.worker_name)

    l.info('Booting up...')

    #
    # Connect to web3
    #

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

    handlers[args.subcommand](w3, args)
    exit(1)

    try:
        if args.mode == 'verify':
            l.info(f'Verifying candidate arbitrages')
            do_verify(w3, job_name, args.worker_name)
        elif args.mode == 'diagnose':
            l.info(f'Diagnosing failures')
            do_diagnose(w3)
        elif args.mode == 'trace':
            assert args.id is not None
            print_trace(w3, args.id)
        elif args.mode == 'connect':
            do_find_arb_termination(w3)
        elif args.mode == 'cleanup':
            do_cleanup()
        elif args.mode == 'measure-tvl':
            pass
        else:
            assert args.mode is None
            seek_candidates(w3, job_name, args.worker_name)
    except:
        l.exception('fatal exception')
        raise

if __name__ == '__main__':
    main()

