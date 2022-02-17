import collections
import typing
import io
import sys
import web3
import os
import web3.types
import web3.contract
import web3._utils.events
import web3._utils.filters
import logging
import logging.handlers
from eth_utils import event_abi_to_log_topic
from tests.conftest import funded_deployer

from utils import ProgressReporter, get_abi, decode_trace_calls, pretty_print_trace

import shooter.deploy
import shooter.composer
import backtest.utils

l = logging.getLogger(__name__)

SAMPLE_ADDR = '0x00000000116579a5Ba59E2F22e77EDE26809B970'
SAMPLE_ADDR_ORIGIN_BLOCK = 13_081_021
END_BLOCK = 14_215_288


erc20: web3.contract.Contract = web3.Web3().eth.contract(
    address=b'\x00' * 20,
    abi=get_abi('erc20.abi.json'),
)
ERC20_TRANSFER_TOPIC = event_abi_to_log_topic(erc20.events.Transfer().abi)

univ2: web3.contract.Contract = web3.Web3().eth.contract(
    address=b'\x00' * 20,
    abi=get_abi('uniswap_v2/IUniswapV2Pair.json')['abi'],
)
UNIV2_SWAP_TOPIC = event_abi_to_log_topic(univ2.events.Swap().abi)

univ3: web3.contract.Contract = web3.Web3().eth.contract(
    address=None,
    abi=get_abi('uniswap_v3/IUniswapV3Pool.json')['abi'],
)
UNIV3_SWAP_TOPIC = event_abi_to_log_topic(univ3.events.Swap().abi)


def get_txns_to_address(w3: web3.Web3) -> typing.List[typing.Tuple[int, bytes]]:
    l.debug(f'getting all transactions to sample shooter')
    fname = f'/mnt/goldphish/tmp/{SAMPLE_ADDR}_txns_{SAMPLE_ADDR_ORIGIN_BLOCK}_to_{END_BLOCK}.csv'
    if not os.path.isfile(fname):
        l.info(f'scanning blockchain (wait a while sorry)')
        fname_tmp = fname + '.tmp'

        with open(fname_tmp, mode='w') as fout:
            dumb_erc20 = w3.eth.contract(abi = get_abi('erc20.abi.json'))
            reporter = ProgressReporter(l, END_BLOCK, SAMPLE_ADDR_ORIGIN_BLOCK)
            blocks_per_query = 100
            for range_start in range(SAMPLE_ADDR_ORIGIN_BLOCK, END_BLOCK, blocks_per_query):
                range_end = range_start + blocks_per_query - 1
                range_end = min(range_end, END_BLOCK)

                filter1: web3._utils.filters.Filter = dumb_erc20.events.Transfer().createFilter(fromBlock=range_start, toBlock=range_end, argument_filters={'to': SAMPLE_ADDR})
                filter2: web3._utils.filters.Filter = dumb_erc20.events.Transfer().createFilter(fromBlock=range_start, toBlock=range_end, argument_filters={'to': SAMPLE_ADDR})

                relevant_txns = set()
                entries = filter1.get_all_entries() + filter2.get_all_entries()
                l.debug(f'Processing {len(entries):,} log entries')
                for xfer in entries:
                    relevant_txns.add((xfer['blockNumber'], xfer['transactionHash']))

                l.debug(f'Have {len(relevant_txns):,} relevant transactions')

                for (block_num, txn_hash) in sorted(relevant_txns, key=lambda x: x[0]):
                    txn = w3.eth.get_transaction(txn_hash)
                    if txn['to'] == SAMPLE_ADDR:
                        fout.write(f'{block_num},{txn_hash.hex()}\n')

                reporter.observe(blocks_per_query)
        os.rename(fname_tmp, fname)
        l.info(f'Done chain scan.')

    ret = []
    with open(fname) as fin:
        for line in fin:
            if line.lstrip().startswith('#'):
                continue
            block_num, txn_hash = line.strip().split(',')
            block_num = int(block_num)
            txn_hash = bytes.fromhex(txn_hash[2:])
            assert len(txn_hash) == 32
            ret.append((block_num, txn_hash))
    return ret


def recover_exchanges(receipt: web3.types.TxReceipt) -> typing.Tuple[typing.Set, typing.Set]:
    """
    Extract all used exchange addresses.
    Returns (uniswap_v2, uniswap_v3)
    """
    univ2_exchanges = set()
    univ3_exchanges = set()
    for log in receipt['logs']:
        if len(log['topics']) > 0:
            if log['topics'][0] == UNIV3_SWAP_TOPIC:
                univ3_exchanges.add(log['address'])
            elif log['topics'][0] == UNIV2_SWAP_TOPIC:
                univ2_exchanges.add(log['address'])
    return (univ2_exchanges, univ3_exchanges)


def recover_token_pairs(logs: typing.List[web3.types.LogReceipt]) -> typing.Dict[str, typing.Tuple[str, str]]:
    addr_to_tokens_in = collections.defaultdict(lambda: set())
    addr_to_tokens_out = collections.defaultdict(lambda: set())
    for log in logs:
        if len(log['topics']) > 0 and log['topics'][0] == ERC20_TRANSFER_TOPIC:
            xfer = erc20.events.Transfer().processLog(log)
            addr_to_tokens_in[xfer['args']['to']].add(log['address'])
            addr_to_tokens_out[xfer['args']['from']].add(log['address'])

    ret = {}
    for addr in set(addr_to_tokens_in.keys()).intersection(addr_to_tokens_out.keys()):
        ins = addr_to_tokens_in[addr]
        outs = addr_to_tokens_out[addr]
        if len(ins) == 1 and len(outs) == 1:
            token_in = next(ins.__iter__())
            token_out = next(outs.__iter__())
            pair = tuple(
                sorted(
                    [token_in, token_out],
                    key=lambda x: bytes.fromhex(x[2:]),
                )
            )
            ret[addr] = pair
    return ret


def get_arbitrages_from_sample(w3: web3.Web3) -> typing.Generator[web3.types.TxReceipt, None, None]:
    all_txns = get_txns_to_address(w3)
    l.debug(f'Have {len(all_txns)} transactions to go through')

    with open('/mnt/goldphish/tmp/rejected_hashes.csv', mode='w') as fout:
        fout.write('# transaction hash, rejection reason\n')
        progress_reporter = ProgressReporter(l, len(all_txns), 0)
        for _, txn_hash in all_txns:
            try:
                receipt = w3.eth.get_transaction_receipt(txn_hash)

                # ensure the transaction didn't revert
                if receipt['status'] == 0:
                    fout.write(f'{txn_hash.hex()},reverted\n')
                    continue
                
                # ensure there are at least TWO exchange actions (and at least one uniswap v3)
                univ2_exchanges, univ3_exchanges = recover_exchanges(receipt)
                if len(univ3_exchanges) + len(univ2_exchanges) < 2:
                    fout.write(f'{txn_hash.hex()},not enough exchange actions\n')
                    continue
                if len(univ3_exchanges) < 1:
                    fout.write(f'{txn_hash.hex()},no uniswap v3 action\n')
                    continue
                
                # This arbitrage may have used an exchange to take profits. As a crude metric, ensure
                # all exchanged tokens are both bought once and sold exactly once
                addr_to_tokens_in = collections.defaultdict(lambda: set())
                addr_to_tokens_out = collections.defaultdict(lambda: set())
                for log in receipt['logs']:
                    if len(log['topics']) > 0 and log['topics'][0] == ERC20_TRANSFER_TOPIC:
                        xfer = erc20.events.Transfer().processLog(log)
                        addr_to_tokens_in[xfer['args']['to']].add(log['address'])
                        addr_to_tokens_out[xfer['args']['from']].add(log['address'])

                sold_tokens = collections.defaultdict(lambda: 0)
                bought_tokens = collections.defaultdict(lambda: 0)
                for exchange in univ2_exchanges.union(univ3_exchanges):
                    assert len(addr_to_tokens_out[exchange]) == 1
                    assert len(addr_to_tokens_in[exchange]) == 1
                    sold_token = next(addr_to_tokens_in[exchange].__iter__())
                    bought_token = next(addr_to_tokens_out[exchange].__iter__())
                    sold_tokens[sold_token] += 1
                    bought_tokens[bought_token] += 1
                for exchanged_token in set(sold_tokens.keys()).union(bought_tokens.keys()):
                    if sold_tokens[exchanged_token] != 1 or bought_tokens[exchanged_token] != 1:
                        fout.write(f'{txn_hash.hex()},at least one token was bought or sold more than once\n')
                        continue
                
                yield receipt
            finally:
                progress_reporter.observe(1)

def reshoot_arbitrage(w3: web3.Web3, receipt: web3.types.TxReceipt, fout: io.TextIOWrapper):
    # if receipt['transactionHash'].hex() != '0x69e1c1004ef3d7dc227c9bf7c677132828fd772b3fd3b2b722a91fdcec27d6ab':
    #     return
    l.debug(f'Re-shooting {receipt["transactionHash"].hex()}')
    # attempt to re-run this arbitrage using our own shooter, and record the results

    # recover all the exchanges
    univ2_exchanges, univ3_exchanges = recover_exchanges(receipt)

    # recover (token0, token1) pairs for each exchange
    pairs = recover_token_pairs(receipt['logs'])

    # decode the arbitrage -- [(exchange, is_v2, amount_in, amount_out, token_in, token_out)]
    arbitrage_chain: typing.List[shooter.composer.ExchangeRecord] = []

    for log in receipt['logs']:
        found = False
        if len(log['topics']) > 0 and log['topics'][0] == UNIV2_SWAP_TOPIC:
            found = True
            exc = univ2.events.Swap().processLog(log)
            exchange = log['address']
            is_v2 = True
            zero_for_one = exc['args']['amount0In'] > 0
            if zero_for_one:
                assert exc['args']['amount1In'] == 0
                assert exc['args']['amount0Out'] == 0
                amount_in = exc['args']['amount0In']
                amount_out = exc['args']['amount1Out']
                (token_in, token_out) = pairs[exchange]
            else:
                assert exc['args']['amount0In'] == 0
                assert exc['args']['amount1Out'] == 0
                amount_in = exc['args']['amount1In']
                amount_out = exc['args']['amount0Out']
                (token_out, token_in) = pairs[exchange]
        elif len(log['topics']) > 0 and log['topics'][0] == UNIV3_SWAP_TOPIC:
            found = True
            exc = univ3.events.Swap().processLog(log)
            exchange = log['address']
            is_v2 = False
            zero_for_one = exc['args']['amount0'] > 0
            if zero_for_one:
                amount_in = exc['args']['amount0']
                amount_out = -exc['args']['amount1']
                (token_in, token_out) = pairs[exchange]
            else:
                amount_in = exc['args']['amount1']
                amount_out = -exc['args']['amount0']
                (token_out, token_in) = pairs[exchange]
        if found:
    # decode the arbitrage -- [(exchange, is_v2, amount_in, amount_out, token_in, token_out)]
            arbitrage_chain.append(shooter.composer.ExchangeRecord(
                address=exchange,
                is_uniswap_v2=is_v2,
                amount_in=amount_in,
                amount_out=amount_out,
                token_in=token_in,
                token_out=token_out,
            ))

    assert len(arbitrage_chain) == len(univ2_exchanges) + len(univ3_exchanges)
    assert len(arbitrage_chain) >= 2
    assert len(arbitrage_chain) <= 3
    
    # poor man's context manager
    for w3_fork in backtest.utils.get_ganache_fork(w3, receipt['blockNumber'] - 1):
        backtest.utils.replay_to_txn(
            w3,
            w3_fork,
            receipt,
        )
        # deploy shooter
        deployer = backtest.utils.funded_deployer()
        shooter_addr = shooter.deploy.deploy_shooter(
            w3_fork,
            deployer,
            max_priority=3,
            max_fee_total=w3.toWei('1', 'ether'),
        )
        l.debug('deployed shooter on new chain')

        # wrap some ether
        weth: web3.contract.Contract = w3_fork.eth.contract(
            address='0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
            abi=get_abi('weth9/WETH9.json')['abi'],
        )
        wrap = weth.functions.deposit().buildTransaction({'value': 100, 'from': deployer.address})
        wrap_hash = w3_fork.eth.send_transaction(wrap)
        wrap_receipt = w3_fork.eth.wait_for_transaction_receipt(wrap_hash)
        assert wrap_receipt['status'] == 1

        # transfer to shooter
        xfer = weth.functions.transfer(deployer.address, 100).buildTransaction({'from': deployer.address})
        xfer_hash = w3_fork.eth.send_transaction(xfer)
        xfer_receipt = w3_fork.eth.wait_for_transaction_receipt(xfer_hash)
        assert xfer_receipt['status'] == 1

        shot = shooter.composer.construct(
            arbitrage_chain,
            coinbase_xfer=0,
            target_block=w3_fork.eth.get_block('latest')['number'] + 1,
        )
        l.debug(f'About to shoot {shot.hex()}')

        txn: web3.types.TxParams = {
            'chainId': w3_fork.eth.chain_id,
            'from': deployer.address,
            'to': shooter_addr,
            'value': 0,
            'nonce': w3_fork.eth.get_transaction_count(deployer.address),
            'data': shot,
            'gas': 400_000,
            'maxPriorityFeePerGas': 2,
            'maxFeePerGas': 500 * (10 ** 9),
        }
        signed_txn = w3_fork.eth.account.sign_transaction(txn, deployer.key)
        tx_hash = w3_fork.eth.send_raw_transaction(signed_txn.rawTransaction)
        new_receipt = w3_fork.eth.wait_for_transaction_receipt(tx_hash)
        
        if new_receipt['status'] != 1:
            print(new_receipt)

            trace = w3_fork.provider.make_request('debug_traceTransaction', [new_receipt['transactionHash'].hex()])
            with open('/mnt/goldphish/trace.txt', mode='w') as fout:
                for log in trace['result']['structLogs']:
                    fout.write(str(log) + '\n')

            decoded = decode_trace_calls(trace['result']['structLogs'])
            pretty_print_trace(decoded)
            raise Exception('status should be success')



        # ensure our exchanges align with the old one
        new_profits = backtest.utils.parse_logs_for_net_profit(new_receipt['logs'])
        old_profits = backtest.utils.parse_logs_for_net_profit(receipt['logs'])
        assert set(new_profits.keys()) == set(old_profits.keys())
        took_profit_in = None
        for token_addr in new_profits.keys():
            new_shooter_mvmt = new_profits[token_addr].get(shooter_addr, None)
            old_shooter_mvmt = old_profits[token_addr].get(SAMPLE_ADDR, None)
            assert new_shooter_mvmt == old_shooter_mvmt
            if new_shooter_mvmt is not None:
                assert new_shooter_mvmt > 0
                took_profit_in = token_addr
        assert took_profit_in == '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2'

        # log to output
        old_gas = receipt['gasUsed']
        new_gas = new_receipt['gasUsed']
        old_gasprice = receipt['effectiveGasPrice']
        profit_amount = old_profits[took_profit_in][SAMPLE_ADDR]
        fout.write(f'{receipt["blockNumber"]},{receipt["transactionHash"].hex()},{old_gas},{new_gas},{old_gasprice},{profit_amount}\n')

    l.debug('done processing transaction')

def main():
    #
    # Set up logging
    #

    root_logger = logging.getLogger()
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(fmt)
    fh = logging.handlers.WatchedFileHandler(
        '/mnt/goldphish/tmp/log.txt'
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
        with open('/mnt/goldphish/tmp/reshoot_log.txt', mode='w') as fout:
            for a in get_arbitrages_from_sample(w3):
                reshoot_arbitrage(w3, a, fout)
    except Exception as e:
        l.exception('top-level exception')
        raise e

if __name__ == '__main__':
    main()
