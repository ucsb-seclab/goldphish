import collections
import subprocess
import time
import typing
import web3
import web3.types
import logging
from eth_utils import event_abi_to_log_topic
from eth_account import Account
from eth_account.signers.local import LocalAccount

l = logging.getLogger(__name__)

from utils import erc20

ERC20_TRANSFER_TOPIC = event_abi_to_log_topic(erc20.events.Transfer().abi)

def parse_logs_for_net_profit(logs: typing.List[web3.types.LogReceipt]) -> typing.Dict[str, typing.Dict[str, int]]:
    """
    Parse ERC20 Transfer events into net flows.
    Maps (ERC20 address, owner) to net flow (as int)
    """
    ret = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    for log in logs:
        if len(log['topics']) > 0 and log['topics'][0] == ERC20_TRANSFER_TOPIC:
            xfer = erc20.events.Transfer().processLog(log)
            ret[log['address']][xfer['args']['from']] -= xfer['args']['value']
            ret[log['address']][xfer['args']['to']] += xfer['args']['value']
    return {k: dict(v) for k, v in ret.items()}

def funded_deployer() -> LocalAccount:
    ret: LocalAccount = Account.from_key(bytes.fromhex('0xab1179084d3336336d60b2ed654d99a21c2644cadd89fd3034ee592e931e4a77'[2:]))
    return ret

def replay_to_txn(w3: web3.Web3, ganache: web3.Web3, receipt: web3.types.TxReceipt):
    """
    Replay transactions from the top of the block `txn` appears in.
    """
    l.debug('replaying...')
    hashes = []
    for i in range(receipt['transactionIndex']):
        to_replay = w3.eth.get_raw_transaction_by_block(receipt['blockHash'], i)
        hashes.append(ganache.eth.send_raw_transaction(to_replay))
    for h in hashes:
        r = ganache.eth.wait_for_transaction_receipt(h)
    l.debug('done replay')


_next_ganache_port = 4444
def get_ganache_fork(w3: web3.Web3, target_block: int) -> typing.Iterator[web3.Web3]:
    global _next_ganache_port

    old_block = w3.eth.get_block(target_block)
    old_ts = old_block['timestamp']

    assert isinstance(target_block, int)
    assert target_block > 0

    p = subprocess.Popen(
        [
            'yarn',
            'ganache-cli',
            '--fork.url', 'ws://172.17.0.1:8546',
            '--server.ws',
            '--server.port', str(_next_ganache_port),
            '--fork.blockNumber', str(target_block),
            '--wallet.accounts', f'{funded_deployer().key.hex()},{web3.Web3.toWei(100, "ether")}',
            '--chain.chainId', '1',
            '--chain.time', str(old_ts * 1_000), # unit conversion needed for some reason -- blame javascript
            '--miner.coinbase', web3.Web3.toChecksumAddress(b'\xa0' * 20) + ' ',
            '--miner.blockTime', '1',
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    provider = web3.WebsocketProvider(
        f'ws://127.0.0.1:{_next_ganache_port}',
        websocket_timeout=60 * 5,
        websocket_kwargs={
            'max_size': 1024 * 1024 * 1024, # 1 Gb max payload
        },
    )

    w3 = web3.Web3(
        provider
    )

    _next_ganache_port += 1
    assert _next_ganache_port < 9000

    while not w3.isConnected():
        time.sleep(0.1)

    assert w3.isConnected()

    yield w3

    # force kill node, sorry
    p.kill()
    p.wait()
    subprocess.check_call(['killall', 'node'])


