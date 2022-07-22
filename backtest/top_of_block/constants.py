import os
import typing
import web3
import web3.contract
from eth_utils import event_abi_to_log_topic
from pricers.token_balance_changing_logs import CACHE_INVALIDATING_TOKEN_LOGS

from utils import get_abi


IMPORTANT_TOPICS: typing.List[bytes] = []
UNISWAP_IMPORTANT_TOPICS: typing.List[bytes] = []

# Set up the important log topics we'll need to listen
univ2: web3.contract.Contract = web3.Web3().eth.contract(
    address=b'\x00' * 20,
    abi=get_abi('uniswap_v2/IUniswapV2Pair.json')['abi'],
)
UNISWAP_IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ2.events.Mint().abi))
UNISWAP_IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ2.events.Burn().abi))
UNISWAP_IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ2.events.Swap().abi))
UNISWAP_V2_SYNC_TOPIC = event_abi_to_log_topic(univ2.events.Sync().abi)
UNISWAP_IMPORTANT_TOPICS.append(UNISWAP_V2_SYNC_TOPIC)

univ3: web3.contract.Contract = web3.Web3().eth.contract(
    address=None,
    abi=get_abi('uniswap_v3/IUniswapV3Pool.json')['abi'],
)
UNISWAP_IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ3.events.Mint().abi))
UNISWAP_IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ3.events.Burn().abi))
UNISWAP_IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ3.events.Initialize().abi))
UNISWAP_IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ3.events.SetFeeProtocol().abi)) # I'm not sure this is ever used?
UNISWAP_IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ3.events.CollectProtocol().abi)) # I'm not sure this is ever used?
UNISWAP_IMPORTANT_TOPICS.append(event_abi_to_log_topic(univ3.events.Swap().abi))

UNISWAP_IMPORTANT_TOPICS_HEX = ['0x' + x.hex() for x in UNISWAP_IMPORTANT_TOPICS]

IMPORTANT_TOPICS += UNISWAP_IMPORTANT_TOPICS

for val in CACHE_INVALIDATING_TOKEN_LOGS.values():
    for v in val:
        IMPORTANT_TOPICS.append(v)

IMPORTANT_TOPICS_HEX: typing.List[str] = ['0x' + x.hex() for x in IMPORTANT_TOPICS]

univ2_fname = os.path.abspath(os.path.dirname(__file__) + '/../univ2_excs.csv.gz')
assert os.path.isfile(univ2_fname), f'should have file {univ2_fname}'
univ3_fname = os.path.abspath(os.path.dirname(__file__) + '/../univ3_excs.csv.gz')
assert os.path.isfile(univ3_fname)


FNAME_EXCHANGES_WITH_BALANCES = '/mnt/goldphish/exchanges_prefilter.csv'

# profit must be enough to pay for 130k gas @ 20 gwei (both overly optimistic)
MIN_PROFIT_PREFILTER = (130_000) * (20 * (10 ** 9))

