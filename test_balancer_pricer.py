import collections
import datetime
import itertools
import os
import time
import typing
import psycopg2
import tabulate
import web3
import web3.types
import web3._utils.filters
import pricers.balancer
from pricers.balancer import LOG_SWAP_TOPIC, BalancerPricer

web3_host = os.getenv('WEB3_HOST', 'ws://172.17.0.1:8546')

w3 = web3.Web3(web3.WebsocketProvider(
    web3_host,
    websocket_timeout=60 * 5,
    websocket_kwargs={
        'max_size': 1024 * 1024 * 1024, # 1 Gb max payload
    },
))

if not w3.isConnected():
    print('Could not connect to web3')
    exit(1)


pg_host = os.getenv('PSQL_HOST', 'ethereum-measurement-pg')
pg_port = int(os.getenv('PSQL_PORT', '5432'))
db = psycopg2.connect(
    host = pg_host,
    port = pg_port,
    user = 'measure',
    password = 'password',
    database = 'eth_measure_db',
)
db.autocommit = False
print('connected to postgresql')

curr = db.cursor()

SWAP_EXACT_IN_SELECTOR = bytes.fromhex(
    pricers.balancer._base_balancer.functions.swapExactAmountIn(
        web3.Web3.toChecksumAddress(b'\x00'*20),
        0,
        web3.Web3.toChecksumAddress(b'\x00'*20),
        0,
        0,
    ).selector[2:]
)

SWAP_EXACT_IN_TOPIC = SWAP_EXACT_IN_SELECTOR.ljust(32, b'\x00')

top_exchanges = [
    '0x59A19D8c652FA0284f44113D0ff9aBa70bd46fB4',
    '0x9B208194Acc0a8cCB2A8dcafEACfbB7dCc093F81',
    '0xe036CCE08cf4E23D33bC6B18e53Caf532AFa8513',
    '0x16cAC1403377978644e78769Daa49d8f6B6CF565',
    '0x10DD17eCfc86101Eab956E0A443cab3e9C62d9b4',
    '0x4304Ae5Fd14CEc2299caee4E9a4AFbedD046D612',
    '0x3031745E732dcE8fECccc94AcA13D5fa18F1012d',
    '0xf9ab7776C7bAEED1D65f0492fE2bB3951A1787Ef',
    '0xe969991CE475bCF817e01E1AAd4687dA7e1d6F83',
    '0x1B8874BaceAAfba9eA194a625d12E8b270D77016',
    '0x60626db611a9957C1ae4Ac5b7eDE69e24A3B76c5',
    '0x003a70265a3662342010823bEA15Dc84C6f7eD54',
    '0x7c90a3cd7Ec80dd2F633ed562480AbbEEd3bE546',
    '0xbd63d492bbb13d081D680CE1f2957a287FD8c57c',
    '0xEe9A6009B926645D33E10Ee5577E9C8D3C95C165',
    '0xa49b3c7C260ce8A7C665e20Af8aA6E099A86cf8A',
    '0x4Fd2d9d6eF05E13Bf0B167509151A4EC3D4d4b93',
    '0x7aFE74AE3C19f070c109A38C286684256ADC656C',
    '0x6B74Fb4E4b3B177b8e95ba9fA4c3a3121d22fbfB',
    '0x1efF8aF5D577060BA4ac8A29A13525bb0Ee2A3D5',
    '0x72Cd8f4504941Bf8c5a21d1Fd83A96499FD71d2C',
    '0x834fb8276B4E8a24010e2108fDd7F8417C8922bD',
    '0x2471de1547296AaDb02CC1Af84afe369B6F67c87',
    '0x99e582374015c1d2F3C0f98d0763B4B1145772B7',
    '0x9866772A9BdB4Dc9d2c5a4753e8658B8B0Ca1fC3',
    '0xe867bE952ee17d2D294F2de62b13B9F4aF521e9a',
    '0x5aeC4cfF7FC3880Ade1582E5E37Cf89152E70Ace',
    '0xD44082F25F8002c5d03165C5d74B520FBc6D342D',
    '0xA5910940b97B7B8771a01B202583Fd9331cb8Be3',
    '0x1811A719A05D20b6447CA556a54F00f9c14578Be',
    '0x8a649274E4d777FFC6851F13d23A86BBFA2f2Fbf',
    '0xAf71d6c242A00E8364Ea0eF3c007f3413E975011',
    '0xA7d7d09484Fa6e5F497b6b687f979509373c6530',
    '0xB50961E3D6128Fb746ffdC6054046D873194376D',
    '0xB1F9eC02480Dd9e16053B010DFc6E6C4b72Ecad5',
    '0xD3c8DcfcF2A5203f6a3210591daBeA14E181dB2D',
    '0x987D7Cc04652710b74Fff380403f5c02f82e290a',
    '0xc0b2B0C5376Cb2e6f73b473A7CAA341542F707Ce',
    '0x8D5C91324da4C4Ef46b623AE62BF5586de0F4507',
    '0xE5D1fAB0C5596ef846DCC0958d6D0b20E1Ec4498',
    '0xDc7d39628855B6013000C9aF957e6Cd484beda6c',
    '0xc409D34aCcb279620B1acDc05E408e287d543d17',
    '0x0e511Aa1a137AaD267dfe3a6bFCa0b856C1a3682',
    '0x80CbA5Ba9259C08851d94d6bf45e248541fB3e86',
    '0xe26A220a341EAca116bDa64cF9D5638A935ae629',
    '0x41284a88D970D3552A26FaE680692ED40B34010C',
    '0x7Fc95945eAa14e7a2954052A4C9bFBaA79d170AE',
    '0x594415978a756c5b02eABdFF98D867CDDa65e888',
    '0x66c03a9d8c7DF62A05f043cAA4e33629780eaf7a',
    '0xFe793bC3D1Ef8d38934896980254e81d0c5F6239',
]

if False:
    # search for the contract with the most LOG_SWAPs

    counts = collections.defaultdict(lambda: 0)

    start_block = 10000000
    end_block   = 14000000
    batch_size = 100

    for i in itertools.count():
        batch_start_block = start_block + i * batch_size
        batch_end_block = min(end_block, batch_start_block + batch_size - 1)

        if batch_start_block > end_block:
            break

        if i % 50 == 1:
            tab = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:50]
            print(tabulate.tabulate(tab, headers=['Exchange', 'Count']))
            print()

        f: web3._utils.filters.Filter = w3.eth.filter({
            'topics': ['0x' + LOG_SWAP_TOPIC.hex()],
            'fromBlock': batch_start_block,
            'toBlock': batch_end_block
        })

        for log in f.get_all_entries():
            counts[log['address']] += 1

bs: typing.List[BalancerPricer] = []

start_block = 100_000_000_000
for exc in top_exchanges:
    curr.execute('SELECT origin_block FROM balancer_exchanges WHERE address = %s', (bytes.fromhex(exc[2:]),))
    if curr.rowcount == 0:
        print(f'Not a balancer??? {exc}')
    else:
        (origin_block,) = curr.fetchone()
        start_block = min(start_block, origin_block)
        bs.append(BalancerPricer(w3, exc))

if True:
    start_block = 10_000_000

batch_size = 1_0000

latest_block = w3.eth.get_block('latest')['number'] - 10

print(f'Scanning from {start_block:,} to {latest_block:,} ({latest_block - start_block:,} blocks)')

t_start = time.time()

for i in itertools.count():
    batch_start_block = start_block + i * batch_size
    batch_end_block = min(latest_block, batch_start_block + batch_size - 1)

    if batch_start_block > latest_block:
        break

    if i % 10 == 1:
        # do an ETA update
        n_blocks = batch_start_block - start_block
        s_elapsed = time.time() - t_start
        bps = n_blocks / s_elapsed

        n_remaining = latest_block - batch_start_block
        s_remaining = n_remaining / bps
        eta_td = datetime.timedelta(seconds=s_remaining)
        print(f'Processed up to {batch_start_block:,}, ETA {eta_td}')


    f: web3._utils.filters.Filter = w3.eth.filter({
        'address': [b.address for b in bs],
        'fromBlock': batch_start_block,
        'toBlock': batch_end_block
    })

    logs = f.get_all_entries()

    print(f'have {len(logs)} logs')

    logs_by_block: typing.Dict[int, typing.List[web3.types.LogReceipt]] = collections.defaultdict(lambda: [])
    for log in logs:
        logs_by_block[log['blockNumber']].append(log)

    for block_number in sorted(logs_by_block.keys()):
        # gather logs by exchange
        logs_by_exchange: typing.Dict[str, typing.List[web3.types.LogReceipt]] = collections.defaultdict(lambda: [])

        for log in logs_by_block[block_number]:
            logs_by_exchange[log['address']].append(log)


        print(f'observing {block_number:,}')
        for b in bs:
            # observe swaps one by one and attempt to verify our swap computation
            these_logs = logs_by_exchange[b.address]

            while len(these_logs) > 0:
                if len(these_logs[0]['topics']) == 0 or these_logs[0]['topics'][0] != SWAP_EXACT_IN_TOPIC:
                    print(f'ignoring because log was {these_logs[0]["topics"][0].hex()}')
                    # done with leading swaps
                    break

                log = these_logs.pop(0)

                payload = bytes.fromhex(log['data'][136+2:])
                token_in = web3.Web3.toChecksumAddress(payload[12:32])
                token_amount_in = int.from_bytes(payload[33:64], byteorder='big', signed=False)
                token_out = web3.Web3.toChecksumAddress(payload[64+12:96])

                expected_amount_out = b.swap_exact_amount_in(token_in, token_amount_in, token_out, log['blockNumber'] - 1)

                next_log = these_logs.pop(0)
                assert next_log['topics'][0] == pricers.balancer.LOG_SWAP_TOPIC

                parsed = b.contract.events.LOG_SWAP().processLog(next_log)

                assert parsed['args']['tokenIn'] == token_in
                assert parsed['args']['tokenOut'] == token_out
                assert parsed['args']['tokenAmountIn'] == token_amount_in
                assert parsed['args']['tokenAmountOut'] == expected_amount_out, f'Expected {parsed["args"]["tokenAmountOut"]} == {expected_amount_out}'

                print('passed swap check')
                b.observe_block([log, next_log])


            b.observe_block(these_logs)

            if b._public_swap:
                actual_public_swap = b.contract.functions.isPublicSwap().call(block_identifier=block_number)
                assert b._public_swap == actual_public_swap

            if b.finalized is not None:
                actual_finalized = b.contract.functions.isFinalized().call(block_identifier=block_number)
                assert b.finalized == actual_finalized

            if b.tokens is not None:
                actual_tokens = b.contract.functions.getCurrentTokens().call(block_identifier=block_number)
                assert set(actual_tokens) == b.tokens

            if b.swap_fee is not None:
                actual_swap_fee = b.contract.functions.getSwapFee().call(block_identifier=block_number)
                assert b.swap_fee == actual_swap_fee

            for token in sorted(b.token_denorms.keys()):
                actual_denorm = b.contract.functions.getDenormalizedWeight(token).call(block_identifier=block_number)
                assert actual_denorm == b.token_denorms[token]

            for token in sorted(b._balance_cache.keys()):
                # see if balances match
                actual_balance = b.contract.functions.getBalance(token).call(block_identifier=block_number)
                my_balance = b._balance_cache[token]
                assert actual_balance == my_balance, f'expect {actual_balance} == {my_balance} in block {block_number:,}'


print('done')
