import web3

from utils import parse_ganache_trace, pretty_print_trace

w3 = web3.Web3(web3.WebsocketProvider('ws://172.17.0.1:34451', websocket_timeout=60 * 5))

assert w3.isConnected()
print('connected')

txn = w3.eth.get_transaction('0xbed84f1d98cafd98555414e24fccc3e5d9b9b65c01a8fec4a4400d5a6cd64096')
receipt = w3.eth.get_transaction_receipt('0xbed84f1d98cafd98555414e24fccc3e5d9b9b65c01a8fec4a4400d5a6cd64096')

result = w3.provider.make_request('debug_callTrace', ['0xbed84f1d98cafd98555414e24fccc3e5d9b9b65c01a8fec4a4400d5a6cd64096'])

print(result)

trace = parse_ganache_trace(result['result'])
pretty_print_trace(trace, txn, receipt)

