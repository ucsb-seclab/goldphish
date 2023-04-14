import web3

w3 = web3.Web3(web3.WebsocketProvider('ws://10.10.111.111:8546',
        websocket_timeout=60 * 5,
        websocket_kwargs={
            'max_size': 1024 * 1024 * 1024, # 1 Gb max payload
        },
))

result = w3.provider.make_request('debug_traceTransaction', ['0x67609ef436f62e9860e29b8449384e6105118781d5eaf0769a810dd04bab912c', {'enableMemory': True}])
for sl in result['result']['structLogs']:
    print(sl)

