import typing

class ArbitrageCycleExchangeItem(typing.NamedTuple):
    address: str
    amount_in: int
    amount_out: int


class ArbitrageCycleExchange(typing.NamedTuple):
    token_in: str
    token_out: str
    items: typing.List[ArbitrageCycleExchangeItem]

class ArbitrageCycle(typing.NamedTuple):
    cycle: typing.List[ArbitrageCycleExchange]
    profit_token: str
    profit_amount: int
    profit_taker: str


class Arbitrage(typing.NamedTuple):
    txn_hash: bytes
    block_number: int
    gas_used: int
    gas_price: int
    shooter: typing.Optional[str]
    n_cycles: int
    only_cycle: typing.Optional[ArbitrageCycle]    

