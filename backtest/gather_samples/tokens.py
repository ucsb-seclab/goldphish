import copy
import psycopg2.extensions
import web3
import typing
import logging

from backtest.utils import erc20

l = logging.getLogger(__name__)

class Token(typing.NamedTuple):
    id: int
    address: str
    name: str
    symbol: str


_token_cache: typing.Dict[str, Token] = {}
def get_cached_token(address: str) -> typing.Optional[Token]:
    return _token_cache.get(address, None)


def _get_name_and_symbol(w3: web3.Web3, address: str, block_identifier: int) -> typing.Tuple[str, str]:
    # see if we can find the token symbol
    token_contract = w3.eth.contract(address=address, abi=erc20.abi)
    try:
        symbol = token_contract.functions.symbol().call(block_identifier=block_identifier)
    except Exception as e:
        l.debug(f'attempting fallback symbol recovery for {address}')
        # this can sometimes be caused bc symbol() returns bytes32 in old tokens
        abi_cpy = copy.deepcopy(erc20.abi)
        for m in abi_cpy:
            if m['type'] == 'function' and m['name'] == 'symbol' and m['inputs'] == []:
                assert len(m['outputs']) == 1
                m['outputs'][0]['type'] = 'bytes32'
        token_contract = w3.eth.contract(address=address, abi=abi_cpy)

        try:
            symbol = bytes(token_contract.functions.symbol().call(block_identifier=block_identifier))
            symbol = symbol[:symbol.index(b'\x00')].decode('ascii')
        except Exception as e:
            l.exception(e)
            symbol = 'UNKNOWN'
    l.debug(f'Found symbol={repr(symbol)} for address={address}')
    # see if we can find the name
    try:
        name = token_contract.functions.name().call(block_identifier=block_identifier)
    except Exception as e:
        l.debug(f'attempting fallback name recovery for {address}')
        # this can sometimes be caused bc symbol() returns bytes32 in old tokens
        abi_cpy = copy.deepcopy(erc20.abi)
        for m in abi_cpy:
            if m['type'] == 'function' and m['name'] == 'name' and m['inputs'] == []:
                assert len(m['outputs']) == 1
                m['outputs'][0]['type'] = 'bytes32'
        token_contract = w3.eth.contract(address=address, abi=abi_cpy)

        try:
            name = bytes(token_contract.functions.name().call(block_identifier=block_identifier))
            name = name[:name.index(b'\x00')].decode('ascii')
        except Exception as e:
            l.exception(e)
            name = 'UNKNOWN'
    l.debug(f'Found name={repr(name)} for address={address}')

    if '\x00' in name:
        l.debug('Replacing nulls in name')
        name = name.replace('\x00', '')
    if '\x00' in symbol:
        l.debug('Replacing nulls in symbol')
        symbol = symbol.replace('\x00', '')
    
    return (name, symbol)



def get_token(
        w3: web3.Web3,
        curr: psycopg2.extensions.cursor,
        token_address: str,
        block_identifier: int
    ) -> Token:
    if isinstance(token_address, bytes):
        assert len(token_address) == 20
        token_address = w3.toChecksumAddress(token_address)
    if token_address not in _token_cache:
        assert web3.Web3.isChecksumAddress(token_address)

        token_address_bytes = bytes.fromhex(token_address[2:])

        curr.execute(
            '''
            SELECT id, name, symbol FROM tokens WHERE address = %s
            ''',
            (token_address_bytes,),
        )
        # assert curr.rowcount > 0, f'Failed to get token record for {token_address}'
        if curr.rowcount > 0:
            assert curr.rowcount == 1
            id_, name, symbol = curr.fetchone()
        else:
            # we need to insert it
            curr.execute(
                '''
                SELECT id, name, symbol FROM tokens WHERE address = %s
                ''',
                (token_address_bytes,),
            )
            if curr.rowcount > 0:
                # we lost the race, this is ok
                assert curr.rowcount == 1
                id_, name, symbol = curr.fetchone()
            else:
                # we (maybe) won the race
                name, symbol = _get_name_and_symbol(w3, token_address, block_identifier)
                curr.execute(
                    '''
                    INSERT INTO tokens (address, name, symbol) VALUES (%s, %s, %s)
                    RETURNING id
                    ''',
                    (token_address_bytes, name, symbol)
                )
                assert curr.rowcount == 1
                (id_,) = curr.fetchone()
                l.debug(f'Inserted token id={id_} name={repr(name)} symbol={repr(symbol)}')

        _token_cache[token_address] = Token(
            id_, token_address, name, symbol
        )

    return _token_cache[token_address]
