import logging
import typing
import os
import json

l = logging.getLogger(__name__)

_abi_cache = {}

def get_abi(abiname) -> typing.Any:
    """
    Gets an ABI from the `abis` directory by its path name.
    Uses an internal cache when possible.
    """
    if abiname in _abi_cache:
        return _abi_cache[abiname]
    abi_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'abis'
        )
    )
    if not os.path.isdir(abi_dir):
        raise Exception(f'Could not find abi directory at {abi_dir}')

    fname = os.path.join(abi_dir, abiname)
    if not os.path.isfile(fname):
        raise Exception(f'Could not find abi at {fname}')
    
    with open(fname) as fin:
        abi = json.load(fin)
    
    _abi_cache[abiname] = abi
    return abi
