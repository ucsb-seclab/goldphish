import datetime
import typing
import os
import json
import time
import logging
import scipy.stats
import logging
import logging.handlers
import random
import sys
import web3
import web3.types
import web3.contract
import random


from .throttler import BlockThrottle
from .profiling import get_measurement, reset_measurement, profile

l = logging.getLogger(__name__)

WETH_ADDRESS = '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2'
TETHER_ADDRESS = '0xdAC17F958D2ee523a2206206994597C13D831ec7'
USDC_ADDRESS = '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48'
UNI_ADDRESS = '0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984'
WBTC_ADDRESS = '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599'
BALANCER_VAULT_ADDRESS = '0xBA12222222228d8Ba445958a75a0704d566BF2C8'


_abi_cache = {}


# copy-pasted: https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
class ColoredFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    blue = "\033[38;5;111m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\u001b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logging(job_name = None, suppress: typing.List[str] = [], worker_name: typing.Optional[str] = None, root_dir = None, stdout_level = None):
    if root_dir is None:
        root_dir = '/mnt/goldphish'

    if stdout_level is None:
        stdout_level = logging.DEBUG

    logdir = os.path.join(root_dir, 'logs')
    if not os.path.isdir(logdir):
        os.mkdir(logdir)

    sz_date = datetime.datetime.utcnow().isoformat(sep='T')

    if worker_name is None:
        worker_name = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=6))
    if job_name is None:
        fname = os.path.join(logdir, 'log.txt')
    else:
        fname = os.path.join(logdir, f'{job_name}_{sz_date}_{worker_name}.txt')
    root_logger = logging.getLogger()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(ColoredFormatter())
    sh.setLevel(stdout_level)
    fh = logging.handlers.WatchedFileHandler(
        fname
    )
    fmt = logging.Formatter(f'%(asctime)s - {worker_name} - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    root_logger.addHandler(sh)
    root_logger.addHandler(fh)
    root_logger.setLevel(logging.DEBUG)

    # silence some annoying logs from subsystems
    for lname in ['websockets.protocol', 'web3.providers.WebsocketProvider',
                  'web3.RequestManager', 'websockets.server', 'asyncio', 'pika'] + suppress:
        logging.getLogger(lname).setLevel(logging.WARNING)


def read_mem(start, read_len, mem):
    b = b''
    for idx in range(start, start + read_len):
        byte = idx % 32
        word = idx // 32
        if word >= len(mem):
            word_content = '00' * 32
        else:
            word_content = mem[word]
        b += bytes.fromhex(word_content[byte*2:byte*2+2])
    return b


def pretty_print_trace(parsed, txn: web3.types.TxData, receipt: web3.types.TxReceipt):
    print()
    print('-----------------------')
    # compute gas used just to start the call
    invocation_gas = txn['gas'] - parsed['gasStart']
    print(f'Gas usage from invocation: {invocation_gas:,}')

    stack = [(0, x) for x in reversed(parsed['actions'])]
    while len(stack) > 0:
        depth, item = stack.pop()
        padding = '    ' * depth
        if item['type'] == 'REVERT':
            print(padding + 'REVERT ' + web3.Web3.toText(item['message']))
        elif item['type'] == 'RETURN':
            print(padding + 'RETURN')
            for i in range(0, len(item['data']), 32):
                print(padding + '  ' + item['data'][i:i+32].hex())
        elif 'CALL' in item['type']:
            gas_usage =  item['gasStart'] - item['gasEnd']
            print(padding + item['type'] + ' ' + item['callee'] + f' gas consumed = {gas_usage:,} [{item["traceStart"]}, {item["traceEnd"]}]')
            method_sel = item['args'][:4].hex()
            if method_sel == '128acb08':
                print(padding + 'UniswapV3Pool.swap()')
                (_, dec) = uv3.decode_function_input(item['args'])
                print(padding + 'recipient ........ ' + dec['recipient'])
                print(padding + 'zeroForOne ....... ' + str(dec['zeroForOne']))
                print(padding + 'amountSpecified .. ' + str(dec['amountSpecified']))
                print(padding + 'sqrtPriceLimitX96  ' + hex(dec['sqrtPriceLimitX96']))
                print(padding + 'len(calldata) .... ' + str(len(dec['data'])))
                print(padding + 'calldata..' + dec['data'].hex())
            elif method_sel == '022c0d9f':
                print(padding + 'UniswapV2Pool.swap()')
                (_, dec) = uv2.decode_function_input(item['args'])
                print(padding + 'amount0Out ... ' + str(dec['amount0Out']) + ' ' + hex(dec['amount0Out']))
                print(padding + 'amount1Out ... ' + str(dec['amount1Out']) + ' ' + hex(dec['amount1Out']))
                print(padding + 'to ........... ' + dec['to'])
                if len(dec['data']) > 0:
                    print(padding + 'calldata..' + dec['data'].hex())
            else:
                print(padding + method_sel)
                for i in range(4, len(item['args']), 32):
                    print(padding + item['args'][i:i+32].hex())
            for sub_action in reversed(item['actions']):
                stack.append((depth + 1, sub_action))
        elif item['type'] == 'REVERT':
            print(padding + 'REVERT')
        elif item['type'] == 'TRANSFER':
            print(padding + 'TRANSFER')
            print(padding + 'FROM  ' + item['from'])
            print(padding + 'TO    ' + item['to'])
            print(padding + 'VALUE ' + str(item['value']) + f'({hex(item["value"])})')
        print()
    print('-----------------------')


def decode_trace_calls(trace, txn: web3.types.TxData, receipt: web3.types.TxReceipt):
    # sum the gas and see if things square up
    gas_from_call = txn['gas'] - trace[0]['gas']
    print(f'Gas from call: {gas_from_call:,}')
    all_gas_usage = sum(sl['gasCost'] for sl in trace)
    print(f'Trace all_gas_usage {all_gas_usage:,}')
    print(f'Trace sum gas usage {gas_from_call + all_gas_usage:,}')
    print(f'Reported gas usage {receipt["gasUsed"]:,}')

    ctx = {
        'type': 'root',
        'from': txn['from'],
        'callee': txn['to'],
        'traceStart': 0,
        'gasStart': trace[0]['gas'],
        'actions': [],
    }
    ctx_stack = [ctx]
    for i, sl in enumerate(trace):
        if sl['op'] == 'REVERT':
            mem_offset = int(sl['stack'][-1], base=16)
            mem_len = int(sl['stack'][-2], base=16)
            payload = read_mem(mem_offset, mem_len, sl['memory'])
            reason_offset = int.from_bytes(payload[4:36], byteorder='big', signed=False)
            reason_len = int.from_bytes(payload[36:68], byteorder='big', signed=False)
            message = payload[reason_offset + 4 + 32 : reason_offset + 4 + 32 + reason_len]
            ctx['actions'].append({
                'type': 'REVERT',
                'message': message,
            })
        if sl['op'] == 'RETURN':
            ret_offset = int(sl['stack'][-1], base=16)
            ret_len = int(sl['stack'][-1], base=16)
            b = read_mem(ret_offset, ret_len, sl['memory'])
            ctx['actions'].append({
                'type': 'RETURN',
                'data': b,
            })
        if sl['op'] == 'RETURN' or sl['op'] == 'STOP' or sl['op'] == 'REVERT':
            if i + 1 < len(trace):
                ctx['gasEnd'] = trace[i+1]['gas']
            else:
                ctx['gasEnd'] = sl['gas'] - sl['gasCost']
            ctx['traceEnd'] = i
            ctx = ctx_stack.pop()
        if sl['op'] == 'STATICCALL':
            dest = '0x' + sl['stack'][-2].replace('0x', '').lstrip('0').rjust(40, '0')
            arg_offset = int(sl['stack'][-3], base=16)
            arg_len = int(sl['stack'][-4], base=16)
            b = read_mem(arg_offset, arg_len, sl['memory'])
            ctx['actions'].append({
                'type': 'STATICCALL',
                'gasStart': sl['gas'],
                'traceStart': i,
                'callee': web3.Web3.toChecksumAddress(dest),
                'from': ctx['callee'],
                'args': b,
                'actions': []
            })
            ctx_stack.append(ctx)
            ctx = ctx['actions'][-1]
        if sl['op'] == 'DELEGATECALL':
            dest = '0x' + sl['stack'][-2].replace('0x', '').lstrip('0').rjust(40, '0')
            arg_offset = int(sl['stack'][-4], base=16)
            arg_len = int(sl['stack'][-5], base=16)
            b = read_mem(arg_offset, arg_len, sl['memory'])
            ctx['actions'].append({
                'type': 'DELEGATECALL',
                'gasStart': sl['gas'],
                'traceStart': i,
                'callee': web3.Web3.toChecksumAddress(dest),
                'from': ctx['callee'],
                'args': b,
                'actions': []
            })
            ctx_stack.append(ctx)
            ctx = ctx['actions'][-1]
        if sl['op'] == 'CALL':
            dest = '0x' + sl['stack'][-2].replace('0x', '').lstrip('0').rjust(40, '0')
            arg_offset = int(sl['stack'][-4], base=16)
            arg_len = int(sl['stack'][-5], base=16)
            b = read_mem(arg_offset, arg_len, sl['memory'])
            ctx['actions'].append({
                'type': 'CALL',
                'gasStart': sl['gas'],
                'traceStart': i,
                'callee': web3.Web3.toChecksumAddress(dest),
                'from': ctx['callee'],
                'args': b,
                'actions': []
            })
            ctx_stack.append(ctx)
            ctx = ctx['actions'][-1]
        if sl['op'] == 'LOG3' and sl['stack'][-3].replace('0x', '').lstrip('0') == 'ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef':
            from_ = '0x' + sl['stack'][-4].replace('0x', '').lstrip('0').rjust(40, '0')
            to_ = '0x' + sl['stack'][-5].replace('0x', '').lstrip('0').rjust(40, '0')
            argstart = int(sl['stack'][-1], 16)
            arglen = int(sl['stack'][-2], 16)
            val = int.from_bytes(read_mem(argstart, arglen, sl['memory']), byteorder='big', signed=True)
            ctx['actions'].append({
                'type': 'TRANSFER',
                'from': web3.Web3.toChecksumAddress(from_[-40:]),
                'to': web3.Web3.toChecksumAddress(to_[-40:]),
                'value': val,
            })
    return ctx


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
            '../abis'
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


def get_block_logs(w3: web3.Web3, block_identifier: int) -> typing.List[web3.types.LogReceipt]:
    block = w3.eth.get_block(block_identifier)
    
    logs = []
    for txn in block['transactions']:
        receipt = w3.eth.get_transaction_receipt(txn)
        logs.extend(receipt['logs'])
    return logs


_block_timestamp_cache: typing.Dict[int, int] = {}
def get_block_timestamp(w3: web3.Web3, block_number: int) -> int:
    got = _block_timestamp_cache.get(block_number, None)
    
    if got is not None:
        return got
    
    # need to load
    ts = w3.eth.get_block(block_number)['timestamp']
    _block_timestamp_cache[block_number] = ts
    return ts


# taken from https://gist.github.com/thatalextaylor/7408395 on Jan 12th 2022
def pretty_time_delta(seconds):
    sign_string = '-' if seconds < 0 else ''
    seconds = abs(int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '%s%dd%dh%dm%ds' % (sign_string, days, hours, minutes, seconds)
    elif hours > 0:
        return '%s%dh%dm%ds' % (sign_string, hours, minutes, seconds)
    elif minutes > 0:
        return '%s%dm%ds' % (sign_string, minutes, seconds)
    else:
        return '%s%ds' % (sign_string, seconds)


class ProgressReporter:
    """
    Collects and reports progress on a running job.
    """
    _log: logging.Logger
    _total_items: int
    _observed_items: int
    _last_screen_print_ts: float
    _last_observation_ts: float
    _sma_points: typing.Deque[typing.Tuple[float, int]]
    _print_period_sec: float
    _start_val: int

    # We will keep the progress observations until both (A) and (B) are met:
    # (A) - At least MIN_SMA_POINTS are recorded
    # (B) - The oldest observation is more than MAX_SMA_PERIOD_SEC seconds old
    # At which point we delete observations FIFO
    MIN_SMA_POINTS: int = 30
    # 1 hour
    MAX_SMA_PERIOD_SEC: int = 60 * 60 * 1
    
    # For purposes of selective rejection of observations into SMA,
    # this is the target number of points 
    TARGET_N_SMA_POINTS = 1000

    # computed from previous values
    _TARGET_SMA_INTERARRIVAL_TIME = MAX_SMA_PERIOD_SEC / TARGET_N_SMA_POINTS

    def __init__(
            self,
            l: logging.Logger,
            total_items: int,
            start_val: int = 0,
            print_period_sec: float = 60,
        ) -> None:
        assert total_items >= 0
        assert l is not None
        self._log = l
        self._total_items = total_items
        self._last_screen_print_ts = None
        self._last_observation_ts = None
        self._observed_items = 0
        self._start_val = start_val
        self._sma_points = typing.Deque()
        self._print_period_sec = print_period_sec

    def observe(self, n_items = 1) -> bool:
        """
        Observe that `n_items` (default = 1) were processed.
        Logs progress if appropriate.

        Returns: True if printed to screen, otherwise False
        """
        now = time.time()
        self._observed_items += n_items

        self._sma_points.append((now, self._observed_items))

        if self._last_observation_ts is not None:
            interarrival_time = now - self._last_observation_ts
        else:
            interarrival_time = float('inf')

        self._last_observation_ts = now

        ret = False
        if len(self._sma_points) >= 2:
            # we can attempt to print a progress report
            if self._last_screen_print_ts is None or \
                    time.time() > self._last_screen_print_ts + self._print_period_sec:
                # get the estimated items/s and ETA
                fit: scipy.stats = scipy.stats.linregress(
                    [x for x, _ in self._sma_points],
                    [y for _, y in self._sma_points]
                )
                items_ps = fit.slope
                items_remaining = self._total_items - (self._observed_items + self._start_val)
                eta_seconds = items_remaining / items_ps
                eta_pretty = pretty_time_delta(eta_seconds)
                self._log.info(f'Progress: {self._observed_items + self._start_val} / {self._total_items} ({self._observed_items / (self._total_items - self._start_val) * 100 :.2f}%) - ETA {eta_pretty}')
                self._last_screen_print_ts = now
                ret = True

        # Selectively drop last observation to keep around target level
        if interarrival_time < self.__class__._TARGET_SMA_INTERARRIVAL_TIME:
            percent_chance_keep = interarrival_time / self.__class__._TARGET_SMA_INTERARRIVAL_TIME
            if random.random() > percent_chance_keep:
                # drop :(
                    self._sma_points.pop()

        # Clean-up: remove old SMA points if needed
        trash_observations_before_ts = now - self.__class__.MAX_SMA_PERIOD_SEC
        while len(self._sma_points) > self.__class__.MIN_SMA_POINTS and \
                self._sma_points[0][0] < trash_observations_before_ts:
            self._sma_points.popleft()

        return ret


uv3: web3.contract.Contract = web3.Web3().eth.contract(
    address=b'\x00' * 20,
    abi=get_abi('uniswap_v3/IUniswapV3Pool.json')['abi'],
)

uv2: web3.contract.Contract = web3.Web3().eth.contract(
    address=b'\x00' * 20,
    abi=get_abi('uniswap_v2/IUniswapV2Pair.json')['abi'],
)

erc20: web3.contract.Contract = web3.Web3().eth.contract(
    address=b'\x00' * 20,
    abi=get_abi('erc20.abi.json'),
)

