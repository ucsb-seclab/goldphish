"""
Fills information about Uniswap V2 clones
Attempts to identify AMM factories and their deployed
instances.
"""

import itertools
import typing
import psycopg2.extensions
import web3
import web3.types
import web3.contract
import web3._utils.filters
import logging
from scrape.scrapers.log_based.base_log_scraper import PrimeResult, ScrapeResult
from utils import get_abi
from eth_utils import event_abi_to_log_topic


l = logging.getLogger(__name__)


class UniswapV2Clones():
    # the 'original' factory --- which we should ignore
    UNISWAP_V2_FACTORY = '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f'
    known_factories: typing.Dict[str, int]


    def __init__(self) -> None:
        self.generic_factory: web3.contract.Contract = web3.Web3() \
                .eth \
                .contract(address=b'\x00'*20, abi=get_abi('uniswap_v2/IUniswapV2Factory.json')) \

        self.pair_created_event = event_abi_to_log_topic(
            self.generic_factory.events.PairCreated().abi
        )
        self.known_factories = {}


    def prepare(self, curr: psycopg2.extensions.cursor):
        curr.execute(
            '''
            CREATE TABLE IF NOT EXISTS uniswap_v2_clone_factories (
                id      SERIAL PRIMARY KEY NOT NULL,
                address BYTEA NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_uniswap_v2_clone_factories_address ON uniswap_v2_clone_factories USING HASH (address);

            CREATE TABLE IF NOT EXISTS uniswap_v2_clone_exchanges (
                id SERIAL PRIMARY KEY NOT NULL,
                token0_id INTEGER NOT NULL,
                token1_id INTEGER NOT NULL,
                index INTEGER NOT NULL,
                address BYTEA NOT NULL,
                origin_txn BYTEA NOT NULL,
                origin_block INTEGER NOT NULL,
                factory_id INTEGER NOT NULL,
                FOREIGN KEY (factory_id) REFERENCES uniswap_v2_clone_factories (id) ON DELETE CASCADE,
                FOREIGN KEY (token0_id)  REFERENCES tokens (id),
                FOREIGN KEY (token1_id)  REFERENCES tokens (id)
            );
            CREATE INDEX IF NOT EXISTS idx_uni2_clone_token0 ON uniswap_v2_clone_exchanges (token0_id);
            CREATE INDEX IF NOT EXISTS idx_uni2_clone_token1 ON uniswap_v2_clone_exchanges (token1_id);
            CREATE INDEX IF NOT EXISTS idx_uni2_clone_ex_addr ON uniswap_v2_clone_exchanges USING hash (address);
            '''
        )

    def scrape_factories(self, w3: web3.Web3, curr: psycopg2.extensions.cursor, start_block: int, end_block: int):
        # scrape the block range for factory invocations

        found_factories = set()
        verified_factories = set()

        BATCH_SIZE = 10_000
        for i in itertools.count():
            batch_start = start_block + BATCH_SIZE * i
            batch_end_block_inclusive = min(end_block, batch_start + BATCH_SIZE - 1)

            if batch_start > end_block:
                break

            f: web3._utils.filters.Filter = w3.eth.filter({
                'topics': ['0x' + self.pair_created_event.hex()],
                'fromBlock': batch_start,
                'toBlock': batch_end_block_inclusive,
            })
            logs = f.get_all_entries()

            relevant_logs = [log for log in logs if log['address'] != self.__class__.UNISWAP_V2_FACTORY]

            l.debug(f'{batch_start:,} -> {batch_end_block_inclusive:,} got {len(relevant_logs):,} logs')

            for log in relevant_logs:
                parsed = self.generic_factory.events.PairCreated().processLog(log)

                if log['address'] not in found_factories:
                    found_factories.add(log['address'])
                    l.info(f'Found factory {log["address"]}')

                    tr = w3.provider.make_request('debug_traceTransaction', [log['transactionHash'].hex(), {'tracer': 'callTracer'}])
                    created_contracts = []

                    queue = [tr['result']]
                    while len(queue) > 0:
                        item = queue.pop()
                        if 'calls' in item:
                            queue += item['calls']
                        
                        if item['type'] == 'CREATE2':
                            creator = web3.Web3.toChecksumAddress(item['from'])
                            contract_created = web3.Web3.toChecksumAddress(item['to'])
                            if creator == log['address']:
                                created_contracts.append(contract_created)
                    
                    if parsed['args']['pair'] in created_contracts:
                        verified_factories.add(log['address'])
                        curr.execute(
                            '''
                            SELECT id FROM uniswap_v2_clone_factories WHERE address = %s
                            ''',
                            (bytes.fromhex(log['address'][2:]),),
                        )
                        if curr.rowcount >= 1:
                            (factory_id,) = curr.fetchone()
                            l.info(f'Already know about this factory (id={factory_id:,})')
                        else:
                            curr.execute(
                                '''
                                INSERT INTO uniswap_v2_clone_factories (address)
                                VALUES (%s)
                                RETURNING (id)
                                ''',
                                (bytes.fromhex(log['address'][2:]),),
                            )
                            (new_id_,) = curr.fetchone()
                            l.info(f'verified factory deployment {log["address"]} (id={new_id_:,})')
                    else:
                        l.warning(f'failed to verify {log["address"]}')
            
            curr.connection.commit()

        l.info('Done scrape for uniswap v2 clone factories')
        curr.connection.commit()

    def prime(self, curr: psycopg2.extensions.cursor) -> PrimeResult:
        curr.execute(
            '''
            SELECT id, address FROM uniswap_v2_clone_factories
            '''
        )
        l.debug(f'Loading {curr.rowcount:,} uniswap v2 factory clones')
        for id_, baddr in curr:
            address = web3.Web3.toChecksumAddress(baddr.tobytes())
            self.known_factories[address] = id_

        return PrimeResult(set(self.known_factories.keys()))

    def scrape(
                self,
                curr: psycopg2.extensions.cursor,
                w3: web3.Web3,
                logs: typing.List[web3.types.LogReceipt]
            ) -> ScrapeResult:
        token_scraper = TokenScraper(w3, curr)
        relevant_logs = []
        for log in logs:
            # filter out irrelevant logs
            if log['address'] in self.known_factories and len(log['topics']) > 0 and log['topics'][0] == self.pair_created_event:
                relevant_logs.append(log)

        l.debug(f'Have {len(relevant_logs)} relevant logs')

        for log in relevant_logs:
            self.process_factory_event(
                w3,
                curr,
                token_scraper,
                log
            )

    def process_factory_event(
                self,
                w3: web3.Web3,
                curr: psycopg2.extensions.cursor,
                token_scraper: TokenScraper,
                log: web3.types.LogReceipt,
            ) -> typing.Optional[str]:
        receipt = self.generic_factory.events.PairCreated().processLog(log)

        # if we already know about this exchange then skip
        curr.execute(
            "SELECT id FROM uniswap_v2_clone_exchanges WHERE address = %s",
            (bytes.fromhex(receipt['args']['pair'][2:]),)
        )
        if curr.rowcount > 0:
            id_ = curr.fetchone()[0]
            l.debug(f'Already know about this exchange, id={id_}')
            return

        block_number = receipt['blockNumber']

        token0_addr_sz: str = receipt['args']['token0']
        token0_addr = bytes.fromhex(token0_addr_sz[2:])
        token1_addr_sz: str = receipt['args']['token1']
        token1_addr = bytes.fromhex(token1_addr_sz[2:])
        
        token0_id = token_scraper.get_token_id(token0_addr, block_identifier=block_number + 1)
        token1_id = token_scraper.get_token_id(token1_addr, block_identifier=block_number + 1)

        # sanity check
        assert isinstance(token0_id, int)
        assert isinstance(token1_id, int)

        # record the exchange info
        curr.execute(
            """
            INSERT INTO uniswap_v2_clone_exchanges (
                token0_id, token1_id, index, address, origin_txn, origin_block, factory_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING (id)
            """,
            (
                token0_id,
                token1_id,
                0,
                bytes.fromhex(receipt['args']['pair'][2:]),
                receipt['transactionHash'],
                receipt['blockNumber'],
                self.known_factories[log['address']]
            )
        )
        l.info(f'Registered uniswap clone v2 exchange id={curr.fetchone()[0]}')
        return receipt['args']['pair']

