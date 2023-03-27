import typing
import logging
import psycopg2.extensions
import web3
import web3.logs
import web3.contract
import web3._utils.events
from eth_utils import event_abi_to_log_topic

from utils import get_abi
from backtest.gather_samples.tokens import get_token
from .base_log_scraper import BaseLogScraper, ScrapeResult, PrimeResult

l = logging.getLogger(__name__)


class EqualizerScraper(BaseLogScraper):
    name = 'equalizer'
    factory_addr = '0xF14421F0BCf9401d8930872C2d44d8e67e40529a'
    factory_abi: typing.Dict
    exchange_abi: typing.Dict
    factory: web3.contract.Contract
    exchanges: typing.Set[str]

    def __init__(self) -> None:
        super().__init__()
        self.factory_abi = get_abi('uniswap_v2/IUniswapV2Factory.json')
        self.exchanges = set()
        self.pool_created_topic = event_abi_to_log_topic(web3.Web3().eth.contract(address=b'\x00'*20, abi=self.factory_abi).events.PairCreated().abi)

    def prime(self, curr: psycopg2.extensions.cursor):
        curr.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.__class__.name}_exchanges (
                id SERIAL PRIMARY KEY NOT NULL,
                token0_id INTEGER NOT NULL,
                token1_id INTEGER NOT NULL,
                index INTEGER NOT NULL,
                address BYTEA NOT NULL,
                origin_txn BYTEA NOT NULL,
                origin_block INTEGER NOT NULL,
                FOREIGN KEY (token0_id) REFERENCES tokens (id),
                FOREIGN KEY (token1_id) REFERENCES tokens (id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_{self.__class__.name}_exchanges_ex_tokens ON {self.__class__.name}_exchanges (token0_id, token1_id);
            CREATE INDEX IF NOT EXISTS idx_{self.__class__.name}_exchanges_token0 ON {self.__class__.name}_exchanges (token0_id);
            CREATE INDEX IF NOT EXISTS idx_{self.__class__.name}_exchanges_token1 ON {self.__class__.name}_exchanges (token1_id);
            CREATE INDEX IF NOT EXISTS idx_{self.__class__.name}_exchanges_ex_addr ON {self.__class__.name}_exchanges USING hash (address);
            """
        )

        return PrimeResult(
            [self.__class__.factory_addr],
        )

    def scrape(
                self,
                curr: psycopg2.extensions.cursor,
                w3: web3.Web3,
                logs: typing.List[typing.Dict]
            ) -> ScrapeResult:
        factory: web3.contract.Contract = w3.eth.contract(
            address = self.__class__.factory_addr,
            abi = self.factory_abi,
        )
        relevant_logs = []
        for log in logs:
            # filter out irrelevant logs
            if log['address'] == factory.address and len(log['topics']) > 0 and log['topics'][0] == self.pool_created_topic:
                relevant_logs.append(log)

        l.debug(f'Have {len(relevant_logs)} relevant logs')

        new_addrs = set()
        for log in relevant_logs:
            if log['address'] == factory.address:
                created_addr = self.process_factory_event(
                    w3,
                    curr,
                    factory,
                    log
                )
                if created_addr is not None:
                    new_addrs.add(created_addr)
        # return ScrapeResult(new_addrs)
        return ScrapeResult(set())

    def process_factory_event(
                self,
                w3: web3.Web3,
                curr: psycopg2.extensions.cursor,
                factory: web3.contract.Contract,
                log
            ) -> typing.Optional[str]:
        receipt = factory.events.PairCreated().processLog(log)

        # if we already know about this exchange then skip
        curr.execute(
            f"SELECT id FROM {self.__class__.name}_exchanges WHERE address = %s",
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
        
        token0_id = get_token(w3, curr, token0_addr, block_identifier=block_number + 1).id
        token1_id = get_token(w3, curr, token1_addr, block_identifier=block_number + 1).id

        # sanity check
        assert isinstance(token0_id, int)
        assert isinstance(token1_id, int)

        # record the exchange info
        curr.execute(
            f"""
            INSERT INTO {self.__class__.name}_exchanges (
                token0_id, token1_id, index, address, origin_txn, origin_block
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING (id)
            """,
            (
                token0_id,
                token1_id,
                0,
                bytes.fromhex(receipt['args']['pair'][2:]),
                receipt['transactionHash'],
                receipt['blockNumber'],
            )
        )
        l.info(f'Registered {self.__class__.name}_exchanges exchange id={curr.fetchone()[0]}')
        return receipt['args']['pair']
