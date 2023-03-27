import typing
import logging
import psycopg2.extensions
import web3
import web3.logs
import web3.contract
import web3._utils.events

from utils import get_abi
from backtest.gather_samples.tokens import get_token
from .base_log_scraper import BaseLogScraper, ScrapeResult, PrimeResult

l = logging.getLogger(__name__)

class OrionV2Scraper(BaseLogScraper):
    factory_abi: typing.Dict
    exchange_abi: typing.Dict
    factory: web3.contract.Contract
    exchanges: typing.Set[str]

    def __init__(self) -> None:
        super().__init__()
        self.factory_abi = get_abi('uniswap_v2/IUniswapV2Factory.json')
        self.exchanges = set()

    def prime(self, curr: psycopg2.extensions.cursor):
        curr.execute(
            """
            CREATE TABLE IF NOT EXISTS orion_v2_exchanges (
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
            CREATE UNIQUE INDEX IF NOT EXISTS idx_orion_v2_ex_tokens ON orion_v2_exchanges (token0_id, token1_id);
            CREATE INDEX IF NOT EXISTS idx_uni2_token0 ON orion_v2_exchanges (token0_id);
            CREATE INDEX IF NOT EXISTS idx_uni2_token1 ON orion_v2_exchanges (token1_id);
            CREATE INDEX IF NOT EXISTS idx_uni2_ex_addr ON orion_v2_exchanges USING hash (address);
            """
        )

        return PrimeResult(
            ['0x5FA0060FcfEa35B31F7A5f6025F0fF399b98Edf1'],
        )

    def scrape(
                self,
                curr: psycopg2.extensions.cursor,
                w3: web3.Web3,
                logs: typing.List[typing.Dict]
            ) -> ScrapeResult:
        factory: web3.contract.Contract = w3.eth.contract(
            address = '0x5FA0060FcfEa35B31F7A5f6025F0fF399b98Edf1',
            abi = self.factory_abi,
        )
        relevant_logs = []
        for log in logs:
            # filter out irrelevant logs
            if log['address'] == factory.address:
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
            "SELECT id FROM orion_v2_exchanges WHERE address = %s",
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
            """
            INSERT INTO orion_v2_exchanges (
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
        l.info(f'Registered orion_v2_exchanges exchange id={curr.fetchone()[0]}')
        return receipt['args']['pair']
