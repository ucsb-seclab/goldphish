"""
Scrapes CroDeFi exchange addresses
"""


import typing
import logging
import psycopg2.extensions
import web3
import web3.logs
import web3.contract
import web3._utils.events
from eth_utils import event_abi_to_log_topic

from utils import get_abi
from .base_log_scraper import BaseLogScraper, ScrapeResult, PrimeResult


l = logging.getLogger(__name__)

class BancorV2Scraper(BaseLogScraper):
    exchange_abi: typing.Dict
    factory: web3.contract.Contract

    def __init__(self) -> None:
        super().__init__()
        self.factory = web3.Web3().eth.contract(
            address = '0x4ade0e57Bc2E129f62547af4d620fB40d28EA269',
            abi = get_abi('bancor_v2/converterfactory.abi.json'),
        )
        self.pair_created_topic = event_abi_to_log_topic(self.factory.events.NewConverter().abi)

    def prime(self, curr: psycopg2.extensions.cursor):
        curr.execute(
            """
            CREATE TABLE IF NOT EXISTS bancor_v2_exchanges (
                id SERIAL PRIMARY KEY NOT NULL,
                address BYTEA NOT NULL,
                origin_txn BYTEA NOT NULL,
                origin_block INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_bancor_v2_exchanges_ex_addr ON bancor_v2_exchanges USING hash (address);
            """
        )

        return PrimeResult(['0x4ade0e57Bc2E129f62547af4d620fB40d28EA269'])


    def scrape(
                self,
                curr: psycopg2.extensions.cursor,
                w3: web3.Web3,
                logs: typing.List[typing.Dict]
            ) -> ScrapeResult:
        relevant_logs = []
        for log in logs:
            # filter out irrelevant logs
            if log['address'] == self.factory.address and len(log['topics']) > 0 and log['topics'][0] == self.pair_created_topic:
                relevant_logs.append(log)

        l.debug(f'Have {len(relevant_logs)} relevant logs')

        for log in relevant_logs:
            self.process_factory_event(
                w3,
                curr,
                log
            )
        return ScrapeResult(set())

    def process_factory_event(
                self,
                w3: web3.Web3,
                curr: psycopg2.extensions.cursor,
                log
            ) -> typing.Optional[str]:
        receipt = self.factory.events.NewConverter().processLog(log)

        # if we already know about this exchange then skip
        curr.execute(
            "SELECT id FROM bancor_v2_exchanges WHERE address = %s",
            (bytes.fromhex(receipt['args']['_converter'][2:]),)
        )
        if curr.rowcount > 0:
            id_ = curr.fetchone()[0]
            l.debug(f'Already know about this exchange, id={id_}')
            return

        # record the exchange info
        curr.execute(
            """
            INSERT INTO bancor_v2_exchanges (
                address, origin_txn, origin_block
            )
            VALUES (%s, %s, %s)
            RETURNING (id)
            """,
            (
                bytes.fromhex(receipt['args']['_converter'][2:]),
                receipt['transactionHash'],
                receipt['blockNumber'],
            )
        )
        l.info(f'Registered bancor_v2_exchanges exchange id={curr.fetchone()[0]}')
        return receipt['args']['_converter']
