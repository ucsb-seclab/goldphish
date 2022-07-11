import typing

class BlockObservationResult(typing.NamedTuple):
    pair_prices_updated: typing.List[typing.Tuple[str, str]]
    swap_enabled: typing.Optional[bool]
    gradual_weight_adjusting_scheduled: typing.Optional[bool]
