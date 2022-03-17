"""
Basic utils for performance profiling, mostly in units of time
"""

import time
import typing
import logging

ENABLED = False
PRINT_INTERVAL_SECONDS = 2 * 60

_global_profile: typing.Dict[str, float] = {}
_last_log: float = 0

l = logging.getLogger(__name__)


def maybe_log():
    """
    If enough time has passed since last log, emits a new log report and clears
    all profiling info.
    """
    global _last_log
    if not ENABLED:
        return

    if time.time() < _last_log + PRINT_INTERVAL_SECONDS:
        # too soon
        return
    
    for k in sorted(_global_profile.keys()):
        l.debug(f'profile name="{k}" seconds={_global_profile[k]}')
        _global_profile[k] = 0

    _last_log = time.time()


def get_measurement(name: str) -> typing.Optional[float]:
    """
    Gets the measurement named 'name'; defaults to 0
    """
    return _global_profile.get(name, 0)


def reset_measurement(name: str):
    """
    Sets this measurement back to zero
    """
    if not ENABLED:
        return
    _global_profile[name] = 0


def inc_measurement(name: str, elapsed: float):
    """
    increase measurement by the given amount
    """
    if not ENABLED:
        return
    _global_profile[name] = _global_profile.get(name, 0) + elapsed


class ProfilerContextManager:
    _name: str
    _start: float

    def __init__(self, name: str) -> None:
        self._name = name
        self._start = None

    def __enter__(self) -> None:
        if not ENABLED:
            return
        self._start = time.time()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not ENABLED:
            return
        elapsed = time.time() - self._start
        _global_profile[self._name] = _global_profile.get(self._name, 0) + elapsed


def profile(name: str) -> ProfilerContextManager:
    """
    Returns a context manager profiling functionality
    """
    return ProfilerContextManager(name)

