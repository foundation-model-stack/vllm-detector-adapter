"""Utility helpers shared by the test suite."""
from __future__ import annotations

import socket
import time
from typing import Callable, TypeVar

__all__ = ["get_random_port", "wait_until", "TaskFailedError"]

T = TypeVar("T")
Predicate = Callable[[], bool]


class TaskFailedError(RuntimeError):
    """Raised when the background server task exits unexpectedly."""


def get_random_port() -> int:
    """Get an unused TCP port"""
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def wait_until(
    predicate: Predicate,
    *,
    timeout: float = 30.0,
    interval: float = 0.5,
) -> None:
    """
    Poll predicate until it returns True or timeout seconds elapse.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            if predicate():
                return
        except Exception:
            pass

        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for condition")

        time.sleep(interval)
