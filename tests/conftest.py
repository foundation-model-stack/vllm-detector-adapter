"""
Pytest fixtures for spinning up a live vllm-detector-adapter HTTP server
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Generator
import argparse
import asyncio
import signal
import sys
import threading
import traceback

# Third Party
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils import FlexibleArgumentParser
import pytest
import requests

# Local
from .utils import TaskFailedError, get_random_port, wait_until
from vllm_detector_adapter.api_server import add_chat_detection_params, run_server
from vllm_detector_adapter.utils import LocalEnvVarArgumentParser


@pytest.fixture(scope="session")
def http_server_port() -> int:
    """Port for the http server"""
    return get_random_port()


@pytest.fixture(scope="session")
def http_server_url(http_server_port: int) -> str:
    """Url for the http server"""
    return f"http://localhost:{http_server_port}"


@pytest.fixture
def args(monkeypatch, http_server_port: int) -> argparse.Namespace:
    """Mimic: python -m vllm_detector_adapter.api_server --model <MODEL> …"""
    # Use a 'tiny' model for test purposes
    model_name = "facebook/opt-125m"

    mock_argv = [
        "__main__.py",
        "--model",
        model_name,
        f"--port={http_server_port}",
        "--host=localhost",
        "--dtype=float32",
        "--device=cpu",
    ]
    monkeypatch.setattr(sys, "argv", mock_argv, raising=False)

    # Build parser like __main__ in api_server.py
    base_parser = FlexibleArgumentParser(description="vLLM server setup for pytest.")
    parser = LocalEnvVarArgumentParser(parser=make_arg_parser(base_parser))
    parser = add_chat_detection_params(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    return args


@pytest.fixture
def _servers(
    args: argparse.Namespace,
    http_server_url: str,
    monkeypatch,
) -> Generator[None, None, None]:
    """Start server in background thread"""
    loop = asyncio.new_event_loop()
    task: asyncio.Task | None = None

    # Patch signal handling so child threads don’t touch the OS handler table
    monkeypatch.setattr(loop, "add_signal_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr(signal, "signal", lambda *args, **kwargs: None)

    def target() -> None:
        nonlocal task
        task = loop.create_task(run_server(args))
        try:
            print("[conftest] starting run server...", flush=True)
            loop.run_until_complete(task)
        except Exception as e:
            print("[conftest] server failed to start:", e, flush=True)
            traceback.print_exc
            raise
        finally:
            loop.close()

    t = threading.Thread(target=target, name="vllm-detector-server")
    t.start()

    def _health() -> bool:
        if task and task.done():
            raise TaskFailedError(task.exception())
        requests.get(f"{http_server_url}/health", timeout=1).raise_for_status()
        return True

    try:
        wait_until(_health, timeout=120.0, interval=1.0)
        # tests execute with live server
        yield
    finally:
        if task:
            task.cancel()
        t.join()


@pytest.fixture
def api_base_url(_servers, http_server_url: str) -> str:
    """Pulls up the server and returns the URL to tests"""
    return http_server_url
