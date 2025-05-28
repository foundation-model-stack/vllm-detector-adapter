"""This module bridges the gap between tgis adapter and vllm and allows us to run
vllm-detector-adapter + tgis-adater grpc server together

Most of this file is taken from: https://github.com/opendatahub-io/vllm-tgis-adapter/blob/main/src/vllm_tgis_adapter/__main__.py
"""
# Future
from __future__ import annotations

# Standard
from concurrent.futures import FIRST_COMPLETED
from typing import TYPE_CHECKING
import argparse
import asyncio
import contextlib
import importlib.util
import inspect
import os
import traceback

# Third Party
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai import api_server
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser
import uvloop

if TYPE_CHECKING:
    import socket

    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.protocol import AsyncEngineClient

# Local
from vllm_detector_adapter.api_server import (
    add_chat_detection_params,
    init_app_state_with_detectors,
)
from vllm_detector_adapter.logging import init_logger

TIMEOUT_KEEP_ALIVE = 5
TGIS_ADAPTER_LIBRARY_NAME = "vllm_tgis_adapter"

logger = init_logger("vllm_detector_adapter.start_with_tgis_adapter")


# Check if tgis_adapter is installed or not
if not importlib.util.find_spec(TGIS_ADAPTER_LIBRARY_NAME):
    logger.error("{} library not installed".format(TGIS_ADAPTER_LIBRARY_NAME))
    exit(1)
else:
    # Third Party
    from vllm_tgis_adapter.grpc import run_grpc_server
    from vllm_tgis_adapter.http import build_http_server
    from vllm_tgis_adapter.tgis_utils.args import (
        EnvVarArgumentParser,
        add_tgis_args,
        postprocess_tgis_args,
    )
    from vllm_tgis_adapter.tgis_utils.logs import add_logging_wrappers
    from vllm_tgis_adapter.utils import check_for_failed_tasks, write_termination_log

# Note: this function references run_http_server in
# vllm-tgis-adapter directly. https://github.com/opendatahub-io/vllm-tgis-adapter/blob/1d62372b78f24e156d6a748f30bb7273d8364532/src/vllm_tgis_adapter/http.py#L20C1-L48C24
async def run_http_server(
    args: argparse.Namespace,
    engine: AsyncLLMEngine | AsyncEngineClient,
    sock: socket.socket | None = None,
    **uvicorn_kwargs,  # noqa: ANN003
) -> None:
    # modified copy of vllm.entrypoints.openai.api_server.run_server that
    # allows passing of the engine

    app = api_server.build_app(args)
    if hasattr(engine, "get_vllm_config"):
        vllm_config = await engine.get_vllm_config()
        await init_app_state_with_detectors(
            engine, vllm_config, app.state, args
        )
    else:
        model_config = await engine.get_model_config()
        await init_app_state_with_detectors(
            engine, model_config, app.state, args
        )

    serve_kwargs = {
        "host": args.host,
        "port": args.port,
        "log_level": args.uvicorn_log_level,
        "timeout_keep_alive": TIMEOUT_KEEP_ALIVE,
        "ssl_keyfile": args.ssl_keyfile,
        "ssl_certfile": args.ssl_certfile,
        "ssl_ca_certs": args.ssl_ca_certs,
        "ssl_cert_reqs": args.ssl_cert_reqs,
    }
    serve_kwargs.update(uvicorn_kwargs)

    # should only be used in versions of vllm >= 0.7.3
    if "sock" in inspect.getfullargspec(serve_http).args:
        serve_kwargs["sock"] = sock

    shutdown_coro = await serve_http(app, **serve_kwargs)

    # launcher.serve_http returns a shutdown coroutine to await
    # (The double await is intentional)
    await shutdown_coro


async def start_servers(args: argparse.Namespace) -> None:
    """This function starts both http_server (openai + vllm-detector-adapter) and
    gRPC server (tgis-adapter)
    """
    loop = asyncio.get_running_loop()

    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    sock_addr = (args.host or "", args.port)
    sock = api_server.create_server_socket(sock_addr)

    tasks: list[asyncio.Task] = []
    async with api_server.build_async_engine_client(args) as engine:
        add_logging_wrappers(engine)

        vllm_server = await build_http_server(args, engine)

        http_server_task = loop.create_task(
            run_http_server(args, engine, sock),
            name="http_server",
        )
        # The http server task will catch interrupt signals for us
        tasks.append(http_server_task)

        grpc_server_task = loop.create_task(
            run_grpc_server(args, engine, vllm_server),
            name="grpc_server",
        )
        tasks.append(grpc_server_task)

        runtime_error = None
        with contextlib.suppress(asyncio.CancelledError):
            # Both server tasks will exit normally on shutdown, so we await
            # FIRST_COMPLETED to catch either one shutting down.
            await asyncio.wait(tasks, return_when=FIRST_COMPLETED)
            if engine and engine.errored and not engine.is_running:
                # both servers shut down when an engine error
                # is detected, with task done and exception handled
                # here we just notify of that error and let servers be
                runtime_error = RuntimeError(
                    "AsyncEngineClient error detected, this may be caused by an \
                        unexpected error in serving a request. \
                        Please check the logs for more details."
                )

        failed_task = check_for_failed_tasks(tasks)

        # Once either server shuts down, cancel the other
        for task in tasks:
            task.cancel()

        # Final wait for both servers to finish
        await asyncio.wait(tasks)

        # Raise originally-failed task if applicable
        if failed_task:
            name, coro_name = failed_task.get_name(), failed_task.get_coro().__name__
            exception = failed_task.exception()
            raise RuntimeError(f"Failed task={name} ({coro_name})") from exception

        if runtime_error:
            raise runtime_error


def run_and_catch_termination_cause(
    loop: asyncio.AbstractEventLoop, task: asyncio.Task
) -> None:
    try:
        loop.run_until_complete(task)
    except Exception:
        # Report the first exception as cause of termination
        msg = traceback.format_exc()
        write_termination_log(
            msg, os.getenv("TERMINATION_LOG_DIR", "/dev/termination-log")
        )
        raise


if __name__ == "__main__":
    parser = FlexibleArgumentParser("vLLM TGIS GRPC + OpenAI REST api server")
    # convert to our custom env var arg parser
    parser = EnvVarArgumentParser(parser=make_arg_parser(parser))
    parser = add_tgis_args(parser)
    parser = add_chat_detection_params(parser)
    args = postprocess_tgis_args(parser.parse_args())
    assert args is not None

    # logger.info("vLLM version %s", f"{vllm.__version__}")
    logger.info("args: %s", args)

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    loop = asyncio.new_event_loop()
    task = loop.create_task(start_servers(args))
    run_and_catch_termination_cause(loop, task)
