# Standard
from argparse import Namespace
import inspect
import signal

# Third Party
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.datastructures import State
from vllm.config import ModelConfig
from vllm.engine.arg_utils import nullable_str
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai import api_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.protocol import ErrorResponse
from vllm.entrypoints.openai.reasoning_parsers import ReasoningParserManager
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.utils import FlexibleArgumentParser, is_valid_ipv6_address, set_ulimit
from vllm.version import __version__ as VLLM_VERSION
import uvloop

# Local
from vllm_detector_adapter import generative_detectors
from vllm_detector_adapter.logging import init_logger
from vllm_detector_adapter.protocol import (
    ChatDetectionRequest,
    ContentsDetectionRequest,
    ContentsDetectionResponse,
    ContextAnalysisRequest,
    DetectionResponse,
)

TIMEOUT_KEEP_ALIVE = 5  # seconds

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = init_logger("vllm_detector_adapter.api_server")

# Use original vllm router and add to it
router = api_server.router


def chat_detection(
    request: Request,
) -> generative_detectors.base.ChatCompletionDetectionBase:
    return request.app.state.detectors_serving_chat_detection


async def init_app_state_with_detectors(
    engine_client: EngineClient,
    model_config: ModelConfig,
    state: State,
    args: Namespace,
) -> None:
    """Add detection capabilities to app state"""
    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    if args.disable_log_requests:
        request_logger = None
    else:
        request_logger = RequestLogger(max_log_len=args.max_log_len)

    base_model_paths = [
        BaseModelPath(name=name, model_path=args.model) for name in served_model_names
    ]

    resolved_chat_template = load_chat_template(args.chat_template)
    state.openai_serving_models = OpenAIServingModels(
        engine_client=engine_client,
        model_config=model_config,
        base_model_paths=base_model_paths,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
    )

    # Use vllm app state init
    # init_app_state became async in https://github.com/vllm-project/vllm/pull/11727
    # ref. https://github.com/opendatahub-io/vllm-tgis-adapter/pull/207
    maybe_coroutine = api_server.init_app_state(
        engine_client, model_config, state, args
    )
    if inspect.isawaitable(maybe_coroutine):
        await maybe_coroutine

    generative_detector_class = generative_detectors.MODEL_CLASS_MAP[args.model_type]

    # Add chat detection
    state.detectors_serving_chat_detection = generative_detector_class(
        args.task_template,
        args.output_template,
        engine_client,
        model_config,
        state.openai_serving_models,
        args.response_role,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_auto_tools=args.enable_auto_tool_choice,
        tool_parser=args.tool_call_parser,
        enable_prompt_tokens_details=args.enable_prompt_tokens_details,
    )


async def run_server(args, **uvicorn_kwargs) -> None:
    """Server should include all vllm supported endpoints and any
    newly added detection endpoints, much of this parsing code
    is taken directly from the vllm API server
    ref. https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py"""
    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    valid_tool_parses = ToolParserManager.tool_parsers.keys()
    if args.enable_auto_tool_choice and args.tool_call_parser not in valid_tool_parses:
        raise KeyError(
            f"invalid tool call parser: {args.tool_call_parser} "
            f"(chose from {{ {','.join(valid_tool_parses)} }})"
        )

    valid_reasoning_parses = ReasoningParserManager.reasoning_parsers.keys()
    if args.enable_reasoning and args.reasoning_parser not in valid_reasoning_parses:
        raise KeyError(
            f"invalid reasoning parser: {args.reasoning_parser} "
            f"(chose from {{ {','.join(valid_reasoning_parses)} }})"
        )

    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    sock_addr = (args.host or "", args.port)
    sock = api_server.create_server_socket(sock_addr)

    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    set_ulimit()

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    async with api_server.build_async_engine_client(args) as engine_client:
        # Use vllm build_app which adds middleware
        app = api_server.build_app(args)

        model_config = await engine_client.get_model_config()
        await init_app_state_with_detectors(
            engine_client, model_config, app.state, args
        )

        def _listen_addr(a: str) -> str:
            if is_valid_ipv6_address(a):
                return "[" + a + "]"
            return a or "0.0.0.0"

        logger.info(
            "Starting vLLM API server on http://%s:%d",
            _listen_addr(sock_addr[0]),
            sock_addr[1],
        )

        shutdown_task = await serve_http(
            app,
            sock=sock,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    await shutdown_task

    sock.close()


@router.post("/api/v1/text/chat")
async def create_chat_detection(request: ChatDetectionRequest, raw_request: Request):
    """Support chat detection endpoint"""

    detector_response = await chat_detection(raw_request).chat(request, raw_request)

    if isinstance(detector_response, ErrorResponse):
        # ErrorResponse includes code and message, corresponding to errors for the detectorAPI
        return JSONResponse(
            content=detector_response.model_dump(), status_code=detector_response.code
        )

    elif isinstance(detector_response, DetectionResponse):
        return JSONResponse(content=detector_response.model_dump())

    return JSONResponse({})


@router.post("/api/v1/text/context/doc")
async def create_context_doc_detection(
    request: ContextAnalysisRequest, raw_request: Request
):
    """Support context analysis endpoint"""

    detector_response = await chat_detection(raw_request).context_analyze(
        request, raw_request
    )

    if isinstance(detector_response, ErrorResponse):
        # ErrorResponse includes code and message, corresponding to errors for the detectorAPI
        return JSONResponse(
            content=detector_response.model_dump(), status_code=detector_response.code
        )

    elif isinstance(detector_response, DetectionResponse):
        return JSONResponse(content=detector_response.model_dump())

    return JSONResponse({})


@router.post("/api/v1/text/contents")
async def create_contents_detection(
    request: ContentsDetectionRequest, raw_request: Request
):
    """Support content analysis endpoint"""

    detector_response = await chat_detection(raw_request).content_analysis(
        request, raw_request
    )
    if isinstance(detector_response, ErrorResponse):
        # ErrorResponse includes code and message, corresponding to errors for the detectorAPI
        return JSONResponse(
            content=detector_response.model_dump(), status_code=detector_response.code
        )

    elif isinstance(detector_response, ContentsDetectionResponse):
        return JSONResponse(content=detector_response.model_dump())

    return JSONResponse({})


def add_chat_detection_params(parser):
    parser.add_argument(
        "--task-template",
        type=nullable_str,
        default=None,
        help="The file path to the task template, "
        "or the template in single-line form "
        "for the specified model",
    )
    parser.add_argument(
        "--output-template",
        type=nullable_str,
        default=None,
        help="The file path to the output template, "
        "or the template in single-line form "
        "for the specified model",
    )
    parser.add_argument(
        "--model-type",
        type=generative_detectors.ModelTypes,
        choices=[
            member.lower() for member in generative_detectors.ModelTypes._member_names_
        ],
        default=generative_detectors.ModelTypes.LLAMA_GUARD,
        help="The model type of the generative model",
    )
    return parser


if __name__ == "__main__":

    # Verify vllm compatibility
    # Local
    from vllm_detector_adapter import package_validate

    package_validate.verify_vllm_compatibility()

    # NOTE(simon):
    # This section should be in sync with vllm/scripts.py for CLI entrypoints.
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)

    # Add chat detection params
    parser = add_chat_detection_params(parser)

    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))
