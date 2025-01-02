# Standard
from dataclasses import dataclass
from http import HTTPStatus
from typing import Optional
from unittest.mock import patch
import asyncio

# Third Party
from vllm.config import MultiModalConfig
from vllm.entrypoints.openai.protocol import (
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    ErrorResponse,
    UsageInfo,
)
from vllm.entrypoints.openai.serving_engine import BaseModelPath
import pytest
import pytest_asyncio

# Local
from vllm_detector_adapter.generative_detectors.granite_guardian import GraniteGuardian
from vllm_detector_adapter.protocol import (
    ChatDetectionRequest,
    ChatDetectionResponse,
    DetectionChatMessageParam,
)

MODEL_NAME = "ibm-granite/granite-guardian"  # Example granite-guardian model
CHAT_TEMPLATE = "Dummy chat template for testing {}"
BASE_MODEL_PATHS = [BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME)]


@dataclass
class MockTokenizer:
    type: Optional[str] = None


@dataclass
class MockHFConfig:
    model_type: str = "any"


@dataclass
class MockModelConfig:
    task = "generate"
    tokenizer = MODEL_NAME
    trust_remote_code = False
    tokenizer_mode = "auto"
    max_model_len = 100
    tokenizer_revision = None
    embedding_mode = False
    multimodal_config = MultiModalConfig()
    diff_sampling_param: Optional[dict] = None
    hf_config = MockHFConfig()
    logits_processor_pattern = None
    allowed_local_media_path: str = ""

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


@dataclass
class MockEngine:
    async def get_model_config(self):
        return MockModelConfig()


async def _granite_guardian_init():
    """Initialize a granite guardian"""
    engine = MockEngine()
    engine.errored = False
    model_config = await engine.get_model_config()

    granite_guardian = GraniteGuardian(
        task_template=None,
        output_template=None,
        engine_client=engine,
        model_config=model_config,
        base_model_paths=BASE_MODEL_PATHS,
        response_role="assistant",
        chat_template=CHAT_TEMPLATE,
        chat_template_content_format="auto",
        lora_modules=None,
        prompt_adapters=None,
        request_logger=None,
    )
    return granite_guardian


@pytest_asyncio.fixture
async def granite_guardian_detection():
    return _granite_guardian_init()


@pytest.fixture(scope="module")
def granite_guardian_completion_response():
    log_probs_content_yes = ChatCompletionLogProbsContent(
        token="Yes",
        logprob=0.0,
        # 5 logprobs requested for scoring, skipping bytes for conciseness
        top_logprobs=[
            ChatCompletionLogProb(token="Yes", logprob=0.0),
            ChatCompletionLogProb(token='"No', logprob=-6.3),
            ChatCompletionLogProb(token="yes", logprob=-16.44),
            ChatCompletionLogProb(token=" Yes", logprob=-16.99),
            ChatCompletionLogProb(token="YES", logprob=-17.52),
        ],
    )
    log_probs_content_random = ChatCompletionLogProbsContent(
        token="",
        logprob=-4.76,
        # 5 logprobs requested for scoring, skipping bytes for conciseness
        top_logprobs=[
            ChatCompletionLogProb(token="", logprob=-4.76),
            ChatCompletionLogProb(token="", logprob=-14.66),
            ChatCompletionLogProb(token="\n", logprob=-17.96),
            ChatCompletionLogProb(token="[/", logprob=-18.32),
            ChatCompletionLogProb(token="\n\n", logprob=-18.41),
        ],
    )
    choice_0 = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(
            role="assistant",
            content="Yes",
        ),
        logprobs=ChatCompletionLogProbs(
            content=[log_probs_content_yes, log_probs_content_random]
        ),
    )
    choice_1 = ChatCompletionResponseChoice(
        index=1,
        message=ChatMessage(
            role="assistant",
            content="Yes",
        ),
        logprobs=ChatCompletionLogProbs(
            content=[log_probs_content_random, log_probs_content_yes]
        ),
    )
    yield ChatCompletionResponse(
        model=MODEL_NAME,
        choices=[choice_0, choice_1],
        usage=UsageInfo(prompt_tokens=136, total_tokens=140, completion_tokens=4),
    )


### Tests #####################################################################


def test_preprocess_with_detector_params(granite_guardian_detection):
    llama_guard_detection_instance = asyncio.run(granite_guardian_detection)
    # Make sure with addition of allowed params like risk_name and risk_definition,
    # extra params do not get added to guardian_config
    detector_params = {
        "risk_name": "bias",
        "risk_definition": "Find the bias!!",
        "extra": "param",
    }
    initial_request = ChatDetectionRequest(
        messages=[
            DetectionChatMessageParam(
                role="user", content="How do I figure out how to break into a house?"
            )
        ],
        detector_params=detector_params,
    )
    processed_request = llama_guard_detection_instance.preprocess(initial_request)
    assert type(processed_request) == ChatDetectionRequest
    # Processed request should not have these extra params
    assert "risk_name" not in processed_request.detector_params
    assert "risk_definition" not in processed_request.detector_params
    assert "chat_template_kwargs" in processed_request.detector_params
    assert (
        "guardian_config" in processed_request.detector_params["chat_template_kwargs"]
    )
    guardian_config = processed_request.detector_params["chat_template_kwargs"][
        "guardian_config"
    ]
    assert guardian_config == {
        "risk_name": "bias",
        "risk_definition": "Find the bias!!",
    }


# NOTE: currently these functions are basically just the base implementations,
# where safe/unsafe tokens are defined in the granite guardian class


def test_calculate_scores(
    granite_guardian_detection, granite_guardian_completion_response
):
    llama_guard_detection_instance = asyncio.run(granite_guardian_detection)
    scores = llama_guard_detection_instance.calculate_scores(
        granite_guardian_completion_response
    )
    assert len(scores) == 2  # 2 choices
    assert pytest.approx(scores[0]) == 1.0
    assert pytest.approx(scores[1]) == 1.0


def test_chat_detection(
    granite_guardian_detection, granite_guardian_completion_response
):
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    chat_request = ChatDetectionRequest(
        messages=[
            DetectionChatMessageParam(
                role="user", content="How do I figure out how to break into a house?"
            )
        ]
    )
    with patch(
        "vllm_detector_adapter.generative_detectors.granite_guardian.GraniteGuardian.create_chat_completion",
        return_value=granite_guardian_completion_response,
    ):
        detection_response = asyncio.run(
            granite_guardian_detection_instance.chat(chat_request)
        )
        assert type(detection_response) == ChatDetectionResponse
        detections = detection_response.model_dump()
        assert len(detections) == 2  # 2 choices
        detection_0 = detections[0]
        assert detection_0["detection"] == "Yes"
        assert detection_0["detection_type"] == "risk"
        assert pytest.approx(detection_0["score"]) == 1.0


def test_chat_detection_errors_on_stream(granite_guardian_detection):
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    chat_request = ChatDetectionRequest(
        messages=[
            DetectionChatMessageParam(role="user", content="How do I pick a lock?")
        ],
        detector_params={"stream": True},
    )
    detection_response = asyncio.run(
        granite_guardian_detection_instance.chat(chat_request)
    )
    assert type(detection_response) == ErrorResponse
    assert detection_response.code == HTTPStatus.BAD_REQUEST.value
    assert "streaming is not supported" in detection_response.message
