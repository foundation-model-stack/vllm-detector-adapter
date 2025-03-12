# Standard
from dataclasses import dataclass
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
    UsageInfo,
)
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
import jinja2
import pytest
import pytest_asyncio

# Local
from vllm_detector_adapter.generative_detectors.base import ChatCompletionDetectionBase
from vllm_detector_adapter.protocol import (
    ContentsDetectionRequest,
    ContentsDetectionResponse,
)

MODEL_NAME = "openai-community/gpt2"
CHAT_TEMPLATE = "Dummy chat template for testing {}"
BASE_MODEL_PATHS = [BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME)]


@dataclass
class MockTokenizer:
    type: Optional[str] = None
    chat_template: str = CHAT_TEMPLATE


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

    async def get_tokenizer(self):
        return MockTokenizer()


async def _async_serving_detection_completion_init():
    """Initialize a chat completion base with string templates"""
    engine = MockEngine()
    engine.errored = False
    model_config = await engine.get_model_config()
    models = OpenAIServingModels(
        engine_client=engine,
        model_config=model_config,
        base_model_paths=BASE_MODEL_PATHS,
    )

    detection_completion = ChatCompletionDetectionBase(
        task_template="hello {{user_text}}",
        output_template="bye {{text}}",
        engine_client=engine,
        model_config=model_config,
        models=models,
        response_role="assistant",
        chat_template=CHAT_TEMPLATE,
        chat_template_content_format="auto",
        request_logger=None,
    )
    setattr(detection_completion, "RISK_BANK_VAR_NAME", "DUMMY_RISK_BANK")
    return detection_completion


@pytest_asyncio.fixture
async def detection_base():
    return _async_serving_detection_completion_init()


@pytest.fixture(scope="module")
def completion_response():
    log_probs_content_no = ChatCompletionLogProbsContent(
        token="no",
        logprob=-0.0013,
        # 5 logprobs requested for scoring, skipping bytes for conciseness
        top_logprobs=[
            ChatCompletionLogProb(token="no", logprob=-0.053),
            ChatCompletionLogProb(token="0", logprob=-6.61),
            ChatCompletionLogProb(token="1", logprob=-16.90),
            ChatCompletionLogProb(token="2", logprob=-17.39),
            ChatCompletionLogProb(token="3", logprob=-17.61),
        ],
    )
    log_probs_content_yes = ChatCompletionLogProbsContent(
        token="yes",
        logprob=-0.0013,
        # 5 logprobs requested for scoring, skipping bytes for conciseness
        top_logprobs=[
            ChatCompletionLogProb(token="yes", logprob=-0.0013),
            ChatCompletionLogProb(token="0", logprob=-6.61),
            ChatCompletionLogProb(token="1", logprob=-16.90),
            ChatCompletionLogProb(token="2", logprob=-17.39),
            ChatCompletionLogProb(token="3", logprob=-17.61),
        ],
    )
    choice_0 = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(
            role="assistant",
            content="no",
        ),
        logprobs=ChatCompletionLogProbs(content=[log_probs_content_no]),
    )
    choice_1 = ChatCompletionResponseChoice(
        index=1,
        message=ChatMessage(
            role="assistant",
            content="yes",
        ),
        logprobs=ChatCompletionLogProbs(content=[log_probs_content_yes]),
    )
    yield ChatCompletionResponse(
        model=MODEL_NAME,
        choices=[choice_0, choice_1],
        usage=UsageInfo(prompt_tokens=200, total_tokens=206, completion_tokens=6),
    )


### Tests #####################################################################


def test_async_serving_detection_completion_init(detection_base):
    detection_completion = asyncio.run(detection_base)
    assert detection_completion.chat_template == CHAT_TEMPLATE

    # tests load_template
    task_template = detection_completion.task_template
    assert type(task_template) == jinja2.environment.Template
    assert task_template.render(({"user_text": "moose"})) == "hello moose"

    output_template = detection_completion.output_template
    assert type(output_template) == jinja2.environment.Template
    assert output_template.render(({"text": "moose"})) == "bye moose"


def test_content_analysis_success(detection_base, completion_response):
    base_instance = asyncio.run(detection_base)

    content_request = ContentsDetectionRequest(
        contents=["Where do I find geese?", "You could go to Canada"]
    )

    scores = [0.9, 0.1]
    response = (completion_response, scores, "risk")
    with patch(
        "vllm_detector_adapter.generative_detectors.base.ChatCompletionDetectionBase.process_chat_completion_with_scores",
        return_value=response,
    ):
        result = asyncio.run(base_instance.content_analysis(content_request))
        assert isinstance(result, ContentsDetectionResponse)
        detections = result.model_dump()
        assert len(detections) == 2
        # For first content
        assert detections[0][0]["detection"] == "no"
        assert detections[0][0]["score"] == 0.9
        assert detections[0][0]["start"] == 0
        assert detections[0][0]["end"] == len(content_request.contents[0])
        assert detections[0][0]["metadata"] == {}

        # 2nd choice as 2nd label
        assert detections[0][1]["detection"] == "yes"
        assert detections[0][1]["score"] == 0.1
        assert detections[0][1]["start"] == 0
        assert detections[0][1]["end"] == len(content_request.contents[0])
        assert detections[0][1]["metadata"] == {}
        # For 2nd content, we are only testing 1st detection for simplicity
        # Note: detection is same, because of how mock is working.
        assert detections[1][0]["detection"] == "no"
        assert detections[1][0]["score"] == 0.9
        assert detections[1][0]["start"] == 0
        assert detections[1][0]["end"] == len(content_request.contents[1])
        assert detections[1][0]["metadata"] == {}
