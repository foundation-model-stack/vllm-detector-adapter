# Standard
from dataclasses import dataclass
from http import HTTPStatus
from typing import Optional
from unittest.mock import patch
import asyncio
import json

# Third Party
from jinja2.exceptions import TemplateError, UndefinedError
from vllm.config import MultiModalConfig
from vllm.entrypoints.openai.protocol import (
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    ErrorResponse,
    UsageInfo,
)
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
import pytest
import pytest_asyncio

# Local
from vllm_detector_adapter.generative_detectors.granite_guardian import GraniteGuardian
from vllm_detector_adapter.protocol import (
    ChatDetectionRequest,
    ContentsDetectionRequest,
    ContextAnalysisRequest,
    DetectionChatMessageParam,
    DetectionResponse,
    GenerationDetectionRequest,
    Tool,
    ToolCall,
    ToolCallFunctionObject,
    ToolFunctionObject,
)
from vllm_detector_adapter.utils import DetectorType

MODEL_NAME = "ibm-granite/granite-guardian"  # Example granite-guardian model
CHAT_TEMPLATE = '{%- set risk_bank = ({"social_bias": { "user": "User text 1", "assistant": "Assistant text 1"},"jailbreak": { "user": "User text 2", "assistant": "Assistant text 2"},"profanity": {  "user": "User text 3",  "assistant": "Assistant text 3"}}) %}'
BASE_MODEL_PATHS = [BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME)]

CONTENT = "Where do I find geese?"
CONTEXT_DOC = "Geese can be found in lakes, ponds, and rivers"

# Tool and tool call functions
TOOL_FUNCTION = ToolFunctionObject(
    description="Fetches a list of comments",
    name="comment_list",
    parameters={"foo": "bar", "sun": "moon"},
)
TOOL_CALL_FUNCTION = ToolCallFunctionObject(
    name="comment_list",
    arguments='{"awname_id":456789123,"count":15,"goose":"moose"}',
)
TOOL = Tool(type="function", function=TOOL_FUNCTION)
TOOL_CALL = ToolCall(id="tool_call", type="function", function=TOOL_CALL_FUNCTION)
USER_CONTENT_TOOLS = "Fetch the first 15 comments for the video with ID 456789123"


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


async def _granite_guardian_init():
    """Initialize a granite guardian"""
    engine = MockEngine()
    engine.errored = False
    model_config = await engine.get_model_config()
    models = OpenAIServingModels(
        engine_client=engine,
        model_config=model_config,
        base_model_paths=BASE_MODEL_PATHS,
    )

    granite_guardian = GraniteGuardian(
        task_template=None,
        output_template=None,
        engine_client=engine,
        model_config=model_config,
        models=models,
        response_role="assistant",
        chat_template=CHAT_TEMPLATE,
        chat_template_content_format="auto",
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


# Initialized per function since response could be updated
@pytest.fixture(scope="function")
def granite_guardian_completion_response_3_2():
    log_probs_content_no = ChatCompletionLogProbsContent(
        token="No",
        logprob=-2.8,
        # 5 logprobs requested for scoring, skipping bytes for conciseness
        top_logprobs=[
            ChatCompletionLogProb(token="No", logprob=-2.8),
            ChatCompletionLogProb(token="Yes", logprob=-0.09),
            ChatCompletionLogProb(token="assistant", logprob=-5.00),
            ChatCompletionLogProb(token="You", logprob=-5.88),
            ChatCompletionLogProb(token=" Yes", logprob=-6.05),
        ],
    )
    log_probs_content_random = ChatCompletionLogProbsContent(
        token="<",
        logprob=0.00,
        # 5 logprobs requested for scoring, skipping bytes for conciseness
        top_logprobs=[
            ChatCompletionLogProb(token="<", logprob=0.00),
            ChatCompletionLogProb(token=" <", logprob=-15.52),
            ChatCompletionLogProb(token="</", logprob=-15.64),
            ChatCompletionLogProb(token="Assistant", logprob=-15.98),
            ChatCompletionLogProb(token="<<", logprob=-16.73),
        ],
    )
    choice_0 = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(
            role="assistant",
            content="No\n<confidence> High </confidence>",
        ),
        logprobs=ChatCompletionLogProbs(
            content=[log_probs_content_no, log_probs_content_random]
        ),
    )
    choice_1 = ChatCompletionResponseChoice(
        index=1,
        message=ChatMessage(
            role="assistant",
            content="Yes\n<confidence> Low </confidence>",
        ),
        logprobs=ChatCompletionLogProbs(
            content=[log_probs_content_random, log_probs_content_no]
        ),
    )
    yield ChatCompletionResponse(
        model=MODEL_NAME,
        choices=[choice_0, choice_1],
        usage=UsageInfo(prompt_tokens=142, total_tokens=162, completion_tokens=20),
    )


@pytest.fixture(scope="function")
def granite_guardian_completion_response_3_3_plus_no_think():
    """No think/trace output in a Granite Guardian 3.3+ response"""
    log_probs_content_yes = ChatCompletionLogProbsContent(
        token=" yes",
        logprob=0.00,
        # 5 logprobs requested for scoring, skipping bytes for conciseness
        top_logprobs=[
            ChatCompletionLogProb(token=" yes", logprob=0.00),
            ChatCompletionLogProb(token=" no", logprob=-13.25),
            ChatCompletionLogProb(token="yes", logprob=-14.63),
            ChatCompletionLogProb(token=" Yes", logprob=-16.37),
            ChatCompletionLogProb(token=" yeah", logprob=-17.06),
        ],
    )
    log_probs_content_random = ChatCompletionLogProbsContent(
        token="<",
        logprob=0.00,
        # 5 logprobs requested for scoring, skipping bytes for conciseness
        top_logprobs=[
            ChatCompletionLogProb(token="<", logprob=0.00),
            ChatCompletionLogProb(token="</", logprob=-17.88),
            ChatCompletionLogProb(token="<!--", logprob=-18.25),
            ChatCompletionLogProb(token="<!", logprob=-18.5),
            ChatCompletionLogProb(token=" <", logprob=-18.5),
        ],
    )
    choice_0 = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(
            role="assistant",
            content="<think>\n</think>\n<score> yes </score>",
        ),
        logprobs=ChatCompletionLogProbs(
            content=[log_probs_content_yes, log_probs_content_random]
        ),
    )
    choice_1 = ChatCompletionResponseChoice(
        index=1,
        message=ChatMessage(
            role="assistant",
            content="<think>\n</think>\n<score> yes </score>",
        ),
        logprobs=ChatCompletionLogProbs(
            content=[log_probs_content_random, log_probs_content_yes]
        ),
    )
    yield ChatCompletionResponse(
        model=MODEL_NAME,
        choices=[choice_0, choice_1],
        usage=UsageInfo(prompt_tokens=124, total_tokens=172, completion_tokens=48),
    )


@pytest.fixture(scope="function")
def granite_guardian_completion_response_3_3_plus_think():
    """Granite Guardian 3.3+ response with think/trace output"""
    log_probs_content_yes = ChatCompletionLogProbsContent(
        token=" yes",
        logprob=0.00,
        # 5 logprobs requested for scoring, skipping bytes for conciseness
        top_logprobs=[
            ChatCompletionLogProb(token=" yes", logprob=0.00),
            ChatCompletionLogProb(token="yes", logprob=-14.00),
            ChatCompletionLogProb(token=" '", logprob=-14.43),
            ChatCompletionLogProb(token=" indeed", logprob=-14.93),
            ChatCompletionLogProb(token=" Yes", logprob=-15.56),
        ],
    )
    log_probs_content_random = ChatCompletionLogProbsContent(
        token="<",
        logprob=0.00,
        # 5 logprobs requested for scoring, skipping bytes for conciseness
        top_logprobs=[
            ChatCompletionLogProb(token="<", logprob=0.00),
            ChatCompletionLogProb(token="First", logprob=-12.12),
            ChatCompletionLogProb(token="The", logprob=-13.63),
            ChatCompletionLogProb(token=" <", logprob=-13.87),
            ChatCompletionLogProb(token=" First", logprob=-16.43),
        ],
    )
    choice_0 = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(
            role="assistant",
            content="<think> First analyzing the user request...there is a risk associated, so the score is yes.\n</think>\n<score> yes </score>",
        ),
        logprobs=ChatCompletionLogProbs(
            content=[log_probs_content_yes, log_probs_content_random]
        ),
    )
    choice_1 = ChatCompletionResponseChoice(
        index=1,
        message=ChatMessage(
            role="assistant",
            content="<think> Since there is no risk associated, the score is no.\n</think>\n<score> no </score>",
        ),
        logprobs=ChatCompletionLogProbs(
            content=[log_probs_content_random, log_probs_content_yes]
        ),
    )
    yield ChatCompletionResponse(
        model=MODEL_NAME,
        choices=[choice_0, choice_1],
        usage=UsageInfo(prompt_tokens=122, total_tokens=910, completion_tokens=788),
    )


### Tests #####################################################################

#### Private tools request tests


def test__make_tools_request(granite_guardian_detection):
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    detector_params = {"risk_name": "function_call", "n": 3}
    tool_2 = Tool(type="function", function=TOOL_FUNCTION)
    request = ChatDetectionRequest(
        messages=[
            DetectionChatMessageParam(
                role="user",
                content=USER_CONTENT_TOOLS,
            ),
            DetectionChatMessageParam(role="assistant", tool_calls=[TOOL_CALL]),
        ],
        tools=[TOOL, tool_2],
        detector_params=detector_params,
    )
    processed_request = granite_guardian_detection_instance._make_tools_request(request)
    assert type(processed_request) == ChatDetectionRequest
    assert processed_request.tools == []  # Tools were removed
    assert (
        processed_request.detector_params == detector_params
    )  # detector_params untouched

    assert len(processed_request.messages) == 3
    first_message = processed_request.messages[0]
    # Make sure first message is tools
    assert first_message["role"] == "tools"
    tools_content = json.loads(first_message["content"])
    assert len(tools_content) == 2  # 2 tool functions since 2 tools
    assert tools_content[0]["description"] == TOOL_FUNCTION["description"]
    assert tools_content[0]["name"] == TOOL_FUNCTION["name"]
    assert tools_content[0]["parameters"] == TOOL_FUNCTION["parameters"]
    # Second message - user
    assert processed_request.messages[1]["role"] == "user"
    assert (
        processed_request.messages[1]["content"]
        == "Fetch the first 15 comments for the video with ID 456789123"
    )
    # Last message - assistant
    last_message = processed_request.messages[2]
    assert last_message["role"] == "assistant"
    assistant_content = json.loads(last_message["content"])
    assert len(assistant_content) == 1  # 1 tool_call function
    assert assistant_content[0]["name"] == TOOL_CALL_FUNCTION["name"]
    assert assistant_content[0]["arguments"] == json.loads(
        TOOL_CALL_FUNCTION["arguments"]
    )  # Note: tool_calls in the OpenAI API have str arguments


def test__make_tools_request_no_tool_calls(granite_guardian_detection):
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    request = ChatDetectionRequest(
        messages=[
            DetectionChatMessageParam(
                role="user",
                content=USER_CONTENT_TOOLS,
            ),
            DetectionChatMessageParam(role="assistant", content="Random content!"),
        ],
        tools=[TOOL],
        detector_params={"risk_name": "function_call", "n": 2},
    )
    processed_request = granite_guardian_detection_instance._make_tools_request(request)
    assert type(processed_request) == ErrorResponse
    assert processed_request.error.code == HTTPStatus.BAD_REQUEST
    assert (
        "no assistant message was provided with tool_calls for analysis"
        in processed_request.error.message
    )


def test__make_tools_request_random_risk(granite_guardian_detection):
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    detector_params = {"risk_name": "social_bias", "n": 2}
    request = ChatDetectionRequest(
        messages=[
            DetectionChatMessageParam(
                role="user",
                content=USER_CONTENT_TOOLS,
            ),
            DetectionChatMessageParam(role="assistant", tool_calls=[TOOL_CALL]),
        ],
        tools=[TOOL],
        detector_params=detector_params,
    )
    processed_request = granite_guardian_detection_instance._make_tools_request(request)
    assert type(processed_request) == ErrorResponse
    assert processed_request.error.code == HTTPStatus.BAD_REQUEST
    assert (
        "tools analysis is not supported with given risk"
        in processed_request.error.message
    )


#### Private metadata extraction tests


def test__extract_no_metadata(
    granite_guardian_detection, granite_guardian_completion_response
):
    # Older Granite Guardian versions do not provide info like confidence
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    choice_index = 0
    content = granite_guardian_completion_response.choices[choice_index].message.content
    # NOTE: private function tested here
    metadata = granite_guardian_detection_instance._extract_tag_info(
        granite_guardian_completion_response, choice_index, content
    )
    assert metadata == {}
    # Content unchanged
    assert (
        granite_guardian_completion_response.choices[choice_index].message.content
        == content
    )


def test__extract_tag_info_with_confidence(
    granite_guardian_detection, granite_guardian_completion_response_3_2
):
    # In Granite Guardian 3.2, confidence info is provided
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    choice_index = 0
    content = granite_guardian_completion_response_3_2.choices[
        choice_index
    ].message.content
    # NOTE: private function tested here
    metadata = granite_guardian_detection_instance._extract_tag_info(
        granite_guardian_completion_response_3_2, choice_index, content
    )
    assert metadata == {"confidence": "High"}
    # Content should be updated
    assert (
        granite_guardian_completion_response_3_2.choices[choice_index].message.content
        == "No"
    )


def test__extract_tag_info_no_think_result_and_score(
    granite_guardian_detection, granite_guardian_completion_response_3_3_plus_no_think
):
    # In Granite Guardian 3.3+, think and score tags are provided
    # This response has the think tags but no think results
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    choice_index = 0
    content = granite_guardian_completion_response_3_3_plus_no_think.choices[
        choice_index
    ].message.content
    # NOTE: private function tested here
    metadata = granite_guardian_detection_instance._extract_tag_info(
        granite_guardian_completion_response_3_3_plus_no_think, choice_index, content
    )
    assert metadata == {}  # think is basically empty with the newline
    # Content should be updated without the tags
    assert (
        granite_guardian_completion_response_3_3_plus_no_think.choices[
            choice_index
        ].message.content
        == "yes"
    )


def test__extract_tag_info_think_result_and_score(
    granite_guardian_detection, granite_guardian_completion_response_3_3_plus_think
):
    # In Granite Guardian 3.3+, think and score tags are provided
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    choice_index = 0
    content = granite_guardian_completion_response_3_3_plus_think.choices[
        choice_index
    ].message.content
    # NOTE: private function tested here
    metadata = granite_guardian_detection_instance._extract_tag_info(
        granite_guardian_completion_response_3_3_plus_think, choice_index, content
    )
    assert metadata == {
        "think": "First analyzing the user request...there is a risk associated, so the score is yes."
    }
    # Content should be updated without the tags
    assert (
        granite_guardian_completion_response_3_3_plus_think.choices[
            choice_index
        ].message.content
        == "yes"
    )


#### Helper function tests


def test_preprocess_chat_request_with_detector_params(granite_guardian_detection):
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
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
    processed_request = granite_guardian_detection_instance.preprocess_request(
        initial_request, fn_type=DetectorType.TEXT_CHAT
    )
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


def test_preprocess_chat_request_with_custom_criteria_detector_params(
    granite_guardian_detection,
):
    # Guardian 3.3+ parameters
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    detector_params = {
        "custom_criteria": "Here is some custom criteria",
        "custom_scoring_schema": "If text meets criteria say yes",
        "foo": "bar",
    }
    initial_request = ChatDetectionRequest(
        messages=[
            DetectionChatMessageParam(
                role="user", content="How do I figure out how to break into a house?"
            )
        ],
        detector_params=detector_params,
    )
    processed_request = granite_guardian_detection_instance.preprocess_request(
        initial_request, fn_type=DetectorType.TEXT_CHAT
    )
    assert type(processed_request) == ChatDetectionRequest
    # Processed request should not have these extra params
    assert "custom_criteria" not in processed_request.detector_params
    assert "custom_scoring_schema" not in processed_request.detector_params
    assert "chat_template_kwargs" in processed_request.detector_params
    assert (
        "guardian_config" in processed_request.detector_params["chat_template_kwargs"]
    )
    guardian_config = processed_request.detector_params["chat_template_kwargs"][
        "guardian_config"
    ]
    assert guardian_config == {
        "custom_criteria": "Here is some custom criteria",
        "custom_scoring_schema": "If text meets criteria say yes",
    }


def test_preprocess_chat_request_with_extra_chat_template_kwargs(
    granite_guardian_detection,
):
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    # Make sure other chat_template_kwargs do not get overwritten
    detector_params = {
        "risk_name": "bias",
        "risk_definition": "Find the bias!!",
        "chat_template_kwargs": {"foo": "bar"},
    }
    initial_request = ChatDetectionRequest(
        messages=[
            DetectionChatMessageParam(
                role="user", content="How do I figure out how to break into a house?"
            )
        ],
        detector_params=detector_params,
    )
    processed_request = granite_guardian_detection_instance.preprocess_request(
        initial_request, fn_type=DetectorType.TEXT_CHAT
    )
    assert type(processed_request) == ChatDetectionRequest
    # Processed request should not have these extra params
    assert "risk_name" not in processed_request.detector_params
    assert "risk_definition" not in processed_request.detector_params
    assert "chat_template_kwargs" in processed_request.detector_params
    assert "foo" in processed_request.detector_params["chat_template_kwargs"]
    assert processed_request.detector_params["chat_template_kwargs"]["foo"] == "bar"
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


#### Helper function request->chat_completion_request tests


def test_request_to_chat_completion_request_prompt_analysis(granite_guardian_detection):
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    context_request = ContextAnalysisRequest(
        content=CONTENT,
        context_type="docs",
        context=[
            "extra!",
            CONTEXT_DOC,
        ],  # additionally test that multiple contexts are concatenated
        detector_params={
            "n": 2,
            "chat_template_kwargs": {
                "guardian_config": {"risk_name": "context_relevance"}
            },
        },
    )
    chat_request = (
        granite_guardian_detection_instance._request_to_chat_completion_request(
            context_request, MODEL_NAME, fn_type=DetectorType.TEXT_CONTEXT_DOC
        )
    )
    assert type(chat_request) == ChatCompletionRequest
    assert len(chat_request.messages) == 2
    assert chat_request.messages[0]["role"] == "context"
    assert chat_request.messages[0]["content"] == "extra! " + CONTEXT_DOC
    assert chat_request.messages[1]["role"] == "user"
    assert chat_request.messages[1]["content"] == CONTENT
    assert chat_request.model == MODEL_NAME
    # detector_paramas
    assert chat_request.n == 2
    assert (
        chat_request.chat_template_kwargs["guardian_config"]["risk_name"]
        == "context_relevance"
    )


def test_request_to_chat_completion_request_response_analysis(
    granite_guardian_detection,
):
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    context_request = ContextAnalysisRequest(
        content=CONTENT,
        context_type="docs",
        context=[CONTEXT_DOC],
        detector_params={
            "n": 3,
            "chat_template_kwargs": {"guardian_config": {"risk_name": "groundedness"}},
        },
    )
    chat_request = (
        granite_guardian_detection_instance._request_to_chat_completion_request(
            context_request, MODEL_NAME, fn_type=DetectorType.TEXT_CONTEXT_DOC
        )
    )
    assert type(chat_request) == ChatCompletionRequest
    assert chat_request.messages[0]["role"] == "context"
    assert chat_request.messages[0]["content"] == CONTEXT_DOC
    assert chat_request.messages[1]["role"] == "assistant"
    assert chat_request.messages[1]["content"] == CONTENT
    assert chat_request.model == MODEL_NAME
    # detector_paramas
    assert chat_request.n == 3
    assert (
        chat_request.chat_template_kwargs["guardian_config"]["risk_name"]
        == "groundedness"
    )


def test_request_to_chat_completion_request_response_analysis_criteria_id(
    granite_guardian_detection,
):
    # Guardian 3.3 parameters
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    context_request = ContextAnalysisRequest(
        content=CONTENT,
        context_type="docs",
        context=[CONTEXT_DOC],
        detector_params={
            "n": 3,
            "chat_template_kwargs": {
                "guardian_config": {"criteria_id": "groundedness"}
            },
        },
    )
    chat_request = (
        granite_guardian_detection_instance._request_to_chat_completion_request(
            context_request, MODEL_NAME, fn_type=DetectorType.TEXT_CONTEXT_DOC
        )
    )
    assert type(chat_request) == ChatCompletionRequest
    assert chat_request.messages[0]["role"] == "context"
    assert chat_request.messages[0]["content"] == CONTEXT_DOC
    assert chat_request.messages[1]["role"] == "assistant"
    assert chat_request.messages[1]["content"] == CONTENT
    assert chat_request.model == MODEL_NAME
    # detector_paramas
    assert chat_request.n == 3
    assert (
        chat_request.chat_template_kwargs["guardian_config"]["criteria_id"]
        == "groundedness"
    )


def test_request_to_chat_completion_request_empty_kwargs(granite_guardian_detection):
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    context_request = ContextAnalysisRequest(
        content=CONTENT,
        context_type="docs",
        context=[CONTEXT_DOC],
        detector_params={"n": 2, "chat_template_kwargs": {}},  # no guardian config
    )
    chat_request = (
        granite_guardian_detection_instance._request_to_chat_completion_request(
            context_request, MODEL_NAME, fn_type=DetectorType.TEXT_CONTEXT_DOC
        )
    )
    assert type(chat_request) == ErrorResponse
    assert chat_request.error.code == HTTPStatus.BAD_REQUEST
    assert (
        "No risk_name or criteria_id for context analysis" in chat_request.error.message
    )


def test_request_to_chat_completion_request_empty_guardian_config(
    granite_guardian_detection,
):
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    context_request = ContextAnalysisRequest(
        content=CONTENT,
        context_type="docs",
        context=[CONTEXT_DOC],
        detector_params={"n": 2, "chat_template_kwargs": {"guardian_config": {}}},
    )
    chat_request = (
        granite_guardian_detection_instance._request_to_chat_completion_request(
            context_request, MODEL_NAME, fn_type=DetectorType.TEXT_CONTEXT_DOC
        )
    )
    assert type(chat_request) == ErrorResponse
    assert chat_request.error.code == HTTPStatus.BAD_REQUEST
    assert (
        "No risk_name or criteria_id for context analysis" in chat_request.error.message
    )


def test_request_to_chat_completion_request_missing_risk_name_and_criteria_id(
    granite_guardian_detection,
):
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    context_request = ContextAnalysisRequest(
        content=CONTENT,
        context_type="docs",
        context=[CONTEXT_DOC],
        detector_params={
            "n": 3,
            "chat_template_kwargs": {"guardian_config": {"risk_definition": "hi"}},
        },
    )
    chat_request = (
        granite_guardian_detection_instance._request_to_chat_completion_request(
            context_request, MODEL_NAME, fn_type=DetectorType.TEXT_CONTEXT_DOC
        )
    )
    assert type(chat_request) == ErrorResponse
    assert chat_request.error.code == HTTPStatus.BAD_REQUEST
    assert (
        "No risk_name or criteria_id for context analysis" in chat_request.error.message
    )


def test_request_to_chat_completion_request_unsupported_risk_name(
    granite_guardian_detection,
):
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    context_request = ContextAnalysisRequest(
        content=CONTENT,
        context_type="docs",
        context=[CONTEXT_DOC],
        detector_params={
            "n": 2,
            "chat_template_kwargs": {"guardian_config": {"risk_name": "foo"}},
        },
    )
    chat_request = (
        granite_guardian_detection_instance._request_to_chat_completion_request(
            context_request, MODEL_NAME, fn_type=DetectorType.TEXT_CONTEXT_DOC
        )
    )
    assert type(chat_request) == ErrorResponse
    assert chat_request.error.code == HTTPStatus.BAD_REQUEST
    assert (
        "risk_name or criteria_id foo is not compatible with context analysis"
        in chat_request.error.message
    )


#### Helper function preprocess content tests


def test_preprocess_content_request_with_detector_params(granite_guardian_detection):
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    # Make sure with addition of allowed params like risk_name and risk_definition,
    # extra params do not get added to guardian_config
    detector_params = {
        "risk_name": "bias",
        "risk_definition": "Find the bias!!",
        "extra": "param",
    }
    initial_request = ContentsDetectionRequest(
        contents=["Where do I find geese?", "You could go to Canada"],
        detector_params=detector_params,
    )
    processed_requests = granite_guardian_detection_instance.preprocess_request(
        initial_request, fn_type=DetectorType.TEXT_CONTENT
    )
    # Contents detector generates several requests per content
    assert type(processed_requests) == list
    assert len(processed_requests) == 2
    processed_request = processed_requests[0]
    assert type(processed_request) == ChatCompletionRequest
    # Processed request should not have these extra params
    assert not hasattr(processed_request, "risk_name")
    assert not hasattr(processed_request, "risk_definition")
    assert hasattr(processed_request, "chat_template_kwargs")
    assert "guardian_config" in processed_request.chat_template_kwargs
    guardian_config = processed_request.chat_template_kwargs["guardian_config"]
    assert guardian_config == {
        "risk_name": "bias",
        "risk_definition": "Find the bias!!",
    }
    # other extra params should get added
    assert hasattr(processed_request, "extra")
    assert processed_request.extra == "param"


#### Helper function post process tests


def test_post_process_completion_no_metadata(
    granite_guardian_detection, granite_guardian_completion_response
):
    # Older Granite Guardian versions do not provide info like confidence
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    dummy_scores = [0.2, 0.2]
    (chat_completion_response, _, _, metadata_list) = asyncio.run(
        granite_guardian_detection_instance.post_process_completion_results(
            granite_guardian_completion_response, dummy_scores, "risk"
        )
    )
    assert len(metadata_list) == 2  # 2 choices
    # Both empty dicts since there was no extra response info
    assert metadata_list[0] == {}
    assert metadata_list[1] == {}
    # Chat completion response should be unchanged
    assert chat_completion_response.choices[0].message.content == "Yes"
    assert chat_completion_response.choices[1].message.content == "Yes"


def test_post_process_completion_with_confidence(
    granite_guardian_detection, granite_guardian_completion_response_3_2
):
    # In Granite Guardian 3.2, confidence info is provided
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    dummy_scores = [0.2, 0.2]
    (chat_completion_response, _, _, metadata_list) = asyncio.run(
        granite_guardian_detection_instance.post_process_completion_results(
            granite_guardian_completion_response_3_2, dummy_scores, "risk"
        )
    )
    assert len(metadata_list) == 2  # 2 choices
    assert metadata_list[0] == {"confidence": "High"}
    assert metadata_list[1] == {"confidence": "Low"}
    # Chat completion response should be updated
    assert chat_completion_response.choices[0].message.content == "No"
    assert chat_completion_response.choices[1].message.content == "Yes"


def test_post_process_completion_with_no_think_result_and_score(
    granite_guardian_detection, granite_guardian_completion_response_3_3_plus_no_think
):
    # In Granite Guardian 3.3+, think and score tags are provided
    # This response has the think tags but no think results
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    dummy_scores = [0.2, 0.2]
    (chat_completion_response, _, _, metadata_list) = asyncio.run(
        granite_guardian_detection_instance.post_process_completion_results(
            granite_guardian_completion_response_3_3_plus_no_think, dummy_scores, "risk"
        )
    )
    assert len(metadata_list) == 2  # 2 choices
    # Empty think content
    assert metadata_list[0] == {}
    assert metadata_list[1] == {}
    # Chat completion response should be updated
    assert chat_completion_response.choices[0].message.content == "yes"
    assert chat_completion_response.choices[1].message.content == "yes"


def test_post_process_completion_with_think_result_and_score(
    granite_guardian_detection, granite_guardian_completion_response_3_3_plus_think
):
    # In Granite Guardian 3.3+, think and score tags are provided
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    dummy_scores = [0.2, 0.2]
    (chat_completion_response, _, _, metadata_list) = asyncio.run(
        granite_guardian_detection_instance.post_process_completion_results(
            granite_guardian_completion_response_3_3_plus_think, dummy_scores, "risk"
        )
    )
    assert len(metadata_list) == 2  # 2 choices
    # Think content
    assert metadata_list[0] == {
        "think": "First analyzing the user request...there is a risk associated, so the score is yes."
    }
    assert metadata_list[1] == {
        "think": "Since there is no risk associated, the score is no."
    }
    # Chat completion response should be updated
    assert chat_completion_response.choices[0].message.content == "yes"
    assert chat_completion_response.choices[1].message.content == "no"


#### Context analysis tests


def test_context_analyze(
    granite_guardian_detection, granite_guardian_completion_response
):
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    context_request = ContextAnalysisRequest(
        content=CONTENT,
        context_type="docs",
        context=[CONTEXT_DOC],
        detector_params={
            "n": 2,
            "risk_name": "groundedness",
        },
    )
    with patch(
        "vllm_detector_adapter.generative_detectors.granite_guardian.GraniteGuardian.create_chat_completion",
        return_value=granite_guardian_completion_response,
    ):
        detection_response = asyncio.run(
            granite_guardian_detection_instance.context_analyze(context_request)
        )
        assert type(detection_response) == DetectionResponse
        detections = detection_response.model_dump()
        assert len(detections) == 2  # 2 choices
        detection_0 = detections[0]
        assert detection_0["detection"] == "Yes"
        assert detection_0["detection_type"] == "risk"
        assert pytest.approx(detection_0["score"]) == 1.0
        assert detection_0["metadata"] == {}


def test_context_analyze_with_confidence(
    granite_guardian_detection, granite_guardian_completion_response_3_2
):
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    context_request = ContextAnalysisRequest(
        content=CONTENT,
        context_type="docs",
        context=[CONTEXT_DOC],
        detector_params={
            "n": 2,
            "risk_name": "groundedness",
        },
    )
    with patch(
        "vllm_detector_adapter.generative_detectors.granite_guardian.GraniteGuardian.create_chat_completion",
        return_value=granite_guardian_completion_response_3_2,
    ):
        detection_response = asyncio.run(
            granite_guardian_detection_instance.context_analyze(context_request)
        )
        assert type(detection_response) == DetectionResponse
        detections = detection_response.model_dump()
        assert len(detections) == 2  # 2 choices
        detection_0 = detections[0]
        assert detection_0["detection"] == "No"
        assert detection_0["detection_type"] == "risk"
        assert pytest.approx(detection_0["score"]) == 0.9377647
        assert detection_0["metadata"] == {"confidence": "High"}


def test_context_analyze_template_kwargs(
    granite_guardian_detection, granite_guardian_completion_response
):
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    context_request = ContextAnalysisRequest(
        content=CONTENT,
        context_type="docs",
        context=[CONTEXT_DOC],
        detector_params={
            "n": 2,
            "chat_template_kwargs": {"guardian_config": {"risk_name": "groundedness"}},
        },
    )
    with patch(
        "vllm_detector_adapter.generative_detectors.granite_guardian.GraniteGuardian.create_chat_completion",
        return_value=granite_guardian_completion_response,
    ):
        detection_response = asyncio.run(
            granite_guardian_detection_instance.context_analyze(context_request)
        )
        assert type(detection_response) == DetectionResponse
        detections = detection_response.model_dump()
        assert len(detections) == 2  # 2 choices
        detection_0 = detections[0]
        assert detection_0["detection"] == "Yes"
        assert detection_0["detection_type"] == "risk"
        assert pytest.approx(detection_0["score"]) == 1.0


def test_context_analyze_unsupported_risk(
    granite_guardian_detection, granite_guardian_completion_response
):
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    context_request = ContextAnalysisRequest(
        content=CONTENT,
        context_type="docs",
        context=[CONTEXT_DOC],
        detector_params={
            "n": 2,
            "risk_name": "boo",
        },
    )
    with patch(
        "vllm_detector_adapter.generative_detectors.granite_guardian.GraniteGuardian.create_chat_completion",
        return_value=granite_guardian_completion_response,
    ):
        detection_response = asyncio.run(
            granite_guardian_detection_instance.context_analyze(context_request)
        )
        assert type(detection_response) == ErrorResponse
        assert detection_response.error.code == HTTPStatus.BAD_REQUEST
        assert (
            "risk_name or criteria_id boo is not compatible with context analysis"
            in detection_response.error.message
        )


#### Generation analysis tests


def test_generation_analyze(
    granite_guardian_detection, granite_guardian_completion_response
):
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    detection_request = GenerationDetectionRequest(
        prompt="Where is the moose?",
        generated_text="Maybe Canada?",
        detector_params={
            "n": 2,
        },
    )
    with patch(
        "vllm_detector_adapter.generative_detectors.granite_guardian.GraniteGuardian.create_chat_completion",
        return_value=granite_guardian_completion_response,
    ):
        detection_response = asyncio.run(
            granite_guardian_detection_instance.generation_analyze(detection_request)
        )
        assert type(detection_response) == DetectionResponse
        detections = detection_response.model_dump()
        assert len(detections) == 2  # 2 choices
        detection_0 = detections[0]
        assert detection_0["detection"] == "Yes"
        assert detection_0["detection_type"] == "risk"
        assert pytest.approx(detection_0["score"]) == 1.0
        assert detection_0["metadata"] == {}


def test_generation_analyze_with_confidence(
    granite_guardian_detection, granite_guardian_completion_response_3_2
):
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    detection_request = GenerationDetectionRequest(
        prompt="Where is the moose?",
        generated_text="Maybe Canada?",
        detector_params={
            "n": 2,
        },
    )
    with patch(
        "vllm_detector_adapter.generative_detectors.granite_guardian.GraniteGuardian.create_chat_completion",
        return_value=granite_guardian_completion_response_3_2,
    ):
        detection_response = asyncio.run(
            granite_guardian_detection_instance.generation_analyze(detection_request)
        )
        assert type(detection_response) == DetectionResponse
        detections = detection_response.model_dump()
        assert len(detections) == 2  # 2 choices
        detection_0 = detections[0]
        assert detection_0["detection"] == "No"
        assert detection_0["detection_type"] == "risk"
        assert pytest.approx(detection_0["score"]) == 0.9377647
        assert detection_0["metadata"] == {"confidence": "High"}


#### Chat detection tests


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
        assert type(detection_response) == DetectionResponse
        detections = detection_response.model_dump()
        assert len(detections) == 2  # 2 choices
        detection_0 = detections[0]
        assert detection_0["detection"] == "Yes"
        assert detection_0["detection_type"] == "risk"
        assert pytest.approx(detection_0["score"]) == 1.0


def test_chat_detection_with_confidence(
    granite_guardian_detection, granite_guardian_completion_response_3_2
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
        return_value=granite_guardian_completion_response_3_2,
    ):
        detection_response = asyncio.run(
            granite_guardian_detection_instance.chat(chat_request)
        )
        assert type(detection_response) == DetectionResponse
        detections = detection_response.model_dump()
        assert len(detections) == 2  # 2 choices
        detection_0 = detections[0]
        assert detection_0["detection"] == "No"
        assert detection_0["detection_type"] == "risk"
        assert pytest.approx(detection_0["score"]) == 0.9377647
        assert detection_0["metadata"] == {"confidence": "High"}
        detection_1 = detections[1]
        assert detection_1["detection"] == "Yes"
        assert detection_1["detection_type"] == "risk"
        assert pytest.approx(detection_1["score"]) == 0.9377647
        assert detection_1["metadata"] == {"confidence": "Low"}


def test_chat_detection_with_no_think_and_score_results(
    granite_guardian_detection, granite_guardian_completion_response_3_3_plus_no_think
):
    # This response has the think tags but no think results
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
        return_value=granite_guardian_completion_response_3_3_plus_no_think,
    ):
        detection_response = asyncio.run(
            granite_guardian_detection_instance.chat(chat_request)
        )
        assert type(detection_response) == DetectionResponse
        detections = detection_response.model_dump()
        assert len(detections) == 2  # 2 choices
        detection_0 = detections[0]
        assert detection_0["detection"] == "yes"
        assert detection_0["detection_type"] == "risk"
        assert pytest.approx(detection_0["score"]) == 0.9999982
        assert detection_0["metadata"] == {}
        detection_1 = detections[1]
        assert detection_1["detection"] == "yes"
        assert detection_1["detection_type"] == "risk"
        assert pytest.approx(detection_1["score"]) == 0.9999982
        assert detection_1["metadata"] == {}


def test_chat_detection_with_think_and_score_results(
    granite_guardian_detection, granite_guardian_completion_response_3_3_plus_think
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
        return_value=granite_guardian_completion_response_3_3_plus_think,
    ):
        detection_response = asyncio.run(
            granite_guardian_detection_instance.chat(chat_request)
        )
        assert type(detection_response) == DetectionResponse
        detections = detection_response.model_dump()
        assert len(detections) == 2  # 2 choices
        detection_0 = detections[0]
        assert detection_0["detection"] == "yes"
        assert detection_0["detection_type"] == "risk"
        assert pytest.approx(detection_0["score"]) == 1.0
        assert detection_0["metadata"] == {
            "think": "First analyzing the user request...there is a risk associated, so the score is yes."
        }
        detection_1 = detections[1]
        assert detection_1["detection"] == "no"
        assert detection_1["detection_type"] == "risk"
        assert pytest.approx(detection_1["score"]) == 1.0
        assert detection_1["metadata"] == {
            "think": "Since there is no risk associated, the score is no."
        }


def test_chat_detection_with_tools(
    granite_guardian_detection, granite_guardian_completion_response
):
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    chat_request = ChatDetectionRequest(
        messages=[
            DetectionChatMessageParam(
                role="user",
                content=USER_CONTENT_TOOLS,
            ),
            DetectionChatMessageParam(role="assistant", tool_calls=[TOOL_CALL]),
        ],
        tools=[TOOL],
        detector_params={"risk_name": "function_call", "n": 2},
    )
    with patch(
        "vllm_detector_adapter.generative_detectors.granite_guardian.GraniteGuardian.create_chat_completion",
        return_value=granite_guardian_completion_response,
    ):
        detection_response = asyncio.run(
            granite_guardian_detection_instance.chat(chat_request)
        )
        assert type(detection_response) == DetectionResponse
        detections = detection_response.model_dump()
        assert len(detections) == 2  # 2 choices


def test_chat_detection_with_tools_criteria_id(
    granite_guardian_detection, granite_guardian_completion_response
):
    # Guardian 3.3 parameters
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    chat_request = ChatDetectionRequest(
        messages=[
            DetectionChatMessageParam(
                role="user",
                content=USER_CONTENT_TOOLS,
            ),
            DetectionChatMessageParam(role="assistant", tool_calls=[TOOL_CALL]),
        ],
        tools=[TOOL],
        detector_params={"criteria_id": "function_call", "n": 2},
    )
    with patch(
        "vllm_detector_adapter.generative_detectors.granite_guardian.GraniteGuardian.create_chat_completion",
        return_value=granite_guardian_completion_response,
    ):
        detection_response = asyncio.run(
            granite_guardian_detection_instance.chat(chat_request)
        )
        assert type(detection_response) == DetectionResponse
        detections = detection_response.model_dump()
        assert len(detections) == 2  # 2 choices


def test_chat_detection_with_tools_wrong_risk(
    granite_guardian_detection, granite_guardian_completion_response
):
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    chat_request = ChatDetectionRequest(
        messages=[
            DetectionChatMessageParam(
                role="user",
                content=USER_CONTENT_TOOLS,
            ),
            DetectionChatMessageParam(role="assistant", tool_calls=[TOOL_CALL]),
        ],
        tools=[TOOL],
        detector_params={},
    )
    with patch(
        "vllm_detector_adapter.generative_detectors.granite_guardian.GraniteGuardian.create_chat_completion",
        return_value=granite_guardian_completion_response,
    ):
        detection_response = asyncio.run(
            granite_guardian_detection_instance.chat(chat_request)
        )
        assert type(detection_response) == ErrorResponse


#### Base class functionality tests

# NOTE: currently these functions are basically just the base implementations,
# where safe/unsafe tokens are defined in the granite guardian class


def test_calculate_scores(
    granite_guardian_detection, granite_guardian_completion_response
):
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    scores = granite_guardian_detection_instance.calculate_scores(
        granite_guardian_completion_response
    )
    assert len(scores) == 2  # 2 choices
    assert pytest.approx(scores[0]) == 1.0
    assert pytest.approx(scores[1]) == 1.0


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
    assert detection_response.error.code == HTTPStatus.BAD_REQUEST.value
    assert "streaming is not supported" in detection_response.error.message


def test_chat_detection_errors_on_jinja_template_error(granite_guardian_detection):
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    chat_request = ChatDetectionRequest(
        messages=[
            DetectionChatMessageParam(role="user", content="How do I pick a lock?")
        ],
    )
    with patch(
        "vllm_detector_adapter.generative_detectors.granite_guardian.GraniteGuardian.create_chat_completion",
        side_effect=TemplateError(),
    ):
        detection_response = asyncio.run(
            granite_guardian_detection_instance.chat(chat_request)
        )
        assert type(detection_response) == ErrorResponse
        assert detection_response.error.code == HTTPStatus.BAD_REQUEST.value
        assert "Template error" in detection_response.error.message


def test_chat_detection_errors_on_undefined_jinja_error(granite_guardian_detection):
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)
    chat_request = ChatDetectionRequest(
        messages=[
            DetectionChatMessageParam(role="user", content="How do I pick a lock?")
        ],
    )
    with patch(
        "vllm_detector_adapter.generative_detectors.granite_guardian.GraniteGuardian.create_chat_completion",
        side_effect=UndefinedError(),  # class of TemplateError
    ):
        detection_response = asyncio.run(
            granite_guardian_detection_instance.chat(chat_request)
        )
        assert type(detection_response) == ErrorResponse
        assert detection_response.error.code == HTTPStatus.BAD_REQUEST.value
        assert "Template error" in detection_response.error.message


def test_risk_bank_extraction(granite_guardian_detection):
    granite_guardian_detection_instance = asyncio.run(granite_guardian_detection)

    risk_bank_objs = asyncio.run(
        granite_guardian_detection_instance._get_predefined_risk_bank()
    )
    assert len(risk_bank_objs) == 3
    assert risk_bank_objs[0].key.value == "social_bias"
