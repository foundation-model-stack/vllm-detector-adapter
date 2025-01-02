# Standard
from dataclasses import dataclass
from typing import Optional
import asyncio

# Third Party
from vllm.config import MultiModalConfig
from vllm.entrypoints.openai.serving_engine import BaseModelPath
import jinja2
import pytest_asyncio

# Local
from vllm_detector_adapter.generative_detectors.base import ChatCompletionDetectionBase

MODEL_NAME = "openai-community/gpt2"
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


async def _async_serving_detection_completion_init():
    """Initialize a chat completion base with string templates"""
    engine = MockEngine()
    engine.errored = False
    model_config = await engine.get_model_config()

    detection_completion = ChatCompletionDetectionBase(
        task_template="hello {{user_text}}",
        output_template="bye {{text}}",
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
    return detection_completion


@pytest_asyncio.fixture
async def detection_base():
    return _async_serving_detection_completion_init()


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
