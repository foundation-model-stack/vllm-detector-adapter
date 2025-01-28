# Standard
from http import HTTPStatus
from typing import Optional, Union

# Third Party
from fastapi import Request
from pydantic import ValidationError
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ErrorResponse

# Local
from vllm_detector_adapter.generative_detectors.base import ChatCompletionDetectionBase
from vllm_detector_adapter.logging import init_logger
from vllm_detector_adapter.protocol import (
    ChatDetectionRequest,
    ChatDetectionResponse,
    ContextAnalysisRequest,
)

logger = init_logger(__name__)


class GraniteGuardian(ChatCompletionDetectionBase):

    DETECTION_TYPE = "risk"
    # User text pattern in task template
    USER_TEXT_PATTERN = "user_text"

    # Model specific tokens
    SAFE_TOKEN = "No"
    UNSAFE_TOKEN = "Yes"

    def preprocess_chat_request(
        self, request: ChatDetectionRequest
    ) -> Union[ChatDetectionRequest, ErrorResponse]:
        """Granite guardian specific parameter updates for risk name and risk definition"""
        # Validation that one of the 'defined' risks is requested will be
        # done through the chat template on each request. Errors will
        # be propagated for chat completion separately
        guardian_config = {}
        if not request.detector_params:
            return request

        if risk_name := request.detector_params.pop("risk_name", None):
            guardian_config["risk_name"] = risk_name
        if risk_definition := request.detector_params.pop("risk_definition", None):
            guardian_config["risk_definition"] = risk_definition
        if guardian_config:
            logger.debug("guardian_config {} provided for request", guardian_config)
            # Move the risk name and/or risk definition to chat_template_kwargs
            # to be propagated to tokenizer.apply_chat_template during
            # chat completion
            request.detector_params["chat_template_kwargs"] = {
                "guardian_config": guardian_config
            }

        return request

    async def context_analyze(
        self,
        request: ContextAnalysisRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[ChatDetectionResponse, ErrorResponse]:
        """Function used to call chat detection and provide a /context/doc response"""
        # # Fetch model name from super class: OpenAIServing
        # model_name = self.models.base_model_paths[0].name

        # # Apply task template if it exists
        # if self.task_template:
        #     request = self.apply_task_template(request)
        #     if isinstance(request, ErrorResponse):
        #         # Propagate any request problems that will not allow
        #         # task template to be applied
        #         return request

        # # Much of this chat completions processing is similar to the
        # # .chat case and could be refactored if reused further in the future
        pass

    def to_chat_completion_request(self, model_name: str):
        """Function to convert context analysis request to openai chat completion request"""
        # Can only process one context currently - TODO: validate
        # For now, context_type is ignored but is required for the detection endpoint
        # TODO: 'context' is not a generally supported 'role' in the openAI API
        # TODO: messages are much more specific to risk type
        messages = [
            {"role": "context", "content": self.context[0]},
            {"role": "assistant", "content": self.content},
        ]

        # Try to pass all detector_params through as additional parameters to chat completions
        # without additional validation or parameter changes as in ChatDetectionRequest above
        try:
            return ChatCompletionRequest(
                messages=messages,
                model=model_name,
                **self.detector_params,
            )
        except ValidationError as e:
            return ErrorResponse(
                message=repr(e.errors()[0]),
                type="BadRequestError",
                code=HTTPStatus.BAD_REQUEST.value,
            )
