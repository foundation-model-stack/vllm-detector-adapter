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

    # Risks associated with context analysis
    PROMPT_CONTEXT_ANALYSIS_RISKS = ["context_relevance"]
    RESPONSE_CONTEXT_ANALYSIS_RISKS = ["groundedness"]

    def preprocess(
        self, request: Union[ChatDetectionRequest, ContextAnalysisRequest]
    ) -> Union[ChatDetectionRequest, ContextAnalysisRequest, ErrorResponse]:
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

    def preprocess_chat_request(
        self, request: ChatDetectionRequest
    ) -> Union[ChatDetectionRequest, ErrorResponse]:
        """Granite guardian chat request preprocess is just detector parameter updates"""
        return self.preprocess(request)

    def request_to_chat_completion_request(
        self, request: ContextAnalysisRequest, model_name: str
    ) -> Union[ChatCompletionRequest, ErrorResponse]:
        NO_RISK_NAME_MESSAGE = "No risk_name for context analysis"

        risk_name = None
        if (
            "chat_template_kwargs" not in request.detector_params
            or "guardian_config" not in request.detector_params["chat_template_kwargs"]
        ):
            return ErrorResponse(
                message=NO_RISK_NAME_MESSAGE,
                type="BadRequestError",
                code=HTTPStatus.BAD_REQUEST.value,
            )
        # Use risk name to determine message format
        if guardian_config := request.detector_params["chat_template_kwargs"][
            "guardian_config"
        ]:
            risk_name = guardian_config["risk_name"]
        else:
            # Leaving off risk name can lead to model/template errors
            return ErrorResponse(
                message=NO_RISK_NAME_MESSAGE,
                type="BadRequestError",
                code=HTTPStatus.BAD_REQUEST.value,
            )

        if len(request.context) > 1:
            # The API supports more than one context text but currently chat completions
            # will only take one context
            logger.warning("More than one context provided. Only the last will be used")
        context_text = request.context[-1]
        content = request.content
        # The "context" role is not an officially support OpenAI role
        # Messages must be in precise ordering, or model/template errors may occur
        if risk_name in self.RESPONSE_CONTEXT_ANALYSIS_RISKS:
            # Response analysis
            messages = [
                {"role": "context", "content": context_text},
                {"role": "assistant", "content": content},
            ]
        elif risk_name in self.PROMPT_CONTEXT_ANALYSIS_RISKS:
            # Prompt analysis
            messages = [
                {"role": "user", "content": content},
                {"role": "context", "content": context_text},
            ]
        else:
            # Return error if risks are not appropriate [or could default to one of the above analyses]
            return ErrorResponse(
                message="risk_name {} is not compatible with context analysis".format(
                    risk_name
                ),
                type="BadRequestError",
                code=HTTPStatus.BAD_REQUEST.value,
            )

        # Try to pass all detector_params through as additional parameters to chat completions
        # without additional validation or parameter changes, similar to ChatDetectionRequest processing
        try:
            return ChatCompletionRequest(
                messages=messages,
                model=model_name,
                **request.detector_params,
            )
        except ValidationError as e:
            return ErrorResponse(
                message=repr(e.errors()[0]),
                type="BadRequestError",
                code=HTTPStatus.BAD_REQUEST.value,
            )

    async def context_analyze(
        self,
        request: ContextAnalysisRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[ChatDetectionResponse, ErrorResponse]:
        """Function used to call chat detection and provide a /context/doc response"""
        # Fetch model name from super class: OpenAIServing
        model_name = self.models.base_model_paths[0].name

        # Apply task template if it exists
        if self.task_template:
            request = self.apply_task_template(request)
            if isinstance(request, ErrorResponse):
                # Propagate any request problems that will not allow
                # task template to be applied
                return request

        # Make model-dependent adjustments for the request
        request = self.preprocess(request)

        # Since particular chat messages are dependent on Granite Guardian risk definitions,
        # the processing is done here rather than in a separate, general to_chat_completion_request
        # for all context analysis requests.
        chat_completion_request = self.request_to_chat_completion_request(
            request, model_name
        )
        if isinstance(chat_completion_request, ErrorResponse):
            # Propagate any request problems
            return chat_completion_request

        # Much of this chat completions processing is similar to the
        # .chat case and could be refactored if reused further in the future

        # Return an error for streaming for now. Since the detector API is unary,
        # results would not be streamed back anyway. The chat completion response
        # object would look different, and content would have to be aggregated.
        if chat_completion_request.stream:
            return ErrorResponse(
                message="streaming is not supported for the detector",
                type="BadRequestError",
                code=HTTPStatus.BAD_REQUEST.value,
            )

        # Manually set logprobs to True to calculate score later on
        # NOTE: this is supposed to override if user has set logprobs to False
        # or left logprobs as the default False
        chat_completion_request.logprobs = True
        # NOTE: We need top_logprobs to be enabled to calculate score appropriately
        # We override this and not allow configuration at this point. In future, we may
        # want to expose this configurable to certain range.
        chat_completion_request.top_logprobs = 5

        logger.debug("Request to chat completion: %s", chat_completion_request)

        # Call chat completion
        chat_response = await self.create_chat_completion(
            chat_completion_request, raw_request
        )
        logger.debug("Raw chat completion response: %s", chat_response)
        if isinstance(chat_response, ErrorResponse):
            # Propagate chat completion errors directly
            return chat_response

        # Apply output template if it exists
        if self.output_template:
            chat_response = self.apply_output_template(chat_response)

        # Calculate scores
        scores = self.calculate_scores(chat_response)

        return ChatDetectionResponse.from_chat_completion_response(
            chat_response, scores, self.DETECTION_TYPE
        )
