# Standard
from http import HTTPStatus
from typing import Optional, Union

# Third Party
from fastapi import Request
from pydantic import ValidationError
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ErrorResponse

# Local
from vllm_detector_adapter.detector_dispatcher import detector_dispatcher
from vllm_detector_adapter.generative_detectors.base import ChatCompletionDetectionBase
from vllm_detector_adapter.logging import init_logger
from vllm_detector_adapter.protocol import (
    ChatDetectionRequest,
    ContextAnalysisRequest,
    DetectionResponse,
)
from vllm_detector_adapter.utils import DetectorType

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

    ##### Private functions ###################################################

    def __preprocess(
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

    ##### General chat completion output processing functions ##################

    @detector_dispatcher(types=[DetectorType.TEXT_CHAT])
    def preprocess_request(
        self, request: ChatDetectionRequest
    ) -> Union[ChatDetectionRequest, ErrorResponse]:
        """Granite guardian chat request preprocess is just detector parameter updates"""
        return self.__preprocess(request)

    @detector_dispatcher(types=[DetectorType.TEXT_CONTEXT_DOC])
    def _request_to_chat_completion_request(
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
            # The detector API for context docs detection supports more than one context text
            # but currently chat completions will only take one context. Here, we concatenate
            # multiple contexts together if provided. Models will error if the user request
            # exceeds the model's context length
            logger.warning("More than one context provided. Concatenating contexts.")
        context_text = " ".join(request.context)  # Will not affect single context case
        content = request.content
        # The "context" role is not an officially supported OpenAI role, so this is specific
        # to Granite Guardian. Messages must also be in precise ordering, or model/template
        # errors may occur.
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
            # Return error if risk names are not expected ones
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
    ) -> Union[DetectionResponse, ErrorResponse]:
        """Function used to call chat detection and provide a /context/doc response"""
        # Fetch model name from super class: OpenAIServing
        model_name = self.models.base_model_paths[0].name

        # Task template not applied for context analysis at this time
        # Make model-dependent adjustments for the request
        request = self.__preprocess(request)

        # Since particular chat messages are dependent on Granite Guardian risk definitions,
        # the processing is done here rather than in a separate, general to_chat_completion_request
        # for all context analysis requests.
        chat_completion_request = self._request_to_chat_completion_request(
            request, model_name, fn_type=DetectorType.TEXT_CONTEXT_DOC
        )
        if isinstance(chat_completion_request, ErrorResponse):
            # Propagate any request problems
            return chat_completion_request

        # Calling chat completion and processing of scores is currently
        # the same as for the /chat case
        return await self.process_chat_completion_with_scores(
            chat_completion_request, raw_request
        )
