# Standard
from http import HTTPStatus
from pathlib import Path
from typing import List, Optional, Union
import codecs
import math

# Third Party
from fastapi import Request
from vllm.entrypoints.openai.protocol import ChatCompletionResponse, ErrorResponse
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
import jinja2
import torch

# Local
from vllm_detector_adapter.logging import init_logger
from vllm_detector_adapter.protocol import (
    ChatDetectionRequest,
    ContextAnalysisRequest,
    DetectionResponse,
)

logger = init_logger(__name__)

START_PROB = 1e-50


class ChatCompletionDetectionBase(OpenAIServingChat):
    """Base class for developing chat completion based detectors"""

    def __init__(self, task_template: str, output_template: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.jinja_env = jinja2.Environment()
        self.task_template = self.load_template(task_template)

        self.output_template = self.load_template(output_template)

    ##### Template functions ###################################################

    def load_template(self, template_path: Optional[Union[Path, str]]) -> str:
        """Function to load template
        Note: this function currently is largely taken from the chat template method
        in vllm.entrypoints.chat_utils
        """
        if template_path is None:
            return None
        try:
            with open(template_path, "r") as f:
                resolved_template = f.read()
                # Addition to vllm's original load chat template method
                # This prevents additional escaping of template characters
                # such as \n (newlines)
                resolved_template = codecs.decode(resolved_template, "unicode-escape")
        except OSError as e:
            if isinstance(template_path, Path):
                raise

            JINJA_CHARS = "{}\n"
            if not any(c in template_path for c in JINJA_CHARS):
                msg = (
                    f"The supplied template ({template_path}) "
                    f"looks like a file path, but it failed to be "
                    f"opened. Reason: {e}"
                )
                raise ValueError(msg) from e

            # If opening a file fails, set template to be args to
            # ensure we decode so our escape are interpreted correctly
            resolved_template = codecs.decode(template_path, "unicode_escape")

        logger.info("Using supplied template:\n%s", resolved_template)
        return self.jinja_env.from_string(resolved_template)

    def apply_output_template(
        self, response: ChatCompletionResponse
    ) -> Union[ChatCompletionResponse, ErrorResponse]:
        """Apply output parsing template for the response"""
        return response

    ##### Chat request processing functions ####################################

    def apply_task_template_to_chat(
        self, request: ChatDetectionRequest
    ) -> Union[ChatDetectionRequest, ErrorResponse]:
        """Apply task template on the chat request"""
        return request

    def preprocess_chat_request(
        self, request: ChatDetectionRequest
    ) -> Union[ChatDetectionRequest, ErrorResponse]:
        """Preprocess chat request"""
        return request

    ##### General chat completion output processing functions ##################

    def calculate_scores(self, response: ChatCompletionResponse) -> List[float]:
        """Extract scores from logprobs of the raw chat response"""
        safe_token_prob = START_PROB
        unsafe_token_prob = START_PROB

        choice_scores = []

        # TODO: consider if this part can be optimized despite nested response structure
        for choice in response.choices:
            # Each choice will have logprobs for tokens
            for logprob_info_i in choice.logprobs.content:
                # NOTE: open-ai chat completion performs a max operation over top log probs
                # and puts that result in `logprobs`, whereas we need to do a sum over these as
                # per discussion with granite team. So we are pulling in `top_logprobs`
                for top_logprob in logprob_info_i.top_logprobs:
                    token = top_logprob.token
                    if token.strip().lower() == self.SAFE_TOKEN.lower():
                        safe_token_prob += math.exp(top_logprob.logprob)
                    if token.strip().lower() == self.UNSAFE_TOKEN.lower():
                        unsafe_token_prob += math.exp(top_logprob.logprob)

            probabilities = torch.softmax(
                torch.tensor([math.log(safe_token_prob), math.log(unsafe_token_prob)]),
                dim=0,
            )

            # We calculate "probability of risk" here, therefore, only return probability related to
            # unsafe_token_prob. Use .item() to get tensor float
            choice_scores.append(probabilities[1].item())

        return choice_scores

    async def process_chat_completion_with_scores(
        self, chat_completion_request, raw_request
    ) -> Union[DetectionResponse, ErrorResponse]:
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

        return DetectionResponse.from_chat_completion_response(
            chat_response, scores, self.DETECTION_TYPE
        )

    ##### Detection methods ####################################################
    # Base implementation of other detection endpoints like content can go here

    async def chat(
        self,
        request: ChatDetectionRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[DetectionResponse, ErrorResponse]:
        """Function used to call chat detection and provide a /chat response"""

        # Fetch model name from super class: OpenAIServing
        model_name = self.models.base_model_paths[0].name

        # Apply task template if it exists
        if self.task_template:
            request = self.apply_task_template_to_chat(request)
            if isinstance(request, ErrorResponse):
                # Propagate any request problems that will not allow
                # task template to be applied
                return request

        # Optionally make model-dependent adjustments for the request
        request = self.preprocess_chat_request(request)

        chat_completion_request = request.to_chat_completion_request(model_name)
        if isinstance(chat_completion_request, ErrorResponse):
            # Propagate any request problems
            return chat_completion_request

        return await self.process_chat_completion_with_scores(
            chat_completion_request, raw_request
        )

    async def context_analyze(
        self,
        request: ContextAnalysisRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[DetectionResponse, ErrorResponse]:
        """Function used to call chat detection and provide a /context/doc response"""
        # Return "not implemented" here since context analysis may not
        # generally apply to all models at this time
        return ErrorResponse(
            message="context analysis is not supported for the detector",
            type="NotImplementedError",
            code=HTTPStatus.NOT_IMPLEMENTED.value,
        )
