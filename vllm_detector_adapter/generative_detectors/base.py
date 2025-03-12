# Standard
from http import HTTPStatus
from pathlib import Path
from typing import List, Optional, Tuple, Union
import asyncio
import codecs
import math

# Third Party
from fastapi import Request
from jinja2.exceptions import TemplateError
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
import jinja2
import torch

# Local
from vllm_detector_adapter.detector_dispatcher import detector_dispatcher
from vllm_detector_adapter.logging import init_logger
from vllm_detector_adapter.protocol import (
    ROLE_OVERRIDE_PARAM_NAME,
    ChatDetectionRequest,
    ContentsDetectionRequest,
    ContentsDetectionResponse,
    ContentsDetectionResponseObject,
    ContextAnalysisRequest,
    DetectionResponse,
    GenerationDetectionRequest,
)
from vllm_detector_adapter.utils import DetectorType

logger = init_logger(__name__)

START_PROB = 1e-50

DEFAULT_ROLE_FOR_CONTENTS_DETECTION = "user"


class ChatCompletionDetectionBase(OpenAIServingChat):
    """Base class for developing chat completion based detectors"""

    def __init__(
        self,
        task_template: str,
        output_template: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.jinja_env = jinja2.Environment()
        self.task_template = self.load_template(task_template)

        self.output_template = self.load_template(output_template)

    ##### Template functions ###################################################

    async def _get_predefined_risk_bank(self) -> List[jinja2.nodes.Pair]:
        """Get the list of risks defined in the chat template"""

        if not hasattr(self, "RISK_BANK_VAR_NAME"):
            raise ValueError(
                f"RISK_BANK_VAR_NAME is not defined for {self.__class__.__name__} type of models"
            )

        if self.chat_template:
            # use chat template directly, since it might have been provided to override
            # default model's chat template
            chat_template = self.chat_template
        else:
            # use model's default chat template
            # NOTE: we need to get tokenizer separately to support LoRA adapters
            tokenizer = await self.engine_client.get_tokenizer()
            chat_template = tokenizer.chat_template

        ast = self.jinja_env.parse(chat_template)
        risk_bank_objs = []

        # Note: jinja2.nodes.Assign is type of node
        for node in ast.find_all(jinja2.nodes.Assign):
            if node.target.name == self.RISK_BANK_VAR_NAME:
                risk_node = node
                for i in risk_node.node.items:
                    risk_bank_objs.append(i)

        return risk_bank_objs

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

    # Usage of detector_dispatcher allows same function name to be called for different types of
    # detectors with different arguments and implementation.
    @detector_dispatcher(
        types=[
            DetectorType.TEXT_CHAT,
            DetectorType.TEXT_CONTENT,
            DetectorType.TEXT_GENERATION,
        ]
    )
    def apply_task_template(
        self, request: ChatDetectionRequest
    ) -> Union[ChatDetectionRequest, ErrorResponse]:
        """Apply task template on the chat request"""
        return request

    @detector_dispatcher(types=[DetectorType.TEXT_CHAT, DetectorType.TEXT_GENERATION])
    def preprocess_request(  # noqa: F811
        self, request: Union[ChatDetectionRequest, GenerationDetectionRequest]
    ) -> Union[ChatDetectionRequest, GenerationDetectionRequest, ErrorResponse]:
        """Preprocess chat request or generation detection request"""
        # pylint: disable=redefined-outer-name
        return request

    ##### Contents request processing functions ####################################

    @detector_dispatcher(types=[DetectorType.TEXT_CONTENT])
    def preprocess_request(  # noqa: F811
        self, request: ContentsDetectionRequest
    ) -> Union[List[ChatCompletionRequest], ErrorResponse]:
        """Preprocess contents request and convert it into appropriate chat requests"""
        # pylint: disable=redefined-outer-name
        # Fetch model name from super class: OpenAIServing
        model_name = self.models.base_model_paths[0].name

        # Fetch role override from detector_params, otherwise use default
        role = request.detector_params.pop(
            ROLE_OVERRIDE_PARAM_NAME, DEFAULT_ROLE_FOR_CONTENTS_DETECTION
        )

        batch_requests = [
            ChatCompletionRequest(
                messages=[{"role": role, "content": content}],
                model=model_name,
                **request.detector_params,
            )
            for content in request.contents
        ]

        return batch_requests

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
    ) -> Union[Tuple[ChatCompletionResponse, List[float], str], ErrorResponse]:
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
        try:
            chat_response = await self.create_chat_completion(
                chat_completion_request, raw_request
            )
        except TemplateError as e:
            # Propagate template errors including those from raise_exception in the chat_template.
            # UndefinedError, a subclass of TemplateError, can happen due to a variety of reasons -
            # e.g. for Granite Guardian it is not limited but including the following
            # for a particular risk definition: unexpected number of messages, unexpected
            # ordering of messages, unexpected roles used for particular messages.
            # Users _may_ be able to correct some of these errors by changing the input
            # but the error message may not be directly user-comprehensible
            chat_response = ErrorResponse(
                message=e.message or "Template error",
                type="BadRequestError",
                code=HTTPStatus.BAD_REQUEST.value,
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

        return chat_response, scores, self.DETECTION_TYPE

    async def post_process_completion_results(
        self, response: ChatCompletionResponse, scores: List[float], detection_type: str
    ) -> Tuple[ChatCompletionResponse, List[float], str, str]:
        """Function to process the results of chat completion and to divide it
        into logical blocks that can be converted into different detection result
        objects

        NOTE: This function is kept async to allow consistent usage with llama-guard's implementation
        and in case this function needs to access other async function or
        execute heavier tasks in future.

        Args:
            response: ChatCompletionResponse,
            scores: List[float],
            detection_type: str
        Returns:
            response: ChatCompletionResponse
            scores: List[float]
            detection_type: str
            metadata: List[dict] or None
        """
        metadata_list = None
        return response, scores, detection_type, metadata_list

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
            request = self.apply_task_template(request, fn_type=DetectorType.TEXT_CHAT)
            if isinstance(request, ErrorResponse):
                # Propagate any request problems that will not allow
                # task template to be applied
                return request

        # Optionally make model-dependent adjustments for the request
        request = self.preprocess_request(request, fn_type=DetectorType.TEXT_CHAT)

        chat_completion_request = request.to_chat_completion_request(model_name)
        if isinstance(chat_completion_request, ErrorResponse):
            # Propagate any request problems
            return chat_completion_request

        result = await self.process_chat_completion_with_scores(
            chat_completion_request, raw_request
        )

        if isinstance(result, ErrorResponse):
            # Propagate any errors from OpenAI API
            return result
        else:
            (
                chat_response,
                scores,
                detection_type,
                metadata,
            ) = await self.post_process_completion_results(*result)

        return DetectionResponse.from_chat_completion_response(
            chat_response,
            scores,
            detection_type,
            metadata_per_choice=metadata,
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

    async def content_analysis(
        self,
        request: ContentsDetectionRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[ContentsDetectionResponse, ErrorResponse]:
        """Function used to call chat detection and provide a /text/contents response"""

        # Apply task template if it exists
        if self.task_template:
            request = self.apply_task_template(
                request, fn_type=DetectorType.TEXT_CONTENT
            )
            if isinstance(request, ErrorResponse):
                # Propagate any request problems that will not allow
                # task template to be applied
                return request

        # Since real batch processing function doesn't exist at the time of writing,
        # we are just going to collect all the text from content request and create
        # separate ChatCompletionRequests and then fire them up and wait asynchronously.
        # This mirrors how batching is handled in run_batch function in entrypoints/openai/
        # in vLLM codebase.
        completion_requests = self.preprocess_request(
            request, fn_type=DetectorType.TEXT_CONTENT
        )

        # Fire up all the completion requests asynchronously.
        tasks = [
            asyncio.create_task(
                self.process_chat_completion_with_scores(
                    completion_request, raw_request
                )
            )
            for completion_request in completion_requests
        ]

        # Gather all the results
        # NOTE: The results are guaranteed to be in order of requests
        results = await asyncio.gather(*tasks)

        # If there is any error, return that otherwise, return the whole response
        # properly formatted.
        processed_result = []
        for result_idx, result in enumerate(results):
            # NOTE: we are only sending 1 of the error results
            # and not every or not cumulative
            if isinstance(result, ErrorResponse):
                return result
            else:
                (
                    response,
                    new_scores,
                    detection_type,
                    metadata,
                ) = await self.post_process_completion_results(*result)

                new_result = (
                    ContentsDetectionResponseObject.from_chat_completion_response(
                        response,
                        new_scores,
                        detection_type,
                        request.contents[result_idx],
                        metadata_per_choice=metadata,
                    )
                )

                processed_result.append(new_result)

        return ContentsDetectionResponse(root=processed_result)

    async def generation_analyze(
        self,
        request: GenerationDetectionRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[DetectionResponse, ErrorResponse]:
        """Function used to call chat detection and provide a /generation response"""

        # Fetch model name from super class: OpenAIServing
        model_name = self.models.base_model_paths[0].name

        # Apply task template if it exists
        if self.task_template:
            request = self.apply_task_template(
                request, fn_type=DetectorType.TEXT_GENERATION
            )
            if isinstance(request, ErrorResponse):
                # Propagate any request problems that will not allow
                # task template to be applied
                return request

        # Optionally make model-dependent adjustments for the request
        request = self.preprocess_request(request, fn_type=DetectorType.TEXT_GENERATION)

        chat_completion_request = request.to_chat_completion_request(model_name)
        if isinstance(chat_completion_request, ErrorResponse):
            # Propagate any request problems
            return chat_completion_request

        result = await self.process_chat_completion_with_scores(
            chat_completion_request, raw_request
        )

        if isinstance(result, ErrorResponse):
            # Propagate any errors from OpenAI API
            return result
        else:
            (
                chat_response,
                scores,
                detection_type,
                metadata,
            ) = await self.post_process_completion_results(*result)

        return DetectionResponse.from_chat_completion_response(
            chat_response,
            scores,
            detection_type,
            metadata_per_choice=metadata,
        )
