# Standard
from typing import Optional
import asyncio
import copy

# Third Party
from fastapi import Request
from vllm.entrypoints.openai.protocol import ErrorResponse

# Local
from vllm_detector_adapter.generative_detectors.base import ChatCompletionDetectionBase
from vllm_detector_adapter.logging import init_logger
from vllm_detector_adapter.protocol import (
    ContentsDetectionRequest,
    ContentsDetectionResponse,
)
from vllm_detector_adapter.utils import DetectorType

logger = init_logger(__name__)


class LlamaGuard(ChatCompletionDetectionBase):

    DETECTION_TYPE = "risk"

    # Model specific tokens
    SAFE_TOKEN = "safe"
    UNSAFE_TOKEN = "unsafe"

    def __post_process_result(self, responses, scores, detection_type):
        """Function to process chat completion results for content type detection.

        Args:
            responses: ChatCompletionResponse,
            scores: List[float],
            detection_type: str,
        Returns:
            Tuple(
                responses: ChatCompletionResponse,
                scores: List[float],
                detection_type,
            )
        """
        # NOTE: Llama-guard returns specific safety categories in the last line and in a csv format
        # this is guided by the prompt definition of the model, so we expect llama_guard to adhere to it
        # atleast for Llama-Guard-3 (latest at the time of writing)

        # NOTE: The concept of "choice" doesn't exist for content type detector API, so
        # we will essentially flatten out the responses, so different categories in 1 choice
        # will also look like another choice.

        new_choices = []
        new_scores = []

        for i, choice in enumerate(responses.choices):
            content = choice.message.content
            if self.UNSAFE_TOKEN in content:
                # We will create multiple results for each unsafe category
                # in addition to "unsafe" as a category itself
                # NOTE: need to deepcopy, otherwise, choice will get overwritten
                unsafe_choice = copy.deepcopy(choice)
                unsafe_choice.message.content = self.UNSAFE_TOKEN

                new_choices.append(unsafe_choice)
                new_scores.append(scores[i])

                # Fetch categories as the last line in the response available in csv format
                for category in content.splitlines()[-1].split(","):
                    category_choice = copy.deepcopy(choice)
                    category_choice.message.content = category
                    new_choices.append(category_choice)
                    # NOTE: currently using same score as "unsafe"
                    # but we need to see if we can revisit this to get better score
                    new_scores.append(scores[i])
            else:
                # "safe" case
                new_choices.append(choice)
                new_scores.append(scores[i])

        responses.choices = new_choices
        return (responses, new_scores, detection_type)

    async def content_analysis(
        self,
        request: ContentsDetectionRequest,
        raw_request: Optional[Request] = None,
    ):
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

        # Since separate batch processing function doesn't exist at the time of writing,
        # we are just going to collect all the text from content request and fire up
        # separate requests and wait asynchronously.
        # This mirrors how batching is handled in run_batch function in entrypoints/openai/
        # in vLLM codebase.
        completion_requests = self.preprocess_request(
            request, fn_type=DetectorType.TEXT_CONTENT
        )

        # Send all the completion requests asynchronously.
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
        categorized_results = []
        for result in results:
            # NOTE: we are only sending 1 of the error results
            # and not every or not cumulative
            if isinstance(result, ErrorResponse):
                return result
            else:
                # Process results to split out safety categories into separate objects
                categorized_results.append(self.__post_process_result(*result))

        return ContentsDetectionResponse.from_chat_completion_response(
            categorized_results, request.contents
        )
