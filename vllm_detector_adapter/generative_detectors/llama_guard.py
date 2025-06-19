# Standard
from http import HTTPStatus
from typing import Optional, Union

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

logger = init_logger(__name__)


class LlamaGuard(ChatCompletionDetectionBase):

    DETECTION_TYPE = "risk"

    # Model specific tokens
    SAFE_TOKEN = "safe"
    UNSAFE_TOKEN = "unsafe"

    # Risk Bank name defined in the chat template
    RISK_BANK_VAR_NAME = "categories"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize risk_bank
        self.risk_bank = None

    ##### Private / Internal functions ###################################################

    async def __get_risk_bank(self):
        # Process risk_bank_objs

        if not self.risk_bank:
            logger.info("Risk bank not found in cache. Generating now.")
            risk_bank_objs = await self._get_predefined_risk_bank()

            if risk_bank_objs:
                risk_bank = {
                    risk.key.value: risk.value.value for risk in risk_bank_objs
                }
            else:
                logger.warning(
                    f"{self.__class__.__name__} is missing the RISK_BANK_VAR_NAME variable"
                )
                risk_bank = {}

            # Store this risk bank with the class for quick future use
            self.risk_bank = risk_bank

        return self.risk_bank

    ##### General overriding request / response processing functions ##################

    async def post_process_completion_results(self, response, scores, detection_type):
        """Function to process chat completion results for content type detection.

        Args:
            response: ChatCompletionResponse,
            scores: List[float],
            detection_type: str
        Returns:
            response: ChatCompletionResponse
            scores: List[float]
            detection_type: str
            metadata_per_choice: List[dict]
        """
        # NOTE: Llama-guard returns specific safety categories in the last line and in a csv format
        # this is guided by the prompt definition of the model, so we expect llama_guard to adhere to it
        # atleast for Llama-Guard-3 (latest at the time of writing)

        # In this function, we will basically remove those "safety" category from output and later on
        # move them to evidences.

        new_choices = []
        new_scores = []
        metadata_per_choice = []

        risk_bank = await self.__get_risk_bank()

        # NOTE: we are flattening out choices here as different categories
        for i, choice in enumerate(response.choices):
            content = choice.message.content
            metadata = {}
            if self.UNSAFE_TOKEN in content:
                choice.message.content = self.UNSAFE_TOKEN
                new_choices.append(choice)
                new_scores.append(scores[i])
                # Process categories
                metadata[self.RISK_BANK_VAR_NAME] = []
                for category in content.splitlines()[-1].split(","):
                    if category in risk_bank:
                        category_name = risk_bank.get(category)
                        metadata[self.RISK_BANK_VAR_NAME].append(category_name)
                    else:
                        logger.warning(
                            f"Category {category} not found in risk bank for model {self.__class__.__name__}"
                        )
            else:
                # "safe" case
                new_choices.append(choice)
                new_scores.append(scores[i])
            metadata_per_choice.append(metadata)

        response.choices = new_choices
        return response, new_scores, detection_type, metadata_per_choice

    ##### Overriding model-class specific endpoint functionality ##################

    async def content_analysis(
        self,
        request: ContentsDetectionRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[ContentsDetectionResponse, ErrorResponse]:
        """Function used to call chat detection and provide a /text/contents response"""

        # Because conversation roles are expected to alternate between 'user' and 'assistant'
        # validate whether role_override was passed as a detector_param, which is invalid
        # since explicitly overriding the conversation roles will result in an error.
        if "role_override" in request.detector_params:
            return ErrorResponse(
                message="role_override is an invalid parameter for llama guard",
                type="BadRequestError",
                code=HTTPStatus.BAD_REQUEST.value,
            )

        return await super().content_analysis(request, raw_request)
