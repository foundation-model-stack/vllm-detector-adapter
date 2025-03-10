# Standard
from http import HTTPStatus
from typing import Dict, List, Optional

# Third Party
from pydantic import BaseModel, Field, RootModel, ValidationError
from typing_extensions import Required, TypedDict
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)

##### [FMS] Detection API types #####
# Endpoints are as documented https://foundation-model-stack.github.io/fms-guardrails-orchestrator/?urls.primaryName=Detector+API#/

######## Contents Detection types (currently unused) for the /text/contents detection endpoint

ROLE_OVERRIDE_PARAM_NAME = "role_override"


class ContentsDetectionRequest(BaseModel):
    contents: List[str] = Field(
        examples=[
            "Hi is this conversation guarded",
            "Yes, it is",
        ]
    )
    # Parameter passthrough
    # NOTE: this endpoint does support the optional `role_override` parameter
    # which allows use of different role when making a call to the guardrails LLM
    # via chat/completions
    detector_params: Optional[Dict] = {}


class ContentsDetectionResponseObject(BaseModel):
    start: int = Field(examples=[0])
    end: int = Field(examples=[10])
    text: str = Field(examples=["text"])
    detection: str = Field(examples=["positive"])
    detection_type: str = Field(examples=["simple_example"])
    score: float = Field(examples=[0.5])
    # Metadata for additional model information provided by the model
    metadata: Optional[Dict] = {}

    @staticmethod
    def from_chat_completion_response(
        response, scores, detection_type, req_content: str
    ):
        """Function to convert openai chat completion response to [fms] contents detection
        response object

        Args:
            response: ChatCompletionResponse,
                Chat completion response object
            scores: List[float],
                Scores for individual responses
            detection_type: str,
                Type of the detection
            req_content: str
               Input content in the request
        Returns:
            List[ContentsDetectionResponseObject]
        """

        detection_responses = []
        start = 0
        end = len(req_content)

        for i, choice in enumerate(response.choices):
            content = choice.message.content
            # NOTE: for providing spans, we currently consider entire generated text as a span.
            # This is because, at the time of writing, the generative guardrail models does not
            # provide specific information about input text, which can be used to deduce spans.
            if content and isinstance(content, str):
                response_object = ContentsDetectionResponseObject(
                    detection_type=detection_type,
                    detection=content.strip(),
                    start=start,
                    end=end,
                    text=req_content,
                    score=scores[i],
                )
                detection_responses.append(response_object)
            else:
                # This case should be unlikely but we handle it since a detection
                # can't be returned without the content
                # A partial response could be considered in the future
                # but that would likely not look like the current ErrorResponse
                return ErrorResponse(
                    message=f"Choice {i} from chat completion does not have content. \
                        Consider updating input and/or parameters for detections.",
                    type="BadRequestError",
                    code=HTTPStatus.BAD_REQUEST.value,
                )

        return detection_responses


class ContentsDetectionResponse(RootModel):
    # The root attribute is used here so that the response will appear
    # as a list instead of a list nested under a key
    root: List[List[ContentsDetectionResponseObject]]


##### Chat Detection types for the /text/chat detection endpoint ###############


class DetectionChatMessageParam(TypedDict):
    # The role of the message's author
    role: Required[str]

    # The contents of the message
    content: str


class ChatDetectionRequest(BaseModel):
    # Chat messages
    messages: List[DetectionChatMessageParam] = Field(
        examples=[
            DetectionChatMessageParam(
                role="user", content="Hi is this conversation guarded"
            ),
            DetectionChatMessageParam(role="assistant", content="Yes, it is"),
        ]
    )

    # Parameters to pass through to chat completions, optional
    detector_params: Optional[Dict] = {}

    def to_chat_completion_request(self, model_name: str):
        """Function to convert [fms] chat detection request to openai chat completion request"""
        messages = [
            {"role": message["role"], "content": message["content"]}
            for message in self.messages
        ]

        # Try to pass all detector_params through as additional parameters to chat completions.
        # We do not try to provide validation or changing of parameters here to not be dependent
        # on chat completion API changes. As of vllm >= 0.6.5, extra fields are allowed
        try:
            return ChatCompletionRequest(
                messages=messages,
                # NOTE: below is temporary
                model=model_name,
                **self.detector_params,
            )
        except ValidationError as e:
            return ErrorResponse(
                message=repr(e.errors()[0]),
                type="BadRequestError",
                code=HTTPStatus.BAD_REQUEST.value,
            )


##### Context Analysis Detection types for the /text/context/docs detection endpoint


class ContextAnalysisRequest(BaseModel):
    # Content to run detection on
    content: str = Field(examples=["What is a moose?"])
    # Type of context - url or docs (for text documents)
    context_type: str = Field(examples=["docs", "url"])
    # Context of type context_type to run detection on
    context: List[str] = Field(
        examples=[
            "https://en.wikipedia.org/wiki/Moose",
            "https://www.nwf.org/Educational-Resources/Wildlife-Guide/Mammals/Moose",
        ]
    )
    # Parameters to pass through to chat completions, optional
    detector_params: Optional[Dict] = {}

    # NOTE: currently there is no general to_chat_completion_request
    # since the chat completion roles and messages are fairly tied
    # to particular models' risk definitions. If a general strategy
    # is identified, it can be implemented here.


##### Generation Detection types for the /text/generation detection endpoint


class GenerationDetectionRequest(BaseModel):
    # Prompt to a model
    prompt: str = Field(examples=["What is a moose?"])
    # Generated text from a model
    generated_text: str = Field(examples=["A moose is not a goose."])
    # Parameters to pass through to chat completions, optional
    detector_params: Optional[Dict] = {}

    def to_chat_completion_request(self, model_name: str):
        """Function to convert [fms] generation detection request to openai chat completion request"""
        messages = [
            {"role": "user", "content": self.prompt},
            {"role": "assistant", "content": self.generated_text},
        ]

        # Try to pass all detector_params through as additional parameters to chat completions.
        # We do not try to provide validation or changing of parameters here to not be dependent
        # on chat completion API changes.
        try:
            return ChatCompletionRequest(
                messages=messages,
                # NOTE: below is temporary
                model=model_name,
                **self.detector_params,
            )
        except ValidationError as e:
            return ErrorResponse(
                message=repr(e.errors()[0]),
                type="BadRequestError",
                code=HTTPStatus.BAD_REQUEST.value,
            )


##### General detection response objects #######################################


class DetectionResponseObject(BaseModel):
    detection: str = Field(examples=["positive"])
    detection_type: str = Field(examples=["simple_example"])
    score: float = Field(examples=[0.5])


class DetectionResponse(RootModel):
    # The root attribute is used here so that the response will appear
    # as a list instead of a list nested under a key
    root: List[DetectionResponseObject]

    @staticmethod
    def from_chat_completion_response(
        response: ChatCompletionResponse, scores: List[float], detection_type: str
    ):
        """Function to convert openai chat completion response to [fms] chat detection response"""
        detection_responses = []
        for i, choice in enumerate(response.choices):
            content = choice.message.content
            if content and isinstance(content, str):
                response_object = DetectionResponseObject(
                    detection_type=detection_type,
                    detection=content.strip(),
                    score=scores[i],
                ).model_dump()
                detection_responses.append(response_object)
            else:
                # This case should be unlikely but we handle it since a detection
                # can't be returned without the content
                # A partial response could be considered in the future
                # but that would likely not look like the current ErrorResponse
                return ErrorResponse(
                    message=f"Choice {i} from chat completion does not have content. \
                        Consider updating input and/or parameters for detections.",
                    type="BadRequestError",
                    code=HTTPStatus.BAD_REQUEST.value,
                )

        return DetectionResponse(root=detection_responses)
