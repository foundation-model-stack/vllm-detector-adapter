# Standard
from http import HTTPStatus
from typing import Dict, List, Optional

# Third Party
from pydantic import BaseModel, Field, RootModel, ValidationError
from typing_extensions import NotRequired, Required, TypedDict
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorInfo,
    ErrorResponse,
)

##### [FMS] Detection API types #####
# Endpoints are as documented https://foundation-model-stack.github.io/fms-guardrails-orchestrator/?urls.primaryName=Detector+API#/

######## Contents Detection types (currently unused) for the /text/contents detection endpoint

ROLE_OVERRIDE_PARAM_NAME = "role_override"


class ContentsDetectionRequest(BaseModel):
    # Contents list to run detections on
    contents: List[str] = Field(
        examples=[
            "Hi is this conversation guarded",
            "Yes, it is",
        ]
    )
    # Parameter passthrough
    # NOTE: this endpoint supports the optional `role_override` parameter,
    # which allows use of a different role when making a call to the guardrails LLM
    # via `/chat/completions`
    detector_params: Optional[Dict] = {}


class ContentsDetectionResponseObject(BaseModel):
    # Start index of the text corresponding to the detection
    start: int = Field(examples=[0])
    # End index of the text corresponding to the detection
    end: int = Field(examples=[10])
    # Text corresponding to the detection
    text: str = Field(examples=["text"])
    # Detection label
    detection: str = Field(examples=["positive"])
    # Detection type
    detection_type: str = Field(examples=["simple_example"])
    # Score of detection
    score: float = Field(examples=[0.5])
    # Optional metadata for additional model information provided by the model
    metadata: Optional[Dict] = {}

    @staticmethod
    def from_chat_completion_response(
        response, scores, detection_type, req_content: str, metadata_per_choice=None
    ):
        """Function to convert openai chat completion response to [fms] contents detection
        response object

        Args:
            response: ChatCompletionResponse
                Chat completion response object
            scores: List[float]
                Scores for individual responses
            detection_type: str
                Type of the detection
            req_content: str
               Input content in the request
            metadata_per_choice: Optional[List[Dict]]
                Optional list of dicts containing metadata response provided by the model,
                one dict per choice
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
            if isinstance(content, str) and content.strip():
                response_object = ContentsDetectionResponseObject(
                    detection_type=detection_type,
                    detection=content.strip(),
                    start=start,
                    end=end,
                    text=req_content,
                    score=scores[i],
                    metadata=metadata_per_choice[i] if metadata_per_choice else {},
                ).model_dump()
                detection_responses.append(response_object)

            else:
                # This case should be unlikely but we handle it since a detection
                # can't be returned without the content
                # A partial response could be considered in the future
                # but that would likely not look like the current ErrorResponse
                return ErrorResponse(
                    error=ErrorInfo(
                        message=f"Choice {i} from chat completion does not have content. \
                        Consider updating input and/or parameters for detections.",
                        type="BadRequestError",
                        code=HTTPStatus.BAD_REQUEST.value,
                    )
                )

        return detection_responses


class ContentsDetectionResponse(RootModel):
    # The root attribute is used here so that the response will appear
    # as a list instead of a list nested under a key
    root: List[List[ContentsDetectionResponseObject]]

    @staticmethod
    def from_chat_completion_response(results, contents: List[str], *args, **kwargs):
        contents_detection_responses = []

        for content_idx, (response, scores, detection_type) in enumerate(results):
            detection_responses = (
                ContentsDetectionResponseObject.from_chat_completion_response(
                    response,
                    scores,
                    detection_type,
                    contents[content_idx],
                    *args,
                    **kwargs,
                )
            )
            if isinstance(detection_responses, ErrorResponse):
                return detection_responses

            contents_detection_responses.append(detection_responses)

        return ContentsDetectionResponse(root=contents_detection_responses)


##### Chat Detection types for the /text/chat detection endpoint ###############
# OpenAI API referenced for portions is at: https://github.com/openai/openai-openapi/blob/master/openapi.yaml


class ToolCallFunctionObject(TypedDict):
    # Name of function to call
    name: str
    # Arguments to call function with, as generated by the model in JSON format
    arguments: str


# Corresponds to ChatCompletionMessageToolCall for OpenAI API
class ToolCall(TypedDict):
    # Tool Call ID
    id: str
    # Tool type, only `function` is currently supported
    type: str
    # Function called by model in JSON format
    function: ToolCallFunctionObject


class DetectionChatMessageParam(TypedDict):
    # The role of the message's author
    role: Required[str]
    # Content of the message,
    content: NotRequired[str]
    # Optional generated tool calls, including function calls,
    # to be evaluated against other message content and/or tools
    # definitions provided on detection
    tool_calls: NotRequired[List[ToolCall]]


# Corresponds to FunctionObject for OpenAI API
class ToolFunctionObject(TypedDict):
    # Description of what the function does
    description: str
    # Name of function to be called
    name: str
    # Parameters accepted by the function
    parameters: Dict
    # Whether to enable strict schema adherence
    strict: NotRequired[bool]


# Corresponds to ChatCompletionTool for OpenAI API
class Tool(TypedDict):
    # Tool type, only `function` is currently supported
    type: str

    # Function definition
    function: ToolFunctionObject


class ChatDetectionRequest(BaseModel):
    # Chat messages
    messages: List[DetectionChatMessageParam] = Field(
        examples=[
            [
                DetectionChatMessageParam(
                    role="user", content="Hi is this conversation guarded"
                ),
                DetectionChatMessageParam(role="assistant", content="Yes, it is"),
            ]
        ]
    )

    # Optional list of tools definitions to provide for evaluation during detection
    tools: Optional[List[Tool]] = []

    # Parameters to pass through to chat completions, optional
    detector_params: Optional[Dict] = {}

    def to_chat_completion_request(self, model_name: str):
        """Function to convert [fms] chat detection request to openai chat completion request"""
        # Note: For current usage, we do not provide `tools` to the
        # completion request. `tools` here are not meant to be
        # called directly by the detection model, but used specifically
        # in evaluation against messages in the request.
        messages = []
        for message in self.messages:
            # OpenAI messages do not require content if other fields like tool_calls
            # exist, but after pre-processing, here we still expect requests to have
            # content
            if "content" not in message:
                return ErrorResponse(
                    error=ErrorInfo(
                        message="message missing content",
                        type="BadRequestError",
                        code=HTTPStatus.BAD_REQUEST.value,
                    )
                )
            messages.append({"role": message["role"], "content": message["content"]})

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
                error=ErrorInfo(
                    message=repr(e.errors()[0]),
                    type="BadRequestError",
                    code=HTTPStatus.BAD_REQUEST.value,
                )
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
                error=ErrorInfo(
                    message=repr(e.errors()[0]),
                    type="BadRequestError",
                    code=HTTPStatus.BAD_REQUEST.value,
                )
            )


##### General detection response objects #######################################


class DetectionResponseObject(BaseModel):
    # Detection label
    detection: str = Field(examples=["positive"])
    # Detection type
    detection_type: str = Field(examples=["simple_example"])
    # Score of detection
    score: float = Field(examples=[0.5])
    # Optional metadata for additional model information provided by the model
    metadata: Optional[Dict] = {}


class DetectionResponse(RootModel):
    # The root attribute is used here so that the response will appear
    # as a list instead of a list nested under a key
    root: List[DetectionResponseObject]

    @staticmethod
    def from_chat_completion_response(
        response: ChatCompletionResponse,
        scores: List[float],
        detection_type: str,
        metadata_per_choice: Optional[List[Dict]] = None,
    ):
        """Function to convert openai chat completion response to [fms] chat detection response"""
        detection_responses = []
        for i, choice in enumerate(response.choices):
            content = choice.message.content
            if isinstance(content, str) and content.strip():
                response_object = DetectionResponseObject(
                    detection_type=detection_type,
                    detection=content.strip(),
                    score=scores[i],
                    metadata=metadata_per_choice[i] if metadata_per_choice else {},
                ).model_dump()
                detection_responses.append(response_object)
            else:
                # This case should be unlikely but we handle it since a detection
                # can't be returned without the content
                # A partial response could be considered in the future
                # but that would likely not look like the current ErrorResponse
                return ErrorResponse(
                    error=ErrorInfo(
                        message=f"Choice {i} from chat completion does not have content. \
                        Consider updating input and/or parameters for detections.",
                        type="BadRequestError",
                        code=HTTPStatus.BAD_REQUEST.value,
                    )
                )

        return DetectionResponse(root=detection_responses)
