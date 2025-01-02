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

##### [FMS] Detection API types
# NOTE: This currently works with the /chat detection endpoint

######## Contents Detection types


class ContentsDetectionRequest(BaseModel):
    contents: List[str] = Field(
        examples=[
            "Hi is this conversation guarded",
            "Yes, it is",
        ]
    )


class ContentsDetectionResponseObject(BaseModel):
    start: int = Field(examples=[0])
    end: int = Field(examples=[10])
    text: str = Field(examples=["text"])
    detection: str = Field(examples=["positive"])
    detection_type: str = Field(examples=["simple_example"])
    score: float = Field(examples=[0.5])


######## Chat Detection types


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


class ChatDetectionResponseObject(BaseModel):
    detection: str = Field(examples=["positive"])
    detection_type: str = Field(examples=["simple_example"])
    score: float = Field(examples=[0.5])


class ChatDetectionResponse(RootModel):
    # The root attribute is used here so that the response will appear
    # as a list instead of a list nested under a key
    root: List[ChatDetectionResponseObject]

    @staticmethod
    def from_chat_completion_response(
        response: ChatCompletionResponse, scores: List[float], detection_type: str
    ):
        """Function to convert openai chat completion response to [fms] chat detection response"""
        detection_responses = []
        for i, choice in enumerate(response.choices):
            content = choice.message.content
            if content and isinstance(content, str):
                response_object = ChatDetectionResponseObject(
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

        return ChatDetectionResponse(root=detection_responses)
