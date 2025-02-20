# Standard
from http import HTTPStatus

# Third Party
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    ErrorResponse,
    UsageInfo,
)

# Local
from vllm_detector_adapter.protocol import (
    ChatDetectionRequest,
    ContentsDetectionResponse,
    ContentsDetectionResponseObject,
    DetectionChatMessageParam,
    DetectionResponse,
)

MODEL_NAME = "org/model-name"

### Tests #####################################################################

#### Chat detection request tests


def test_chat_detection_to_completion_request():
    chat_request = ChatDetectionRequest(
        messages=[
            DetectionChatMessageParam(
                role="user", content="How do I search for moose?"
            ),
            DetectionChatMessageParam(
                role="assistant", content="You could go to Canada"
            ),
        ],
        detector_params={"n": 3, "temperature": 0.5},
    )
    request = chat_request.to_chat_completion_request(MODEL_NAME)
    assert type(request) == ChatCompletionRequest
    assert request.messages[0]["role"] == "user"
    assert request.messages[0]["content"] == "How do I search for moose?"
    assert request.messages[1]["role"] == "assistant"
    assert request.messages[1]["content"] == "You could go to Canada"
    assert request.model == MODEL_NAME
    assert request.temperature == 0.5
    assert request.n == 3


def test_chat_detection_to_completion_request_unknown_params():
    chat_request = ChatDetectionRequest(
        messages=[
            DetectionChatMessageParam(role="user", content="How do I search for moose?")
        ],
        detector_params={"moo": 2},
    )
    request = chat_request.to_chat_completion_request(MODEL_NAME)
    # As of vllm >= 0.6.5, extra fields are allowed
    assert type(request) == ChatCompletionRequest


#### General response tests


def test_response_from_completion_response():
    # Simplified response without logprobs since not needed for this method
    choice_0 = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(
            role="assistant",
            content="  moose",
        ),
    )
    choice_1 = ChatCompletionResponseChoice(
        index=1,
        message=ChatMessage(
            role="assistant",
            content="goose\n\n",
        ),
    )
    response = ChatCompletionResponse(
        model=MODEL_NAME,
        choices=[choice_0, choice_1],
        usage=UsageInfo(prompt_tokens=136, total_tokens=140, completion_tokens=4),
    )
    scores = [0.3, 0.7]
    detection_type = "type"
    detection_response = DetectionResponse.from_chat_completion_response(
        response, scores, detection_type
    )
    assert type(detection_response) == DetectionResponse
    detections = detection_response.model_dump()
    assert len(detections) == 2  # 2 choices
    detection_0 = detections[0]
    assert detection_0["detection"] == "moose"
    assert detection_0["detection_type"] == "type"
    assert detection_0["score"] == 0.3
    detection_1 = detections[1]
    assert detection_1["detection"] == "goose"
    assert detection_1["detection_type"] == "type"
    assert detection_1["score"] == 0.7


def test_response_from_completion_response_missing_content():
    choice_0 = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(
            role="assistant",
            content="  moose",
        ),
    )
    choice_1 = ChatCompletionResponseChoice(
        index=1, message=ChatMessage(role="assistant")
    )
    response = ChatCompletionResponse(
        model=MODEL_NAME,
        choices=[choice_0, choice_1],
        usage=UsageInfo(prompt_tokens=136, total_tokens=140, completion_tokens=4),
    )
    scores = [0.3, 0.7]
    detection_type = "type"
    detection_response = DetectionResponse.from_chat_completion_response(
        response, scores, detection_type
    )
    assert type(detection_response) == ErrorResponse
    assert (
        "Choice 1 from chat completion does not have content"
        in detection_response.message
    )
    assert detection_response.code == HTTPStatus.BAD_REQUEST.value


def test_response_from_single_content_detection_response():
    choice = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(
            role="assistant",
            content="  moose",
        ),
    )
    chat_response = ChatCompletionResponse(
        model=MODEL_NAME,
        choices=[choice],
        usage=UsageInfo(prompt_tokens=136, total_tokens=140, completion_tokens=4),
    )
    contents = ["sample sentence"]
    scores = [0.9]
    detection_type = "risk"

    expected_response = ContentsDetectionResponse(
        root=[
            [
                ContentsDetectionResponseObject(
                    start=0,
                    end=len(contents[0]),
                    score=scores[0],
                    text=contents[0],
                    detection="moose",
                    detection_type=detection_type,
                )
            ]
        ]
    )
    detection_response = ContentsDetectionResponse.from_chat_completion_response(
        [(chat_response, scores, detection_type)], contents
    )
    assert isinstance(detection_response, ContentsDetectionResponse)
    assert detection_response == expected_response


def test_response_from_multi_contents_detection_response():
    choice_content_0 = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(
            role="assistant",
            content="  moose",
        ),
    )
    choice_content_1 = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(
            role="assistant",
            content="  goose",
        ),
    )
    chat_response_0 = ChatCompletionResponse(
        model=MODEL_NAME,
        choices=[choice_content_0],
        usage=UsageInfo(prompt_tokens=136, total_tokens=140, completion_tokens=4),
    )
    chat_response_1 = ChatCompletionResponse(
        model=MODEL_NAME,
        choices=[choice_content_1],
        usage=UsageInfo(prompt_tokens=136, total_tokens=140, completion_tokens=4),
    )

    contents = ["sample sentence 1", "sample sentence 2"]
    # scores for each content is a list of scores (for multi-label)
    scores = [[0.9], [0.6]]
    detection_type = "risk"

    content_response_0 = [
        ContentsDetectionResponseObject(
            start=0,
            end=len(contents[0]),
            score=scores[0][0],
            text=contents[0],
            detection="moose",
            detection_type=detection_type,
        )
    ]
    content_response_1 = [
        ContentsDetectionResponseObject(
            start=0,
            end=len(contents[1]),
            score=scores[1][0],
            text=contents[1],
            detection="goose",
            detection_type=detection_type,
        )
    ]
    expected_response = ContentsDetectionResponse(
        root=[content_response_0, content_response_1]
    )
    detection_response = ContentsDetectionResponse.from_chat_completion_response(
        [
            (chat_response_0, scores[0], detection_type),
            (chat_response_1, scores[1], detection_type),
        ],
        contents,
    )
    assert isinstance(detection_response, ContentsDetectionResponse)
    assert detection_response == expected_response


def test_response_from_single_content_detection_missing_content():
    choice_content_0 = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(
            role="assistant",
        ),
    )
    choice_content_1 = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(
            role="assistant",
            content="  goose",
        ),
    )
    chat_response_0 = ChatCompletionResponse(
        model=MODEL_NAME,
        choices=[choice_content_0],
        usage=UsageInfo(prompt_tokens=136, total_tokens=140, completion_tokens=4),
    )
    chat_response_1 = ChatCompletionResponse(
        model=MODEL_NAME,
        choices=[choice_content_1],
        usage=UsageInfo(prompt_tokens=136, total_tokens=140, completion_tokens=4),
    )

    contents = ["sample sentence 1", "sample sentence 2"]
    # scores for each content is a list of scores (for multi-label)
    scores = [[0.9], [0.6]]
    detection_type = "risk"

    detection_response = ContentsDetectionResponse.from_chat_completion_response(
        [
            (chat_response_0, scores[0], detection_type),
            (chat_response_1, scores[1], detection_type),
        ],
        contents,
    )
    assert type(detection_response) == ErrorResponse
    assert (
        "Choice 0 from chat completion does not have content"
        in detection_response.message
    )
    assert detection_response.code == HTTPStatus.BAD_REQUEST.value
