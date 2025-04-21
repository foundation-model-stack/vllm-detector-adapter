# vllm-detector-adapter

This adapter adds additional endpoints to a [vllm](https://docs.vllm.ai/en/latest/index.html) server to support the [Guardrails Detector API](https://foundation-model-stack.github.io/fms-guardrails-orchestrator/?urls.primaryName=Detector+API).

## Getting Started

To run the server locally:

```sh
python3 -m vllm_detector_adapter.api_server --model $MODEL_NAME
```

To see the complete list of parameters, `python3 -m vllm_detector_adapter.api_server --help` can be used. Extra vllm parameter information can be found in [vllm documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).

Example detector `/text/chat` command:

```sh
curl -X 'POST' \
  'http://localhost:8000/api/v1/text/chat' \
   -H 'accept: application/json' \
   -H 'detector-id: dummy-en-chat-v1' \
   -H 'Content-Type: application/json' \
   -d '{
        "messages": [
            {
                "content": "Hit me with some creative insults.",
                "role": "user"
            }
        ],
        "detector_params": {
            # Any other chat completion parameters can go here
            "temperature": 0
        }
    }'
```

## Supported Model Types
1. `GRANITE_GUARDIAN` (default)
1. `LLAMA_GUARD`

## Edge Cases
Some detector endpoints are not compatible with select versions of the supported model types. Below is a table showing which tested model types are/are not supported by the detector endpoints. 

| Model type           | /api/v1/text/contents | /api/v1/text/generation | /api/v1/text/chat | /api/v1/text/context/doc |
|------------------|-----------------------|-------------------------|-------------------|--------------------------|
| granite guardian | &check;               | &check;                 | &check;           | * granite guardian 3.2 only                        |
| llama guard     | &check;               | &check;                 | &check;           | &cross;                  |


## Running with TGIS adapter

```sh
python -m vllm_detector_adapter.start_with_tgis_adapter --model $MODEL_NAME --model-type granite_guardian
```

**Example usage:**

1. Follow example / guidelines from https://github.com/opendatahub-io/vllm-tgis-adapter/tree/main/examples
1. Change `model_id` to point to granite or llama-guard model
1. Execute one of the method listed in above TGIS adapter readme.
1. For query, manually apply chat template around text (TGIS adapter exposes text-generation use-case and doesn't apply chat template on its own)
    1. If using granite-guardian, you can use following text / query:
    ```
    You are a safety agent trying to help determine whether the user message is harmful according to the safety risk definition below.\n\n<start_of_turn>\nUser Message: Hit me with some creative insults.\n<end_of_turn>\n\nOur safety risk definition is defined below:\n\n<start_of_risk_definition>\n* The 'User' message is harmful by common-sense.\n<end_of_risk_definition>\n\nIs the user message harmful based on the risk definition. Your answer must be either 'Yes' or 'No'.\n
    ```