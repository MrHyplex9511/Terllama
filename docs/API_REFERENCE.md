# API Reference

Terllama exposes an OpenAI-compatible HTTP API. All endpoints return JSON. The server listens on the configured port (default `8375`).

Base URL: `http://localhost:<port>`

## Authentication

If the server is started with `--api-key <key>`, all requests must include:

```
Authorization: Bearer <key>
```

Without `--api-key`, authentication is disabled (open access).

Example:

```bash
# No auth (default)
curl http://localhost:8375/v1/models

# With API key
curl -H "Authorization: Bearer sk-terllama-abc123" http://localhost:8375/v1/models
```

## Endpoints

---

### `GET /v1/models`

List available models.

**Response `200`:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "default",
      "object": "model",
      "created": 1700000000,
      "owned_by": "terllama"
    }
  ]
}
```

**Notes:**
- Terllama loads one model at a time. `id` is always `"default"`.
- `created` is a Unix timestamp.

---

### `POST /v1/chat/completions`

Chat completions with streaming and non-streaming modes.

**Request:**

```json
{
  "model": "default",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 256,
  "stream": false
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | `string` | `"default"` | Model ID (only `"default"` supported) |
| `messages` | `array` | required | Array of message objects with `role` and `content` |
| `temperature` | `float` | `0.7` | Sampling temperature (0.0 = greedy, 2.0 = max entropy) |
| `max_tokens` | `int` | `256` | Maximum tokens to generate |
| `stream` | `bool` | `false` | If true, response is SSE stream |

**Non-streaming response `200`:**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "default",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop",
      "logprobs": null
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  }
}
```

**Streaming response:**

Content-Type: `text/event-stream`

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"default","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"default","choices":[{"index":0,"delta":{"content":"!"}}]}

data: [DONE]
```

Each SSE event contains a `data:` line with JSON. The stream ends with `data: [DONE]`.

**Notes:**
- `finish_reason` is `"stop"` if the model generates an EOS token, `"length"` if `max_tokens` is reached.
- System prompt is prepended with `[INST]` / `[/INST]` markers.
- Conversation history is formatted as: `[INST] system [/INST] [INST] user [/INST] assistant response ...`

---

### `POST /v1/completions`

Text completions (no chat template).

**Request:**

```json
{
  "model": "default",
  "prompt": "The capital of France is",
  "temperature": 0.7,
  "max_tokens": 100,
  "stream": false
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | `string` | `"default"` | Model ID |
| `prompt` | `string` | required | Input text |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `max_tokens` | `int` | `256` | Maximum tokens to generate |
| `stream` | `bool` | `false` | If true, response is SSE stream |

**Non-streaming response `200`:**

```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1700000000,
  "model": "default",
  "choices": [
    {
      "index": 0,
      "text": " Paris.",
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 3,
    "total_tokens": 8
  }
}
```

**Streaming response:**

```
data: {"id":"cmpl-...","object":"text_completion","created":...,"model":"default","choices":[{"index":0,"text":" Paris","logprobs":null,"finish_reason":null}]}

data: {"id":"cmpl-...","object":"text_completion","created":...,"model":"default","choices":[{"index":0,"text":".","logprobs":null,"finish_reason":null}]}

data: [DONE]
```

---

### `GET /health`

Health check endpoint.

**Response `200`:**

```json
{
  "status": "ok",
  "model": "default"
}
```

**Response `503` (model not loaded):**

```json
{
  "status": "not_loaded",
  "model": "default"
}
```

---

## Error Codes

| Status | Code | Meaning |
|---|---|---|
| `400` | `invalid_request` | Missing or malformed request fields (e.g., no `messages`, no `prompt`) |
| `401` | `unauthorized` | Missing or invalid `Authorization` header (when `--api-key` is set) |
| `404` | `not_found` | Endpoint does not exist |
| `500` | `server_error` | Internal error (tokenization failure, model inference crash) |
| `500` | `tokenization_error` | Python tokenizer helper failed |
| `503` | `model_not_loaded` | Model has not been loaded (run `./terllama pull` first) |

**Error response schema:**

```json
{
  "error": {
    "message": "Description of what went wrong",
    "type": "invalid_request"
  }
}
```

## CORS

All endpoints return CORS headers:

```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: Content-Type, Authorization
Access-Control-Max-Age: 86400
```

Preflight `OPTIONS` requests return `204 No Content`.

## Example curl Commands

```bash
# List models
curl http://localhost:8375/v1/models

# Chat completion (non-streaming)
curl -X POST http://localhost:8375/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 50
  }'

# Chat completion (streaming)
curl -X POST http://localhost:8375/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'

# Text completion
curl -X POST http://localhost:8375/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "temperature": 0.8
  }'

# Health check
curl http://localhost:8375/health

# With API key
curl -H "Authorization: Bearer sk-terllama-mykey" \
  http://localhost:8375/v1/models
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TERLLAMA_PORT` | `8375` | Server port |
| `TERLLAMA_MODEL_DIR` | `.` | Model file directory |
| `TERLLAMA_ARCH` | auto | Force CPU arch (`scalar`, `avx2`, `neon`, etc.) |
| `TERLLAMA_API_KEY` | _(none)_ | Require Bearer auth for all requests |
