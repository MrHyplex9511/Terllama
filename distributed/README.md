# Terllama Distributed Inference

Native **layer-sharding** across multiple devices. A model's transformer layers
are split into contiguous half-open shards `[start, end)`, each owned by one
`terllama-worker` process. A `terllama-cluster` coordinator fronts them with an
OpenAI-compatible HTTP API and pipelines hidden states between shards per token.

Pipeline layout:

```
rank 0 (is_first):  embedding + layers[0, k)        token id ──► hidden
rank i (middle):    layers[i_k, (i+1)_k)             hidden  ──► hidden
rank N (is_last):   layers + final_norm + lm_head    hidden  ──► logits
```

The workers run the **exact same `transformer_block` chain** as single-device
inference (no core engine files are modified), so a correct cluster produces
bit-identical logits/output to `terllama serve` with the same sampling.

## Build

Additive CMake targets — `BUILD_DISTRIBUTED=ON` by default; the single-device
`terllama` binary is untouched.

```sh
cmake -S . -B build-release && cmake --build build-release -j4
# produces: build-release/terllama-worker, build-release/terllama-cluster
```

## Quickstart (2 local "fake nodes", one model dir)

```sh
# Node 0: first 15 of 30 layers
build-release/terllama-worker --listen 127.0.0.1:9100 \
    --model ~/.terllama/models/terllama-convert-test --shard 0,15

# Node 1: last 15 layers (owns final_norm + lm_head)
build-release/terllama-worker --listen 127.0.0.1:9101 \
    --model ~/.terllama/models/terllama-convert-test --shard 15,30

# Coordinator (OpenAI API on :8375)
build-release/terllama-cluster --workers 127.0.0.1:9100,127.0.0.1:9101 \
    --model ~/.terllama/models/terllama-convert-test --port 8375

curl http://127.0.0.1:8375/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Once upon a time,","max_tokens":32,"temperature":0.7}'
```

Workers can also start empty (no `--model`) and wait for the coordinator's
`POST /load` — useful when the coordinator computes shard boundaries. Shards
are allocated **proportionally to each worker's available RAM** (largest-
remainder), queried at startup via `/health`.

## CLI

`terllama-worker`:

| flag | meaning |
|---|---|
| `--listen HOST:PORT` | address to serve on (default `127.0.0.1:9100`) |
| `--model PATH` | model dir (`model_extra.bin` + `model_decomposed.bin`) or `.gguf`. Omit to wait for `POST /load`. |
| `--shard START,END` | half-open layer range `[START, END)`. Requires `--model`; defaults to full model. |

`terllama-cluster`:

| flag | meaning |
|---|---|
| `--workers h:port[,h:port...]` | worker endpoints, in pipeline order (required) |
| `--model PATH` | model dir or `.gguf` (required — also used to read config + tokenizer) |
| `--port N` | OpenAI API port (default `8375`) |

## API

- `GET /v1/models` — `{data:[{id: <model-basename>, ...}]}`
- `POST /v1/completions` — OpenAI shape; SSE streaming (`stream: true`, `data: [DONE]`)
- `POST /v1/chat/completions` — OpenAI shape; SSE delta chunks
- `GET  /health` (workers) — RAM + shard info
- `POST /load` (workers) — coordinator-driven model load
- `POST /forward` (workers) — binary token/hidden RPC, FP32 payloads
- `POST /reset` (workers) — clears KV cache (coordinator sends it per request)

On worker failure the coordinator returns `502` with
`{"error":{"type":"cluster_error","message":"worker failed: ..."}}`.

## Wire format (`distributed/protocol.h`)

- Control plane: JSON (`/health`, `/load`, `/reset`).
- Tensor plane: binary octet-stream. Forward request =
  `[u32 magic 0x544C4652][u32 seq_pos][u32 input_kind][u32 count][f32 data]`,
  `input_kind 0` = token id (rank 0), `1` = hidden state floats. Response =
  `[u32 magic][u32 count][f32 data]`.

## Notes & known v1 limitations

- KV cache is **local to each worker**; `/reset` is sent per request, so
  concurrent streams are serialized by the coordinator (one generation at a
  time).
- No auto-discovery: the coordinator needs an explicit worker list, and every
  worker must have the model files locally.
- No weight streaming; shards are loaded from each worker's own disk.
- GGUF models load fully on each worker then filter (memory-heavy v1 path);
  ALS (`.bin`) shards skip out-of-range records while reading.
- `temperature < 0.01` uses argmax (deterministic) — matches single-device.
- Workers listen on plaintext HTTP; put them on a trusted network.
