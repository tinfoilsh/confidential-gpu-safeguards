# confidential-gpu-safeguards

Three GPU-based guard models colocated on a single B300 behind a Go reverse
proxy router.

## Architecture

```
                 ┌─────────┐
  shim ──8080──► │ router  │
                 └────┬────┘
          ┌───────────┼───────────┐
          ▼           ▼           ▼
   /lyraixguard  /aprielguard  /qwen3guard
     :8001         :8002         :8003
   vLLM (Qwen3)  vLLM (Mistral) custom transformers
   4B            8B             8B
```

The router does pure path-based proxying — each backend exposes its native API:

| Path prefix      | Backend                    | Native endpoint        |
| ---------------- | -------------------------- | ---------------------- |
| `/lyraixguard/*` | LyraixGuard (vLLM)         | `/v1/chat/completions` |
| `/aprielguard/*` | AprielGuard (vLLM)         | `/v1/chat/completions` |
| `/qwen3guard/*`  | Qwen3Guard-Stream (custom) | `/moderate`            |
| `/health`        | router                     | —                      |
| `/models`        | router                     | —                      |

## Models

| Model             | HF repo                     | Architecture            | Server                          |
| ----------------- | --------------------------- | ----------------------- | ------------------------------- |
| LyraixGuard       | `Lyraix-AI/LyraixGuard-v0`  | Qwen3ForCausalLM (4B)   | vLLM (stock image)              |
| AprielGuard       | `ServiceNow-AI/AprielGuard` | MistralForCausalLM (8B) | vLLM (stock image)              |
| Qwen3Guard-Stream | `Qwen/Qwen3Guard-Stream-8B` | Qwen3ForGuardModel (8B) | custom (`qwen3guard/server.py`) |

LyraixGuard and AprielGuard are standard causal LMs served by vLLM's OpenAI-compatible API. Qwen3Guard-Stream is a classification model with custom `modeling_qwen3_guard.py` (loaded via `trust_remote_code=True`); it's not compatible with vLLM, so it runs in a custom FastAPI server on the vLLM base image.

## Qwen3Guard `/moderate` API

```
POST /qwen3guard/moderate
{
  "messages": [{"role": "user", "content": "..."}]
}

→ 200
{
  "role": "user",
  "result": {
    "risk_level": "Safe",
    "risk_prob": 0.99,
    "category": "Jailbreak",
    "category_prob": 0.01
  },
  "latency_ms": 12.3
}
```

The `role` of the last message determines which classification head is used: `"user"` → query head, `"assistant"` → response head. Risk levels: `Safe`, `Unsafe`, `Controversial`.

## B300 / flashinfer

The two vLLM containers get a tmpfs at `flashinfer_cubin/cubins` (the read-only rootfs fix) and are on the `nvidia` egress network for cubin downloads. Qwen3Guard-Stream runs on the vLLM base image (for Blackwell-compatible PyTorch/CUDA) but doesn't use flashinfer.

## Development

```sh
# Build and test the router
cd router && go vet ./... && go build ./...

# Build the qwen3guard image locally
cd qwen3guard && docker build -t qwen3guard .
```

All MPK values and image digests in `tinfoil-config.yml` are placeholders — populated when models are registered and images are built via the release workflow.
