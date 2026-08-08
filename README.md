# confidential-gpu-safeguards

Four GPU-based guard models colocated on a single GPU (B200/H200 class)
behind a Go reverse proxy router.

## Architecture

```
                 ┌─────────┐
  shim ──8080──► │ router  │
                 └────┬────┘
          ┌───────────┼────────────┬──────────────┐
          ▼           ▼            ▼              ▼
   /lyraixguard  /aprielguard  /qwen3guard   /shieldstral
     :8001         :8002         :8003          :8004
   vLLM (Qwen3)  vLLM (Mistral)  transformers vLLM (Ministral)
   4B            8B              8B           3B
```

The router does pure path-based proxying — each backend exposes its native API:

| Path prefix      | Backend                    | Native endpoint        |
| ---------------- | -------------------------- | ---------------------- |
| `/lyraixguard/*` | LyraixGuard (vLLM)         | `/v1/chat/completions` |
| `/aprielguard/*` | AprielGuard (vLLM)         | `/v1/chat/completions` |
| `/qwen3guard/*`  | Qwen3Guard-Stream (custom) | `/moderate`            |
| `/shieldstral/*` | Shieldstral (vLLM)         | `/v1/chat/completions` |
| `/health`        | router                     | —                      |
| `/models`        | router                     | —                      |

## Models

| Model             | HF repo                        | Architecture                          | Server                          |
| ----------------- | ------------------------------ | ------------------------------------- | ------------------------------- |
| LyraixGuard       | `Lyraix-AI/LyraixGuard-v0`     | Qwen3ForCausalLM (4B)                 | vLLM (stock image)              |
| AprielGuard       | `ServiceNow-AI/AprielGuard`    | MistralForCausalLM (8B)               | vLLM (stock image)              |
| Qwen3Guard-Stream | `Qwen/Qwen3Guard-Stream-8B`    | Qwen3ForGuardModel (8B)               | custom (`qwen3guard/server.py`) |
| Shieldstral       | `mistralai/Shieldstral-1.0-3B` | Mistral3ForConditionalGeneration (3B) | vLLM (stock image)              |

LyraixGuard and AprielGuard are standard causal LMs served by vLLM's OpenAI-compatible API. Qwen3Guard-Stream is a classification model with custom `modeling_qwen3_guard.py` (loaded via `trust_remote_code=True`); it's not compatible with vLLM, so it runs in a custom FastAPI server on the vLLM base image. Shieldstral is Mistral's policy-adaptive multimodal safety classifier (Ministral-3-3B + Pixtral vision encoder), also served by stock vLLM; its HF repo ships weights in both HF and consolidated formats, so the container passes `--tokenizer-mode/--config-format/--load-format mistral` to load the consolidated ones unambiguously (same as voxtral).

## Shieldstral usage

Shieldstral answers a single yes/no safety question per call via standard chat completions. The system prompt is fixed and the user message carries `<Instruct>` (evaluation context), `<Query>` (a yes/no policy question), and `<Document>` (the content). Call with `max_tokens=1` and `logprobs=true, top_logprobs=20`, then renormalize the `yes`/`no` token probabilities into a safety score (threshold 0.5). See the [model card](https://huggingface.co/mistralai/Shieldstral-1.0-3B) for the exact prompts.

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

## Blackwell / flashinfer

The three vLLM containers get a tmpfs at `flashinfer_cubin/cubins` (the read-only rootfs fix) and are on the `nvidia` egress network for cubin downloads: FlashInfer fetches Blackwell TRT-LLM cubins from `edge.urm.nvidia.com` at startup. Needed on Blackwell hosts (B200); vestigial but harmless on Hopper. Qwen3Guard-Stream runs on the vLLM base image (for Blackwell-compatible PyTorch/CUDA) but doesn't use flashinfer.

## Development

```sh
# Build and test the router
cd router && go vet ./... && go build ./...

# Build the qwen3guard image locally
cd qwen3guard && docker build -t qwen3guard .
```

## GPU/host-dependent config

The current config targets a 1×B200 (TDX) host; it should also run unchanged on H200. It previously ran on a B300, which needed extra workarounds (reverted since):

| Config                         | B200/H200 (current)                  | On B300                                                  |
| ------------------------------ | ------------------------------------ | -------------------------------------------------------- |
| `cpus` / `memory`              | 32 / 262144 (`large_1d_new`)         | 32 / 524288 (`extra_large_1d_b300_new`)                  |
| vLLM image                     | `v0.26.0`, default attention backend | `v0.25.1` + `VLLM_USE_V2_MODEL_RUNNER=0` + `TRITON_ATTN` |
| cubins tmpfs + `nvidia` egress | on all vLLM guards                   | Keep                                                     |

- **VM shape**: `cpus`/`memory` must exactly match a published shape in [hardware-measurements](https://github.com/tinfoilsh/hardware-measurements), or clients fail attestation with "no matching hardware platform found". 32cpu/512G exists for both B300 (`extra_large_1d_b300_new`) and non-B300 (`extra_large_1d_new`) hosts. The stack peaks at ~55G RAM during model load, so 32 / 262144 (`large_1d_new`) would also fit — don't use the 64G medium shape.
- **vLLM version/backend**: vLLM v0.23 silently produces corrupted output on B300 (sm_103) regardless of attention backend; v0.22 crashes outright. v0.25.1 V1 + Triton is the org-validated B300 combo (same as gemma4); re-apply it if moving back to B300. On B200/H200 stock v0.26.0 with the default attention backend is fine (and Shieldstral's model card requires vLLM >= 0.26.0).

GPU-independent (don't touch when changing hardware): the qwen3guard `transformers==4.55.0` pin (the model's `trust_remote_code` code breaks under the transformers v5 bundled in the vLLM base image), the router, and the shim config.
