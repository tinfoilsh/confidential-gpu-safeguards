"""Qwen3Guard-Stream inference server (GPU).

Exposes POST /moderate for content moderation using the
Qwen/Qwen3Guard-Stream-8B model. The model uses custom modeling code
(trust_remote_code=True) with separate classification heads for user
queries and assistant responses.

The model is mounted as a verified model pack (MPK) at boot — no
HuggingFace download or egress required.

The /moderate endpoint takes a list of messages (OpenAI chat format),
tokenizes the full conversation, and returns the risk assessment for the
last token (i.e., the last message). The role of the last message
determines which classification head is used: "user" -> query head,
"assistant" -> response head.
"""

import logging
import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

log = logging.getLogger("qwen3guard")
logging.basicConfig(level=logging.INFO)

MODEL_PATH = os.environ.get("MODEL_PATH", "/tinfoil/models/qwen3guard-stream")

_model = None
_tokenizer = None


def load_model():
    global _model, _tokenizer
    from transformers import AutoModel, AutoTokenizer

    log.info("Loading Qwen3Guard-Stream from %s", MODEL_PATH)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    _model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    _model.to("cuda")
    _model.eval()

    # Warmup — force eager weight loading and CUDA kernel compilation.
    text = _tokenizer.apply_chat_template(
        [{"role": "user", "content": "warmup"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    token_ids = _tokenizer.encode(text, return_tensors="pt").reshape(-1)
    result, stream_state = _model.stream_moderate_from_ids(token_ids, "user", None)
    _model.close_stream(stream_state)
    log.info("Model loaded and warmed up (warmup=%s)", result["risk_level"][-1])


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Qwen3Guard-Stream", lifespan=lifespan)


class Message(BaseModel):
    role: str
    content: str


class ModerateRequest(BaseModel):
    messages: list[Message]


class ModerationResult(BaseModel):
    risk_level: str
    risk_prob: float
    category: str
    category_prob: float


class ModerateResponse(BaseModel):
    role: str
    result: ModerationResult
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/moderate", response_model=ModerateResponse)
async def moderate(req: ModerateRequest):
    if _model is None or _tokenizer is None:
        raise HTTPException(503, "Model not loaded. Check server logs.")
    if not req.messages:
        raise HTTPException(400, "messages must not be empty")

    last_msg = req.messages[-1]
    if last_msg.role not in ("user", "assistant"):
        raise HTTPException(
            400, f"role must be 'user' or 'assistant', got '{last_msg.role}'"
        )

    return await run_in_threadpool(_moderate, req.messages, last_msg.role)


def _moderate(messages: list[Message], role: str) -> dict:
    msgs = [{"role": m.role, "content": m.content} for m in messages]
    text = _tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False
    )
    token_ids = _tokenizer.encode(text, return_tensors="pt").reshape(-1)

    start = time.perf_counter()
    result, stream_state = _model.stream_moderate_from_ids(token_ids, role, None)
    _model.close_stream(stream_state)
    latency_ms = (time.perf_counter() - start) * 1000

    # stream_moderate_from_ids returns predictions for every token; take the
    # last one (the assessment of the final message).
    return {
        "role": role,
        "result": {
            "risk_level": result["risk_level"][-1],
            "risk_prob": result["risk_prob"][-1],
            "category": result["category"][-1],
            "category_prob": result["category_prob"][-1],
        },
        "latency_ms": latency_ms,
    }
