# Qwen3Guard-Stream server image.
# Built on the vLLM base image for Blackwell-compatible PyTorch/CUDA.
# ENTRYPOINT [] clears vLLM's entrypoint so we run uvicorn directly.
FROM vllm/vllm-openai:v0.23.0-ubuntu2404@sha256:662e4975c5c9947f8723f4d8f438145971361a480a2ade1919bb9462a9f24088

ENTRYPOINT []

COPY server.py /app/server.py

WORKDIR /app

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8003"]
