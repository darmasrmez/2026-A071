FROM rocm/pytorch:${PYTORCH_VERSION}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

WORKDIR /app
COPY ${MODEL_NAME}/${SIZE} /app

CMD ["python", "${MODEL_NAME}-${SIZE}.py"]