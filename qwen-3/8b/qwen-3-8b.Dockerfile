FROM rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

COPY requirements-qwen3-8b.txt /app/requirements-qwen3-8b.txt
# bitsandbytes may require a ROCm-specific wheel; fall back to finetune_peft_bf16.py if import fails.
RUN uv pip install --system --no-cache -r /app/requirements-qwen3-8b.txt

COPY . /app

# Default: BF16 LoRA. Configuration: environment variables (see docker-compose `environment:`).
# Int8: override compose `command` to `python finetune_peft_int8.py`
# QLoRA: `command` → `python finetune_peft_qlora.py` and set TRAIN_OPTIM=adamw_torch if needed
# Multi-GPU: `accelerate launch --num_processes 2 finetune_peft_bf16.py` (pass same env as compose)
CMD ["python", "finetune_peft_bf16.py"]