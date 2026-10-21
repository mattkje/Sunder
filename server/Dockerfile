# Build from the repository root (docker compose does this): the vendored RVC
# model code lives in tools/convert/models/rvc and is shared with the Mac tools.
#
# CPU by default. For an NVIDIA GPU build with
#   TORCH_INDEX=https://download.pytorch.org/whl/cu124  SEPARATOR_EXTRA=gpu
# (see docker-compose.yml).
FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG SEPARATOR_EXTRA=cpu

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# torch (and the torchvision/torchaudio some dependencies pull in) first, from
# the matching index, so nothing below installs a mismatched or CUDA build.
RUN pip install --no-cache-dir --index-url "$TORCH_INDEX" torch torchvision torchaudio
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt "audio-separator[$SEPARATOR_EXTRA]"

COPY tools/convert/models/rvc ./rvc
COPY tools/convert/models/seedvc ./seedvc
COPY tools/convert/models/__init__.py ./models/__init__.py
COPY tools/convert/models/bs_roformer ./models/bs_roformer
COPY server/sundervoice ./sundervoice

ENV SUNDERVOICE_DATA=/data SUNDERVOICE_MODELS=/data/models HF_HOME=/data/models/hf SEEDVC_CACHE=/data/models/seedvc PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 8765
CMD ["python", "-m", "sundervoice", "serve"]
