# syntax=docker/dockerfile:1
# Official serving GPU base; never layer a new Torch ABI over the training image.
ARG BASE_IMAGE=vllm/vllm-openai@sha256:5b03c7de2951cda2d5fd087457902625eabbe9188478e439561ef34f325b0fd5
FROM ${BASE_IMAGE}
USER root
WORKDIR /opt/part7
# Clear the upstream NCCL override: this image must follow its explicit project lock.
ENV UV_OVERRIDE="" UV_INDEX_STRATEGY=first-index \
    HOME=/home/ray PYTHONPATH=/opt/part7 TOKENIZERS_PARALLELISM=false
RUN apt-get update && apt-get install -y --no-install-recommends git wget tar \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && if ! getent group 1000 >/dev/null; then groupadd --gid 1000 ray; fi \
    && if ! getent passwd 1000 >/dev/null; then \
         useradd --uid 1000 --gid 1000 --home-dir /home/ray --no-create-home --shell /bin/bash ray; fi \
    && mkdir -p /home/ray /opt/part7/assets /opt/part7/results \
    && chown 1000:1000 /home/ray /opt/part7 /opt/part7/assets /opt/part7/results
COPY infrastructure/7d_serve/config.py /opt/part7/infrastructure/7d_serve/config.py
# Fail on an unexpected prebuilt Torch/vLLM ABI instead of silently replacing it.
RUN python -c "import importlib,importlib.metadata as md,sys; c=importlib.import_module('infrastructure.7d_serve.config'); assert sys.version_info[:2]==(3,12); assert all(md.version(k).split('+')[0]==c.CORE_VERSIONS[k] for k in ('torch','vllm')), 'Official image GPU versions differ from candidate'"
COPY infrastructure/7d_serve/environment/ /opt/part7/infrastructure/7d_serve/environment/
# Sync only inside the derived image; host/training environments remain unchanged.
RUN --mount=type=cache,id=part7-7d-modern-uv,target=/root/.cache/uv,sharing=locked \
    UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy uv pip sync --python /usr/bin/python3.12 --break-system-packages \
        infrastructure/7d_serve/environment/requirements.lock \
    && uv pip check --python /usr/bin/python3.12
COPY --chown=1000:1000 infrastructure/ /opt/part7/infrastructure/
COPY --chown=1000:1000 workloads/ /opt/part7/workloads/
USER 1000:1000
# Complete model-free validation before expensive export. No models/clients/runtime.
RUN python -c "import importlib,json; print(json.dumps(importlib.import_module('infrastructure.7d_serve.runtime_check').check_runtime()))"
ENTRYPOINT []
CMD ["bash"]
