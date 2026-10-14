# Cloud build only; context is Part 7. See build_image.sh and documents/7C_RUNBOOK.md.
# Supply the existing compatible CUDA 13 development base by registry digest.
ARG BASE_IMAGE
ARG UV_IMAGE
FROM ${UV_IMAGE} AS uv_binary
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
LABEL org.opencontainers.image.base.name="${BASE_IMAGE}"
USER root
WORKDIR /opt/part7
COPY --from=uv_binary /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates tar \
    && rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/part7/.venv-7a \
    PATH=/opt/part7/.venv-7a/bin:$PATH \
    HOME=/home/ray \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    PYTHONPATH=/opt/part7:/opt/part7/.venv-7a/src/verl \
    TOKENIZERS_PARALLELISM=false
COPY infrastructure/7a_verl/environment/ /opt/part7/infrastructure/7a_verl/environment/
# Build-time dependency installation has no GPU detection, model loading or compilation.
RUN command -v uv && command -v git && command -v tar && nvcc --version | grep -q 'release 13\.' \
    && uv python install 3.12.14 \
    && uv venv --managed-python --python 3.12.14 /opt/part7/.venv-7a \
    && mkdir -p /opt/part7/.venv-7a/src \
    && git clone https://github.com/verl-project/verl.git /opt/part7/.venv-7a/src/verl \
    && git -C /opt/part7/.venv-7a/src/verl checkout --detach bec9ef74768dd201881cd4e54cd0385e87caae27 \
    && uv pip sync --python /opt/part7/.venv-7a/bin/python \
       /opt/part7/infrastructure/7a_verl/environment/requirements.lock \
       --index-url https://pypi.org/simple --only-binary torch,vllm,flash-attn,torchvision,torchaudio \
    && uv pip check --python /opt/part7/.venv-7a/bin/python \
    && mkdir -p /opt/part7/image_metadata /home/ray \
    && uv pip freeze --python /opt/part7/.venv-7a/bin/python > /opt/part7/image_metadata/installed.txt \
    && nvcc --version > /opt/part7/image_metadata/cuda.txt
COPY infrastructure/ /opt/part7/infrastructure/
COPY workloads/ /opt/part7/workloads/
RUN mkdir -p /opt/part7/assets /opt/part7/results \
    && chown -R 1000:1000 /opt/part7 /home/ray /opt/uv-python
# Numeric identity avoids assumptions about usernames in the custom base image.
USER 1000:1000
# KubeRay supplies the startup command; discard custom-base service boot hooks.
ENTRYPOINT []
CMD ["bash"]
