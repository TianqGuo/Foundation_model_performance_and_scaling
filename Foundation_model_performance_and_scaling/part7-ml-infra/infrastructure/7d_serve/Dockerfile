# syntax=docker/dockerfile:1
# Candidate serving overlay. Build and imports must pass before publication/use.
ARG BASE_IMAGE=sclion/part7-training@sha256:b1d8240b348d2a036d311f19fe75784825613607129c1d93526cfa7f530fb8e8
FROM ${BASE_IMAGE}
USER root
WORKDIR /opt/part7
COPY --chown=1000:1000 infrastructure/7d_serve/environment/ /opt/part7/infrastructure/7d_serve/environment/
# Preserve base packages (including flash-attn/CUTLASS). Install the fully resolved
# overlay without dependency re-resolution; pip check verifies the combined stack.
# Do not recursively chown inherited libraries: that duplicates large base layers.
USER 1000:1000
RUN --mount=type=cache,id=part7-7d-uv,target=/home/ray/.cache/uv,uid=1000,gid=1000,sharing=locked \
    UV_LINK_MODE=copy uv pip install --python /opt/part7/.venv-7a/bin/python --no-deps \
        -r infrastructure/7d_serve/environment/requirements.lock \
    && uv pip check --python /opt/part7/.venv-7a/bin/python
COPY --chown=1000:1000 infrastructure/7d_serve/ /opt/part7/infrastructure/7d_serve/
ENTRYPOINT []
CMD ["bash"]
