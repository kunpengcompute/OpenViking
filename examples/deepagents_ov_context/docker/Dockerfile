# syntax=docker/dockerfile:1.9
ARG EULER_IMAGE=openeuler/openeuler:24.03-lts-sp3
FROM roundcube/roundcubemail:1.6.11-apache AS webmail_assets
FROM ${EULER_IMAGE} AS builder
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG BUILD_JOBS=4
ARG OPENVIKING_VERSION=0.4.1
RUN dnf install -y gcc gcc-c++ make cmake pkgconf openssl-devel libffi-devel \
      zlib-devel bzip2-devel xz-devel git curl ca-certificates tar gzip xz unzip \
    && dnf clean all
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy \
    PATH=/opt/openviking/.venv/bin:/root/.cargo/bin:$PATH \
    OV_REQUIRE_RAGFS_BUILD=1 OV_SKIP_STUDIO_BUILD=1 \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPENVIKING=${OPENVIKING_VERSION} \
    CARGO_BUILD_JOBS=${BUILD_JOBS} CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    LITELLM_LOCAL_MODEL_COST_MAP=true
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain 1.91.1
WORKDIR /opt/openviking
COPY README.md LICENSE pyproject.toml setup.py MANIFEST.in Cargo.toml Cargo.lock ./
COPY build_support/ ./build_support/
COPY crates/ ./crates/
COPY src/ ./src/
COPY third_party/ ./third_party/
COPY openviking/ ./openviking/
COPY openviking_cli/ ./openviking_cli/
COPY bot/ ./bot/
ARG UV_HTTP_TIMEOUT=300
ENV UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT}
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.11 && uv venv --seed --python 3.11 .venv
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    uv pip install -e '.[deepagents]' 'langchain-openai>=1,<2' \
    && uv pip check
COPY examples/ ./examples/
COPY tests/ ./tests/
COPY benchmark/ ./benchmark/

FROM ${EULER_IMAGE} AS ov
RUN dnf install -y ca-certificates curl libstdc++ libgcc openssl libffi zlib \
      bzip2-libs xz-libs sqlite-libs findutils \
    && dnf clean all
COPY --from=builder /opt/python /opt/python
COPY --from=builder /opt/openviking/.venv /opt/openviking/.venv
COPY --from=builder /opt/openviking/openviking /opt/openviking/openviking
COPY --from=builder /opt/openviking/openviking_cli /opt/openviking/openviking_cli
COPY --from=builder /opt/openviking/bot /opt/openviking/bot
COPY --from=builder /opt/openviking/examples /opt/openviking/examples
COPY --from=builder /opt/openviking/tests /opt/openviking/tests
COPY --from=builder /opt/openviking/benchmark /opt/openviking/benchmark
ENV PATH=/opt/openviking/.venv/bin:$PATH \
    PYTHONPATH=/opt/openviking UV_PYTHON_INSTALL_DIR=/opt/python \
    LITELLM_LOCAL_MODEL_COST_MAP=true \
    OV_CONFIG_DIR=/config OV_DATA_DIR=/data HOME=/tmp/ov-home \
    OV_CHAT_TOOLS=/opt/ov-tools
WORKDIR /opt/openviking
RUN python -m examples.deepagents_ov_context.scripts.check install
ENTRYPOINT ["python", "-m", "examples.deepagents_ov_context.docker.runtime"]
CMD ["ov"]

FROM ov AS chat
ARG PLAYWRIGHT_DOWNLOAD_HOST=https://cdn.playwright.dev
RUN dnf --disablerepo='*source*' --disablerepo='*debuginfo*' install -y \
      git tar gzip xz unzip openssl nss nspr atk at-spi2-atk \
      cups-libs libdrm libXcomposite libXdamage libXfixes libXrandr \
      libxkbcommon mesa-libgbm alsa-lib pango dovecot \
      php-cli php-pdo php-mbstring php-xml php-intl php-gd php-pecl-zip \
    && dnf clean all
COPY --from=webmail_assets /usr/src/roundcubemail /opt/ov-tools/roundcube
ENV PATH=/opt/ov-tools/bin:/opt/ov-tools/node/bin:/opt/openviking/.venv/bin:$PATH \
    PLAYWRIGHT_BROWSERS_PATH=/opt/ov-tools/playwright \
    NODE_PATH=/opt/ov-tools/node_modules
RUN bash examples/deepagents_ov_context/scripts/deploy.sh install-browser
RUN bash examples/deepagents_ov_context/scripts/deploy.sh install-pptx-renderer
RUN bash examples/deepagents_ov_context/scripts/deploy.sh install-pptx-skill
RUN bash examples/deepagents_ov_context/scripts/deploy.sh install-office-skills \
    && test ! -L benchmark/office \
    && python -m benchmark.office.admin.datasets check \
    && php -r 'foreach (["pdo_sqlite", "mbstring", "dom", "intl", "openssl", "fileinfo"] as $name) { if (!extension_loaded($name)) { fwrite(STDERR, "Missing PHP extension: $name\n"); exit(1); } }' \
    && dovecot --version \
    && python -m pip install pytest pytest-asyncio \
    && python -m pip check \
    && python -m pip freeze > /opt/openviking/image-requirements.txt \
    && chmod -R a+rX /opt/ov-tools /opt/python /opt/openviking \
    && rm -rf /tmp/ov-home \
    && dnf clean all
CMD ["chat"]
