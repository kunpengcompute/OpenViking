FROM swr.cn-north-4.myhuaweicloud.com/cloud_boostkit/openeuler-24.03-lts-sp3:latest

# 基础编译/测试工具与 git（checkout 插件依赖）
RUN dnf install -y --setopt=install_weak_deps=False \
        git \
        make \
        cmake \
        gcc \
        gcc-c++ \
        curl \
        ca-certificates \
        python3-devel \
        python3-pip \
    && dnf clean all

# pip 走华为云镜像
RUN printf '[global]\nindex-url = https://repo.huaweicloud.com/repository/pypi/simple\ntimeout = 120\n' > /etc/pip.conf

# uv 走华为云索引（构建依赖由 pyproject 的 build-system.requires 提供）
RUN mkdir -p /etc/uv \
    && printf '[[index]]\nurl = "https://repo.huaweicloud.com/repository/pypi/simple"\ndefault = true\n' > /etc/uv/uv.toml

# Python 构建后端依赖 + uv（与 pyproject.toml 的 build-system.requires 保持一致）
RUN python3 -m pip install -U setuptools wheel "setuptools-scm>=8.0" "maturin>=1.0,<2.0" uv \
    && uv --version

# Rust 1.91.1（华为云 rustup 镜像，按架构选择 rustup-init）
ENV RUSTUP_DIST_SERVER=https://mirrors.huaweicloud.com/rustup \
    RUSTUP_UPDATE_ROOT=https://mirrors.huaweicloud.com/rustup/rustup \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:$PATH

RUN set -eux; \
    case "$(uname -m)" in \
        aarch64|arm64) TARGET=aarch64-unknown-linux-gnu ;; \
        x86_64|amd64) TARGET=x86_64-unknown-linux-gnu ;; \
        *) echo "unsupported arch: $(uname -m)"; exit 1 ;; \
    esac; \
    curl -sSf -o /tmp/rustup-init "$RUSTUP_DIST_SERVER/rustup/dist/$TARGET/rustup-init"; \
    chmod +x /tmp/rustup-init; \
    /tmp/rustup-init -y --default-toolchain 1.91.1 --profile minimal; \
    rm -f /tmp/rustup-init; \
    rustc --version; \
    cargo --version

# cargo 走国内镜像（rsproxy），运行时的 cargo build / maturin 不再直连 crates.io
RUN mkdir -p /usr/local/cargo \
    && printf '[source.crates-io]\nreplace-with = "rsproxy-sparse"\n\n[source.rsproxy-sparse]\nregistry = "sparse+https://rsproxy.cn/index/"\n\n[net]\ngit-fetch-with-cli = true\n' > /usr/local/cargo/config.toml

# ---------------------------------------------------------------------------
# 预热：Python 依赖 + 构建后端
# ---------------------------------------------------------------------------
ENV UV_PROJECT_ENVIRONMENT=/opt/ov-venv \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=600 \
    UV_HTTP_RETRIES=5 \
    UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple \
    UV_DEFAULT_INDEX=https://repo.huaweicloud.com/repository/pypi/simple

WORKDIR /opt/ov-warm-py
COPY requirements-ci.txt ./
RUN uv venv /opt/ov-venv \
    && uv pip install --python /opt/ov-venv/bin/python -r requirements-ci.txt \
    && uv pip install --python /opt/ov-venv/bin/python "setuptools>=61" wheel "setuptools-scm>=8.0" "maturin>=1.0,<2.0" \
    && chmod -R a+rwX /opt/ov-venv /opt/uv-cache \
    && rm -rf /opt/ov-warm-py
WORKDIR /

# 工具链自检
RUN make --version | head -n1 \
    && cmake --version | head -n1 \
    && gcc --version | head -n1 \
    && python3 --version \
    && git --version
