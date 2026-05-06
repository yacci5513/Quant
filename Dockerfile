# syntax=docker/dockerfile:1.7

# =============================================================================
# Quant — Multi-stage Dockerfile (Apple Silicon arm64 target)
#
# Stages:
#   - base       : 공통 베이스 (Python + tzdata + KST)
#   - builder    : uv로 의존성 설치 (캐시 활용)
#   - runtime    : 프로덕션 런타임 (슬림, 비루트, dev 의존성 제외)
#   - dev        : 개발 환경 (dev + jupyter 포함, 코드는 bind mount)
#
# 빌드 예시:
#   docker build --target runtime -t quant:latest .
#   docker build --target dev     -t quant:dev    .
#
# Compose에서는 service별 target 지정.
# =============================================================================

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.4.30


# -----------------------------------------------------------------------------
# base — 공통 베이스: KST 타임존 + locale 안정화
# -----------------------------------------------------------------------------
FROM --platform=linux/arm64 python:${PYTHON_VERSION}-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Seoul \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# tzdata: KIS 장 시간/캔들 시각 정확성 위해 필수
# ca-certificates: KIS/네이버 등 HTTPS 호출
# curl: healthcheck용
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        tzdata \
        ca-certificates \
        curl \
 && ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime \
 && dpkg-reconfigure -f noninteractive tzdata \
 && rm -rf /var/lib/apt/lists/*


# -----------------------------------------------------------------------------
# builder — uv로 의존성 설치 (numba/pyarrow 빌드를 위한 toolchain 포함)
# -----------------------------------------------------------------------------
FROM base AS builder

ARG UV_VERSION=0.4.30
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never

# uv 바이너리 설치
COPY --from=ghcr.io/astral-sh/uv:0.4.30 /uv /uvx /usr/local/bin/

# numba/pyarrow/numpy 등 휠 미제공 시 빌드 위한 toolchain
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1단계: 의존성만 먼저 설치 (코드 변경과 무관하게 캐시)
#   uv.lock 있으면 frozen, 없으면 일반 sync
COPY pyproject.toml ./
COPY uv.loc[k] ./
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --extra dev --extra jupyter; \
    else \
        uv sync --no-install-project --extra dev --extra jupyter; \
    fi

# 2단계: 프로젝트 코드 복사 후 자기 자신 설치
COPY src ./src
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra dev --extra jupyter


# -----------------------------------------------------------------------------
# runtime — 프로덕션 (슬림, dev/jupyter 제외, 비루트)
# -----------------------------------------------------------------------------
FROM base AS runtime

ARG UV_VERSION=0.4.30
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/opt/venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:0.4.30 /uv /uvx /usr/local/bin/

WORKDIR /app

# 의존성만 (dev/jupyter 제외)
COPY pyproject.toml ./
COPY uv.loc[k] ./
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-dev; \
    else \
        uv sync --no-install-project --no-dev; \
    fi

COPY src ./src
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

# 비루트 유저
RUN groupadd -r quant -g 1000 \
 && useradd -r -u 1000 -g quant -d /app -s /bin/bash quant \
 && mkdir -p /app/data /app/logs \
 && chown -R quant:quant /app /opt/venv

USER quant

# 헬스체크: CLI 핑
HEALTHCHECK --interval=60s --timeout=10s --start-period=20s --retries=3 \
    CMD quant hello || exit 1

CMD ["quant", "hello"]


# -----------------------------------------------------------------------------
# dev — 개발 환경 (dev + jupyter 포함, 코드는 compose에서 bind mount)
# -----------------------------------------------------------------------------
FROM builder AS dev

ENV PATH="/opt/venv/bin:$PATH"

# 개발 편의: vim, git, less
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        vim \
        git \
        less \
        procps \
 && rm -rf /var/lib/apt/lists/*

# 비루트 유저 (호스트 UID 1000과 매칭 — Mac 기본값)
RUN groupadd -r quant -g 1000 \
 && useradd -r -u 1000 -g quant -d /app -s /bin/bash quant \
 && mkdir -p /app/data /app/logs /home/quant/.jupyter \
 && chown -R quant:quant /app /opt/venv /home/quant

USER quant

# 코드는 compose volume으로 마운트되므로 여기서 다시 복사하지 않음
CMD ["bash"]
