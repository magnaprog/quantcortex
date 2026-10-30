# quantcortex application image (Python 3.11).
FROM python:3.11-slim-bookworm@sha256:2e32f7d302adc1c37428355c1e646897c0c53f4fd60b6a551245fb90ee129f91

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/cache \
    XDG_STATE_HOME=/tmp/state

WORKDIR /app

# Runtime library for the optional LightGBM/XGBoost backends.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install from the runtime lock before copying source to reuse this build layer.
COPY requirements/runtime.lock requirements/runtime.lock
RUN python -m pip install --no-cache-dir -r requirements/runtime.lock

RUN groupadd --gid 10001 quantcortex \
    && useradd --uid 10001 --gid quantcortex --create-home quantcortex

COPY --chown=quantcortex:quantcortex README.md LICENSE pyproject.toml poetry.lock ./
COPY --chown=quantcortex:quantcortex requirements/ requirements/
COPY --chown=quantcortex:quantcortex quantcortex/ quantcortex/
COPY --chown=quantcortex:quantcortex scripts/ scripts/
USER quantcortex

CMD ["python", "-c", "import quantcortex.alpha, quantcortex.backtest, quantcortex.data, quantcortex.execution, quantcortex.portfolio, quantcortex.strategies; print('quantcortex container ready')"]
