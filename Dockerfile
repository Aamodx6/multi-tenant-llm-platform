FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN groupadd --gid 10001 appgroup && \
    useradd --uid 10001 --gid appgroup --no-create-home --shell /usr/sbin/nologin appuser

COPY pyproject.toml ./
COPY catalog ./catalog
COPY gateway ./gateway
COPY controller ./controller
COPY observability ./observability
RUN pip install --no-cache-dir .

USER 10001:10001
EXPOSE 8080

CMD ["uvicorn", "gateway.server:app", "--host", "0.0.0.0", "--port", "8080"]
