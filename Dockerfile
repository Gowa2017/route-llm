FROM python:3.12-slim

WORKDIR /app

ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV PIP_TRUSTED_HOST=mirrors.aliyun.com

COPY pyproject.toml uv.lock ./
RUN pip install uv --quiet && uv sync --no-dev --quiet

COPY route_llm/ route_llm/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "route_llm.main:app", "--host", "0.0.0.0", "--port", "8000"]
