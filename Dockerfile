# Official lightweight Python 3.13 image
FROM python:3.13-slim

# Install system dependencies (curl for healthchecks, libgomp1 for LightGBM OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv from official binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install dependencies using uv layer caching
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy application source code, tests, and datasets
COPY src/ ./src/
COPY tests/ ./tests/
COPY data/ ./data/
COPY app.py main.py README.md GEMINI.md ./

# Final sync to register the package and dependencies
RUN uv sync --frozen

# Expose default Streamlit port
EXPOSE 8501

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command: launch interactive Streamlit web dashboard
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
