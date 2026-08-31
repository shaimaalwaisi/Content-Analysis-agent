# The agent, its CLI and its Streamlit app in one image.
#
# Build once, run either way:
#   docker compose up app                          # the UI on :8501
#   docker compose run --rm cli tag --input data/test --limit 10
#
# Only tagging needs ANTHROPIC_API_KEY; the test suite does not.
FROM python:3.11-slim

# Faster, quieter, and no .pyc files owned by root in a mounted volume.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so editing the source does not reinstall them. Every
# wheel here is manylinux, so no compiler is needed and the image stays slim.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# The databases live on a volume, not in the image: a container that is
# thrown away must not take the run history with it.
ENV RESULTS_DB=/data/results.sqlite3 \
    MEMORY_DB=/data/.agent_memory.sqlite3

# Streamlit writes to $HOME; a non-root user needs one it owns. /data is left
# writable by any uid, so `user:` in the compose file can be your own.
RUN useradd --create-home --uid 10001 agent \
    && mkdir -p /data /app/results \
    && chown -R agent:agent /data /app \
    && chmod -R a+rwX /data
USER agent

EXPOSE 8501

# Streamlit's own readiness endpoint, asked with the interpreter that is
# already here rather than an apt-get for curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request as u; \
        u.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
