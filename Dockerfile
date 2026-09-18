# Container image for DermpathGrader. See docs/DEPLOYMENT.md.
FROM python:3.11-slim

# libopenslide0 is only needed by extract_tiles.py, for reading .svs
# whole-slide images. Drop it (and openslide-python below) if tiles are
# exported by ImageScope or QuPath instead.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenslide0 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Case logs must outlive the container. Mount a volume here, or a run's
# record disappears when it stops.
VOLUME ["/app/analysis_logs"]

EXPOSE 8501

# ANTHROPIC_API_KEY is supplied at run time by the host's secrets
# manager. Never bake it into the image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
