FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/hs-guard
COPY pyproject.toml README.md LICENSE LICENSE-FLA.txt NOTICE ./
COPY src ./src
COPY third_party_licenses ./third_party_licenses
RUN pip install --no-cache-dir . httpx==0.28.1
COPY examples/demo /opt/demo
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
    HF_HOME=/tmp/huggingface TRITON_CACHE_DIR=/tmp/triton HOME=/tmp
USER 10001:10001
EXPOSE 8080
WORKDIR /opt/demo
ENTRYPOINT ["python", "-u", "server.py", "--bundle", "/models/HS-Guard-v10", "--host", "0.0.0.0", "--port", "8080"]
