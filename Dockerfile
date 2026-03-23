FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    SHELL=/bin/bash

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apache2-utils \
        ffmpeg \
        git \
        git-lfs \
        nginx \
        rsync \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git lfs install

WORKDIR /opt/sparkvsr_template

COPY runtime-requirements.txt /tmp/runtime-requirements.txt
RUN python3 -m pip install --upgrade pip wheel && \
    python3 -m pip install --no-cache-dir -r /tmp/runtime-requirements.txt && \
    rm -rf /root/.cache/pip

COPY . /opt/sparkvsr_template

RUN chmod +x /opt/sparkvsr_template/start-sparkvsr.sh /opt/sparkvsr_template/restart-sparkvsr.sh && \
    ln -sf /opt/sparkvsr_template/restart-sparkvsr.sh /usr/local/bin/restart-sparkvsr

EXPOSE 7862 8888

CMD ["/opt/sparkvsr_template/start-sparkvsr.sh"]
