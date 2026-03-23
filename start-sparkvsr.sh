#!/bin/bash
set -euo pipefail

LOG_PREFIX="[SparkVSR]"

log() {
    echo "${LOG_PREFIX} $*"
}

SOURCE_DIR=/opt/sparkvsr_template
TARGET_DIR=/workspace/SparkVSR
LOG_DIR="${TARGET_DIR}/logs"
MODEL_STAGE2_DIR="${TARGET_DIR}/checkpoints/sparkvsr-s2/ckpt-500-sft"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"
export SPARKVSR_ACCESS_PORT="${SPARKVSR_ACCESS_PORT:-7862}"
export SPARKVSR_WORKSPACE_DIR="${TARGET_DIR}"
export SPARKVSR_MODEL_PATH="${SPARKVSR_MODEL_PATH:-$MODEL_STAGE2_DIR}"

if [ ! -f "${TARGET_DIR}/app.py" ]; then
    log "Restoring application files to ${TARGET_DIR}"
    mkdir -p "${TARGET_DIR}"
    rsync -a "${SOURCE_DIR}/" "${TARGET_DIR}/"
else
    log "Application files already present in workspace"
fi

mkdir -p \
    "${TARGET_DIR}/checkpoints" \
    "${TARGET_DIR}/in" \
    "${TARGET_DIR}/logs" \
    "${TARGET_DIR}/out" \
    "${HF_HOME}"

download_snapshot() {
    local repo_id="$1"
    local target_dir="$2"
    local label="$3"

    if [ -d "${target_dir}" ] && [ -n "$(find "${target_dir}" -mindepth 1 -maxdepth 1 2>/dev/null)" ]; then
        log "${label} already cached at ${target_dir}"
        return 0
    fi

    log "Downloading ${label} from ${repo_id}"
    mkdir -p "${target_dir}"
    python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["DOWNLOAD_REPO_ID"]
target_dir = os.environ["DOWNLOAD_TARGET_DIR"]
token = os.environ.get("HF_TOKEN") or None

snapshot_download(
    repo_id=repo_id,
    local_dir=target_dir,
    token=token,
    resume_download=True,
)
PY
}

export DOWNLOAD_REPO_ID="JiongzeYu/SparkVSR"
export DOWNLOAD_TARGET_DIR="${SPARKVSR_MODEL_PATH}"
download_snapshot "${DOWNLOAD_REPO_ID}" "${DOWNLOAD_TARGET_DIR}" "SparkVSR Stage-2 checkpoint"

USERNAME="${SPARKVSR_USERNAME:-admin}"
PASSWORD="${SPARKVSR_PASSWORD:-sparkvsr}"
TARGET_PORT="${GRADIO_SERVER_PORT}"
PROXY_PORT="${SPARKVSR_ACCESS_PORT}"

htpasswd -cb /etc/nginx/.htpasswd "${USERNAME}" "${PASSWORD}"

cat > /etc/nginx/conf.d/sparkvsr-auth.conf <<EOF_CONF
server {
    listen ${PROXY_PORT};

    location / {
        auth_basic "SparkVSR Access";
        auth_basic_user_file /etc/nginx/.htpasswd;

        proxy_pass http://127.0.0.1:${TARGET_PORT};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF_CONF

if ! grep -q "include /etc/nginx/conf.d/.*.conf;" /etc/nginx/nginx.conf; then
    sed -i '/http {/a \	include /etc/nginx/conf.d/*.conf;' /etc/nginx/nginx.conf
fi

nginx -t >/dev/null
if ! nginx -s reload >/dev/null 2>&1; then
    nginx -s stop >/dev/null 2>&1 || true
    nginx
fi

APP_LOG="${LOG_DIR}/sparkvsr.log"
log "Launching SparkVSR Gradio app on port ${TARGET_PORT}"
cd "${TARGET_DIR}"
nohup python3 app.py > "${APP_LOG}" 2>&1 &

log "SparkVSR started. Logs: ${APP_LOG}"
log "Auth credentials -> user: ${USERNAME} password: ${PASSWORD}"
log "External access via port ${PROXY_PORT}"

if [ "${SPARKVSR_RESTART_ONLY:-0}" = "1" ]; then
    exit 0
fi

if [ -f "/start.sh" ]; then
    exec /start.sh
else
    exec tail -f "${APP_LOG}"
fi
