#!/bin/bash
set -euo pipefail

pkill -f "python3 app.py" >/dev/null 2>&1 || true
pkill -f "python app.py" >/dev/null 2>&1 || true

export SPARKVSR_RESTART_ONLY=1
/opt/sparkvsr_template/start-sparkvsr.sh
