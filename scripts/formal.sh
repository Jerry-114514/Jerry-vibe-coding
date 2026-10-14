#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/arp
source /etc/network_turbo >/dev/null 2>&1
export HF_ENDPOINT=https://huggingface.co
export HF_HUB_ETAG_TIMEOUT=120
export HF_HUB_DOWNLOAD_TIMEOUT=120
mkdir -p runs/formal-20261014
exec /root/autodl-tmp/arp-venv/bin/python -u tools/pipeline.py --run runs/formal-20261014 --publish >> runs/formal-20261014/console.log 2>&1
