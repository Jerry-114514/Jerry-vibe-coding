#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/arp
source /etc/network_turbo >/dev/null 2>&1
export HF_ENDPOINT=https://huggingface.co
export HF_HUB_ETAG_TIMEOUT=120
export HF_HUB_DOWNLOAD_TIMEOUT=120
mkdir -p runs/preflight
exec >> runs/preflight/recovery.log 2>&1
python_bin=/root/autodl-tmp/arp-venv/bin/python
for attempt in 1 2 3 4 5 6; do
  echo "SETUP_ATTEMPT ${attempt}"
  if "$python_bin" tools/prepare_training.py pool && "$python_bin" tools/preflight.py cpu && "$python_bin" tools/preflight.py gpu; then
    mkdir -p runs/formal-20261014
    exec "$python_bin" -u tools/pipeline.py --run runs/formal-20261014 >> runs/formal-20261014/console.log 2>&1
  fi
  echo "Setup incomplete; cached successful work is preserved."
  sleep 60
done
echo "SETUP_FAILED_AFTER_RETRIES; formal experiment not started."
exit 1
