#!/bin/bash
# Post-reboot verification for nagababa. Read-only. Run:
#   ssh -i ~/Research/TRI/nagababa.pem ubuntu@10.242.9.112 'bash -s' \
#     < ~/Research/TRI/wsmv2/scripts/ec2/verify_nagababa.sh
set -u
echo "== GPUs =="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || echo "DRIVER NOT LOADED"
echo "== Mounts =="
df -h /data /data2 2>/dev/null || echo "NVMe NOT MOUNTED"
echo "== AWS identity (instance role?) =="
aws sts get-caller-identity 2>&1 | head -4
echo "== EGL smoke (needs a python with mujoco later; for now just the lib) =="
ldconfig -p | grep -i "libEGL_nvidia" || echo "libEGL_nvidia MISSING (headless render will fail)"
echo "== Versions =="
uname -r; aws --version 2>&1; ~/.local/bin/uv --version 2>&1 || true
