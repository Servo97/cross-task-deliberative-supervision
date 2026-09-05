#!/bin/bash
# nagababa first-boot bootstrap — g7e.24xlarge (us-west-2a), Ubuntu 24.04.4,
# 4x RTX PRO 6000 Blackwell (PCI 2bb5), 96 vCPU, 1TB RAM, 8GB root + 2x3.5TB instance-store NVMe.
#
# Run FROM THE CMU BOX (one command, takes ~5-8 min, ends in a reboot):
#   ssh -i ~/Research/TRI/nagababa.pem ubuntu@10.242.9.112 'sudo bash -s' \
#     < ~/Research/TRI/wsmv2/scripts/ec2/bootstrap_nagababa.sh
# Then re-ssh after ~1 min and run verify_nagababa.sh (same 'bash -s' pattern, no sudo).
#
# NOTE: /data and /data2 are INSTANCE STORE — wiped on instance STOP (survive reboot).
# Everything on them must stay rebuildable from the content-addressed study store on 141.
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive

# 1) Format + mount the two blank 3.5TB NVMe drives (root disk is 8GB — nothing big goes there)
if ! mountpoint -q /data; then
  mkfs -t ext4 -q -F /dev/nvme1n1
  mkfs -t ext4 -q -F /dev/nvme2n1
  mkdir -p /data /data2
  mount /dev/nvme1n1 /data
  mount /dev/nvme2n1 /data2
  chown ubuntu:ubuntu /data /data2
  grep -q '^/dev/nvme1n1 /data ' /etc/fstab || {
    printf '/dev/nvme1n1 /data ext4 defaults,nofail 0 2\n' >> /etc/fstab
    printf '/dev/nvme2n1 /data2 ext4 defaults,nofail 0 2\n' >> /etc/fstab
  }
fi

# 2) NVIDIA driver. Blackwell (2bb5) needs >=570; prefer newest -server-open, fall back to
#    ubuntu-drivers' recommendation. Full metapackage (not headless) so libnvidia-gl ships ->
#    libEGL_nvidia for MuJoCo headless rendering.
apt-get update -qq
apt-get install -y -qq ubuntu-drivers-common git unzip
DRIVER=$(apt-cache search '^nvidia-driver-[0-9]+-server-open$' | awk '{print $1}' | sort -V | tail -1)
if [ -n "$DRIVER" ]; then
  apt-get install -y -qq "$DRIVER"
else
  ubuntu-drivers install
fi

# 3) AWS CLI v2 (root disk has room for ~250MB; caches cleaned below)
cd /tmp
curl -sO https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip
unzip -qo awscli-exe-linux-x86_64.zip
./aws/install --update
rm -rf aws awscli-exe-linux-x86_64.zip
apt-get clean

# 4) uv for the ubuntu user; all envs/caches live on /data
sudo -u ubuntu bash -c '
  curl -LsSf https://astral.sh/uv/install.sh | sh
  mkdir -p /data/envs /data/code /data/cache/hf /data/cache/uv /data2/evals /data2/videos
  grep -q WSM_NAGABABA ~/.bashrc || cat >> ~/.bashrc <<EOF
# WSM_NAGABABA env layout (instance store; rebuildable from 141)
export HF_HOME=/data/cache/hf
export UV_CACHE_DIR=/data/cache/uv
export MUJOCO_GL=egl
EOF
'

echo "BOOTSTRAP OK — rebooting to load the NVIDIA driver; re-ssh in ~1 min"
reboot
