#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "ERROR: run as root (e.g. sudo $0)" >&2
  exit 1
fi

echo "[1/5] Installing prerequisites..."
apt-get update -y
apt-get install -y ca-certificates curl gnupg

echo "[2/5] Adding NVIDIA Container Toolkit apt repo..."
mkdir -p /usr/share/keyrings
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

source /etc/os-release
distribution="${ID}${VERSION_ID}"
list_url="https://nvidia.github.io/libnvidia-container/stable/deb/${distribution}/libnvidia-container.list"

if curl -fsSL "${list_url}" >/dev/null 2>&1; then
  echo "Using distro-specific repo list: ${list_url}"
  curl -fsSL "${list_url}" \
    | sed 's#^deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
else
  # For newer Ubuntu releases, NVIDIA may not publish a per-distro list.
  # Use the generic list instead.
  echo "WARN: NVIDIA repo does not have an entry for ${distribution}; using generic stable/deb list." >&2
  curl -fsSL "https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list" \
    | sed 's#^deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
fi

echo "[3/5] Installing nvidia-container-toolkit..."
apt-get update -y
apt-get install -y nvidia-container-toolkit

echo "[4/5] Configuring Docker runtime..."
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "[5/5] Verifying GPU access from Docker..."
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

echo "DONE."

