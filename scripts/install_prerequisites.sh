#!/bin/bash

#-----------------------------------------------------------------------------------------------
# Filename: install_prerequisites.sh
#
# @Author: GeonhaPark
# @Affiliation: Real-Time Operating System Laboratory, Seoul National University
# @Created: 07/23/25
# @Original Work: Based on minimal-litert-c repository (https://github.com/SNU-RTOS/minimal-litert-c)
# @Modified by: GeonhaPark on 08/10/25
# @Contact: {nclee,ghpark,thkim}@redwood.snu.ac.kr
#
# @Description: Install script for RTCSA25 tutorial prerequisites
#
#-----------------------------------------------------------------------------------------------

# install dev prerequisites
sudo apt install -y \
    git \
    curl \
    wget \
    python3 \
    python3-pip \
    build-essential \
    clang \
    cmake \
    unzip \
    pkg-config

# install pyenv prerequisites
sudo apt install -y --no-install-recommends \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    ncurses-dev \
    libffi-dev \
    libreadline-dev \
    sqlite3 libsqlite3-dev \
    tk-dev \
    bzip2 libbz2-dev \
    lzma liblzma-dev \
    llvm libncursesw5-dev xz-utils libxml2-dev \
    libxmlsec1-dev 


# install Pyenv
curl -fsSL https://pyenv.run | bash

# update .bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init --path)"\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc
source ~/.bashrc

# install python 3.10.16
pyenv install 3.10.16
pyenv global 3.10.16

# create python virtual environment by using venv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd ${SCRIPT_DIR}/..
PROJECT_ROOT=$(pwd)
echo "[INFO] Project root is ${PROJECT_ROOT}"

VENV_ROOT="${PROJECT_ROOT}/.venv"
python3 -m venv ${VENV_ROOT}
echo "[INFO] Created virtual environment at ${VENV_ROOT}"

source ${VENV_ROOT}/bin/activate
echo "export VENV_ROOT=\"${VENV_ROOT}\"" >> ~/.bashrc
echo "source \$VENV_ROOT/bin/activate" >> ~/.bashrc
echo "[INFO] Activated virtual environment: ${VENV_ROOT}"

# install bazelisk and bazel
if [ ! -f /usr/bin/bazel ]; then
    sudo curl -L https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-arm64 -o /usr/local/bin/bazelisk
    sudo chmod +x /usr/local/bin/bazelisk
    sudo ln -s /usr/local/bin/bazelisk /usr/bin/bazel
else
    echo "[INFO] Bazel already exists, skipping symlink creation..."
fi

# add environment variable HERMETIC_PYTHON_VERSION to fix python during build LiteRT
echo 'export HERMETIC_PYTHON_VERSION=3.10' >> ~/.bashrc

source ~/.bashrc