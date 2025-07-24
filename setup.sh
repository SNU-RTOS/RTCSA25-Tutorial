#!/bin/bash

# Activate the .ws_pip virtual environment
if [ -f ".ws_pip/bin/activate" ]; then
    source .ws_pip/bin/activate
    echo "[INFO] Activated Python virtual environment: .ws_pip"
else
    echo "[ERROR] .ws_pip environment exists but activation script is missing"
    exit 1
fi


########## Generate .env ##########
ROOT_PATH=$(pwd)
EXTERNAL_PATH=${ROOT_PATH}/external
LITERT_PATH=${EXTERNAL_PATH}/litert

cat <<EOF > .env
ROOT_PATH=${ROOT_PATH}
EXTERNAL_PATH=${EXTERNAL_PATH}
LITERT_PATH=${LITERT_PATH}
EOF

echo "[INFO] .env file generated at $(pwd)/.env"

# shellcheck source=/dev/null
source .env

########## Setup env ##########
echo "[INFO] ROOT_PATH: ${ROOT_PATH}"
echo "[INFO] EXTERNAL_PATH: ${EXTERNAL_PATH}"
echo "[INFO] LITERT_PATH: ${LITERT_PATH}"

mkdir -p "${EXTERNAL_PATH}" "${ROOT_PATH}/inc" "${ROOT_PATH}/lib" "${ROOT_PATH}/obj" "${ROOT_PATH}/output" "${ROOT_PATH}/models"

########## Setup external sources ##########
cd "${EXTERNAL_PATH}"
echo "[INFO] Working in: $(pwd)"

## Clone LiteRT
echo "[INFO] Installing LiteRT"
if [ ! -d "${LITERT_PATH}" ]; then
    git clone https://github.com/google-ai-edge/litert.git
    cd "${LITERT_PATH}"
    ./configure
else
    echo "[INFO] LiteRT sources already exist, skipping clone/configure..."
fi


########## Build LiteRT ##########
cd "${ROOT_PATH}/scripts"
./build-litert.sh
./build-litert_gpu_delegate.sh

echo "[INFO] Setup Finished"


# install necessary python packages
echo "[INFO] Installing system packages..."
sudo apt install -y libopencv-dev libjsoncpp-dev

# install python packages into virtual environment
echo "[INFO] Installing Python packages into .ws_pip..."
pip install tensorflow==2.12.0

# download ResNet50
python model_downloader.py

# convert resnet50.h5 to resnet50.tflie
python model_h5_to_tflite.py