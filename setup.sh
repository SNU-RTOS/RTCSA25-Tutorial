#!/bin/bash

#-----------------------------------------------------------------------------------------------
# Filename: setup.sh
#
# @Author: GeonhaPark
# @Affiliation: Real-Time Operating System Laboratory, Seoul National University
# @Created: 07/23/25
# @Original Work: Based on minimal-litert-c repository (https://github.com/SNU-RTOS/minimal-litert-c)
# @Modified by: GeonhaPark on 08/10/25
# @Contact: {nclee,ghpark,thkim}@redwood.snu.ac.kr
#
# @Description: Setup script for RTCSA25 tutorial
#
#-----------------------------------------------------------------------------------------------

########## Generate .env ##########
if [ -f ".env" ]; then
    echo "[INFO] .env file already exists, skipping generation..."
else
    echo "[INFO] Generating .env file..."

    ROOT_PATH=$(pwd)
    EXTERNAL_PATH=${ROOT_PATH}/external
    LITERT_PATH=${EXTERNAL_PATH}/litert

cat <<EOF > .env
ROOT_PATH=${ROOT_PATH}
EXTERNAL_PATH=${EXTERNAL_PATH}
LITERT_PATH=${LITERT_PATH}
EOF

    echo "[INFO] .env file generated at $(pwd)/.env"
fi

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
    git clone https://github.com/Seunmul/LiteRT.git "${LITERT_PATH}"  --depth 1
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
echo "[INFO] Installing Python packages into .venv..."
pip install tensorflow==2.12.0

######### Download ResNet50 ##########
pip install netron

cd ..
# download ResNet50
python3 model_downloader.py

# convert resnet50.h5 to resnet50.tflie
python3 model_h5_to_tflite.py
