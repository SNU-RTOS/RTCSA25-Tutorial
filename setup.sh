#!/bin/bash

source .env

########## Setup env ##########

echo "[INFO] ROOT_PATH: ${ROOT_PATH}"
echo "[INFO] EXTERNAL PATH: ${EXTERNAL_PATH}"
echo "[INFO] LITERT_PATH: ${LITERT_PATH}"

if [ ! -d ${EXTERNAL_PATH} ]; then
    mkdir -p ${EXTERNAL_PATH}
fi

########## Make folders ##########
cd ${ROOT_PATH}
if [ ! -d "inc" ]; then
    mkdir inc    
fi

if [ ! -d "lib" ]; then
    mkdir lib   
fi

if [ ! -d "obj" ]; then
    mkdir obj  
fi

if [ ! -d "output" ]; then
    mkdir output  
fi

########## Setup external sources #########S#

cd ${EXTERNAL_PATH}
pwd

## Clone LiteRT
echo "[INFO] Installing LiteRT"
if [ ! -d "./litert" ]; then
    git clone https://github.com/google-ai-edge/litert.git
    cd ${LITERT_PATH}
    ./configure
else
    echo "[INFO] LiteRT sources are already cloned, skipping ..."
fi


########## Build LiteRT ##########
cd ${ROOT_PATH}/scripts
./build-litert.sh
./build-litert_gpu_delegate.sh

echo "[INFO] Setup Finished"