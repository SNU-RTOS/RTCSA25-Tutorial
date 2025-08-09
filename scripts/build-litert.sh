#!/bin/bash

#-----------------------------------------------------------------------------------------------
# Filename: build-litert.sh
#
# @Author: GeonhaPark
# @Affiliation: Real-Time Operating System Laboratory, Seoul National University
# @Created: 07/23/25
# @Original Work: Based on minimal-litert-c repository (https://github.com/SNU-RTOS/minimal-litert-c)
# @Modified by: Taehyun Kim and Namcheol Lee on 08/06/25
# @Contact: {nclee,ghpark,thkim}@redwood.snu.ac.kr
#
# @Description: LiteRT build script for RTCSA25 tutorial
#
# ──────────────────────────────────────────────────────────────────────────────
cd ..
source .env

# ── Build Configuration ───────────────────────────────────────────────────────
FLATBUFFER_PATH=${LITERT_PATH}/bazel-litert/external/flatbuffers/include/flatbuffers
TENSORFLOW_PATH=${LITERT_PATH}/bazel-litert/external/org_tensorflow/tensorflow

LITERT_LIB_PATH=${LITERT_PATH}/bazel-bin/tflite/libtensorflowlite.so

LITERT_INC_PATH=${LITERT_PATH}/tflite

########## Build ##########
cd ${LITERT_PATH}

echo "[INFO] Build LiteRT .so .."
echo "[INFO] Path: ${LITERT_LIB_PATH}"

cd ${LITERT_PATH}
pwd

# Release mode (Note: -Wno-incompatible-pointer-types could cause undefined behavior)
bazel build -c opt //tflite:tensorflowlite \
    --copt=-Os \
    --copt=-fPIC \
    --copt=-Wno-incompatible-pointer-types \
    --linkopt=-s

########## Make symlink ##########
ln -sf ${LITERT_LIB_PATH} ${ROOT_PATH}/lib/libtensorflowlite.so
ln -sf ${LITERT_INC_PATH} ${ROOT_PATH}/inc
ln -sf ${FLATBUFFER_PATH} ${ROOT_PATH}/inc
ln -sf ${TENSORFLOW_PATH} ${ROOT_PATH}/inc

cd ${ROOT_PATH}/scripts
pwd




