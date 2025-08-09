#!/bin/bash

#-----------------------------------------------------------------------------------------------
# Filename: build-litert_gpu_delegate.sh
#
# @Author: GeonhaPark
# @Affiliation: Real-Time Operating System Laboratory, Seoul National University
# @Created: 07/23/25
# @Original Work: Based on minimal-litert-c repository (https://github.com/SNU-RTOS/minimal-litert-c)
# @Modified by: GeonhaPark on 08/10/25
# @Contact: {nclee,ghpark,thkim}@redwood.snu.ac.kr
#
# @Description: LiteRT GPU delegate build script for RTCSA25 tutorial
#
#-----------------------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd ${SCRIPT_DIR}/..

source .env

# ── Build Configuration ───────────────────────────────────────────────────────

GPU_DELEGATE_LIB_PATH=${LITERT_PATH}/bazel-bin/tflite/delegates/gpu/libtensorflowlite_gpu_delegate.so

########## Build ##########
cd ${LITERT_PATH}

echo "[INFO] Build gpu delegate .so .."
echo "[INFO] Path: ${GPU_DELEGATE_LIB_PATH}"

cd ${LITERT_PATH}
pwd

# Release mode
bazel build -c opt //tflite/delegates/gpu:libtensorflowlite_gpu_delegate.so \
    --copt=-Os \
    --copt=-fPIC \
    --linkopt=-s
bazel shutdown


########## Make symlink ##########
ln -sf ${GPU_DELEGATE_LIB_PATH} ${ROOT_PATH}/lib/libtensorflowlite_gpu_delegate.so

cd ${ROOT_PATH}/scripts
pwd
