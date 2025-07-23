#!/bin/bash
cd ..
source .env
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
