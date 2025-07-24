#!/bin/bash
# This script is used to run the hands-on example
# It builds the inference driver program and runs it
# It is assumed that the litert-c library is already built and available in the lib directory
# The script will log the output of each program to a separate log file

run_main() {
  local model="$1"         # Example: ./models/resnet50.tflite
  local image="$2"         # Example: ./images/_images_1.png

  local model_base
  model_base=$(basename "${model%.*}")
  local logfile="output_inference_driver_${mode}_${model_base}.log"

  (
    exec > >(tee "$logfile") 2>&1

    echo "================================"
    echo "[INFO] Build inference_driver"
    make -f Makefile_inference_driver -j4

    echo "[INFO] Run inference_driver"
    ./output/inference_driver true "$model" "$image"
  )
}


##################### main #####################
run_main ./models/resnet50.tflite ./images/_images_1.png