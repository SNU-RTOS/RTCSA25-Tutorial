#!/bin/bash
#-----------------------------------------------------------------------------------------------
# Filename: run_inference_driver.sh
#
# @Author: Namcheol Lee
# @Affiliation: Real-Time Operating System Laboratory, Seoul National University
# @Created: 07/23/25
# @Contact: {ghpark,thkim,nclee}@redwood.snu.ac.kr
#
# @Description: Script to run inference driver for RTCSA25 tutorial
#-----------------------------------------------------------------------------------------------

set -euo pipefail

# --------- CONFIGURATION ---------
executable="./bin/pipelined_inference_driver_advanced"

# List your submodels in order. Add more to scale stages.
SUBMODELS=(
  "./models/submodel_0.tflite"
  "./models/submodel_1.tflite"
  # "./models/submodel_2.tflite"
  # "./models/submodel_3.tflite"
)

# gpu_usage flags aligned 1:1 with SUBMODELS. "true" => GPU delegate, "false" => XNNPACK
GPU_USAGES=(
  "false"
  "true"
  # "false"
  # "false"
)

class_labels="class_labels.json"

# Base images to round-robin
BASE_IMAGES=(
  "./images/_images_1.png"
  "./images/_images_2.png"
  "./images/_images_3.png"
  "./images/_images_4.png"
  "./images/_images_5.png"
  "./images/_images_6.png"
)

input_period_ms=0
total_inputs=30   # adjust as needed
# ---------------------------------

# ---------- Sanity checks ----------
if [[ ! -x "$executable" ]]; then
  echo "ERROR: Executable not found or not executable: $executable"
  exit 1
fi

if [[ ${#SUBMODELS[@]} -eq 0 ]]; then
  echo "ERROR: Provide at least one submodel"
  exit 1
fi

if [[ ${#SUBMODELS[@]} -ne ${#GPU_USAGES[@]} ]]; then
  echo "ERROR: SUBMODELS and GPU_USAGES must have the same length"
  echo "       SUBMODELS=${#SUBMODELS[@]} GPU_USAGES=${#GPU_USAGES[@]}"
  exit 1
fi

for f in "${SUBMODELS[@]}" "$class_labels" "${BASE_IMAGES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: File not found: $f"
    exit 1
  fi
done
# -----------------------------------

# Build repeated image list (round-robin)
IMAGES=()
for ((i=0; i<total_inputs; i++)); do
  idx=$(( i % ${#BASE_IMAGES[@]} ))
  IMAGES+=("${BASE_IMAGES[$idx]}")
done

# Build CLI per the new program:
# app <num_inference_stages> (<submodel_i_path> <gpu_usage_i>){i=0..K-1} <class_labels_path> <images...> [--input-period=ms]
NUM_INFER=${#SUBMODELS[@]}
PERIOD_ARG="--input-period=${input_period_ms}"

# Assemble command
CMD=( "$executable" "$NUM_INFER" )
for ((i=0; i<NUM_INFER; i++)); do
  CMD+=( "${SUBMODELS[$i]}" "${GPU_USAGES[$i]}" )
done
CMD+=( "$class_labels" )
CMD+=( "${IMAGES[@]}" )
CMD+=( "$PERIOD_ARG" )

# Show and run
echo "Running:"
printf '  %q' "${CMD[@]}"
echo

exec "${CMD[@]}"
