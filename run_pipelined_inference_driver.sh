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
#
#-----------------------------------------------------------------------------------------------

set -e  # Exit if any command fails

# --------- CONFIGURATION ---------
executable="./bin/pipelined_inference_driver"
submodel0="./models/submodel_0.tflite"
submodel0_gpu_usage="false"
submodel1="./models/submodel_1.tflite"
submodel1_gpu_usage="true"
class_labels="class_labels.json"
base_images=(
    "./images/_images_1.png"
    "./images/_images_2.png"
    "./images/_images_3.png"
    "./images/_images_4.png"
    "./images/_images_5.png"
)
input_period_ms=0
total_inputs=500 # adjust as needed
# ---------------------------------

# Sanity check for files
for f in "$submodel0" "$submodel1" "${base_images[@]}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: File not found: $f"
        exit 1
    fi
done

# Build repeated image list (round-robin)
images=()
for ((i=0; i<total_inputs; i++)); do
    index=$(( i % ${#base_images[@]} ))
    images+=("${base_images[$index]}")
done

# Build input-rate argument
period_arg="--input-period=$input_period_ms"

# Show the command
echo "Running: $executable $submodel0 $submodel0_gpu_usage $submodel1 $submodel1_gpu_usage $class_labels ${images[@]} $period_arg"

# Run
"$executable" "$submodel0" "$submodel0_gpu_usage" "$submodel1" "$submodel1_gpu_usage" "$class_labels" "${images[@]}" "$period_arg"
