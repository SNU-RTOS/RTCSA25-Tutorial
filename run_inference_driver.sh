#!/bin/bash
set -e  # Exit if any command fails

# --------- CONFIGURATION ---------
executable="./bin/inference_driver"
model="./models/resnet50.tflite"
gpu_usage="true"
class_labels="class_labels.json"
base_images=(
    "./images/_images_1.png"
    "./images/_images_2.png"
    "./images/_images_3.png"
    "./images/_images_4.png"
    "./images/_images_5.png"
)
input_period_ms=0
total_inputs=100 # adjust as needed
# ---------------------------------

# Sanity check for files
for f in "$model" "${base_images[@]}"; do
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

# Build input-period argument
period_arg="--input-period=$input_period_ms"

# Show the command
echo "Running: $executable $model $gpu_usage $class_labels ${images[@]} $period_arg"

# Run
"$executable" "$model" "$gpu_usage" "$class_labels" "${images[@]}" "$period_arg"
