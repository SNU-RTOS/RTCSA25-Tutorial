#!/bin/bash
set -e  # Exit if any command fails

# --------- CONFIGURATION ---------
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
submodel0="$script_dir/../models/sub_model_1.tflite"
submodel1="$script_dir/../models/sub_model_2.tflite"
original_model="$script_dir/../models/resnet50.tflite"
rate_ms=0
executable="$script_dir/../output/pipelined_inference_driver"

base_images=(
    "$script_dir/../images/_images_1.png"
    "$script_dir/../images/_images_2.png"
    "$script_dir/../images/_images_3.png"
    "$script_dir/../images/_images_4.png"
    "$script_dir/../images/_images_5.png"
)
total_inputs=100
# ---------------------------------

# Sanity check for files
for f in "$submodel0" "$submodel1" "$original_model" "${base_images[@]}"; do
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
rate_arg="--input-rate=$rate_ms"

# Show the command
echo "Running: $executable $submodel0 $submodel1 $original_model $rate_arg ${images[@]}"

# Run
"$executable" "$submodel0" "$submodel1" "$original_model" "$rate_arg" "${images[@]}"
