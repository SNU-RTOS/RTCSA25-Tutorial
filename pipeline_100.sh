#!/bin/bash

# Set model and image paths
MODEL0="./models/sub_model_1.tflite"
MODEL1="./models/sub_model_2.tflite"
IMAGE="./images/_images_1.png"
INPUT_RATE=0

# Generate an argument list with the same image repeated 100 times
IMAGE_ARGS=()
for i in {1..100}; do
    IMAGE_ARGS+=("$IMAGE")
done

# Execute
./output/pipelined_inference_driver "$MODEL0" "$MODEL1" "${IMAGE_ARGS[@]}" --input-rate=$INPUT_RATE
