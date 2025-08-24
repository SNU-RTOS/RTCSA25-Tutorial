# RTCSA25 Tutorial: Deep Software Stack Optimization for AI-Enabled Embedded Systems

## Overview

Objective: This repository provides hands-on exercises on optimizing and deploying LiteRT (formerly TFLite) models on the RUBIK Pi platform, focusing on pipeline parallelism for efficient on-device inference.

Key Topics
- Inference driver and inference runtime
- Model slicing and conversion
- Throughput enhancement via pipelining on heterogeneous accelerators

## Project Structure

```
├── src/                                # Source code directory
│   ├── inference_driver.cpp            
│   ├── instrumentation_harness_utils.cpp                   
│   ├── instrumentation_harness_utils.hpp                   
│   ├── instrumentation_harness.cpp       
│   ├── pipelined_inference_driver.cpp  
│   ├── util.cpp                        
│   └── util.hpp                        
├── models/                             # Directory for DNN models
├── images/                             # Test images
├── scripts/                            # Build and setup scripts
│   ├── build-litert.sh 
│   ├── build-litert_gpu_delegate.sh    
│   └── install_prerequisites.sh
├── class_labels.json                   # JSON file containing the class labels used by ResNet50
├── Makefile                            # Makefile for building executables into bin/
├── model_downloader.py                 # Downloads pretrained resnet50 (.h5 format)
├── model_h5_to_tflite.py               # Converts (.h5) format to (.tflite) format
├── model_slicer.py                     # Model slicer
├── run_inference_driver.sh             # Script for running ./bin/inference_driver
├── run_pipelined_inference_driver.sh   # Script for running ./bin/pipelined_inference_driver
└── setup.sh                            # Environment setup script
```

## Prerequisites

### System Requirements
- Ubuntu 20.04/22.04/24.04 
- Build tools
- OpenCV development libraries
- JSON libraries (jsoncpp)
- Python development tools
- Bazel build system

Run the prerequisite installation script at `~/RTCSA25-Tutorial/`:
```bash
./scripts/install_prerequisites.sh
```

## Environment Setup

```bash
# Run setup script
./setup.sh
```

The setup script automatically:
- Creates and configures `.env` file with your paths
- Clones the LiteRT repository
- Configures the build environment
- Builds the core LiteRT library
- Builds GPU delegate library
- Installs necessary python packages for downloading and slicing the model
- Downloads `resnet50.h5` and converts it into `resnet50.tflite` under `./models` directory

## Source Code

### Build and Run

#### Inference driver
```bash
# Build the inference driver
make inference -j4

# Run the inference driver
./bin/inference_driver <model_path> <gpu_usage> <class_labels_path> <image1_path> [image2_path ... imageN_path] [input_period_ms]

Arguments:
  model_path        Path to .tflite model file
  gpu_usage         true/false to enable GPU delegate
  class_labels_path Path to JSON class labels file
  image_path(s)     One or more image files for inference
  input_period_ms   (Optional) Delay between inputs in milliseconds

# Option 1: Run directly
./bin/inference_driver ./models/resnet50.tflite false class_labels.json ./images/_images_1.png

# Option 2: Run via script (./models/resnet50.tflite, true, 500 inputs, input_period=0)
# Runs the model on GPU
./run_inference_driver.sh
```
#### Instrumentation Harness
```bash
# Build the instrumentation harness
make instrumentation_harness -j4

# Run the instrumentation harness
./bin/instrumentation_harness <model_path> [gpu_usage]

Arguments:
  model_path  Path to .tflite model file
  gpu_usage   (Optional) true/false to enable GPU delegate

# Example
./bin/instrumentation_harness ./models/resnet50.tflite
```

#### Pipelined Inference Driver
```bash
# Build the pipelined inference driver
make pipelined -j4

# Run the pipelined inference driver
./bin/pipelined_inference_driver <submodel0_path> <submodel0_gpu_usage> <submodel1_path> <submodel1_gpu_usage> <class_labels> <image1_path> [image2_path ... imageN_path] [input_period_ms]

Arguments:
  submodel0_path      Path to first sliced submodel (.tflite)
  submodel0_gpu_usage true/false to enable GPU delegate for submodel 0
  submodel1_path      Path to second sliced submodel (.tflite)
  submodel1_gpu_usage true/false to enable GPU delegate for submodel 1
  class_labels        Path to JSON class labels file
  image_path(s)       One or more image files for inference
  input_period_ms     (Optional) Delay between inputs in milliseconds

# Option 1: Run directly
./bin/pipelined_inference_driver ./models/sub_model_0.tflite false ./models/sub_model_1.tflite true class_labels.json ./images/_images_1.png

# Option 2: Run via script (./models/sub_model_0.tflite, false, ./models/sub_model_1.tflite, true, 500 inputs, input_period=0)
# Runs sub-model 0 on CPU and sub-model 1 on GPU
./run_pipelined_inference_driver.sh
```

## Model Slicer (from [DNNPipe](https://github.com/SNU-RTOS/DNNPipe))
A tool for **Slicing** and **Converting** a **.h5** model file into multiple **.tflite** sub-models

### Slicing a DNN model

`model_slicer.py` : Interactively slices a given DNN model into multiple sub-models based on user-defined layer indices
  - Input: DNN model in `.h5` format (e.g., `resnet50.h5`)
  - Output: Sliced sub-models in `.tflite` format
  ```bash
  python model_slicer.py --model-path ./models/resnet50.h5
  ```
### Example
```bash
(.ws_pip) rubikpi@RUBIKPi:~/workspace/RTCSA25-Tutorial$ python model_slicer.py --model-path ./models/resnet50.h5
How many submodels? 4
Enter 3 slicing points for ranges: (1, x1-1), (x1, x2-1), (x2, x3-1), (x3, 176)
Enter x1 x2 x3: 40 80 120
Slicing ranges: [(1, 39), (40, 79), (80, 119), (120, 176)]
Saved sliced tflite model to: ./models/sub_model_0.tflite
Saved sliced tflite model to: ./models/sub_model_1.tflite
Saved sliced tflite model to: ./models/sub_model_2.tflite
Saved sliced tflite model to: ./models/sub_model_3.tflite
```

### Acknowledgement
This code is derived from the original implementation of 
*[DNNPipe: Dynamic Programming-based Optimal DNN Partitioning for Pipelined Inference on IoT Networks]*.

While DNNPipe focuses on optimal automated partitioning using dynamic programming, 
this version provides a manual slicing interface for users to define custom DNN partitioning boundaries for experimentation and prototyping purposes.

DNNPipe: https://www.sciencedirect.com/science/article/pii/S1383762125001341?via%3Dihub

DNNPipe Github Repository: https://github.com/SNU-RTOS/DNNPipe

## References

- [LiteRT Documentation](https://ai.google.dev/edge/litert)
- [LiteRT C++ API](https://www.tensorflow.org/lite/api_docs/cc)
- [OpenCV Documentation](https://docs.opencv.org/)
- [DNNPipe](https://www.sciencedirect.com/science/article/pii/S1383762125001341?via%3Dihub)




