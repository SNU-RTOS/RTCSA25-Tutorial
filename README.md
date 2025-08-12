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

### Dependencies
Run the prerequisite installation script:
```bash
./scripts/install_prerequisites.sh
```

This installs:
- Build tools
- OpenCV development libraries
- JSON libraries (jsoncpp)
- Python development tools
- Bazel build system

## Environment Setup

```bash
# Run setup script
./setup.sh
```

The setup script automatically:
- Create and configure .env file with your paths
- Clones the LiteRT repository
- Configures the build environment
- Builds the core LiteRT library
- Builds GPU delegate library
- Installs necessary python packages for downloading and slicing the model
- Download resnet50.h5 and convert it into resnet50.tflite under ./models directory

## Source Code

### Build and Run

#### Inference driver
```bash
# Build inference driver
make inference -j4

# Run inference driver 
# ./bin/inference_driver <model_path> <gpu_usage> <class_labels_path> <image1_path> [image2_path ... imageN_path] [input_period_ms]
./bin/inference_driver ./models/resnet50.tflite false class_labels.json ./images/_images_1.png
or
# Script for running the inference driver on the GPU with 500 input samples and input_period set to 0
./run_inference_driver.sh
```

#### Instrumentation Harness
```bash
# Build instrumnetation harness
make instrumentation_harness -j4

# Run inference driver
# ./bin/instrumentation_harness <model_path> [gpu_usage]
./bin/instrumentation_harness ./models/resnet50.tflite
```

#### Pipelined Inference Driver
```bash
# Build Pipelined inference drvier
make pipelined -j4

# Run inference driver
# ./bin/pipelined_inference_driver <submodel0_path> <submodel0_gpu_usage> <submodel1_path> <submodel1_gpu_usage> <class_labels> <image1_path> [image2_path ... imageN_path] [input_period_ms]
./bin/pipelined_inference_driver ./models/sub_model_0.tflite false ./models/sub_model_1.tflite true ./images/_images_1.png
or
# Script for running the pipelined inference driver with 500 input samples
# Runs Submodel 0 on the CPU and Submodel 1 on the GPU with input_period set to 0
./run_pipelined_inference_driver.sh
```

## Model Slicer (from [DNNPipe](https://github.com/SNU-RTOS/DNNPipe))
A tool for **Slicing** and **Converting** a **.h5** model file into multiple **.tflite** sub-models

### Slicing a DNN model

`model_slicer.py` : Interactively slices a given DNN model into multiple sub-models based on user-defined layer indices
  - Input: DNN model in `.h5` format (e.g., `resnet50.h5`)
  - Output: Skiced sub-models in `.tflite` formats
  ```bash
  python model_slicer.py --model-path ./models/resnet50.h5
  ```
### Example
```bash
(.ws_pip) rubikpi@RUBIKPi:~/workspace/RTCSA25-Tutorial$ python model_slicer.py --model-path ./models/resnet50.h5
How many submodels? 4
Enter 3 slicing points for ranges: (0, x1), (x1+1, x2), (x2+1, x3), (x3+1, 176)
Enter x1 x2 x3: 40 80 120
Slicing ranges: [(0, 40), (41, 80), (81, 120), (121, 176)]
Saved sliced tflite model to: ./models/sub_model_1.tflite
Saved sliced tflite model to: ./models/sub_model_2.tflite
Saved sliced tflite model to: ./models/sub_model_3.tflite
Saved sliced tflite model to: ./models/sub_model_4.tflite
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




