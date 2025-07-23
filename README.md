# RTCSA25 Tutorial: Deep Software Stack Optimization for AI-Enabled Embedded Systems

A hands-on example for [RTCSA25 Tutorial: Deep Software Stack Optimization for AI-Enabled Embedded Systems].  
Uses LiteRT (formerly TensorFlow Lite) inference with CPU & GPU delegate support.

## Overview

Objective: Hands-on optimizing for running TFLite models on RubikPi

Focus:
- Model conversion and inference
- Model slicing and pipelined execution
- LiteRT-based multi-threaded inference engine  

Target: Students working on edge AI systems using lightweight inference engines


## Features

- ✅ CPU inference with XNNPACK optimization
- ✅ GPU acceleration support
- ✅ OpenCV integration for image processing
- ✅ Automated build and test scripts
- ✅ Model verification utilities

## Project Structure

```
├── src/                                # Source code
│   ├── inference_driver_100.cpp        # CPU inference example that runs the same image 100 times
│   ├── inference_driver.cpp            # Basic CPU inference example
│   ├── pipelining.cpp                  # Pipelined parallel execution example
│   ├── thread_safe_queue.cpp           # Thread-safe queue implementation for multi-threaded communication 
│   ├── util.cpp                        # Utility functions
│   └── util.hpp                        # Utility headers
├── models/                             # TensorFlow Lite models
│   ├── sub_model_1.tflite              # First sliced partition of resnet50 model (generated in section 3)
│   ├── sub_model_1.tflite              # Second sliced partition of resnet50 model (generated in section 3)
│   └── resnet50.tflite                 # resnet50 model
├── images/                             # Test images
├── scripts/                            # Build and setup scripts
│   ├── build-litert.sh 
│   ├── build-litert_gpu_delegate.sh    
│   └── install_prerequisites.sh
├── Makefile_*                          # Makefiles for different targets
├── pipeline_100.sh                     # Script that performs pipelined parallel inference on the same image 100 times
├── setup.sh                            # Environment setup script
└── build_and_run.sh                    # Build and run automation script
```

## Prerequisites

### System Requirements
- Ubuntu 20.04/22.04/24.04 
- Clang  
- Bazel
- Git

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

## Setup and Installation

### 1. Environment Setup
```bash
# Create and configure .env file with your paths
# Example .env content:
# ROOT_PATH=$(pwd)
# EXTERNAL_PATH=${ROOT_PATH}/external
# LITERT_PATH=${EXTERNAL_PATH}/liteRT

# Run setup script
./setup.sh
```

### 2. Build LiteRT
The setup script automatically:
- Clones the LiteRT repository
- Configures the build environment
- Builds the core LiteRT library
- Builds GPU delegate library

## Usage

### Quick Start
```bash
# Build and run CPU inference example
./build_and_run.sh
```

### Manual Build and Run

#### CPU Inference
```bash
# Build CPU version
make -f Makefile_inference_driver -j4

# Run CPU inference
./output/inference_driver ./models/resnet50.tflite ./images/_images_1.png ./labels.json
```

### Run Pipelined Parallel Execution
  ```bash
  (.ws_pip) taebikpi@RUBIKPi:~/workspace/DNNPipe_tutorial$ python model_downloader.py 
  WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
  Model saved as resnet50.h5

  (.ws_pip) taebikpi@RUBIKPi:~/workspace/DNNPipe_tutorial$ python model_slicer.py --model-path ./resnet50.h5
  How many submodels? 4
  Enter 3 slicing points for ranges: (0, x1), (x1+1, x2), (x2+1, x3), (x3+1, 176)
  Enter x1 x2 x3: 40 80 120
  Slicing ranges: [(0, 40), (41, 80), (81, 120), (121, 176)]
  Saved sliced tflite model 1 to: ./submodels/resnet50/sub_model_1.tflite
  Saved sliced tflite model 2 to: ./submodels/resnet50/sub_model_2.tflite
  Saved sliced tflite model 3 to: ./submodels/resnet50/sub_model_3.tflite
  Saved sliced tflite model 4 to: ./submodels/resnet50/sub_model_4.tflite
  ```

## Supported Models

Currently tested with Resnet50

The project supports standard TensorFlow Lite models (.tflite format).

## Output

The application performs image classification and outputs:
- Model loading status
- Input/output tensor information
- Inference timing
- Top-5 predictions with confidence scores

Example output:
```
====== main_cpu ====
🔍 Loading model from: ./models/mobileone_s0.tflite
📊 Model loaded successfully
⚡ Inference time: 15.2ms
🏆 Top predictions:
1. Golden retriever (85.4%)
2. Labrador retriever (12.1%)
3. Nova Scotia duck tolling retriever (1.8%)
```



## Development

### Adding New Models
1. Place your `.tflite` model in the `models/` directory
2. Update the `labels.json` file if needed
3. Modify the input preprocessing in `util.cpp` if required

### Debugging
- Set `TF_CPP_MIN_LOG_LEVEL=0` for verbose logging
- Use the verify utilities to test model compatibility
- Check build logs for compilation issues

## Troubleshooting

### Common Issues

1. **Build Errors**: Ensure all prerequisites are installed
2. **Model Loading Fails**: Check model path and format
3. **GPU Delegate Issues**: Verify GPU drivers and OpenCL support
4. **Missing Libraries**: Run `ldconfig` after installation

### Environment Variables
Create a `.env` file in the project root:
```bash
ROOT_PATH=/path/to/your/project
EXTERNAL_PATH=${ROOT_PATH}/external
LITERT_PATH=${EXTERNAL_PATH}/LiteRT
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with all delegate types
5. Submit a pull request

## License

This project is provided as-is for educational and research purposes.

## References

- [LiteRT Documentation](https://ai.google.dev/edge/litert)
- [TensorFlow Lite C++ API](https://www.tensorflow.org/lite/api_docs/cc)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Resnet50 Model]()




# Model Slicer (from DNNPipe)
Slicing a given DNN model into multiple sub-models based on user-defined layer indices

## Acknowledgement
This code is derived from the original implementation of 
*[DNNPipe: Dynamic Programming-based Optimal DNN Partitioning for Pipelined Inference on IoT Networks]*.

While DNNPipe focuses on optimal automated partitioning using dynamic programming, 
this version provides a manual slicing interface for users to define custom DNN partitioning boundaries for experimentation and prototyping purposes.

DNNPipe: https://www.sciencedirect.com/science/article/pii/S1383762125001341?via%3Dihub

DNNPipe Github Repository: https://github.com/SNU-RTOS/DNNPipe

## System Requirements

- Rubikpi (Debian 13)
- Python 3.10.6
- TensorFlow 2.12.0

## Usage
The model downloader (`model_download.py`) : Downloads a pretrained DNN model (ResNet50) for inference
  - Function: Loads the ResNet50 model with pretrained ImageNet weights and saves it in `.h5` format.
  - Output: `resnet50.h5` – Keras H5 format model file (used as input to the slicer)  
  ```bash
  python model_downloader.py 
  ```

The model partitioner (`model_slicer.py`) : Interactively slices a given DNN model into multiple sub-models based on user-defined layer indices
  - Input: DNN model in `.h5` format (e.g., `resnet50.h5`)
  - Output: Skiced sub-models in `.tflite` formats
  ```bash
  python model_slicer.py --model-path ./resnet50.h5
  ```

## Example
  ```bash
  (.ws_pip) taebikpi@RUBIKPi:~/workspace/DNNPipe_tutorial$ python model_downloader.py 
  WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
  Model saved as resnet50.h5

  (.ws_pip) taebikpi@RUBIKPi:~/workspace/DNNPipe_tutorial$ python model_slicer.py --model-path ./resnet50.h5
  How many submodels? 4
  Enter 3 slicing points for ranges: (0, x1), (x1+1, x2), (x2+1, x3), (x3+1, 176)
  Enter x1 x2 x3: 40 80 120
  Slicing ranges: [(0, 40), (41, 80), (81, 120), (121, 176)]
  Saved sliced tflite model 1 to: ./submodels/resnet50/sub_model_1.tflite
  Saved sliced tflite model 2 to: ./submodels/resnet50/sub_model_2.tflite
  Saved sliced tflite model 3 to: ./submodels/resnet50/sub_model_3.tflite
  Saved sliced tflite model 4 to: ./submodels/resnet50/sub_model_4.tflite
  ```

 ## Pipelining
  ```bash
(.ws_pip) taebikpi@RUBIKPi:~/workspace/DNNPipe_tutorial$ cd ../minimal-litert-c/
(.ws_pip) taebikpi@RUBIKPi:~/workspace/minimal-litert-c$ ./output/pipeline ./models/sub_model_1.tflite ./models/sub_model_2.tflite ./images/_images_161.png 
INFO: Created TensorFlow Lite delegate for GPU.
INFO: Loaded OpenCL library with dlopen.
W/Adreno-GSL (219037,219037): <os_lib_map:1488>:   os_lib_map error: libadreno_app_profiles.so: cannot open shared object file: No such file or directory, on 'libadreno_app_profiles.so'

W/Adreno-CB (219037,219037): <cl_app_profiles_initialize:104>: Failed to load the app profiles library libadreno_app_profiles.so!
INFO: Initialized OpenCL-based API.
INFO: Created 1 GPU delegate kernels.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
[stage1] Started inference thread (model0 / GPU)
[stage0] Started preprocessing thread
[stage2] Started inference thread (model1 / CPU)
[stage0] Loading image: ./images/_images_161.png
[stage0] Preprocessing image: ./images/_images_161.png
[stage0] Enqueuing preprocessed image index: 0
[stage0] Finished preprocessing. Signaling shutdown.
[stage1] Dequeued image index: 0
[stage1] Invoking model0...
[stage1] Enqueuing result for image index: 0
[stage1] Finished inference. Signaling shutdown.
[stage2] Dequeued intermediate result for index: 0
[stage2] Invoking model1...
[stage2] Top-5 prediction for image index 0:
- Class 763 (revolver): 0.998964
- Class 597 (holster): 0.000599704
- Class 764 (rifle): 0.000290624
- Class 413 (assault_rifle): 0.000142021
- Class 596 (hatchet): 8.64901e-07
[stage2] Finished all inference.

[INFO] Elapsed time summary
- main:load_models took 0 ms
- main:build_interpreters took 0 ms
- main:apply_delegate took 315 ms
- main:allocate_tensors took 79 ms
- stage0:load_image took 43 ms
- stage0:preprocess took 1 ms
- stage0:flatten took 0 ms
- stage0:enqueue took 0 ms
- stage0:total took 45 ms
- stage1:copy_input took 1 ms
- stage1:invoke took 30 ms
- stage1:postprocess took 10 ms
- stage1:enqueue took 4 ms
- stage1:total took 46 ms
- stage2:copy_input took 0 ms
- stage2:invoke took 96 ms
- stage2:postprocess took 2 ms
- stage2:total took 99 ms
- main:thread_join took 194 ms
- main:total took 590 ms
  ```  


 
