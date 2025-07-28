# RTCSA25 Tutorial: Deep Software Stack Optimization for AI-Enabled Embedded Systems

A hands-on example for [RTCSA25 Tutorial: Deep Software Stack Optimization for AI-Enabled Embedded Systems].  
Uses LiteRT (formerly TensorFlow Lite) inference with CPU & GPU delegate support.

## Overview

Objective: Hands-on optimizing for running TFLite models on RubikPi

Focus:
- Inference driver
- LiteRT internals
- Model slicing and conversion
- Pipelined inference driver

## Features

- ✅ CPU inference with XNNPACK delegate
- ✅ GPU acceleration support
- ✅ Automated build and test scripts
- ✅ Model verification utilities

## Project Structure

```
├── src/                                # Source code
│   ├── inference_driver.cpp            # General inference driver
│   ├── internals_sample_code.cpp       # Sample code for showing internals of LiteRT
│   ├── internals.cpp                   # Functions for inspecting the internals of LiteRT
│   ├── internals.hpp                   # Header file of inspecting functions
│   ├── pipelined_inference_driver.cpp  # Inference driver for pipelined inference
│   ├── util.cpp                        # Utility functions
│   └── util.hpp                        # Header file of utility functions
├── models/                             # Directory for DNN models
├── images/                             # Test images
├── scripts/                            # Build and setup scripts
│   ├── build-litert.sh 
│   ├── build-litert_gpu_delegate.sh    
│   └── install_prerequisites.sh
├── model_downloader.py                 # Downloads pretrained resnet50 (.h5 format)
├── model_h5_to_tflite.py               # Converts (.h5) format to (.tflite) format
├── model_slicer.py                     # Model slicer
├── Makefile                            # Makefile for generating all outputs
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

## Usage

### Quick Start
```bash
# Build and run inference driver
./build_and_run.sh
```

### Manual Build and Run

#### Inference driver
```bash
# Build inference driver
make inference -j4

# Run inference driver: ./output/inference_driver <gpu_usage> <model path> <image path>
./output/inference_driver true ./models/resnet50.tflite ./images/_images_1.png
```

#### Internals Sample Code
```bash
# Build inference driver
make internals -j4

# Run inference driver
# ./output/internals_sample_code <gpu_usage> <model_path> <image_path> <show_internals>
./output/internals_sample_code true ./models/resnet50.tflite ./images/_images_1.png true
```

#### Pipelined Inference Driver
```bash
# Build Pipelining
make pipelined -j4

# Run inference driver
# ./output/pipelined_inference_driver <submodel_1_path> <submodel_2_path> <image_path>
./output/pipelined_inference_driver ./models/sub_model_1.tflite ./models/sub_model_2.tflite ./images/_images_1.png
```

### Model Slicer

#### Slicing the model

The model partitioner (`model_slicer.py`) : Interactively slices a given DNN model into multiple sub-models based on user-defined layer indices
  - Input: DNN model in `.h5` format (e.g., `resnet50.h5`)
  - Output: Skiced sub-models in `.tflite` formats
  ```bash
  python model_slicer.py --model-path ./models/resnet50.h5
  ```
##### Example
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


## Supported Models

Tested with Resnet50

The project supports standard LiteRT models (.tflite format).

### Downloading the model
```bash
# Downloads pretrained DNN model 'resnet50.h5' and saves the model in './models/'
# During setup, model will be downloaded
python model_downloader.py
```

### Converting to tflite format
```bash
# Converts the original downloaded model format (.h5) to (.tflite) format
# During setup, the conversion will be done
python model_h5_to_tflite.py --h5-path ./models/resnet50.h5 
```


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
- [LiteRT C++ API](https://www.tensorflow.org/lite/api_docs/cc)
- [OpenCV Documentation](https://docs.opencv.org/)
- [DNNPipe](https://www.sciencedirect.com/science/article/pii/S1383762125001341?via%3Dihub)


## Model Slicer (from DNNPipe)
Slicing a given DNN model into multiple sub-models based on user-defined layer indices

### Acknowledgement
This code is derived from the original implementation of 
*[DNNPipe: Dynamic Programming-based Optimal DNN Partitioning for Pipelined Inference on IoT Networks]*.

While DNNPipe focuses on optimal automated partitioning using dynamic programming, 
this version provides a manual slicing interface for users to define custom DNN partitioning boundaries for experimentation and prototyping purposes.

DNNPipe: https://www.sciencedirect.com/science/article/pii/S1383762125001341?via%3Dihub

DNNPipe Github Repository: https://github.com/SNU-RTOS/DNNPipe
