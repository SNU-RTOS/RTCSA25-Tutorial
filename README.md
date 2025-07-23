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