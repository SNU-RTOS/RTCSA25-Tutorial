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


 
