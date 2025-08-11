/*
 * Filename: pipelined_inference_driver.cpp
 *
 * @Author: Namcheol Lee
 * @Affiliation: Real-Time Operating System Laboratory, Seoul National University
 * @Created: 07/23/25
 * @Original Work: Based on DNNPipe repository (https://github.com/SNU-RTOS/DNNPipe)
 * @Contact: nclee@redwood.snu.ac.kr
 *
 * @Description: Pipelined Inference driver codes
 *
 */

#include <iostream>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>
#include "tflite/delegates/xnnpack/xnnpack_delegate.h" 
#include "tflite/delegates/gpu/delegate.h"             
#include "tflite/interpreter_builder.h"
#include "tflite/interpreter.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"
#include "util.hpp"

/* Pipelined Inference Driver
 * This driver demonstrates a pipelined inference workflow using two submodels.
 * There are three stages:
 * 1. Stage 0: Preprocess input on CPU core 
 * 2. Stage 1: Run inference for sub-model 0 on CPU core 
 * 3. Stage 2: Run inference for sub-model 1 on GPU                
 * 4. Stage 3: Postprocess output on CPU core  */

// === Queues for inter-stage communication ===
// stageX_to_stageY_queue: from stageX to stageY
InterStageQueue<IntermediateTensor> stage0_to_stage1_queue;
InterStageQueue<IntermediateTensor> stage1_to_stage2_queue;
InterStageQueue<IntermediateTensor> stage2_to_stage3_queue;

void stage0_thread_function(const std::vector<std::string>& images, int input_period_ms) {
    auto next_wakeup_time = std::chrono::high_resolution_clock::now();
    size_t idx = 0;
    do {
        std::string label = "Stage0 " + std::to_string(idx);
        util::timer_start(label);
        /* Preprocessing */
        // Load image
        cv::Mat image = cv::imread(images[idx]);
        if (image.empty()) {
            std::cerr << "[Stage0] Failed to load image: " << images[idx] << "\n";
            util::timer_stop(label);
            continue;
        }
        
        // Preprocess image
        cv::Mat preprocessed_image = util::preprocess_image_resnet(image, 224, 224);
        if (preprocessed_image.empty()) {
            std::cerr << "[Stage0] Preprocessing failed: " << images[idx] << "\n";
            util::timer_stop(label);
            continue;
        }

        /* Create IntermediateTensor, deep-copy preprocessed_image data into it, 
        *  and move it into stage0_to_stage1_queue */
        // Hint: std::memcpy(destination_ptr, source_ptr, num_bytes);
        IntermediateTensor intermediate_tensor;
        // ======= Write your code here =======
        intermediate_tensor.index = idx;
        intermediate_tensor.data.resize(
            preprocessed_image.total() * preprocessed_image.channels());
        std::memcpy(intermediate_tensor.data.data(), preprocessed_image.ptr<float>(), 
            intermediate_tensor.data.size() * sizeof(float));
        intermediate_tensor.tensor_boundaries = 
            {static_cast<int>(intermediate_tensor.data.size())};
        // ====================================
        stage0_to_stage1_queue.push(std::move(intermediate_tensor));
        ++idx;
        
        util::timer_stop(label);

        // Sleep to control the input rate
        // If next_wakeup_time is in the past, it will not sleep
        next_wakeup_time += std::chrono::milliseconds(input_period_ms);
        std::this_thread::sleep_until(next_wakeup_time);
    } while (idx < images.size());

    // Notify stage1_thread that no more data will be sent
    stage0_to_stage1_queue.signal_shutdown();
} // end of stage0_thread_function

void stage1_thread_function(tflite::Interpreter* interpreter) {
    IntermediateTensor intermediate_tensor;

    while (stage0_to_stage1_queue.pop(intermediate_tensor)) {
        std::string label = "Stage1 " + std::to_string(intermediate_tensor.index);
        util::timer_start(label);

        /* Access the 0th input tensor of the interpreter
        *  and copy the contents of intermediate_tensor.data into it */
        // Hint: std::memcpy(destination_ptr, source_ptr, num_bytes);
        // ======= Write your code here =======
        float *input_tensor = interpreter->typed_input_tensor<float>(0);
        std::memcpy(input_tensor, intermediate_tensor.data.data(), 
            intermediate_tensor.data.size() * sizeof(float));
        // ====================================

        /* Inference */
        // ======= Write your code here =======
        interpreter->Invoke();
        // ====================================

        /* Get output tensors of the interpreter and copy them to an intermediate tensor */
        // Clear data in it for reuse
        intermediate_tensor.data.clear();
        intermediate_tensor.tensor_boundaries.clear();
        // ======= Let's write together =======
        for (size_t i = 0; i < interpreter->outputs().size(); ++i) {
            // Get i-th output tensor object
            TfLiteTensor* output_tensor = interpreter->output_tensor(i);

            // Calculate tensor size
            int size = 1;
            for (int d = 0; d < output_tensor->dims->size; ++d)
                size *= output_tensor->dims->data[d];

            // Resize intermediate_tensor.data and copy output tensor data into it
            int current_boundary = intermediate_tensor.data.size();
            intermediate_tensor.data.resize(current_boundary + size);
            std::memcpy(intermediate_tensor.data.data() + current_boundary,
                output_tensor->data.f,
                size * sizeof(float));
            intermediate_tensor.tensor_boundaries.push_back(current_boundary + size);
        } // end of for loop
        // ====================================
    
        stage1_to_stage2_queue.push(std::move(intermediate_tensor));

        util::timer_stop(label);
    } // end of while loop

    // Notify stage2_thread that no more data will be sent
    stage1_to_stage2_queue.signal_shutdown();
} // end of stage1_thread_function

void stage2_thread_function(tflite::Interpreter* interpreter) {
    IntermediateTensor intermediate_tensor;

    while (stage1_to_stage2_queue.pop(intermediate_tensor)) {
        std::string label = "Stage2 " + std::to_string(intermediate_tensor.index);
        util::timer_start(label);

        /* Copy each sub-tensor from intermediate_tensor.data into 
        *  the corresponding input tensors */
        // ======= Let's write together =======
        for (size_t i = 0; i < interpreter->inputs().size(); ++i) {
            // Get i-th input tensor from the interpreter
            float* input_data = interpreter->typed_input_tensor<float>(i);

            // Copy data from intermediate tensor to i-th input tensor
            int start_idx = (i == 0) ? 0 : intermediate_tensor.tensor_boundaries[i-1];
            int end_idx = intermediate_tensor.tensor_boundaries[i];
            std::memcpy(input_data, intermediate_tensor.data.data() + start_idx,
                (end_idx - start_idx) * sizeof(float));
        } // end of for loop
        // ====================================

        /* Inference */
        // ======= Write your code here =======
        interpreter->Invoke();
        // ====================================

        /* Extract output tensors and copy them into an intermediate tensor */
        // Clear data in it for reuse
        intermediate_tensor.data.clear();
        intermediate_tensor.tensor_boundaries.clear();
        // ======= Let's write together =======
        for (size_t i = 0; i < interpreter->outputs().size(); ++i) {
            // Get i-th output tensor object
            TfLiteTensor* output_tensor = interpreter->output_tensor(i);

            // Calculate tensor size
            int size = 1;
            for (int d = 0; d < output_tensor->dims->size; ++d)
                size *= output_tensor->dims->data[d];

            // Resize intermediate_tensor.data and copy output tensor data into it
            int current_boundary = intermediate_tensor.data.size();
            intermediate_tensor.data.resize(current_boundary + size);
            std::memcpy(intermediate_tensor.data.data() + current_boundary,
                output_tensor->data.f,
                size * sizeof(float));
            intermediate_tensor.tensor_boundaries.push_back(current_boundary + size);
        } // end of for loop
        // ====================================
        stage2_to_stage3_queue.push(std::move(intermediate_tensor));
        util::timer_stop(label);
    } // end of while loop

    stage2_to_stage3_queue.signal_shutdown();    
} // end of stage2_thread_function

void stage3_thread_function(std::unordered_map<int, std::string> class_labels_map) {
    IntermediateTensor intermediate_tensor;

    while (stage2_to_stage3_queue.pop(intermediate_tensor)) {
        std::string tlabel = "Stage3 " + std::to_string(intermediate_tensor.index);
        util::timer_start(tlabel);

        const std::vector<float>& probs = intermediate_tensor.data;

        if ((intermediate_tensor.index + 1) % 10 == 0) {
            auto top_k_indices = util::get_topK_indices(probs, 3);
            std::cout << "\n[stage3] Top-3 prediction for image index "
                      << intermediate_tensor.index << ":\n";
            for (int idx : top_k_indices) {
                std::string label = class_labels_map.count(idx)
                    ? class_labels_map.at(idx)
                    : "unknown";
                std::cout << "- Class " << idx << " (" << label
                          << "): " << probs[idx] << std::endl;
            }
        }

        util::timer_stop(tlabel);
    } // end of while loop
} // end of stage3_thread_function

int main(int argc, char* argv[]) {
    /* Receive user input */
    if (argc < 7)
    {
        std::cerr << "Usage: " << argv[0] 
            << "<submodel0_path> <submodel0_gpu_usage> <submodel1_path> "      
            << "<submodel1_gpu_usage> <class_labels_path> <image_path 1> "      
            << "[image_path 2 ... image_path N] [--input-period=milliseconds]"
            << std::endl;
        return 1;
    }

    const std::string submodel0_path = argv[1];  // Path to sub-model 0
    bool submodel0_gpu_usage = false;
    const std::string gpu_usage_str1 = argv[2];
    if(gpu_usage_str1 == "true"){
        submodel0_gpu_usage = true;
    }

    const std::string submodel1_path = argv[3];  // Path to sub-model 1
    bool submodel1_gpu_usage = false;
    const std::string gpu_usage_str2 = argv[4];
    if(gpu_usage_str2 == "true"){
        submodel1_gpu_usage = true;
    }
    
    // Load class label mapping, used for postprocessing
    const std::string class_labels_path = argv[5];
    auto class_labels_map = util::load_class_labels(class_labels_path.c_str());

    std::vector<std::string> images;    // List of input image paths
    int input_period_ms = 0;            // Input period in milliseconds, default is 0 (no delay)
    for (int i = 6; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--input-period=", 0) == 0)
            input_period_ms = std::stoi(arg.substr(15)); 
        else
            images.push_back(arg);  // Assume it's an image path
    }

    /* Load models */
    // 1. Create a std::unique_ptr<tflite::FlatBufferModel> for each sub-model
    // ======= Write your code here =======
    std::unique_ptr<tflite::FlatBufferModel> submodel0_model = 
        tflite::FlatBufferModel::BuildFromFile(submodel0_path.c_str());
    std::unique_ptr<tflite::FlatBufferModel> submodel1_model = 
        tflite::FlatBufferModel::BuildFromFile(submodel1_path.c_str());
    if (!submodel0_model || !submodel1_model) {
        std::cerr << "Failed to load one or both models" << std::endl;
        return 1;
    }
    // ====================================

    /* Build interpreters */
    // 1. Create an OpResolver
    // 2. Create two interpreter builders, one for each sub-model
    // 3. Build interpreters using the interpreter builders
    // ======= Write your code here =======
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder submodel0_builder(*submodel0_model, resolver);
    tflite::InterpreterBuilder submodel1_builder(*submodel1_model, resolver);
    std::unique_ptr<tflite::Interpreter> submodel0_interpreter;
    std::unique_ptr<tflite::Interpreter> submodel1_interpreter;
    submodel0_builder(&submodel0_interpreter);
    submodel1_builder(&submodel1_interpreter);
    if (!submodel0_interpreter || !submodel1_interpreter) {
        std::cerr << "Failed to initialize interpreters" << std::endl;
        return 1;
    }
    // ====================================

    /* Apply delegate */
    // 1. Create a XNNPACK delegate
    // 2. Apply the delegate to the submodel0 interpreter
    // 3. Create a GPU delegate
    // 4. Apply the GPU delegate to the submodel1 interpreter
    // ======= Write your code here =======
    TfLiteDelegate* xnn_delegate = TfLiteXNNPackDelegateCreate(nullptr);
    if(submodel0_interpreter->ModifyGraphWithDelegate(xnn_delegate) != kTfLiteOk) {
        std::cerr << "Failed to apply XNNPACK delegate to submodel0" << std::endl;
        return 1;
    }
    TfLiteDelegate* gpu_delegate = TfLiteGpuDelegateV2Create(nullptr);
    if(submodel1_interpreter->ModifyGraphWithDelegate(gpu_delegate) != kTfLiteOk) {
        std::cerr << "Failed to apply GPU delegate to submodel1" << std::endl;
        return 1;
    }
    // ====================================

    /* Allocate tensors */
    // 1. Allocate tensors for both interpreters
    // ======= Write your code here =======
    if (submodel0_interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors for submodel0" << std::endl;
        return 1;
    }
    if (submodel1_interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors for submodel1" << std::endl;
        return 1;
    }
    // ====================================

    // Running pipelined inference driver
    util::timer_start("Total Latency");

    /* Create and launch threads */
    // Hint: std::thread thread_name(function name, arguments...);
    // 1. Launch stage0_thread_function in a new thread with images and input_period_ms
    // 2. Launch stage1_thread_function in a new thread with submodel0 interpreter
    // 3. Launch stage2_thread_function in a new thread with submodel1 interpreter
    // 4. Launch stage3_thread_function in a new thread with class_labels_map
    // ======= Write your code here =======
    std::thread stage0_thread(stage0_thread_function, images, input_period_ms);
    std::thread stage1_thread(stage1_thread_function, submodel0_interpreter.get());
    std::thread stage2_thread(stage2_thread_function, submodel1_interpreter.get());
    std::thread stage3_thread(stage3_thread_function, class_labels_map);
    // ====================================

    // Setting CPU affinity for each thread
    util::set_cpu_affinity(stage0_thread, 4); // Stage 0 on CPU core 4
    util::set_cpu_affinity(stage1_thread, 7); // Stage 1 on CPU core 7
    util::set_cpu_affinity(stage2_thread, 5); // Stage 2 on CPU core 5
    util::set_cpu_affinity(stage3_thread, 6); // Stage 3 on CPU core 6

    /* Wait for threads to finish */  
    stage0_thread.join();
    stage1_thread.join();
    stage2_thread.join();
    stage3_thread.join();

    util::timer_stop("Total Latency");

    /* Print average E2E latency and throughput */
    util::print_average_latency("Stage0");
    util::print_average_latency("Stage1");
    util::print_average_latency("Stage2");
    util::print_average_latency("Stage3");
    util::print_throughput("Total Latency", images.size());

    return 0;
}