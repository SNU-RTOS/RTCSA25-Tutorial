/*
 * Filename: inference_driver.cpp
 *
 * @Author: GeonhaPark
 * @Affiliation: Real-Time Operating System Laboratory, Seoul National University
 * @Created: 07/23/25
 * @Original Work: Based on minimal-litert-c repository (https://github.com/SNU-RTOS/minimal-litert-c)
 * @Modified by: Namcheol Lee on 08/06/25
 * @Contact: {nclee,ghpark,thkim}@redwood.snu.ac.kr
 *
 * @Description: Inference driver codes
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

/* ============ Function Naming Convention of LiteRT ============
 * Public C++ class methods: UpperCamelCase (e.g., BuildFromFile)
 * Internal helpers: snake_case (e.g., typed_input_tensor)
 * ============================================================ */

int main(int argc, char *argv[])
{
    /* Receive user input */
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0] 
            << "<model_path> <gpu_usage> <class_labels_path> <image_path 1> "
            << "[image_path 2 ... image_path N] [--input-period=milliseconds]"
            << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];

    bool gpu_usage = false; // If true, GPU delegate is applied
    const std::string gpu_usage_str = argv[2];
    if(gpu_usage_str == "true"){
        gpu_usage = true;
    }
    
    // Load class label mapping, used for postprocessing
    const std::string class_labels_path = argv[3];
    auto class_labels_map = util::load_class_labels(class_labels_path.c_str());

    std::vector<std::string> images;    // List of input image paths
    int input_period_ms = 0;            // Input period in milliseconds, default is 0 (no delay)
    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--input-period=", 0) == 0) // Check for input period argument
            input_period_ms = std::stoi(arg.substr(15));
        else 
            images.push_back(arg);  // Assume it's an image path
    }
    
    /* Load model */
    std::unique_ptr<tflite::FlatBufferModel> model = 
        tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model)
    {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    /* Build interpreter */
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    if (!interpreter)
    {
        std::cerr << "Failed to Initialize Interpreter" << std::endl;
        return 1;
    }

    /* Apply either XNNPACK delegate or GPU delegate */
    TfLiteDelegate* xnn_delegate = TfLiteXNNPackDelegateCreate(nullptr);
    TfLiteDelegate* gpu_delegate = TfLiteGpuDelegateV2Create(nullptr);
    if(gpu_usage) {
        if (interpreter->ModifyGraphWithDelegate(gpu_delegate) == kTfLiteOk)
        {
            // Delete unused delegate
            if(xnn_delegate) TfLiteXNNPackDelegateDelete(xnn_delegate);
        } else {
            std::cerr << "Failed to Apply GPU Delegate" << std::endl;
        }
    } else {
        if (interpreter->ModifyGraphWithDelegate(xnn_delegate) == kTfLiteOk)
        {
            // Delete unused delegate
            if(gpu_delegate) TfLiteGpuDelegateV2Delete(gpu_delegate);
        } else {
            std::cerr << "Failed to Apply XNNPACK Delegate" << std::endl;
        }
    }

    /* Allocate Tensors */
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr << "Failed to Allocate Tensors" << std::endl;
        return 1;
    }

    // Starting inference
    util::timer_start("Total Latency");
    // Initialize next wakeup time
    int count = 0;
    auto next_wakeup_time = std::chrono::high_resolution_clock::now();
    do {
        std::string e2e_label = "E2E" + std::to_string(count);
        std::string preprocess_label = "Preprocessing" + std::to_string(count);
        std::string inference_label = "Inference" + std::to_string(count);
        std::string postprocess_label = "Postprocessing" + std::to_string(count);

        util::timer_start(e2e_label);
        util::timer_start(preprocess_label);
        /* Preprocessing */
        // Load input image
        cv::Mat image = cv::imread(images[count]);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << images[count] << "\n";
            continue;
        }

        // Preprocess input data
        cv::Mat preprocessed_image = 
                util::preprocess_image_resnet(image, 224, 224);

        // Copy preprocessed_image to input_tensor
        float* input_tensor = interpreter->typed_input_tensor<float>(0);
        std::memcpy(input_tensor, preprocessed_image.ptr<float>(), 
                    preprocessed_image.total() * preprocessed_image.elemSize());

        util::timer_stop(preprocess_label);

        util::timer_start(inference_label);
        /* Inference */
        if (interpreter->Invoke() != kTfLiteOk)
        {
            std::cerr << "Failed to invoke the interpreter" << std::endl;
            return 1;
        }
        util::timer_stop(inference_label);

        util::timer_start(postprocess_label);
        /* PostProcessing */
        // Get output tensor
        float *output_tensor = interpreter->typed_output_tensor<float>(0);
        int num_classes = 1000; // Total 1000 classes
        std::vector<float> probs(num_classes);
        std::memcpy(probs.data(), output_tensor, sizeof(float) * num_classes);

        // Print Top-3 predictions every 10 iterations
        if ((count + 1) % 10 == 0) {
            std::cout << "\n[INFO] Top 3 predictions for image index " << count << ":" 
            << std::endl;
            auto top_k_indices = util::get_topK_indices(probs, 3);
            for (int idx : top_k_indices)
            {
                std::string label = 
                    class_labels_map.count(idx) ? class_labels_map[idx] : "unknown";
                std::cout << "- Class " << idx << " (" << label << "): " 
                    << probs[idx] << std::endl;
            }
        }

        util::timer_stop(postprocess_label);
        util::timer_stop(e2e_label);

        // Sleep to control the input rate
        // If next_wakeup_time is in the past, it will not sleep
        next_wakeup_time += std::chrono::milliseconds(input_period_ms);
        std::this_thread::sleep_until(next_wakeup_time);
        ++count;
    } while (count < images.size());
    util::timer_stop("Total Latency");

    /* Print average E2E latency and throughput */
    util::print_average_latency("E2E");
    util::print_average_latency("Preprocessing");
    util::print_average_latency("Inference");
    util::print_average_latency("Postprocessing");
    util::print_throughput("Total Latency", images.size());    

    return 0;
}