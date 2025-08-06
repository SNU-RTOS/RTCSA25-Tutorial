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

/* ================= Variable Naming Convention =================
* Variables that start with a prefix _litert_ 
* are objects or pointers directly related to inference via LiteRT 
* This includes the model, interpreter, delegates, and tensors
* ============================================================ */

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
                << "<model_path> <gpu_usage> <class_labels_path> <image_path 1> " // mandatory arguments
                << "[image_path 2 ... image_path N] [--input-period=milliseconds]"  // optional arguments
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
    int input_period_ms = 0;                    // Input period in milliseconds, default is 0 (no delay)
    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--input-period=", 0) == 0)
            input_period_ms = std::stoi(arg.substr(15));  // Extract input period from --input-period=XX
        else
            images.push_back(arg);  // Assume it's an image path
    }
    
    /* Load model */
    std::unique_ptr<tflite::FlatBufferModel> _litert_model = 
        tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!_litert_model)
    {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    /* Build interpreter */
    tflite::ops::builtin::BuiltinOpResolver _litert_resolver;
    tflite::InterpreterBuilder _litert_builder(*_litert_model, _litert_resolver);
    std::unique_ptr<tflite::Interpreter> _litert_interpreter;
    _litert_builder(&_litert_interpreter);
    if (!_litert_interpreter)
    {
        std::cerr << "Failed to Initialize Interpreter" << std::endl;
        return 1;
    }

    /* Apply either XNNPACK delegate or GPU delegate */
    TfLiteDelegate* _litert_xnn_delegate = TfLiteXNNPackDelegateCreate(nullptr);
    TfLiteDelegate* _litert_gpu_delegate = TfLiteGpuDelegateV2Create(nullptr);
    if(gpu_usage) {
        if (_litert_interpreter->ModifyGraphWithDelegate(_litert_gpu_delegate) == kTfLiteOk)
        {
            // Delete unused delegate
            if(_litert_xnn_delegate) TfLiteXNNPackDelegateDelete(_litert_xnn_delegate);
        } else {
            std::cerr << "Failed to Apply GPU Delegate" << std::endl;
        }
    } else {
        if (_litert_interpreter->ModifyGraphWithDelegate(_litert_xnn_delegate) == kTfLiteOk)
        {
            // Delete unused delegate
            if(_litert_gpu_delegate) TfLiteGpuDelegateV2Delete(_litert_gpu_delegate);
        } else {
            std::cerr << "Failed to Apply XNNPACK Delegate" << std::endl;
        }
    }

    /* Allocate Tensors */
    if (_litert_interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr << "Failed to Allocate Tensors" << std::endl;
        return 1;
    }

    // Starting inference
    util::timer_start("Total Latency");
    auto next_wakeup_time = std::chrono::high_resolution_clock::now(); // Initialize next wakeup time
    for (int i = 0; i < images.size(); i++) {
        std::string e2e_label = "E2E" + std::to_string(i);
        std::string preprocess_label = "Preprocessing" + std::to_string(i);
        std::string inference_label = "Inference" + std::to_string(i);
        std::string postprocess_label = "Postprocessing" + std::to_string(i);

        util::timer_start(e2e_label);
        util::timer_start(preprocess_label);
        /* Preprocessing */
        // Load input image
        cv::Mat image = cv::imread(images[i]);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << images[i] << "\n";
            continue;
        }

        // Preprocess input data
        cv::Mat preprocessed_image = 
                util::preprocess_image_resnet(image, 224, 224);

        // Copy preprocessed_image to input_tensor
        float* _litert_input_tensor = _litert_interpreter->typed_input_tensor<float>(0);
        std::memcpy(_litert_input_tensor, preprocessed_image.ptr<float>(), 
                    preprocessed_image.total() * preprocessed_image.elemSize());

        util::timer_stop(preprocess_label);

        util::timer_start(inference_label);
        /* Inference */
        if (_litert_interpreter->Invoke() != kTfLiteOk)
        {
            std::cerr << "Failed to invoke the interpreter" << std::endl;
            return 1;
        }
        util::timer_stop(inference_label);

        util::timer_start(postprocess_label);
        /* PostProcessing */
        // Get output tensor
        float *_litert_output_tensor = _litert_interpreter->typed_output_tensor<float>(0);
        int num_classes = 1000; // Total 1000 classes
        std::vector<float> probs(num_classes);
        std::memcpy(probs.data(), _litert_output_tensor, sizeof(float) * num_classes);

        // Print Top-3 results
        std::cout << "\n[INFO] Top 3 predictions:" << std::endl;
        auto top_k_indices = util::get_topK_indices(probs, 3);
        for (int idx : top_k_indices)
        {
            std::string label = class_labels_map.count(idx) ? class_labels_map[idx] : "unknown";
            std::cout << "- Class " << idx << " (" << label << "): " << probs[idx] << std::endl;
        }

        util::timer_stop(postprocess_label);
        util::timer_stop(e2e_label);

        // Sleep to control the input rate
        // If next_wakeup_time is in the past, it will not sleep
        next_wakeup_time += std::chrono::milliseconds(input_period_ms);
        std::this_thread::sleep_until(next_wakeup_time);
    } // end of for loop
    util::timer_stop("Total Latency");

    /* Print average E2E latency and throughput */
    util::print_average_latency("E2E");
    util::print_average_latency("Preprocessing");
    util::print_average_latency("Inference");
    util::print_average_latency("Postprocessing");
    util::print_throughput("Total Latency", images.size());    

    return 0;
}