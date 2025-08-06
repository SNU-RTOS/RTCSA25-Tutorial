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
 * 1. Stage 0 Preprocessing (CPU): Load and preprocess input images.
 * 2. Stage 1 Inference (CPU): Run inference on the first submodel and 
    produce intermediate results.
 * 3. Stage 2 Inference (GPU): Take intermediate results from Stage 1 and
    run inference on the second submodel. 
    After the inference, postprocessing of the output is done on the CPU. */

// === Queues for inter-stage communication ===
// stage0_to_stage1_queue: from stage0 to stage1
// stage1_to_stage2_queue: from stage1 to stage2
InterStageQueue<IntermediateTensor> stage0_to_stage1_queue;
InterStageQueue<IntermediateTensor> stage1_to_stage2_queue;

void stage0_function(const std::vector<std::string>& images, int input_period_ms) {
    auto next_wakeup_time = std::chrono::high_resolution_clock::now(); // Initialize next wakeup time

    for (size_t i = 0; i < images.size(); ++i) {
        std::string label = "Stage0 " + std::to_string(i);
        util::timer_start(label);
        /* Preprocessing */
        // Load image
        cv::Mat image = cv::imread(images[i]);
        if (image.empty()) {
            std::cerr << "[Stage0] Failed to load image: " << images[i] << "\n";
            util::timer_stop(label);
            continue;
        }
        
        // Preprocess image
        cv::Mat preprocessed_image = util::preprocess_image_resnet(image, 224, 224);
        if (preprocessed_image.empty()) {
            std::cerr << "[Stage0] Preprocessing failed: " << images[i] << "\n";
            util::timer_stop(label);
            continue;
        }

        /* Push processed tensor into stage0_to_stage1_queue */
        // Copy preprocessed_image to input_tensor
        std::vector<float> input_tensor(preprocessed_image.total() * preprocessed_image.channels());
        std::memcpy(input_tensor.data(), preprocessed_image.ptr<float>(), 
                    input_tensor.size() * sizeof(float));

        // Create IntermediateTensor and push to stage0_to_stage1_queue
        IntermediateTensor intermediate_tensor;
        intermediate_tensor.index = i;
        intermediate_tensor.data = std::move(input_tensor);
        intermediate_tensor.tensor_boundaries = {static_cast<int>(input_tensor.size())};
        stage0_to_stage1_queue.push(intermediate_tensor);

        util::timer_stop(label);

        // Sleep to control the input rate
        // If next_wakeup_time is in the past, it will not sleep
        next_wakeup_time += std::chrono::milliseconds(input_period_ms);
        std::this_thread::sleep_until(next_wakeup_time);
    } // end of for loop

    // Notify stage1_thread that no more data will be sent
    stage0_to_stage1_queue.signal_shutdown();
}


void stage1_function(tflite::Interpreter* interpreter) {
    IntermediateTensor intermediate_tensor;

    while (stage0_to_stage1_queue.pop(intermediate_tensor)) {
        std::string label = "Stage1 " + std::to_string(intermediate_tensor.index);
        util::timer_start(label);

        /* Acess the 0th input tensor of sub-model 0 */
        // ======= Write your code here =======
        float *input_tensor = interpreter->typed_input_tensor<float>(0);
        // ====================================
        std::copy(intermediate_tensor.data.begin(), intermediate_tensor.data.end(), input_tensor);

        /* Inference */
        // ======= Write your code here =======
        interpreter->Invoke();
        // ====================================

        /* Get output tensors and push it into the stage1_to_stage2_queue */
        std::vector<float> flattened_output;
        std::vector<int> bounds{};
        for (int idx : interpreter->outputs()) {
            TfLiteTensor* output_tensor = interpreter->tensor(idx);

            int size = 1;
            for (int d = 0; d < output_tensor->dims->size; ++d)
                size *= output_tensor->dims->data[d];

            int current_boundary = flattened_output.size();
            flattened_output.resize(current_boundary + size);
            std::copy(output_tensor->data.f, output_tensor->data.f + size, flattened_output.begin() + current_boundary);
            bounds.push_back(current_boundary + size);
        } // end of for loop

        intermediate_tensor.data = std::move(flattened_output);
        intermediate_tensor.tensor_boundaries = std::move(bounds);
    
        stage1_to_stage2_queue.push(intermediate_tensor);

        util::timer_stop(label);
    } // end of while loop

    // Notify stage2_thread that no more data will be sent
    stage1_to_stage2_queue.signal_shutdown();
}

void stage2_function(tflite::Interpreter* interpreter, std::unordered_map<int, std::string> class_labels_map) {
    IntermediateTensor intermediate_tensor;

    while (stage1_to_stage2_queue.pop(intermediate_tensor)) {
        std::string label = "Stage2 " + std::to_string(intermediate_tensor.index);
        util::timer_start(label);

        /* Retrieve input tensors from the intermediate tensor */
        size_t num_inputs = interpreter->inputs().size();
        size_t tensors_to_copy = std::min(intermediate_tensor.tensor_boundaries.size(), num_inputs);

        for (size_t tensor_idx = 0; tensor_idx < tensors_to_copy; tensor_idx++) {
            TfLiteTensor* input_tensor = interpreter->input_tensor(tensor_idx);
            float* input_data = interpreter->typed_input_tensor<float>(tensor_idx);
            int start_idx = (tensor_idx == 0) ? 0 : intermediate_tensor.tensor_boundaries[tensor_idx];
            int end_idx = intermediate_tensor.tensor_boundaries[tensor_idx + 1];

            std::copy(intermediate_tensor.data.begin() + start_idx,
                    intermediate_tensor.data.begin() + end_idx,
                    input_data);
        } // end of for loop

        /* Inference */
        // ======= Write your code here =======
        interpreter->Invoke();
        // ====================================

        /* Postprocessing */
        // Access the 0th output tensor
        // ======= Write your code here =======
        float *output_tensor = interpreter->typed_output_tensor<float>(0);
        // ====================================
        int num_classes = 1000;
        std::vector<float> probs(num_classes);
        std::memcpy(probs.data(), output_tensor, sizeof(float) * num_classes);

        // Get top-3 predictions
        auto top_k_indices = util::get_topK_indices(probs, 3);
        std::cout << "\n[stage2] Top-3 prediction for image index " << intermediate_tensor.index << ":\n";
        for (int idx : top_k_indices)
        {
            std::string label = class_labels_map.count(idx) ? class_labels_map[idx] : "unknown";
            std::cout << "- Class " << idx << " (" << label << "): " << probs[idx] << std::endl;
        }

        util::timer_stop(label);
    } // end of while loop
}

int main(int argc, char* argv[]) {
    /* Receive user input */
    if (argc < 7)
    {
        std::cerr << "Usage: " << argv[0] 
                << "<submodel0_path> <submodel0_gpu_usage> <submodel1_path> "       // mandatory arguments
                << "<submodel1_gpu_usage> <class_labels_path> <image_path 1> "      
                << "[image_path 2 ... image_path N] [--input-period=milliseconds]"  // optional arguments
                << std::endl;
        return 1;
    }

    const std::string submodel0_path = argv[1];  // Path to sub-model 0
    bool submodel0_gpu_usage = false; // If true, GPU delegate is applied to submodel0
    const std::string gpu_usage_str1 = argv[2];
    if(gpu_usage_str1 == "true"){
        submodel0_gpu_usage = true;
    }

    const std::string submodel1_path = argv[3];  // Path to sub-model 1
    bool submodel1_gpu_usage = false; // If true, GPU delegate is applied
    const std::string gpu_usage_str2 = argv[4];
    if(gpu_usage_str2 == "true"){
        submodel1_gpu_usage = true;
    }
    
    // Load class label mapping, used for postprocessing
    const std::string class_labels_path = argv[5];
    auto class_labels_map = util::load_class_labels(class_labels_path.c_str());

    std::vector<std::string> images;    // List of input image paths
    int input_period_ms = 0;                    // Input period in milliseconds, default is 0 (no delay)
    for (int i = 6; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--input-period=", 0) == 0)
            input_period_ms = std::stoi(arg.substr(15));  // Extract input period from --input-period=XX
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
    // ====================================

    /* Apply delegate */
    // 1. Create a XNNPACK delegate
    // 2. Apply the delegate to the stage1 interpreter
    // 3. Create a GPU delegate
    // 4. Apply the GPU delegate to the stage2 interpreter
    // ======= Write your code here =======
    TfLiteDelegate* xnn_delegate_1 = TfLiteXNNPackDelegateCreate(nullptr);
    TfLiteDelegate* gpu_delegate_1 = TfLiteGpuDelegateV2Create(nullptr);
    if (submodel0_gpu_usage) {
        if (submodel0_interpreter->ModifyGraphWithDelegate(gpu_delegate_1) == kTfLiteOk) {
            // Delete unused delegate
            if (xnn_delegate_1) TfLiteXNNPackDelegateDelete(xnn_delegate_1);
        } else {
            std::cerr << "Failed to apply GPU delegate to submodel0" << std::endl;
        }
        
    } else {
        if (submodel0_interpreter->ModifyGraphWithDelegate(xnn_delegate_1) == kTfLiteOk) {
            // Delete unused delegate
            if (gpu_delegate_1) TfLiteGpuDelegateV2Delete(gpu_delegate_1);
        } else {
            std::cerr << "Failed to apply XNNPACK delegate to submodel0" << std::endl;
        }
    }
    TfLiteDelegate* xnn_delegate_2 = TfLiteXNNPackDelegateCreate(nullptr);
    TfLiteDelegate* gpu_delegate_2 = TfLiteGpuDelegateV2Create(nullptr);
    if (submodel1_gpu_usage) {
        if (submodel1_interpreter->ModifyGraphWithDelegate(gpu_delegate_2) == kTfLiteOk) {
            // Delete unused delegate
            if (xnn_delegate_2) TfLiteXNNPackDelegateDelete(xnn_delegate_2);
        } else {
            std::cerr << "Failed to apply GPU delegate to submodel1" << std::endl;
        }
    } else {
        if (submodel1_interpreter->ModifyGraphWithDelegate(xnn_delegate_2) == kTfLiteOk) {
            // Delete unused delegate
            if (gpu_delegate_2) TfLiteGpuDelegateV2Delete(gpu_delegate_2);
        } else {
            std::cerr << "Failed to apply XNNPACK delegate to submodel1" << std::endl;
        }
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
    // 1. Launch stage0_function thread which takes images and input_period_ms
    // 2. Launch stage1_function thread which takes stage1 interpreter
    // 3. Launch stage2_function thread which takes stage2 interpreter and class_labels_map
    // ======= Write your code here =======
    std::thread stage0_thread(stage0_function, images, input_period_ms);
    std::thread stage1_thread(stage1_function, submodel0_interpreter.get());
    std::thread stage2_thread(stage2_function, submodel1_interpreter.get(), class_labels_map);
    // ====================================

    /* Wait for threads to finish */  
    // Hint: thread_name.join();
    // ======= Write your code here =======
    stage0_thread.join();
    stage1_thread.join();
    stage2_thread.join();
    // ====================================

    util::timer_stop("Total Latency");

    /* Print average E2E latency and throughput */
    util::print_average_latency("Stage0");
    util::print_average_latency("Stage1");
    util::print_average_latency("Stage2");
    util::print_throughput("Total Latency", images.size());

    return 0;
}