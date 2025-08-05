#include <iostream>
#include <vector>
#include <thread>
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

void stage0_worker(const std::vector<std::string>& images, int rate_ms) {
    auto next_wakeup_time = std::chrono::high_resolution_clock::now(); // Initialize next wakeup time
    std::string label = "Stage 0"; // String variable for util::timer_start/stop

    for (size_t i = 0; i < images.size(); ++i) {
        if(i == 6) util::timer_start(label);

        // Load image
        cv::Mat image = cv::imread(images[i]);

        if (image.empty()) {
            std::cerr << "[stage0] Failed to load image: " << images[i] << "\n";
            util::timer_stop(label);
            continue;
        }
        
        // Preprocess image
        cv::Mat preprocessed_image = util::preprocess_image_resnet(image, 224, 224);
        if (preprocessed_image.empty()) {
            std::cerr << "[stage0] Preprocessing failed: " << images[i] << "\n";
            util::timer_stop(label);
            continue;
        }

        // Copy preprocessed_image to input_tensor
        std::vector<float> input_tensor(preprocessed_image.total() * preprocessed_image.channels());
        std::memcpy(input_tensor.data(), preprocessed_image.ptr<float>(), 
                    input_tensor.size() * sizeof(float));

        // Create IntermediateTensor and push to stage0_to_stage1_queue
        IntermediateTensor intermediate_tensor;
        intermediate_tensor.index = i;
        intermediate_tensor.data = std::move(input_tensor);
        stage0_to_stage1_queue.push(intermediate_tensor);

        if(i == 6) util::timer_stop(label);

        // Sleep to control the input rate
        next_wakeup_time += std::chrono::milliseconds(rate_ms);
        std::this_thread::sleep_until(next_wakeup_time);
    } // end of for loop

    // Notify stage1_worker that no more data will be sent
    stage0_to_stage1_queue.signal_shutdown();
}


void stage1_worker(tflite::Interpreter* interpreter) {
    IntermediateTensor intermediate_tensor;
    uint count = 0;
    std::string label = "Stage 1";
    while (stage0_to_stage1_queue.pop(intermediate_tensor)) {
        if(count == 6) util::timer_start(label);

        // TODO: Finish this line
        // float *input_tensor = ???
        std::copy(intermediate_tensor.data.begin(), intermediate_tensor.data.end(), input_tensor);

        // TODO: Invoke the interpreter
        // ???

        std::vector<float> flattened_output;
        std::vector<int> bounds{0};

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

        if(count == 6) util::timer_stop(label);
        ++count;
    } // end of while loop

    stage1_to_stage2_queue.signal_shutdown();
}

void stage2_worker(tflite::Interpreter* interpreter, std::unordered_map<int, std::string> label_map) {
    IntermediateTensor intermediate_tensor;
    uint count = 0; 
    std::string label = "Stage 2";
    
    while (stage1_to_stage2_queue.pop(intermediate_tensor)) {
        if(count == 6) util::timer_start(label);

        size_t num_inputs = interpreter->inputs().size();
        size_t tensors_to_copy = std::min(intermediate_tensor.tensor_boundaries.size() - 1, num_inputs);

        for (size_t tensor_idx = 0; tensor_idx < tensors_to_copy; tensor_idx++) {
            TfLiteTensor* input_tensor = interpreter->input_tensor(tensor_idx);
            float* input_data = interpreter->typed_input_tensor<float>(tensor_idx);
            int start_idx = intermediate_tensor.tensor_boundaries[tensor_idx];
            int end_idx = intermediate_tensor.tensor_boundaries[tensor_idx + 1];

            std::copy(intermediate_tensor.data.begin() + start_idx,
                    intermediate_tensor.data.begin() + end_idx,
                    input_data);
        } // end of for loop

        // TODO: Invoke the interpreter
        // ???

        // TODO: Finish this line
        // float *output_tensor = ???
        int num_classes = 1000;
        std::vector<float> probs(num_classes);
        std::memcpy(probs.data(), output_tensor, sizeof(float) * num_classes);

        // Get top-3 predictions
        auto top_k_indices = util::get_topK_indices(probs, 3);
        if(count < 5) {
            std::cout << "\n[stage2] Top-3 prediction for image index " << intermediate_tensor.index << ":\n";
            for (int idx : top_k_indices)
            {
                std::string label = label_map.count(idx) ? label_map[idx] : "unknown";
                std::cout << "- Class " << idx << " (" << label << "): " << probs[idx] << std::endl;
            }
        }

        if(count == 6) util::timer_stop(label);
        ++count;
    } // end of while loop
}

void inference_driver_worker(const std::vector<std::string>& images, 
    tflite::Interpreter* interpreter, std::unordered_map<int, std::string> label_map) {
    std::string label = "Inference Driver";

    for (size_t i = 0; i < images.size(); ++i) {
        if(i == 6) util::timer_start(label);

        cv::Mat image = cv::imread(images[i]);

        if (image.empty()) {
            std::cerr << "[stage0] Failed to load image: " << images[i] << "\n";
            if(i == 6) util::timer_stop(label);
            continue;
        }
        
        cv::Mat preprocessed_image = util::preprocess_image_resnet(image, 224, 224);
        if (preprocessed_image.empty()) {
            std::cerr << "[stage0] Preprocessing failed: " << images[i] << "\n";
            if(i == 6) util::timer_stop(label);
            continue;
        }

        // Copy preprocessed_image to input_tensor
        // TODO: Finish this line
        // float *input_tensor = ???
        std::memcpy(input_tensor, preprocessed_image.ptr<float>(), 
                    preprocessed_image.total() * preprocessed_image.elemSize());


        // TODO: Invoke the interpreter
        // ???

        // TODO: Finish this line
        // float *output_tensor = ???
        int num_classes = 1000; // Total 1000 classes
        std::vector<float> probs(num_classes);
        std::memcpy(probs.data(), output_tensor, sizeof(float) * num_classes);

        auto top_k_indices = util::get_topK_indices(probs, 3);
        if(i < 5) {
            std::cout << "\n[Inference Driver] Top-3 prediction for image index " << i << ":\n";
            for (int idx : top_k_indices)
            {
                std::string label = label_map.count(idx) ? label_map[idx] : "unknown";
                std::cout << "- Class " << idx << " (" << label << "): " << probs[idx] << std::endl;
            }
        }

        if(i == 6) util::timer_stop(label);
    } // end of for loop
}

int main(int argc, char* argv[]) {
    /* Receive user input */
    const char* submodel0_path = argv[1];  // Path to sub-model 0
    const char* submodel1_path = argv[2];  // Path to sub-model 1
    std::vector<std::string> images;    // List of input image paths
    int rate_ms = 0;                    // Input rate in milliseconds, default is 0 (no delay)

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--input-rate=", 0) == 0)
            rate_ms = std::stoi(arg.substr(13));  // Extract rate from --input-rate=XX
        else
            images.push_back(arg);  // Assume it's an image path
    }

    /* Load models */
    // TODO: Write your code here 

    /* Build interpreters */
    // TODO: Write your code here 

    /* Apply delegate */
    // TODO: Write your code here 

    /* Allocate tensors */
    // TODO: Write your code here 

    // Running pipelined inference driver
    util::timer_start("Pipelined Inference Driver Total");

    auto label_map = util::load_class_labels("class_names.json"); // Variable for postprocessing
    /* Create and launch threads */
    // TODO: Write your code here 
    // Hint: std::thread thread_name(function name, arguments...);

    /* Wait for threads to finish */  
    // TODO: Write your code here 
    // Hint: thread_name.join();

    util::timer_stop("Pipelined Inference Driver Total");

    // Deallocate delegates
    if (gpu_delegate) TfLiteGpuDelegateV2Delete(gpu_delegate);

    /* ==================================================== */
    /* Set up inference runtime for normal inference driver */
    const char* original_model_path = "./models/resnet50.tflite";

    // Load model
    // TODO: Write your code here 

    // Build interpreter
    // TODO: Write your code here 

    // Apply GPU delegate
    // TODO: Write your code here 

    // Allocate tensors
    // TODO: Write your code here 

    util::timer_start("Inference Driver Total");
    // Create and launch inference driver thread
    // TODO: Write your code here 
    // Hint: std::thread thread_name(function name, arguments...);
    
    // Wait for inference driver thread to finish
    // TODO: Write your code here 
    // Hint: thread_name.join();
    
    util::timer_stop("Inference Driver Total");

    // Deallocate GPU delegate
    // TODO: Write your code here 

    /* ==================================================== */
    /* Compare performance */
    // Print all timers
    util::print_all_timers();

    // Compare throughput between inference driver and pipelined inference driver
    util::compare_throughput("Inference Driver Total", "Pipelined Inference Driver Total", images.size());
    
    // Compare the ratio of the longest stage in pipelined inference to the E2E latency of a normal inference
    std::vector<std::string> stage_labels = {"Stage 0", "Stage 1", "Stage 2"};
    util::compare_latency(stage_labels, "Inference Driver");
    /* ==================================================== */

    return 0;
}

