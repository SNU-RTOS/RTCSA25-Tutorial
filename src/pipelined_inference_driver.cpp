#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <pthread.h>
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
// queue0: from stage0 to stage1
// queue1: from stage1 to stage2
InterStageQueue<IntermediateTensor> queue0;
InterStageQueue<IntermediateTensor> queue1;

void stage0_worker(const std::vector<std::string>& images, int rate_ms) {
    std::cout << "[stage0] Started preprocessing thread\n";
    auto next_wakeup_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < images.size(); ++i) {
        std::string label = "Image " + std::to_string(i) + " Stage 0";
        if(i == 5) util::timer_start(label);

        cv::Mat origin_image = cv::imread(images[i]);

        if (origin_image.empty()) {
            std::cerr << "[stage0] Failed to load image: " << images[i] << "\n";
            util::timer_stop(label);
            continue;
        }
        
        cv::Mat preprocessed_image = util::preprocess_image_resnet(origin_image, 224, 224);
        if (preprocessed_image.empty()) {
            std::cerr << "[stage0] Preprocessing failed: " << images[i] << "\n";
            util::timer_stop(label);
            continue;
        }

        std::vector<float> input_vector(preprocessed_image.total() * preprocessed_image.channels());
        std::memcpy(input_vector.data(), preprocessed_image.ptr<float>(), input_vector.size() * sizeof(float));

        IntermediateTensor intermediate_tensor;
        intermediate_tensor.index = i;
        intermediate_tensor.data = std::move(input_vector);
        queue0.push(intermediate_tensor);

        if(i == 5) util::timer_stop(label);

        next_wakeup_time += std::chrono::milliseconds(rate_ms);
        std::this_thread::sleep_until(next_wakeup_time);
    }

    std::cout << "[stage0] Finished preprocessing. Signaling shutdown.\n";
    queue0.signal_shutdown();
}


void stage1_worker(tflite::Interpreter* interpreter) {
    std::cout << "[stage1] Started inference thread (submodel0)\n";
    IntermediateTensor intermediate_tensor;
    uint count = 0;
    while (queue0.pop(intermediate_tensor)) {
        std::string label = "Image " + std::to_string(count) + " Stage 1";
        if(count == 5) util::timer_start(label);

        float* input = interpreter->typed_input_tensor<float>(0);
        std::copy(intermediate_tensor.data.begin(), intermediate_tensor.data.end(), input);

        interpreter->Invoke();

        std::vector<float> flat;
        std::vector<int> bounds{0};

        for (int idx : interpreter->outputs()) {
            TfLiteTensor* t = interpreter->tensor(idx);

            int sz = 1;
            for (int d = 0; d < t->dims->size; ++d)
                sz *= t->dims->data[d];

            int prev = flat.size();
            flat.resize(prev + sz);
            std::copy(t->data.f, t->data.f + sz, flat.begin() + prev);
            bounds.push_back(prev + sz);
        }

        intermediate_tensor.data = std::move(flat);
        intermediate_tensor.tensor_boundaries = std::move(bounds);
    
        queue1.push(intermediate_tensor);

        if(count == 5) util::timer_stop(label);
        ++count;
    }

    std::cout << "[stage1] Finished inference. Signaling shutdown.\n";
    queue1.signal_shutdown();
}



void stage2_worker(tflite::Interpreter* interpreter, std::unordered_map<int, std::string> label_map) {
    std::cout << "[stage2] Started inference thread (submodel1)\n";
    IntermediateTensor intermediate_tensor;
    uint count = 0; 
    while (queue1.pop(intermediate_tensor)) {
        std::string label = "Image " + std::to_string(count) + " Stage 2";
        if(count == 5) util::timer_start(label);

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
        }

        interpreter->Invoke();

        // !TODO: The postprocessing part is different from the one of the inference driver
        TfLiteTensor* out = interpreter->tensor(interpreter->outputs()[0]);
        int numel = 1;
        for (int d = 0; d < out->dims->size; ++d)
            numel *= out->dims->data[d];

        std::vector<float> out_data(numel);
        std::copy(interpreter->typed_output_tensor<float>(0),
                  interpreter->typed_output_tensor<float>(0) + numel,
                  out_data.begin());

        // !TODO: Is printing out the value everytime necessary?
        auto top_k_indices = util::get_topK_indices(out_data, 3);
        if(count < 5) {
            std::cout << "\n[stage2] Top-3 prediction for image index " << intermediate_tensor.index << ":\n";
            for (int idx : top_k_indices)
            {
                std::string label = label_map.count(idx) ? label_map[idx] : "unknown";
                std::cout << "- Class " << idx << " (" << label << "): " << out_data[idx] << std::endl;
            }
        }

        if(count == 5) util::timer_stop(label);
        ++count;
    }

    std::cout << "[stage2] Finished all inference.\n";
}

void inference_driver_worker(const std::vector<std::string>& images, tflite::Interpreter* interpreter, std::unordered_map<int, std::string> label_map) {

    for (size_t i = 0; i < images.size(); ++i) {
        std::string label = "Image " + std::to_string(i) + " Inference Driver";
        if(i == 5) util::timer_start(label);

        cv::Mat origin_image = cv::imread(images[i]);

        if (origin_image.empty()) {
            std::cerr << "[stage0] Failed to load image: " << images[i] << "\n";
            if(i==5) util::timer_stop(label);
            continue;
        }
        
        cv::Mat preprocessed_image = util::preprocess_image_resnet(origin_image, 224, 224);
        if (preprocessed_image.empty()) {
            std::cerr << "[stage0] Preprocessing failed: " << images[i] << "\n";
            if(i==5) util::timer_stop(label);
            continue;
        }

        // Copy preprocessed_image to input_tensor
        float* input_tensor = interpreter->typed_input_tensor<float>(0);
        std::memcpy(input_tensor, preprocessed_image.ptr<float>(), 
                    preprocessed_image.total() * preprocessed_image.elemSize());


        interpreter->Invoke();

        float *output_tensor = interpreter->typed_output_tensor<float>(0);
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

        if(i == 5) util::timer_stop(label);
    }
}

int main(int argc, char* argv[]) {
    const char* submodel0_path = argv[1];  // Path to first model (used in stage1)
    const char* submodel1_path = argv[2];  // Path to second model (used in stage2)
    const char* original_model_path = argv[3]; // Path to the original model
    std::vector<std::string> images;    // List of input image paths
    int rate_ms = 0;                    // Input rate in milliseconds

    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--input-rate=", 0) == 0)
            rate_ms = std::stoi(arg.substr(13));  // Extract rate from --input-rate=XX
        else
            images.push_back(arg);  // Treat as image file path
    }

    auto submodel0 = tflite::FlatBufferModel::BuildFromFile(submodel0_path);
    auto submodel1 = tflite::FlatBufferModel::BuildFromFile(submodel1_path);
    auto original_model = tflite::FlatBufferModel::BuildFromFile(original_model_path);

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> stage1_interpreter, stage2_interpreter, original_internpreter;
    tflite::InterpreterBuilder(*submodel0, resolver)(&stage1_interpreter);
    tflite::InterpreterBuilder(*submodel1, resolver)(&stage2_interpreter);
    tflite::InterpreterBuilder(*original_model, resolver)(&original_internpreter);

    TfLiteGpuDelegateOptionsV2 opts = TfLiteGpuDelegateOptionsV2Default();
    TfLiteDelegate* gpu_delegate = TfLiteGpuDelegateV2Create(&opts);
    stage2_interpreter->ModifyGraphWithDelegate(gpu_delegate);

    TfLiteDelegate* gpu_delegate_for_original_model = TfLiteGpuDelegateV2Create(&opts);
    original_internpreter->ModifyGraphWithDelegate(gpu_delegate_for_original_model);

    stage1_interpreter->AllocateTensors(); // Inside this function, XNNPACK delegate is automatically applied
    stage2_interpreter->AllocateTensors();
    original_internpreter->AllocateTensors();

    auto label_map = util::load_class_labels("class_names.json"); // An unordered_map of class indices to labels for postprocessing

    // Running inference driver
    util::timer_start("Normal Inference Total");
    std::thread t3(inference_driver_worker, std::ref(images), original_internpreter.get(), label_map);
    t3.join();
    util::timer_stop("Normal Inference Total");

    // Running pipelined inference driver
    util::timer_start("Pipeliend Inference Total");
    std::thread t0(stage0_worker, std::ref(images), rate_ms);
    std::thread t1(stage1_worker, stage1_interpreter.get());
    std::thread t2(stage2_worker, stage2_interpreter.get(), label_map);

    t0.join();
    t1.join();
    t2.join();
    util::timer_stop("Pipeliend Inference Total");

    // !TODO: Add a utility function that compares the ratio of the throughputs
    // and the ratio between the longest stage time and E2E latency of the inference driver

    if (gpu_delegate) TfLiteGpuDelegateV2Delete(gpu_delegate);
    if (gpu_delegate_for_original_model) TfLiteGpuDelegateV2Delete(gpu_delegate_for_original_model);

    util::print_all_timers();

    return 0;
}

