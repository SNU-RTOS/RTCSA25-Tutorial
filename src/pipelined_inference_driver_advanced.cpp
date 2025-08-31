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
 * There are four stages:
 * 1. Stage 0: Preprocess input on CPU core 4
 * 2. Stage 1: Run inference for submodel 0 on CPU core 7
 * 3. Stage 2: Run inference for submodel 1 on GPU                
 * 4. Stage 3: Postprocess output on CPU core 6 */

// === Queues for inter-stage communication ===
// inter_stage_queues[0] is stage0->stage1, inter_stage_queues[i] is stage i -> i+1
std::vector<std::unique_ptr<InterStageQueue<StageOutput>>> inter_stage_queues;

void stage0_worker(
    const std::vector<std::string>& images, 
    int input_period_ms,
     InterStageQueue<StageOutput>& out_queue) 
{
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

        /* Create an StageOutput, copy preprocessed_image data into it, 
        *  and push it into out_queue */
        // Hint: std::memcpy(destination_ptr, source_ptr, num_bytes);
        StageOutput stage_output;
        // ======= Write your code here =======
        stage_output.index = idx;
        stage_output.data.resize(
            preprocessed_image.total() * preprocessed_image.channels());
        std::memcpy(stage_output.data.data(), preprocessed_image.ptr<float>(), 
            stage_output.data.size() * sizeof(float));
        stage_output.tensor_end_offsets = 
            {static_cast<int>(stage_output.data.size())};
        // ====================================
        out_queue.push(std::move(stage_output));
        ++idx;
        
        util::timer_stop(label);

        // Sleep to control the input rate
        // If next_wakeup_time is in the past, it will not sleep
        next_wakeup_time += std::chrono::milliseconds(input_period_ms);
        std::this_thread::sleep_until(next_wakeup_time);
    } while (idx < images.size());

    // Notify stage1_thread that no more data will be sent
    out_queue.signal_shutdown();
} // end of stage0_worker

void stage_worker(
    int stage_idx, 
    tflite::Interpreter* interpreter, 
    InterStageQueue<StageOutput>& in_queue,
    InterStageQueue<StageOutput>& out_queue) 
{
    StageOutput stage_output;

    while (in_queue.pop(stage_output)) {
        std::string label = "Stage" +  std::to_string(stage_idx) + " " + std::to_string(stage_output.index);
        std::string label_invoke = "Invoke" +  std::to_string(stage_idx) + " " + std::to_string(stage_output.index);
        util::timer_start(label);

        /* Access the 0th input tensor of the interpreter as a float pointer
        *  and copy the contents of stage_output.data into it */
        // Hint: std::memcpy(destination_ptr, source_ptr, num_bytes);
        // ======= Write your code here =======
        for (size_t i = 0; i < interpreter->inputs().size(); ++i) {
            // Get i-th input tensor from the interpreter
            float* input_data = interpreter->typed_input_tensor<float>(i);

            // Copy data from stage_output to i-th input tensor
            int start_idx = (i == 0) ? 0 : stage_output.tensor_end_offsets[i-1];
            int end_idx = stage_output.tensor_end_offsets[i];
            std::memcpy(input_data, stage_output.data.data() + start_idx,
                (end_idx - start_idx) * sizeof(float));
        } // end of for loop
        // ====================================

        util::timer_start(label_invoke);
        /* Inference */
        // ======= Write your code here =======
        interpreter->Invoke();
        // ====================================
        util::timer_stop(label_invoke);

        /* Extract data from the interpreter's output tensors and copy into a StageOutput */
        // Clear data in it for reuse
        stage_output.data.clear();
        stage_output.tensor_end_offsets.clear();
        // ======= Write your code here =======
        for (size_t i = 0; i < interpreter->outputs().size(); ++i) {
            // Get i-th output tensor object
            TfLiteTensor* output_tensor = interpreter->output_tensor(i);

            // Calculate the number of elements in the tensor
            int num_elements = 1;
            for (int d = 0; d < output_tensor->dims->size; ++d)
                num_elements *= output_tensor->dims->data[d];

            // Resize stage_output.data and copy output tensor data into it
            int current_data_length = stage_output.data.size();
            stage_output.data.resize(current_data_length + num_elements);
            std::memcpy(stage_output.data.data() + current_data_length,
                output_tensor->data.f,
                num_elements * sizeof(float));
            stage_output.tensor_end_offsets.push_back(current_data_length + num_elements);
        } // end of for loop
        // ====================================
    
        out_queue.push(std::move(stage_output));

        util::timer_stop(label);
    } // end of while loop

    // Notify next thread that no more data will be sent
    out_queue.signal_shutdown();
} // end of stage_worker

void stageK_worker(
    std::unordered_map<int, std::string> class_labels_map,
    InterStageQueue<StageOutput>& in_queue) 
{
    StageOutput stage_output;

    while (in_queue.pop(stage_output)) {
        std::string tlabel = "StageK " + std::to_string(stage_output.index);
        util::timer_start(tlabel);

        const std::vector<float>& probs = stage_output.data;

        if ((stage_output.index + 1) % 10 == 0) {
            auto top_k_indices = util::get_topK_indices(probs, 3);
            std::cout << "\n[stageK] Top-3 prediction for image index "
                      << stage_output.index << ":\n";
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
} // end of stageK_worker

int main(int argc, char* argv[]) {
    if (argc < 8) {
        std::cerr
            << "Usage: " << argv[0]
            << " <num_inference_stages>"
               " (<submodel_i_path> <gpu_usage_i>){i=0..K-1}"
               " <class_labels_path> <image1> [image2 ... imageN]"
               " [--input-period=ms]\n";
        return 1;
    }

    int argi = 1;
    int num_infer = 0;
    try { num_infer = std::stoi(argv[argi++]); }
    catch (...) {
        std::cerr << "First argument must be an integer num_inference_stages\n";
        return 1;
    }
    if (num_infer < 1) {
        std::cerr << "Total stages must be at least 3. Set num_inference_stages >= 1.\n";
        return 1;
    }

    struct ModelSpec {
        std::string submodel_path;
        bool gpu_usage;  // true -> GPU, false -> XNNPACK
    };
    std::vector<ModelSpec> specs;
    specs.reserve(num_infer);

    for (int i = 0; i < num_infer; ++i) {
        if (argi + 1 >= argc) {
            std::cerr << "Missing arguments for submodel " << i << "\n";
            return 1;
        }
        std::string submodel_path = argv[argi++];
        std::string gpu_s = argv[argi++];
        bool gpu_usage = (gpu_s == "true");
        if (!(gpu_s == "true" || gpu_s == "false")) {
            std::cerr << "gpu_usage for submodel " << i << " must be true or false\n";
            return 1;
        }
        specs.push_back({submodel_path, gpu_usage});
    }

    if (argi >= argc) {
        std::cerr << "Missing class_labels_path\n";
        return 1;
    }
    const std::string class_labels_path = argv[argi++];

    std::vector<std::string> images;
    int input_period_ms = 0;
    for (; argi < argc; ++argi) {
        std::string a = argv[argi];
        if (a.rfind("--input-period=", 0) == 0) {
            try { input_period_ms = std::stoi(a.substr(15)); }
            catch (...) { std::cerr << "Bad --input-period value\n"; return 1; }
            if (input_period_ms < 0) input_period_ms = 0;
        } else {
            images.push_back(a);
        }
    }
    if (images.empty()) {
        std::cerr << "Provide at least one image\n";
        return 1;
    }

    // Load labels
    auto class_labels_map = util::load_class_labels(class_labels_path.c_str());

    // Build models and interpreters
    tflite::ops::builtin::BuiltinOpResolver resolver;

    std::vector<std::unique_ptr<tflite::FlatBufferModel>> submodels;
    submodels.reserve(num_infer);
    for (int i = 0; i < num_infer; ++i) {
        auto submodel = tflite::FlatBufferModel::BuildFromFile(specs[i].submodel_path.c_str());
        if (!submodel) {
            std::cerr << "Failed to load model: " << specs[i].submodel_path << "\n";
            return 1;
        }
        submodels.push_back(std::move(submodel));
    }

    std::vector<std::unique_ptr<tflite::Interpreter>> interpreters(num_infer);
    for (int i = 0; i < num_infer; ++i) {
        tflite::InterpreterBuilder builder(*submodels[i], resolver);
        if (builder(&interpreters[i]) != kTfLiteOk || !interpreters[i]) {
            std::cerr << "Failed to create interpreter for stage " << i << "\n";
            return 1;
        }
    }

    // Apply delegates according to gpu_usage
    std::vector<TfLiteDelegate*> delegates(num_infer, nullptr);
    for (int i = 0; i < num_infer; ++i) {
        if (specs[i].gpu_usage) {
            // TfLiteGpuDelegateOptionsV2 gopt = TfLiteGpuDelegateOptionsV2Default();
            delegates[i] = TfLiteGpuDelegateV2Create(nullptr);
            if (!delegates[i] ||
                interpreters[i]->ModifyGraphWithDelegate(delegates[i]) != kTfLiteOk) {
                std::cerr << "Failed to apply GPU delegate to submodel " << i << "\n";
                return 1;
            } else {
                std::cout << "Applied GPU delegate to submodel " << i << "\n" << std::endl;
            }
        } else {
            // TfLiteXNNPackDelegateOptions xopt = TfLiteXNNPackDelegateOptionsDefault();
            delegates[i] = TfLiteXNNPackDelegateCreate(nullptr);
            if (!delegates[i] ||
                interpreters[i]->ModifyGraphWithDelegate(delegates[i]) != kTfLiteOk) {
                std::cerr << "Failed to apply XNNPACK delegate to submodel " << i << "\n";
                return 1;
            } else {
                std::cout << "Applied XNNPACK delegate to submodel " << i << "\n" << std::endl;
            }
        }
    }

    // Allocate tensors
    for (int i = 0; i < num_infer; ++i) {
        if (interpreters[i]->AllocateTensors() != kTfLiteOk) {
            std::cerr << "Failed to allocate tensors for submodel " << i << "\n";
            return 1;
        }
    }

    inter_stage_queues.clear();
    inter_stage_queues.resize(num_infer + 1);
    for (auto& inter_stage_queue: inter_stage_queues)
        inter_stage_queue = std::make_unique<InterStageQueue<StageOutput>>();

    // Launch threads
    util::timer_start("Total Latency");
    std::vector<std::thread> threads;
    threads.reserve(num_infer + 2);

    // Stage 0 producer
    threads.emplace_back(stage0_worker, images, input_period_ms, std::ref(*inter_stage_queues[0]));

    // Inference stages 1..num_infer
    for (int i = 0; i < num_infer; ++i) {
        int stage_id = i + 1; // keep your metric naming
        threads.emplace_back(stage_worker,
                             stage_id,
                             interpreters[i].get(),
                             std::ref(*inter_stage_queues[i]),
                             std::ref(*inter_stage_queues[i + 1]));
    }

    // Postprocessing stage
    threads.emplace_back(stageK_worker, class_labels_map, std::ref(*inter_stage_queues[num_infer]));

    // CPU affinity example plan
    const std::vector<int> core_plan = {4,7,5,6,3,0,1,2};
    auto pick_core = [&](int idx){ return core_plan[idx % core_plan.size()]; };

    util::set_cpu_affinity(threads[0], pick_core(0)); // stage0
    for (int i = 0; i < num_infer; ++i)
        util::set_cpu_affinity(threads[1 + i], pick_core(1 + i)); // stage1..K - 1
    util::set_cpu_affinity(threads.back(), pick_core(1 + num_infer)); // stage K

    for (auto& th : threads) th.join();
    util::timer_stop("Total Latency");

    // Metrics
    util::print_average_latency("Stage0");
    for (int i = 0; i < num_infer; ++i) {
        std::string s = "Stage" + std::to_string(i + 1);
        std::string invoke_latency = "Invoke" + std::to_string(i + 1);
        util::print_average_latency(s);
        util::print_average_latency(invoke_latency);
    }
    util::print_average_latency("StageK");
    util::print_throughput("Total Latency", images.size());

    // Clean up delegates
    for (int i = 0; i < num_infer; ++i) {
        if (!delegates[i]) continue;
        if (specs[i].gpu_usage) TfLiteGpuDelegateV2Delete(delegates[i]);
        else TfLiteXNNPackDelegateDelete(delegates[i]);
    }
    return 0;
}