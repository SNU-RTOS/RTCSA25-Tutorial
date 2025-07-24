#include "tflite/delegates/xnnpack/xnnpack_delegate.h" //for xnnpack delegate
#include "tflite/delegates/gpu/delegate.h"             // for gpu delegate
#include "tflite/model_builder.h"
#include "tflite/interpreter_builder.h"
#include "tflite/interpreter.h"
#include "tflite/kernels/register.h"
#include "tflite/model.h"
#include "util.hpp"
#include "thread_safe_queue.hpp"

#include <opencv2/opencv.hpp> //opencv
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <pthread.h>

// --- Data container used to pass results between pipeline stages ---
struct IntermediateResult {
    int index;                            // Index of the input image (used for tracking)
    std::vector<float> data;             // Flattened data (input/output tensor contents)
    std::vector<int> tensor_boundaries;  // Marks boundaries between multiple output tensors (if any)
};

// --- Thread-safe queues for inter-stage communication ---
// queue0: connects stage0 (preprocessing) to stage1 (first inference)
// queue1: connects stage1 to stage2 (final inference)
ThreadSafeQueue<IntermediateResult> queue0;
ThreadSafeQueue<IntermediateResult> queue1;


void stage0_worker(const std::vector<std::string>& images, int rate_ms) {
    std::cout << "[stage0] Started preprocessing thread\n";
    auto next_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < images.size(); ++i) {
        util::timer_start("stage0:total");

        std::cout << "[stage0] Loading image: " << images[i] << std::endl;
        util::timer_start("stage0:load_image");
        cv::Mat origin_image = cv::imread(images[i]);
        util::timer_stop("stage0:load_image");

        if (origin_image.empty()) {
            std::cerr << "[stage0] Failed to load image: " << images[i] << "\n";
            util::timer_stop("stage0:total");
            continue;
        }

        std::cout << "[stage0] Preprocessing image: " << images[i] << std::endl;
        util::timer_start("stage0:preprocess");
        cv::Mat preprocessed_image = util::preprocess_image_resnet(origin_image, 224, 224);
        util::timer_stop("stage0:preprocess");

        if (preprocessed_image.empty()) {
            std::cerr << "[stage0] Preprocessing failed: " << images[i] << "\n";
            util::timer_stop("stage0:total");
            continue;
        }

        util::timer_start("stage0:flatten");
        std::vector<float> input_vector(preprocessed_image.total() * preprocessed_image.channels());
        std::memcpy(input_vector.data(), preprocessed_image.ptr<float>(), input_vector.size() * sizeof(float));
        util::timer_stop("stage0:flatten");

        IntermediateResult ir;
        ir.index = i;
        ir.data = std::move(input_vector);

        std::cout << "[stage0] Enqueuing preprocessed image index: " << ir.index << std::endl;
        util::timer_start("stage0:enqueue");
        queue0.push(ir);
        util::timer_stop("stage0:enqueue");

        util::timer_stop("stage0:total");

        next_time += std::chrono::milliseconds(rate_ms);
        std::this_thread::sleep_until(next_time);
    }

    std::cout << "[stage0] Finished preprocessing. Signaling shutdown.\n";
    queue0.signal_shutdown();
}


void stage1_worker(tflite::Interpreter* interp) {
    std::cout << "[stage1] Started inference thread (model0 / GPU)\n";
    IntermediateResult ir;

    while (queue0.pop(ir)) {
        util::timer_start("stage1:total");

        std::cout << "[stage1] Dequeued image index: " << ir.index << std::endl;

        util::timer_start("stage1:copy_input");
        float* input = interp->typed_input_tensor<float>(0);
        std::copy(ir.data.begin(), ir.data.end(), input);
        util::timer_stop("stage1:copy_input");

        std::cout << "[stage1] Invoking model0...\n";
        util::timer_start("stage1:invoke");
        interp->Invoke();
        util::timer_stop("stage1:invoke");

        util::timer_start("stage1:postprocess");
        std::vector<float> flat;
        std::vector<int> bounds{0};

        for (int idx : interp->outputs()) {
            TfLiteTensor* t = interp->tensor(idx);

            int sz = 1;
            for (int d = 0; d < t->dims->size; ++d)
                sz *= t->dims->data[d];

            int prev = flat.size();
            flat.resize(prev + sz);
            std::copy(t->data.f, t->data.f + sz, flat.begin() + prev);
            bounds.push_back(prev + sz);
        }

        ir.data = std::move(flat);
        ir.tensor_boundaries = std::move(bounds);
        util::timer_stop("stage1:postprocess");

        util::timer_start("stage1:enqueue");
        std::cout << "[stage1] Enqueuing result for image index: " << ir.index << std::endl;
        queue1.push(ir);
        util::timer_stop("stage1:enqueue");

        util::timer_stop("stage1:total");
    }

    std::cout << "[stage1] Finished inference. Signaling shutdown.\n";
    queue1.signal_shutdown();
}



void stage2_worker(tflite::Interpreter* interp) {
    std::cout << "[stage2] Started inference thread (model1 / CPU)\n";
    IntermediateResult ir;

    while (queue1.pop(ir)) {
        util::timer_start("stage2:total");

        std::cout << "[stage2] Dequeued intermediate result for index: " << ir.index << std::endl;

        util::timer_start("stage2:copy_input");
        size_t num_inputs = interp->inputs().size();
        size_t tensors_to_copy = std::min(ir.tensor_boundaries.size() - 1, num_inputs);

        for (size_t tensor_idx = 0; tensor_idx < tensors_to_copy; tensor_idx++) {
            TfLiteTensor* input_tensor = interp->input_tensor(tensor_idx);
            float* input_data = interp->typed_input_tensor<float>(tensor_idx);
            int start_idx = ir.tensor_boundaries[tensor_idx];
            int end_idx = ir.tensor_boundaries[tensor_idx + 1];

            std::copy(ir.data.begin() + start_idx,
                    ir.data.begin() + end_idx,
                    input_data);
        }
        util::timer_stop("stage2:copy_input");


        std::cout << "[stage2] Invoking model1...\n";
        util::timer_start("stage2:invoke");
        interp->Invoke();
        util::timer_stop("stage2:invoke");

        util::timer_start("stage2:postprocess");
        TfLiteTensor* out = interp->tensor(interp->outputs()[0]);
        int numel = 1;
        for (int d = 0; d < out->dims->size; ++d)
            numel *= out->dims->data[d];

        std::vector<float> out_data(numel);
        std::copy(interp->typed_output_tensor<float>(0),
                  interp->typed_output_tensor<float>(0) + numel,
                  out_data.begin());

        std::cout << "[stage2] Top-5 prediction for image index " << ir.index << ":\n";
        auto label_map = util::load_class_labels("labels.json");
        auto top_k_indices = util::get_topK_indices(out_data, 5);
        for (int idx : top_k_indices)
        {
            std::string label = label_map.count(idx) ? label_map[idx] : "unknown";
            std::cout << "- Class " << idx << " (" << label << "): " << out_data[idx] << std::endl;
        }

        util::timer_stop("stage2:postprocess");

        util::timer_stop("stage2:total");
    }

    std::cout << "[stage2] Finished all inference.\n";
}


int main(int argc, char* argv[]) {
    util::timer_start("main:total");

    // --- [1] Parse model paths and input arguments ---
    const char* model0_path = argv[1];  // Path to first model (used in stage1)
    const char* model1_path = argv[2];  // Path to second model (used in stage2)
    std::vector<std::string> images;    // List of input image paths
    int rate_ms = 0;                    // Input rate in milliseconds

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--input-rate=", 0) == 0)
            rate_ms = std::stoi(arg.substr(13));  // Extract rate from --input-rate=XX
        else
            images.push_back(arg);  // Treat as image file path
    }

    util::timer_start("main:load_models");
    auto model0 = tflite::FlatBufferModel::BuildFromFile(model0_path);
    auto model1 = tflite::FlatBufferModel::BuildFromFile(model1_path);
    util::timer_stop("main:load_models");

    util::timer_start("main:build_interpreters");
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interp0, interp1;
    tflite::InterpreterBuilder(*model0, resolver)(&interp0);
    tflite::InterpreterBuilder(*model1, resolver)(&interp1);
    util::timer_stop("main:build_interpreters");

    interp0->SetNumThreads(4);
    interp1->SetNumThreads(1);

    util::timer_start("main:apply_delegate");
    TfLiteGpuDelegateOptionsV2 opts = TfLiteGpuDelegateOptionsV2Default();
    TfLiteDelegate* gpu = TfLiteGpuDelegateV2Create(&opts);
    interp1->ModifyGraphWithDelegate(gpu);
    util::timer_stop("main:apply_delegate");

    util::timer_start("main:allocate_tensors");
    interp0->AllocateTensors();
    interp1->AllocateTensors();
    util::timer_stop("main:allocate_tensors");

    util::timer_start("main:thread_join");
    std::thread t0(stage0_worker, std::ref(images), rate_ms);
    std::thread t1(stage1_worker, interp0.get());
    std::thread t2(stage2_worker, interp1.get());

    t0.join();
    t1.join();
    t2.join();
    util::timer_stop("main:thread_join");

    if (gpu) TfLiteGpuDelegateV2Delete(gpu);

    util::timer_stop("main:total");

    // --- [10] Print all collected timing results ---
    util::print_all_timers();

    return 0;
}

