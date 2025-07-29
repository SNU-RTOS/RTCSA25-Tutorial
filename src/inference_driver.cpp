#include <iostream>
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
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <gpu_usage> <model_path> <image_path>" << std::endl;
        return 1;
    }

    bool gpu_usage = false; // If true, GPU delegate is applied
    const std::string gpu_usage_str = argv[1];
    if(gpu_usage_str == "true"){
        gpu_usage = true;
    }
    const std::string model_path = argv[2];
    const std::string image_path = argv[3];
    const std::string class_names_path = "./class_names.json";

    /* Load .tflite model */
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
    bool delegate_applied = false;
    if(gpu_usage) {
        if (_litert_interpreter->ModifyGraphWithDelegate(_litert_gpu_delegate) == kTfLiteOk)
        {
            delegate_applied = true;
        } else {
            std::cerr << "Failed to Apply GPU Delegate" << std::endl;
        }
    } else {
        if (_litert_interpreter->ModifyGraphWithDelegate(_litert_xnn_delegate) == kTfLiteOk)
        {
            delegate_applied = true;
        } else {
            std::cerr << "Failed to Apply XNNPACK Delegate" << std::endl;
        }
    }

    if (_litert_interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr << "Failed to Allocate Tensors" << std::endl;
        return 1;
    }

    /* Load input image */
    cv::Mat origin_image = cv::imread(image_path);
    if (origin_image.empty())
        throw std::runtime_error("Failed to load image: " + image_path);

    /* Preprocessing */
    util::timer_start("E2E Total(Pre+Inf+Post)");
    util::timer_start("Preprocessing");

    // Preprocess input data
    cv::Mat preprocessed_image = 
            util::preprocess_image_resnet(origin_image, 224, 224); // Input tensor shape: [3, 224, 224]

    // Copy preprocessed_image to input_tensor
    float* _litert_input_tensor = _litert_interpreter->typed_input_tensor<float>(0);
    std::memcpy(_litert_input_tensor, preprocessed_image.ptr<float>(), 
                preprocessed_image.total() * preprocessed_image.elemSize());

    util::timer_stop("Preprocessing");

    /* Inference */
    util::timer_start("Inference");

    if (_litert_interpreter->Invoke() != kTfLiteOk)
    {
        std::cerr << "Failed to invoke interpreter" << std::endl;
        return 1;
    }

    util::timer_stop("Inference");

    /* PostProcessing */
    util::timer_start("Postprocessing");

    // Get output tensor
    float *_litert_output_tensor = _litert_interpreter->typed_output_tensor<float>(0); // 1x1000 tensor
    int num_classes = 1000; // Total 1000 classes
    std::vector<float> probs(num_classes);
    std::memcpy(probs.data(), _litert_output_tensor, sizeof(float) * num_classes);

    util::timer_stop("Postprocessing");
    util::timer_stop("E2E Total(Pre+Inf+Post)");

    /* Print Results */
    // Load class label mapping
    auto label_map = util::load_class_labels(class_names_path);

    // Print Top-5 results
    std::cout << "\n[INFO] Top 5 predictions:" << std::endl;
    auto top_k_indices = util::get_topK_indices(probs, 5);
    for (int idx : top_k_indices)
    {
        std::string label = label_map.count(idx) ? label_map[idx] : "unknown";
        std::cout << "- Class " << idx << " (" << label << "): " << probs[idx] << std::endl;
    }

    /* Print Timers */
    util::print_all_timers();

    /* Deallocate delegates */
    if (_litert_xnn_delegate)
    {
        TfLiteXNNPackDelegateDelete(_litert_xnn_delegate);
    }
    if (_litert_gpu_delegate)
    {
        TfLiteGpuDelegateV2Delete(_litert_gpu_delegate);
    }
    return 0;
}