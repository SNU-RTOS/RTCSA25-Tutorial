#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"
#include "tflite/delegates/gpu/delegate.h"
#include "tflite/interpreter_builder.h"
#include "tflite/interpreter.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"
#include "util.hpp"

int main(int argc, char *argv[])
{
    if (argc != 4)
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
    const std::string label_path = "./labels.json";

    /* Load model */
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
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

    /* Apply either XNNPACK delegate or GPU delegate */
    TfLiteDelegate *xnn_delegate = TfLiteXNNPackDelegateCreate(nullptr);
    TfLiteDelegate* gpu_delegate = TfLiteGpuDelegateV2Create(nullptr);
    bool delegate_applied = false;
    if(gpu_usage) {
        if (interpreter->ModifyGraphWithDelegate(gpu_delegate) == kTfLiteOk)
        {
            delegate_applied = true;
        } else {
            std::cerr << "Failed to Apply GPU Delegate" << std::endl;
        }
    } else {
        if (interpreter->ModifyGraphWithDelegate(xnn_delegate) == kTfLiteOk)
        {
            delegate_applied = true;
        } else {
            std::cerr << "Failed to Apply XNNPACK Delegate" << std::endl;
        }
    }

    util::print_model_summary(interpreter.get(), delegate_applied);

    /* Allocate Tensor */
    if (!interpreter || interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr << "Failed to initialize interpreter" << std::endl;
        return 1;
    }

    /* Load input image */
    cv::Mat origin_image = cv::imread(image_path);
    if (origin_image.empty())
        throw std::runtime_error("Failed to load image: " + image_path);

    /* Preprocessing */
    util::timer_start("E2E Total(Pre+Inf+Post)");
    util::timer_start("Preprocessing");

    // Get input tensor info
    TfLiteTensor *input_tensor = interpreter->input_tensor(0);
    int input_height = input_tensor->dims->data[1];
    int input_width = input_tensor->dims->data[2];
    int input_channels = input_tensor->dims->data[3];

    std::cout << "\n[INFO] Input shape  : ";
    util::print_tensor_shape(input_tensor);
    std::cout << std::endl;

    // Preprocess input data
    cv::Mat preprocessed_image = 
            util::preprocess_image_resnet(origin_image, input_height, input_width);

    // Copy HWC float32 cv::Mat to TFLite input tensor
    float *input_tensor_value = interpreter->typed_input_tensor<float>(0);
    std::memcpy(input_tensor_value, preprocessed_image.ptr<float>(), preprocessed_image.total() * preprocessed_image.elemSize());

    util::timer_stop("Preprocessing");

    /* Inference */
    util::timer_start("Inference");

    if (interpreter->Invoke() != kTfLiteOk)
    {
        std::cerr << "Failed to invoke interpreter" << std::endl;
        return 1;
    }

    util::timer_stop("Inference");

    /* PostProcessing */
    util::timer_start("Postprocessing");

    // Get output tensor
    TfLiteTensor *output_tensor = interpreter->output_tensor(0);
    std::cout << "[INFO] Output shape : ";
    util::print_tensor_shape(output_tensor);
    std::cout << std::endl;

    float *output_tensor_value = interpreter->typed_output_tensor<float>(0);
    int num_classes = output_tensor->dims->data[1];

    std::vector<float> probs(num_classes);
    std::memcpy(probs.data(), output_tensor_value, sizeof(float) * num_classes);

    util::timer_stop("Postprocessing");
    util::timer_stop("E2E Total(Pre+Inf+Post)");

    /* Print Results */
    // Load class label mapping
    auto label_map = util::load_class_labels(label_path);

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
    if (xnn_delegate)
    {
        TfLiteXNNPackDelegateDelete(xnn_delegate);
    }
    if (gpu_delegate)
    {
        TfLiteGpuDelegateV2Delete(gpu_delegate);
    }
    return 0;
}