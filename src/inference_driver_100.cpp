// xnn-delegate-main
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>

#include "opencv2/opencv.hpp" //opencv

#include "tflite/delegates/xnnpack/xnnpack_delegate.h" //for xnnpack delegate
#include "tflite/model_builder.h"
#include "tflite/interpreter_builder.h"
#include "tflite/interpreter.h"
#include "tflite/kernels/register.h"
#include "tflite/model.h"
#include "util.hpp"

int main(int argc, char *argv[])
{
    std::cout << "====== main_cpu ====" << std::endl;

    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];
    const std::string label_path = "./labels.json";

    /* Load model */
    util::timer_start("Load Model");
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model)
    {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    util::timer_stop("Load Model");

    /* Build interpreter */
    util::timer_start("Build Interpreter");
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    util::timer_stop("Build Interpreter");

    util::print_model_summary(interpreter.get(), false);

    /* Apply XNNPACK delegate */
    util::timer_start("Apply Delegate");
    TfLiteXNNPackDelegateOptions xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
    TfLiteDelegate *xnn_delegate = TfLiteXNNPackDelegateCreate(&xnnpack_opts);
    bool delegate_applied = false;
    if (interpreter->ModifyGraphWithDelegate(xnn_delegate) == kTfLiteOk)
    {
        delegate_applied = true;
    }
    else
    {
        std::cerr << "Failed to Apply XNNPACK Delegate" << std::endl;
    }
    util::timer_stop("Apply Delegate");

    util::print_model_summary(interpreter.get(), delegate_applied);

    /* Allocate Tensor */
    util::timer_start("Allocate Tensor");
    if (!interpreter || interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr << "Failed to initialize interpreter" << std::endl;
        return 1;
    }
    util::timer_stop("Allocate Tensor");

    // Prepare label map
    auto label_map = util::load_class_labels(label_path);

    // Get input tensor info
    TfLiteTensor *input_tensor = interpreter->input_tensor(0);
    int input_height = input_tensor->dims->data[1];
    int input_width = input_tensor->dims->data[2];
    int input_channels = input_tensor->dims->data[3];

    std::cout << "\n[INFO] Input shape  : ";
    util::print_tensor_shape(input_tensor);
    std::cout << std::endl;

    util::timer_start("Total 100 Iterations");

    for (int iter = 0; iter < 100; ++iter)
    {
        util::timer_start("Iteration Total");

        // Load image each time (same file)
        util::timer_start("Load Input Image");
        cv::Mat origin_image = cv::imread(image_path);
        util::timer_stop("Load Input Image");
        if (origin_image.empty())
        {
            std::cerr << "Failed to load image: " << image_path << std::endl;
            return 1;
        }

        // Preprocessing
        util::timer_start("Preprocessing");
        cv::Mat preprocessed_image = util::preprocess_image_resnet(origin_image, input_height, input_width);
        util::timer_stop("Preprocessing");

        if (preprocessed_image.empty())
        {
            std::cerr << "Preprocessing failed" << std::endl;
            return 1;
        }

        // Copy into tensor
        float *input_tensor_buffer = interpreter->typed_input_tensor<float>(0);
        std::memcpy(input_tensor_buffer, preprocessed_image.ptr<float>(),
                    preprocessed_image.total() * preprocessed_image.elemSize());

        // Inference
        util::timer_start("Inference");
        if (interpreter->Invoke() != kTfLiteOk)
        {
            std::cerr << "Failed to invoke interpreter (iter " << iter << ")" << std::endl;
            return 1;
        }
        util::timer_stop("Inference");

        // Postprocessing
        util::timer_start("Postprocessing");
        TfLiteTensor *output_tensor = interpreter->output_tensor(0);
        float *logits = interpreter->typed_output_tensor<float>(0);
        int num_classes = output_tensor->dims->data[1];

        std::vector<float> probs(num_classes);
        //util::softmax(logits, probs, num_classes);
        std::memcpy(probs.data(), logits, sizeof(float) * num_classes);
        util::timer_stop("Postprocessing");

        std::cout << "\n[INFO] Top 5 predictions [iteration " << iter << "]:" << std::endl;
        auto top_k_indices = util::get_topK_indices(probs, 5);
        for (int idx : top_k_indices)
        {
            std::string label = label_map.count(idx) ? label_map[idx] : "unknown";
            std::cout << "- Class " << idx << " (" << label << "): " << probs[idx] << std::endl;
        }

        util::timer_stop("Iteration Total");
    }

    util::timer_stop("Total 100 Iterations");

    util::print_all_timers();
    std::cout << "========================" << std::endl;

    /* Deallocate delegate */
    if (xnn_delegate)
    {
        TfLiteXNNPackDelegateDelete(xnn_delegate);
    }
    return 0;
}