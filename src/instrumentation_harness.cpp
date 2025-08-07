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
#include "instrumentation_harness_utils.hpp"

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
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] 
                << "<model_path> [gpu_usage]" // mandatory arguments
                << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];

    bool gpu_usage = false; // If true, GPU delegate is applied
    if(argc == 3) {
        const std::string gpu_usage_str = argv[2];
        if(gpu_usage_str == "true"){
            gpu_usage = true;
        }
    }

    /* Load .tflite model */
    std::unique_ptr<tflite::FlatBufferModel> _litert_model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!_litert_model)
    {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    instrumentation::inspect_model_loading();

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
    instrumentation::inspect_interpreter_instantiation(_litert_model.get(), _litert_resolver, _litert_interpreter.get());
    instrumentation::inspect_interpreter(_litert_interpreter.get());

    /* Apply either XNNPACK delegate or GPU delegate */
    TfLiteDelegate* _litert_xnn_delegate = TfLiteXNNPackDelegateCreate(nullptr);
    TfLiteDelegate* _litert_gpu_delegate = TfLiteGpuDelegateV2Create(nullptr);
    bool delegate_applied = false;
    if(gpu_usage) {
        if (_litert_interpreter->ModifyGraphWithDelegate(_litert_gpu_delegate) == kTfLiteOk)
        {
            delegate_applied = true;
            // Delete unused delegate
            if(_litert_xnn_delegate) TfLiteXNNPackDelegateDelete(_litert_xnn_delegate);
        } else {
            std::cerr << "Failed to Apply GPU Delegate" << std::endl;
        }
    } else {
        if (_litert_interpreter->ModifyGraphWithDelegate(_litert_xnn_delegate) == kTfLiteOk)
        {
            delegate_applied = true;
            // Delete unused delegate
            if(_litert_gpu_delegate) TfLiteGpuDelegateV2Delete(_litert_gpu_delegate);
        } else {
            std::cerr << "Failed to Apply XNNPACK Delegate" << std::endl;
        }
    }
    instrumentation::inspect_interpreter_with_delegate(_litert_interpreter.get());

    /* Allocate Tensor */
    instrumentation::inspect_tensors(_litert_interpreter.get(), "Before Allocate Tensors");
    if (_litert_interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr << "Failed to Allocate Tensors" << std::endl;
        return 1;
    }
    instrumentation::inspect_tensors(_litert_interpreter.get(), "After Allocate Tensors");

    /* Preprocessing */
    // We skip the actual preprocessing step here, as it is not the focus of this code

    /* Inference */
    // We skip the actual inference step here, as it is not the focus of this code
    instrumentation::inspect_inference(_litert_interpreter.get());

    /* PostProcessing */
    // We skip the actual postprocessing step here, as it is not the focus of this code

    return 0;
}