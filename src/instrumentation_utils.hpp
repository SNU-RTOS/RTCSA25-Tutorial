// inference driver internals
#include <iostream>
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"
#include "tflite/delegates/gpu/delegate.h"
#include "tflite/interpreter_builder.h"
#include "tflite/interpreter.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"
#include "tensorflow/compiler/mlir/lite/version.h" // TFLITE_SCHEMA_VERSION is defined inside

namespace instrumentation {
    void inspect_model_loading();

    void inspect_interpreter_instantiation(const tflite::FlatBufferModel* model,
                                            const tflite::ops::builtin::BuiltinOpResolver& resolver,
                                            const tflite::Interpreter* interpreter);

    void inspect_interpreter(const tflite::Interpreter* interpreter);

    void inspect_interpreter_with_delegate(const tflite::Interpreter* interpreter);

    void inspect_tensors(tflite::Interpreter* interpreter, const std::string& stage);

    void inspect_inference(const tflite::Interpreter* interpreter);
} // namespace internals
