// inference driver internals
#include <iostream>
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"
#include "tflite/delegates/gpu/delegate.h"
#include "tflite/interpreter_builder.h"
#include "tflite/interpreter.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"
#include "tensorflow/compiler/mlir/lite/version.h" // TFLITE_SCHEMA_VERSION is defined inside

void PrintLoadModel();

void PrintInterpreterInstantiation(const tflite::FlatBufferModel* model,
                                    const tflite::ops::builtin::BuiltinOpResolver& resolver,
                                    const tflite::Interpreter* interpreter);

void PrintInterpreterAfterInstantiation(const tflite::Interpreter* interpreter);

void PrintAfterDelegateApplication(const tflite::Interpreter* interpreter);

void PrintTensorsInfo(tflite::Interpreter* interpreter, const std::string& stage);

void PrintExecutionPlan(const tflite::Interpreter* interpreter);