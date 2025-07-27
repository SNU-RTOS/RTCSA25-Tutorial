// inference driver internals
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
#include "tensorflow/compiler/mlir/lite/version.h" // TFLITE_SCHEMA_VERSION is defined inside

/*
void util::check_proc_maps() {
    // std::string cmd = "ls -l /proc/self";
    pid_t pid = getpid();
    std::stringstream cmd;
    cmd << "cat /proc/" << pid << "/maps | grep tflite";
    std::array<char, 256> buffer;
    std::string result;

    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) {
        std::cerr << "popen() failed!" << std::endl;
    }

    // Equivalent to calling "cat /proc/<pid>/maps | grep tflite"
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        std::cout << buffer.data();  // directly print line
    }

    pclose(pipe);
}
*/

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
    const std::string class_names_path = "./class_names.json";

    /* Load model */
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    /* The model file is mapped to the user-space memory of the process */
    util::check_proc_maps(); // Equivalent to "cat /proc/<pid>/maps | grep tflite"
    // std::this_thread::sleep_for(std::chrono::milliseconds(10000));

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
    /* ======================================================================================================== */
    /* Code snippet for simulating what happens when the operator of the interpreter builder is called */
    // 1. Model Validation

    // Get the root object of the FlatBuffer model.
    // This provides access to the serialized model data
    // (e.g., subgraphs, tensors, operators)
    const tflite::Model* model_root = model->GetModel();
    std::cout << "Schema version of the model: " << model_root->version() 
    << "\nSupported schema version: " << TFLITE_SCHEMA_VERSION << std::endl;

    // 2. Operator mapping
    const auto* op_codes = model_root->operator_codes(); // It is a vector of tflite::OperatorCode
    std::cout << "\nTotal " << op_codes->size() << " operators in the model" << std::endl;

    for (int i = 0; i < op_codes->size(); i++) {
        const auto* opcode = op_codes->Get(i); // The i th operator code in the op_codes
        auto builtin_code = opcode->builtin_code(); // An enum indicating the type of the operator like CONV_2D, RELU, etc.
        std::string op_name = tflite::EnumNameBuiltinOperator(builtin_code);
        int op_version = opcode->version(); // Version of the operator
        const TfLiteRegistration* reg = resolver.FindOp(builtin_code, op_version); // Checks whether the OpResolver supports the operator
        
        std::cout << "[" << i << "] " << op_name << ", version: " << op_version 
        << ", supported: " << (reg ? "Y" : "N") << std::endl;
    }

    // 3. Internal data instantiation
    // 3-1. Extracts subgraph information from the model
    const auto* subGraphs = model_root->subgraphs();
    std::cout << "\nNumber of subgraphs: " << subGraphs->size() << std::endl;
    for( int i = 0; i < subGraphs->size(); i++){
        // Note: tflite::SubGraph is for FlatBuffer serialized subgraph info
        // and tflite::Subgraph is for subgraph class that the interpreter uses
        const tflite::SubGraph* subGraph = subGraphs->Get(i); // Gets the i th SubGraph of the model
        std::cout << "SubGraph [" << i << "] " 
            << (subGraph->name() ? subGraph->name()->str() : "(unnamed)") << std::endl;
        // The space for subgraphs are reserved in the interpreter

        // 3-2. Parse tensor information from the buffer information in the SubGraph
        // verifies the information and sets tensor variables for a subgraph
        const auto* buffers = model_root->buffers(); // Global raw data about weights, bias, and others, shared across subgraphs
        const auto* tensors = subGraph->tensors(); // Tensor data structure that contains shape, type, pointer to a buffer. Not shared across subgraphs

        std::cout << "Total " << tensors->size() << " tensors in SubGraph [" << i << "]" << std::endl;
        for(int i = 0; i < tensors->size(); i++) {
            const auto* tensor = tensors->Get(i);
            int buffer_index = tensor->buffer();
            const auto* buffer = buffers->Get(buffer_index);

            std::string name = tensor->name() ? tensor->name()->str() : "(unnamed)";
            std::string type = tflite::EnumNameTensorType(tensor->type());


            std::cout << "Tensor [" << i << "] " << name
                    << ", type = " << type
                    << ", shape = [";
            if (tensor->shape()) {
                for (int d = 0; d < tensor->shape()->size(); ++d) {
                    std::cout << tensor->shape()->Get(d);
                    if (d < tensor->shape()->size() - 1) std::cout << ", ";
                }
            }
            std::cout << "]"
                    << ", buffer = " << buffer_index;

            // Check if buffer contains actual data
            // If does it is a read-only tensor
            // If not it is a read-write tensor
            if (buffer && buffer->data() && buffer->data()->size() > 0) {
                std::cout << " (has data, size = " << buffer->data()->size() << ")";
            } else {
                std::cout << " (no data)";
            }
            std::cout << std::endl;

            // When a tensor is valid the it is saved in the subgraph's tensor variables
            // If any of the tensors is invalid, an error is raised
        }

        // 3-3. Parses node information in the SubGraph, which is a vector of node indices in execution order
        const auto* operators = subGraph->operators(); // A vector that contains the operators of the subgraph in execution order
        std::cout << "\nTotal " << operators->size() << " operators in SubGraph [" << i << "]" << std::endl;
        for(int i = 0; i < operators->size(); i++) {
            const auto* op = operators->Get(i); // Gets the i th operator in the vector
            int opcode_index = op->opcode_index(); // Gets the operator code of the operator
            const auto* opcode = op_codes->Get(opcode_index);
            std::string op_name = tflite::EnumNameBuiltinOperator(opcode->builtin_code());

            std::cout << "Node [" << i << "]: " << op_name << "\n";

            // Inputs
            std::cout << "  Input tensors: ";
            if (op->inputs()) {
                for (int j = 0; j < op->inputs()->size(); ++j) {
                    std::cout << op->inputs()->Get(j) << " ";
                }
            } else {
                std::cout << "(none)";
            }
            std::cout << "\n";

            // Intermediates
            std::cout << "  Intermediate tensors: ";
            if (op->intermediates()) {
                for (int j = 0; j < op->intermediates()->size(); ++j) {
                    std::cout << op->intermediates()->Get(j) << " ";
                }
            } else {
                std::cout << "(none)";
            }
            std::cout << "\n";

            // Outputs
            std::cout << "  Output tensors: ";
            if (op->outputs()) {
                for (int j = 0; j < op->outputs()->size(); ++j) {
                    std::cout << op->outputs()->Get(j) << " ";
                }
            } else {
                std::cout << "(none)";
            }
            std::cout << "\n";
        }
    } 

    // Now let's check the interpreter, if it is correctly instantiated as we saw through the above code
    std::cout << "\nNumber of subgraphs: " << interpreter->subgraphs_size() << std::endl;
    std::cout << "Number of nodes of subgraph 0: " << interpreter->nodes_size() << std::endl; // Internally returns only the value of subgraph 0
    std::cout << "Execution plan size of subgraph 0: " << interpreter->execution_plan().size() << std::endl; // Internally returns only the value of subgraph 0
    for (int i = 0; i < interpreter->execution_plan().size(); i++) {
        const auto* node_and_reg = interpreter->node_and_registration(i);
        if (!node_and_reg) {
            std::cerr << "Failed to get node " << i << std::endl;
            continue;
        }

        // const TfLiteNode& node = node_and_reg->first;
        const TfLiteRegistration& registration = node_and_reg->second;

        std::cout << "Node " << i << ": " 
            << tflite::EnumNameBuiltinOperator(static_cast<tflite::BuiltinOperator>(registration.builtin_code));

        std::cout << std::endl;
    } // this for loop is implemented as util::print_execution_plan()
    /* ======================================================================================================== */

    /* Apply XNNPACK delegate */
    TfLiteXNNPackDelegateOptions xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
    TfLiteDelegate *xnn_delegate = TfLiteXNNPackDelegateCreate(&xnnpack_opts);
    bool delegate_applied = true;
    if (interpreter->ModifyGraphWithDelegate(xnn_delegate) != kTfLiteOk)
    {
        delegate_applied = false;
    }

    /* ======================================================================================================== */
    /* Code snippet for checking how the subgraph changes after applying a delegate */
    std::cout << "\nNumber of nodes of subgraph 0: " << interpreter->nodes_size() << std::endl;
    for(int node_index = 0; node_index < interpreter->nodes_size(); node_index++) {
        const auto* node_and_reg = interpreter->node_and_registration(node_index);
        
        const TfLiteNode& node = node_and_reg->first;
        const TfLiteRegistration& registration = node_and_reg->second;

        std::cout << "Node " << node_index << ": "
        << tflite::EnumNameBuiltinOperator(static_cast<tflite::BuiltinOperator>(registration.builtin_code))
        << std::endl;
    }

    std::cout << "\nExecution plan size of subgraph 0: " << interpreter->execution_plan().size() << std::endl;
    util::print_execution_plan(interpreter);
    // input and output tensors of the delegate node
    {
        // Number of tensors
        std::cout << "\nNumber of tensors in subgraph 0: " << interpreter->tensors_size() << std::endl;

        for (int i = 0; i < interpreter->tensors_size(); ++i) {
            const TfLiteTensor* t = interpreter->tensor(i);
            if (true/*t->allocation_type != kTfLiteNone*/) {
                std::cout << "Tensor " << i << " is used by interpreter, alloc: ";
                switch(t->allocation_type) {
                    case kTfLiteMemNone: 
                        std::cout << "kTfLiteMemNone" << std::endl;
                        break;
                    case kTfLiteMmapRo: 
                        std::cout << "kTfLiteMmapRo" << std::endl;
                        break;
                    case kTfLiteArenaRw:
                        std::cout << "kTfLiteArenaRw" << std::endl;
                        break;
                    case kTfLiteArenaRwPersistent: 
                        std::cout << "kTfLiteArenaRwPersistent" << std::endl;
                        break;
                    case kTfLiteDynamic: 
                        std::cout << "kTfLiteDynamic" << std::endl;
                        break;
                    case kTfLitePersistentRo: 
                        std::cout << "kTfLitePersistentRo" << std::endl;
                        break;
                    case kTfLiteCustom: 
                        std::cout << "kTfLiteCustom" << std::endl;
                        break;
                    default: 
                        std::cout << "Unknown" << std::endl;
                        break;
                }
            }
        }

        const TfLiteNode& node = (interpreter->node_and_registration(50))->first; // We already know the node index
        const TfLiteRegistration& reg = (interpreter->node_and_registration(50))->second;
        
        // Access input tensors
        std::cout << "\nInputs:\n";
        for (int i = 0; i < node.inputs->size; ++i) {
            int tensor_index = node.inputs->data[i];
            const TfLiteTensor* tensor = interpreter->tensor(tensor_index);
            std::cout << tensor_index << " (type: " << TfLiteTypeGetName(tensor->type)
                    << ", dims: [";
            for (int d = 0; d < tensor->dims->size; ++d) {
                std::cout << tensor->dims->data[d];
                if (d != tensor->dims->size - 1) std::cout << ", ";
            }
            std::cout << "])\n";
        }
        std::cout << std::endl;

        // Access intermediate tensors
        std::cout << "Intermediates:\n";
        for (int i = 0; i < node.intermediates->size; ++i) {
            int tensor_index = node.intermediates->data[i];
            const TfLiteTensor* tensor = interpreter->tensor(tensor_index);
            std::cout << tensor_index << " (type: " << TfLiteTypeGetName(tensor->type)
                    << ", dims: [";
            for (int d = 0; d < tensor->dims->size; ++d) {
                std::cout << tensor->dims->data[d];
                if (d != tensor->dims->size - 1) std::cout << ", ";
            }
            std::cout << "]) ";
        }
        std::cout << std::endl;

        // Access intermediate tensors
        std::cout << "Temporaries:\n";
        for (int i = 0; i < node.temporaries->size; ++i) {
            int tensor_index = node.temporaries->data[i];
            const TfLiteTensor* tensor = interpreter->tensor(tensor_index);
            std::cout << tensor_index << " (type: " << TfLiteTypeGetName(tensor->type)
                    << ", dims: [";
            for (int d = 0; d < tensor->dims->size; ++d) {
                std::cout << tensor->dims->data[d];
                if (d != tensor->dims->size - 1) std::cout << ", ";
            }
            std::cout << "]) ";
        }
        std::cout << std::endl;

        // Access output tensors
        std::cout << "Outputs:\n";
        for (int i = 0; i < node.outputs->size; ++i) {
            int tensor_index = node.outputs->data[i];
            const TfLiteTensor* tensor = interpreter->tensor(tensor_index);
            std::cout << tensor_index << " (type: " << TfLiteTypeGetName(tensor->type)
                    << ", dims: [";
            for (int d = 0; d < tensor->dims->size; ++d) {
                std::cout << tensor->dims->data[d];
                if (d != tensor->dims->size - 1) std::cout << ", ";
            }
            std::cout << "]) ";
        }
        std::cout << std::endl;
    }
    /* ======================================================================================================== */

    /* Allocate Tensor */
    // Types of tensors, what happens inside it
    // How the memory space for ArenaRW tensors changes after AllocateTensors() is called
    if (!interpreter || interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr << "Failed to initialize interpreter" << std::endl;
        return 1;
    }

    /* ======================================================================================================== */
    // Code snippet for simulating what happens during AllocateTensors
    // Memory planner is created per subgraph
    // ArenaRW tensors
    // Sum of memory space compared to the actual memory space that is suitable
    // XNNPACK delegate 쓰면 ArenaRW tensor가 하나 밖에 없어서 줄어들 일이 없음
    /* ======================================================================================================== */


    util::print_model_summary(interpreter.get(), delegate_applied);

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
    cv::Mat preprocessed_image = util::preprocess_image(origin_image, input_height, input_width);

    // Copy HWC float32 cv::Mat to TFLite input tensor
    float *input_tensor_buffer = interpreter->typed_input_tensor<float>(0);
    std::memcpy(input_tensor_buffer, preprocessed_image.ptr<float>(),
                preprocessed_image.total() * preprocessed_image.elemSize());

    util::timer_stop("Preprocessing");

    /* Inference */
    util::timer_start("Inference");

    /* ======================================================================================================== */
    // 여기서는 call stack 보여주는게 좋을 듯

    /* ======================================================================================================== */
    if (interpreter->Invoke() != kTfLiteOk)
    {
        std::cerr << "Failed to invoke interpreter" << std::endl;
        /* Deallocate delegate */
        if (xnn_delegate)
        {
            TfLiteXNNPackDelegateDelete(xnn_delegate);
        }
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

    float *logits = interpreter->typed_output_tensor<float>(0);
    int num_classes = output_tensor->dims->data[1];

    std::vector<float> probs(num_classes);
    util::softmax(logits, probs, num_classes);

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
    std::cout << "========================" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // Sleep for 1 second to see the output before exit

    /* Deallocate delegate */
    if (xnn_delegate)
    {
        TfLiteXNNPackDelegateDelete(xnn_delegate);
    }


    return 0;
}