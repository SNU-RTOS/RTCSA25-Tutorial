// inference driver internals
#include <iostream>
#include "internals_sample.hpp"

namespace internals {
/* Load .tflite Model */
// Equivalent to "cat /proc/<process_id>/maps | grep tflite"
void inspect_model_loading() {
    std::cout << "\n==== Model Loading ====" << std::endl;
    pid_t pid = getpid();
    std::stringstream cmd;
    cmd << "cat /proc/" << pid << "/maps | grep tflite";
    std::array<char, 256> buffer;
    std::string result;

    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) {
        std::cerr << "popen() failed!" << std::endl;
    }

    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        std::cout << buffer.data();
    }

    pclose(pipe);    
    
    std::cout << "Press Enter to continue...";
    std::cin.get();  // Wait for Enter
}

/* Build Interpreter */
void inspect_interpreter_instantiation(const tflite::FlatBufferModel* model,
                                    const tflite::ops::builtin::BuiltinOpResolver& resolver,
                                    const tflite::Interpreter* interpreter) {
    std::cout << "\n==== Interpreter Instantiation ====" << std::endl;
    // 1. Model Validation
    // Get the root object of the FlatBuffer model.
    // This provides access to the serialized model data
    // (e.g., subgraphs, tensors, operators)
    const tflite::Model* model_root = model->GetModel();
    std::cout << "\nStep 1: Model Validation" << std::endl;
    std::cout << "\nSchema version of the model: " << model_root->version() 
    << "\nSupported schema version: " << TFLITE_SCHEMA_VERSION << std::endl;

    std::cout << "Press Enter to continue...";
    std::cin.get();  // Wait for Enter

    // 2. Operator mapping
    std::cout << "\nStep 2: Operator Mapping" << std::endl;
    const auto* op_codes = model_root->operator_codes(); // It is a vector of tflite::OperatorCode
    std::cout << "\nTotal " << op_codes->size() << " operators in the model" << std::endl;

    for (int i = 0; i < op_codes->size(); i++) {
        const auto* opcode = op_codes->Get(i); // The i th operator code in the op_codes
        auto builtin_code = opcode->builtin_code(); // An enum indicating the type of the operator like CONV_2D, RELU, etc.
        std::string op_name = tflite::EnumNameBuiltinOperator(builtin_code);
        int op_version = opcode->version(); // Version of the operator
        const TfLiteRegistration* reg = resolver.FindOp(builtin_code, op_version); // Checks whether the OpResolver supports the operator
        
        std::cout << "[" << i << "] " << op_name << ", version: " << op_version 
        << ", supported (Y/N): " << (reg ? "Y" : "N") << std::endl;
    }
    std::cout << "Press Enter to continue...";
    std::cin.get();  // Wait for Enter

    // 3. Internal data instantiation
    std::cout << "\nStep 3: Internal Data Instantiation" << std::endl;
    // 3-1. Extracts subgraph information from the model
    std::cout << "\nStep 3-1: Subgraph Extraction" << std::endl;
    const auto* subGraphs = model_root->subgraphs();
    std::cout << "\nNumber of subgraphs: " << subGraphs->size() << std::endl;
    std::cout << "Press Enter to continue...";
    std::cin.get();  // Wait for Enter

    for( int i = 0; i < subGraphs->size(); i++){
        // Note: tflite::SubGraph is for FlatBuffer serialized subgraph info
        // and tflite::Subgraph is for subgraph class that the interpreter uses
        const tflite::SubGraph* subGraph = subGraphs->Get(i); // Gets the i th SubGraph of the model
        std::cout << "SubGraph [" << i << "] " 
            << (subGraph->name() ? subGraph->name()->str() : "(unnamed)") << std::endl;
        // The space for subgraphs are reserved in the interpreter

        // 3-2. Parse tensor information from the buffer information in the SubGraph
        std::cout << "\nStep 3-2: Tensor Extraction" << std::endl;
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
        std::cout << "Press Enter to continue...";
        std::cin.get();  // Wait for Enter

        // 3-3. Parses node information in the SubGraph, which is a vector of node indices in execution order
        std::cout << "\nStep 3-3: Node Extraction" << std::endl;
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
        if(i != 0) {
            std::cout << "Press Enter to continue to next subgraph...";
            std::cin.get();  // Wait for Enter
        }
    }
    std::cout << "Press Enter to continue...";
    std::cin.get();  // Wait for Enter
}

void inspect_interpreter(const tflite::Interpreter* interpreter) {
    std::cout << "\n==== Interpreter Inspection ====" << std::endl;
    // Now let's check the interpreter, if it is correctly instantiated as we saw through the above code
    std::cout << "\nNumber of subgraphs: " << interpreter->subgraphs_size() << std::endl;
    std::cout << "Number of nodes of subgraph 0: " << interpreter->nodes_size() << std::endl; // Internally returns only the value of subgraph 0
    std::cout << "Number of tensors in subgraph 0: " << interpreter->tensors_size() << std::endl; // Number of tensors
    std::cout << "Execution plan size of subgraph 0: " << interpreter->execution_plan().size() << std::endl; // Internally returns only the value of subgraph 0
    for (int i = 0; i < interpreter->execution_plan().size(); i++) {
        const auto* node_and_reg = interpreter->node_and_registration(i);
        if (!node_and_reg) {
            std::cerr << "Failed to get node " << i << std::endl;
            continue;
        }

        const TfLiteRegistration& registration = node_and_reg->second;

        std::cout << "Node " << i << ": " 
            << tflite::EnumNameBuiltinOperator(static_cast<tflite::BuiltinOperator>(registration.builtin_code));

        std::cout << std::endl;
    } 
    std::cout << "Press Enter to continue...";
    std::cin.get();  // Wait for Enter
}

/* Apply Delegate */
std::unordered_set<int> used_tensor_indices; // for tensor allocation during allocate tensors
void inspect_interpreter_with_delegate(const tflite::Interpreter* interpreter) {
    std::cout << "\n==== Inspect Interpreter with Delegate ====" << std::endl;
    std::cout << "\nNumber of nodes of subgraph 0: " << interpreter->nodes_size() << std::endl;
    for(int node_index = 0; node_index < interpreter->nodes_size(); node_index++) {
        const auto* node_and_reg = interpreter->node_and_registration(node_index);
        const TfLiteRegistration& registration = node_and_reg->second;

        std::cout << "Node " << node_index << ": "
        << tflite::EnumNameBuiltinOperator(static_cast<tflite::BuiltinOperator>(registration.builtin_code))
        << std::endl;
    }
    std::cout << "Press Enter to continue...";
    std::cin.get();  // Wait for Enter

    std::cout << "\nExecution plan size of subgraph 0: " << interpreter->execution_plan().size() << std::endl;
    for (int i = 0; i < interpreter->execution_plan().size(); i++) {
        const auto* node_and_reg = interpreter->node_and_registration(interpreter->execution_plan()[i]);
        if (!node_and_reg) {
            std::cerr << "Failed to get node " << interpreter->execution_plan()[i] << std::endl;
            continue;
        }

        const TfLiteRegistration& registration = node_and_reg->second;

        std::cout << "Node " << interpreter->execution_plan()[i] << ": " 
            << tflite::EnumNameBuiltinOperator(static_cast<tflite::BuiltinOperator>(registration.builtin_code));

        std::cout << std::endl;
    } 
    std::cout << "Press Enter to continue...";
    std::cin.get();  // Wait for Enter

    // input and output tensors of the delegate node
    {
        const TfLiteNode& node = (interpreter->node_and_registration(interpreter->nodes_size()-1))->first; // Get the last node
        int tensor_count = 0;
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
            tensor_count++;
            used_tensor_indices.insert(tensor_index); // Track used tensors for allocation
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
            tensor_count++;
            used_tensor_indices.insert(tensor_index); // Track used tensors for allocation
        }
        std::cout << std::endl;
        std::cout << "\nTotal " << tensor_count << " tensors are used." << std::endl;
    }
    std::cout << "Press Enter to continue...";
    std::cin.get();  // Wait for Enter
}

/* Allocate Tensors */
void inspect_tensors(tflite::Interpreter* interpreter, const std::string& stage) {

    auto AllocTypeToStr = [](TfLiteAllocationType type) -> std::string {
        switch (type) {
            case kTfLiteMmapRo: return "kTfLiteMmapRo";
            case kTfLiteArenaRw: return "kTfLiteArenaRw";
            default: return "Unknown";
        }
    };

    std::cout << "\n==== " << stage << " (used tensors only) ====" << std::endl;
    for (size_t i = 0; i < interpreter->tensors_size(); i++) {
        if (!used_tensor_indices.count(i)) continue;  // Skip unused tensors
        TfLiteTensor* tensor = interpreter->tensor(i);
        if (!tensor) continue;

        void* data_ptr = tensor->data.raw;  // Do NOT call typed_tensor before allocation

        std::cout << "Tensor " << i << ": " 
                  << (tensor->name ? tensor->name : "unnamed")
                  << " | AllocType: " << AllocTypeToStr(tensor->allocation_type);

        if (tensor->dims) {
            std::cout << " | Shape: [";
            for (int d = 0; d < tensor->dims->size; d++) {
                std::cout << tensor->dims->data[d] << (d < tensor->dims->size - 1 ? ", " : "");
            }
            std::cout << "]";
        }

        std::cout << " | Bytes: " << tensor->bytes;
        std::cout << " | Address: " << data_ptr << std::endl;
    }
    std::cout << "Press Enter to continue...";
    std::cin.get();  // Wait for Enter
}

/* Invoke */
void inspect_inference(const tflite::Interpreter* interpreter) {
    std::cout << "==== Invoke ====" << std::endl;
    const auto& plan = interpreter->execution_plan();
    for (size_t i = 0; i < plan.size(); i++) {
        int node_index = plan[i];
        auto node_and_reg = interpreter->node_and_registration(node_index);
        const TfLiteRegistration* reg = &node_and_reg->second;  // FIXED

        std::string op_name = reg->custom_name
            ? reg->custom_name
            : tflite::EnumNameBuiltinOperator(static_cast<tflite::BuiltinOperator>(reg->builtin_code));
        std::cout << i << ": Node " << node_index << " -> " << op_name << std::endl;

        // Heuristic: check if it's a delegate node
        bool is_delegate = (reg->custom_name && std::string(reg->custom_name).find("Delegate") != std::string::npos);
        if (is_delegate && node_and_reg->first.user_data) {
            auto* subgraph = reinterpret_cast<tflite::Subgraph*>(node_and_reg->first.user_data);
            std::cout << "   (Delegate subgraph with " << subgraph->nodes_size() << " internal nodes)" << std::endl;
            for (int j = 0; j < subgraph->nodes_size(); j++) {
                auto* internal_node_and_reg = subgraph->node_and_registration(j);
                const TfLiteRegistration& internal_reg = internal_node_and_reg->second;
                std::string sub_op = internal_reg.custom_name
                    ? internal_reg.custom_name
                    : tflite::EnumNameBuiltinOperator(static_cast<tflite::BuiltinOperator>(internal_reg.builtin_code));
                std::cout << "     - Subnode " << j << " -> " << sub_op << std::endl;
            }
        }
    }
    std::cout << "Press Enter to continue...";
    std::cin.get();  // Wait for Enter
}

} // namespace internals