"""
Filename: model_slicer.py

@Author: Woobean Seo
@Affiliation: Real-Time Operating System Laboratory, Seoul National University
@Modified by: Taehyun Kim on 07/31/25
@Contact: thkim@redwood.snu.ac.kr

@Description: Model slicer for RTCSA25 tutorial
"""
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import load_model
import argparse

# Suppress unessary TensorFlow warnings
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Function to partition a Keras model into submodels based on specified layer indices
def DNNPartitioning(model, start, end, prev_outputs):
    """
    Parameters:
        model (tf.keras.Model): Full Keras model to be partitioned.
        start (int): Index of the first layer in the partitioned submodel.
        end (int): Index of the last layer (inclusive) in the partitioned submodel.
        prev_outputs (dict): {layer name → output tensor} from the previous submodel stage.

    Key data structures:
        stage_inputs (dict): {layer name → tf.keras.Input tensor}, inputs to the current stage.
        reused_tensors (dict): {layer name → tensor}, output tensors reused within the same stage (e.g., for skip connections).
        stage_outputs (dict): {layer name → tensor}, output tensors needed as input for the next stage (cross-stage connection).
        current_tensor (tensor or list of tensors): intermediate result tensor(s) propagated through current submodel.

    Mechanism:
        - The DNNPartitioning function iteratively applies layers between the specified start and end indices to construct a submodel.
        - It creates tf.keras.Input tensors from the previous stage’s outputs and maps them to the new computation graph of the current stage.
        - Output tensors that are reused within the same stage are cached in reused_tensors, 
          while those needed by the next stage are collected in stage_outputs to preserve skip connections and cross-stage dependencies.
    """
    
    # Initialize data structures
    stage_inputs = {}        
    reused_tensors = {}      
    stage_outputs = {}      
    
    # Create tf.keras.Input tensors from previous stage outputs and store for intra-stage reuse.
    for layer_name in prev_outputs.keys():
        input_shape = prev_outputs[layer_name].shape[1:]
        input_tensor = tf.keras.layers.Input(shape=input_shape, name=layer_name)
        
        # Initialize stage_inputs and reused_tensors
        stage_inputs[layer_name] = input_tensor
        reused_tensors[layer_name] = input_tensor

    # Determine how to initialize the first input tensor `current_tensor`
    # Multiple stage inputs
    if isinstance(model.layers[start].input, list): 
        current_tensor = []
        for stage_input in model.layers[start].input:
            layer_name = stage_input.name.split('/')[0]
            
            # Initialize current_tensor and stage_outputs
            current_tensor.append(stage_inputs[layer_name])
            stage_outputs[layer_name] = stage_inputs[layer_name]
    # Single stage input
    else: 
        if len(stage_inputs) == 1:
            current_tensor = next(iter(stage_inputs.values()))
        else:
            layer_name = model.layers[start].input.name.split('/')[0]
            current_tensor = stage_inputs[layer_name]

    # Iterate over layers in the specified range
    for i in range(start, end):
        # Retrieve the inbound layers connected as inputs to the current layer
        layer = model.layers[i]
        inbound_node = layer._inbound_nodes[0]
        inbound_layers = inbound_node.inbound_layers

        # Handle case where the layer has multiple inputs
        if isinstance(inbound_layers, list) and len(inbound_layers) > 1:
            for inbound_layer in inbound_layers:
                source_layer = model.get_layer(inbound_layer.name)  
                source_idx = model.layers.index(source_layer)

                if source_idx != i - 1:
                    if source_layer.name in reused_tensors:
                        try:
                            current_tensor = layer(current_tensor)
                        except (ValueError, TypeError):
                            try:
                                current_tensor = layer([current_tensor, reused_tensors[source_layer.name]])
                            except:
                                current_tensor = layer(current_tensor, reused_tensors[source_layer.name])
                    elif source_layer.name in stage_inputs:
                        current_tensor = layer([current_tensor, stage_inputs[source_layer.name]])
                    else:
                        current_tensor = layer(current_tensor)
        else:
            # Single input layer case
            try:
                source_layer = model.get_layer(inbound_node.inbound_layers.name)
                current_tensor = layer(reused_tensors[source_layer.name])
            except (TypeError, KeyError):
                try:
                    current_tensor = layer(current_tensor)
                except (TypeError, ValueError):
                    current_tensor = layer(current_tensor[0])

        # For multiple outbound connections
        if len(layer._outbound_nodes) > 1:
            destination_idx = model.layers.index(layer._outbound_nodes[1].outbound_layer)
            # If the output of the layer is used again within the current stage
            if destination_idx < end:
                reused_tensors[layer.name] = current_tensor
            # If the output of the layer is needed in the next stage
            else:
                stage_outputs[layer.name] = current_tensor
        # For single outbound connection        
        else:    
            if i != len(model.layers) - 1:
                destination_idx = model.layers.index(layer._outbound_nodes[0].outbound_layer)
                # If the output is not used by the immediately next layer
                if destination_idx != i + 1:
                    # If the output is reused later within the current stage
                    if destination_idx < end:
                        reused_tensors[layer.name] = current_tensor
                    # If the output is needed by a later stage
                    else:
                        stage_outputs[layer.name] = current_tensor

    # Construct submodel
    # If there are additional outputs needed for the next stage (e.g., skip connections)
    if stage_outputs:
        current_tensor = [current_tensor] + list(stage_outputs.values())
    # If the current stage produces only a single output tensor
    else:
        try:
            current_tensor = list(current_tensor)[0]
        except TypeError:
            pass

    # Create and return the submodel
    submodel = tf.keras.models.Model(inputs=list(stage_inputs.values()), outputs=current_tensor)
    return submodel

# Function to create a sample input tensor for the model
def create_sample_input(shape=(1, 224, 224, 3)):
    return np.random.rand(*shape)

# Prepare inputs for the next stage based on the outputs of the current submodel
def prepare_next_stage_inputs(sub_model, stage_outputs):
    next_inputs = {}
    for model_output, actual_output in zip(sub_model.outputs, stage_outputs):
        layer_name = model_output._keras_history[0].name
        next_inputs[layer_name] = actual_output
    return next_inputs

# Save the sliced submodel as a TFLite model
def save_models(output_dir, model_name, sub_model, stage_num, coral=False, single_stage=False):
    converter = tf.lite.TFLiteConverter.from_keras_model(sub_model)
    tflite_model = converter.convert()

    os.makedirs(output_dir, exist_ok=True)

    if single_stage:
        filename = f"{model_name}.tflite"
    else:
        filename = f"sub_model_{stage_num+1}.tflite"

    tflite_path = os.path.join(output_dir, filename)

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved sliced tflite model to: {tflite_path}")

    return tflite_model

# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Pipeline Stage')
    parser.add_argument('--model-path', type=str, required=True, help='Path to sourceal h5 model')
    parser.add_argument('--output-dir', type=str, default='./models')
    return parser.parse_args()

# Function to get slicing points from the user
def get_slicing_points_from_user(num_layers):
    n = int(input("How many submodels? ").strip())
    if n < 1:
        raise ValueError("Submodel count must be >= 1")

    if n == 1:
        points = [0, num_layers - 1]
    else:
        ranges = [f"(0, x1)"]
        for i in range(1, n - 1):
            ranges.append(f"(x{i}+1, x{i+1})")
        ranges.append(f"(x{n-1}+1, {num_layers - 1})")
        range_str = ', '.join(ranges)

        x_list = ' '.join([f"x{i}" for i in range(1, n)])
        print(f"Enter {n-1} slicing points for ranges: {range_str}")
        user_input = input(f"Enter {x_list}: ").strip()
        cuts = sorted(int(x) for x in user_input.split())

        points = [0] + cuts + [num_layers - 1]
    
    # Create partitioning points to match the internal model slicing logic
    slice_pairs = [(points[i], points[i+1]) for i in range(len(points)-1)]
    partitioning_points = [1] + [end + 1 for _, end in slice_pairs]

    # For display: show actual slicing ranges in terms of layers
    slice_ranges = [
        (points[i] + (0 if i == 0 else 1), points[i+1])
        for i in range(len(points)-1)
    ]

    if n==1:
        print("No slicing needed. Just converting the model.")
    else:
        print(f"Slicing ranges: {slice_ranges}")
    
    return n, partitioning_points


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Load the model from the given path without compiling (for inference/slicing only)
    model = load_model(args.model_path, compile=False)
    model_file = os.path.basename(args.model_path)
    model_name = os.path.splitext(model_file)[0]
    
    # Get total number of layers in the model
    num_layers = len(model.layers)

    # Ask the user how many submodels they want to split into
    n, partitioning_points = get_slicing_points_from_user(num_layers)

    # Create a sample dummy input tensor for the first submodel
    sample_input = create_sample_input()
    num_stages = len(partitioning_points) - 1
    
    sub_models = []
    tflite_models = []

    # Perform slicing and model conversion per stage
    for i in range(num_stages):
        # Prepare inputs for current stage: either dummy input or previous stage's output
        if i == 0:
            stage_inputs = {model.layers[0].name: sample_input}
        else:
            stage_inputs = prepare_next_stage_inputs(sub_models[i-1], sub_models[i-1].outputs)

        # Slice the model using DNNPartitioning
        sub_model = DNNPartitioning(model, partitioning_points[i], partitioning_points[i+1], stage_inputs)
        sub_models.append(sub_model)

        # Convert and save the sliced submodel to TFLite
        single_stage = (n == 1)
        tflite_models.append(save_models(args.output_dir, model_name, sub_model, i, single_stage=single_stage))

        
if __name__ == "__main__":
    main()
