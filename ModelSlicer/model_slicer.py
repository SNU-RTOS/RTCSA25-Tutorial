"""
Filename: model_slicer.py

@Author: Woobean Seo
@Affiliation: Real-Time Operating System Laboratory, Seoul National University
@Modified by: Taehyun Kim on 07/22/25
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
absl.logging.set_verbosity(absl.logging.ERROR)  # Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def DNNPartitioning(model, start, end, prev_outputs):
    # Dictionary to hold Input() layers for this submodel
    stage_inputs = {}
    # Tensors used within the current stage for skip connections
    intra_stage_skips = {}
    # Tensors that must be passed to the next stage (inter-stage skip connections)
    inter_stage_skips = {}

    # Create new tf.keras.Input layers based on outputs from the previous stage
    for inp in prev_outputs.keys():
        input_shape = prev_outputs[inp].shape[1:]
        stage_inputs[inp] = tf.keras.layers.Input(shape=input_shape, name=inp)

    # Initialize intra-stage skips with direct stage inputs
    for stage_input in stage_inputs.keys():
        intra_stage_skips[stage_input] = stage_inputs[stage_input]

    # Determine how to initialize the first input tensor `x`
    if isinstance(model.layers[start].input, list):
        # Multiple input tensors
        temp = []
        for stage_input in model.layers[start].input:
            key = stage_input.name.split('/')[0]
            temp.append(stage_inputs[key])
            inter_stage_skips[key] = stage_inputs[key]
        x = temp
    else:
        # Single input tensor
        if len(stage_inputs) == 1:
            x = next(iter(stage_inputs.values()))
        else:
            key = model.layers[start].input.name.split('/')[0]
            x = stage_inputs[key]

    # Iterate over layers in the specified range
    for i in range(start, end):
        layer = model.layers[i]
        inbound_node = layer._inbound_nodes[0]
        inbound_layers = inbound_node.inbound_layers

        # Handle case where the layer has multiple inputs
        if isinstance(inbound_layers, list) and len(inbound_layers) > 1:
            for inbound_layer in inbound_layers:
                origin_layer = model.get_layer(inbound_layer.name)
                origin_idx = model.layers.index(origin_layer)

                if origin_idx != i - 1:
                    if origin_layer.name in intra_stage_skips:
                        try:
                            x = layer(x)
                        except (ValueError, TypeError):
                            try:
                                x = layer([x, intra_stage_skips[origin_layer.name]])
                            except:
                                x = layer(x, intra_stage_skips[origin_layer.name])
                    elif origin_layer.name in inter_stage_skips:
                        x = layer([x, inter_stage_skips[origin_layer.name]])
                    else:
                        x = layer(x)
        else:
            # Single input layer case
            try:
                origin_layer = model.get_layer(inbound_node.inbound_layers.name)
                x = layer(intra_stage_skips[origin_layer.name])
            except (TypeError, KeyError):
                try:
                    x = layer(x)
                except (TypeError, ValueError):
                    x = layer(x[0])

        # Decide if the current layer output is needed for skip connections
        if len(layer._outbound_nodes) > 1:
            dest_idx = model.layers.index(layer._outbound_nodes[1].outbound_layer)
            if dest_idx < end:
                intra_stage_skips[layer.name] = x
            else:
                inter_stage_skips[layer.name] = x
        else:
            # For single outbound connection
            if i != len(model.layers) - 1:
                dest_idx = model.layers.index(layer._outbound_nodes[0].outbound_layer)
                if dest_idx != i + 1:
                    if dest_idx < end:
                        intra_stage_skips[layer.name] = x
                    else:
                        inter_stage_skips[layer.name] = x

    # Construct submodel
    if inter_stage_skips:
        # If multiple outputs including skip connections
        x = [x] + list(inter_stage_skips.values())
    else:
        # If single output
        try:
            x = list(x)[0]
        except TypeError:
            pass

    # Create and return the submodel
    submodel = tf.keras.models.Model(inputs=list(stage_inputs.values()), outputs=x)
    return submodel


def create_sample_input(shape=(1, 224, 224, 3)):
    return np.random.rand(*shape)

def prepare_next_stage_inputs(sub_model, stage_outputs):
    next_inputs = {}
    for model_output, actual_output in zip(sub_model.outputs, stage_outputs):
        layer_name = model_output._keras_history[0].name
        next_inputs[layer_name] = actual_output
    return next_inputs

def save_models(output_dir, model_name, sub_model, stage_num, coral=False):
    converter = tf.lite.TFLiteConverter.from_keras_model(sub_model)
    tflite_model = converter.convert()

    # Create output directory if it does not exist
    output_path = os.path.join(output_dir, model_name)
    os.makedirs(output_path, exist_ok=True)

    tflite_path = os.path.join(output_path, f"sub_model_{stage_num+1}.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved sliced tflite model {stage_num+1} to: {tflite_path}")
    return tflite_model

def parse_arguments():
    parser = argparse.ArgumentParser(description='Pipeline Stage')
    parser.add_argument('--model-path', type=str, required=True, help='Path to original h5 model')
    parser.add_argument('--output-dir', type=str, default='./submodels')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Load the model from the given path without compiling (for inference/slicing only)
    model = load_model(args.model_path, compile=False)
    model_file = os.path.basename(args.model_path)
    model_name = os.path.splitext(model_file)[0]
    
    # Get total number of layers in the model
    num_layers = len(model.layers)

    # Ask the user how many submodels they want to split into
    n = int(input("How many submodels? ").strip())

    if n < 1:
        raise ValueError("Submodel count must be >= 1")

    # If only one submodel is requested, no slicing is needed
    if n == 1:
        points = [0, num_layers - 1]
    else:
        # Dynamically generate slicing range guide for user display
        ranges = [f"(0, x1)"]
        for i in range(1, n - 1):
            ranges.append(f"(x{i}+1, x{i+1})")
        ranges.append(f"(x{n-1}+1, {num_layers - 1})")
        range_str = ', '.join(ranges)

        # Display input format prompt for slicing points
        x_list = ' '.join([f"x{i}" for i in range(1, n)])
        print(f"Enter {n-1} slicing points for ranges: {range_str}")
        user_input = input(f"Enter {x_list}: ").strip()
        cuts = sorted(int(x) for x in user_input.split())

        # Include start and end layer indices
        points = [0] + cuts + [num_layers - 1]

    # Build list of (start, end) layer index pairs for each slice
    slice_pairs = [(points[i], points[i+1]) for i in range(len(points)-1)]

    # Create partitioning points to match the internal model slicing logic
    partitioning_points = [1] + [end + 1 for _, end in slice_pairs]

    # For display: show actual slicing ranges in terms of layers
    slice_ranges = [
        (points[i] + (0 if i == 0 else 1), points[i+1])
        for i in range(len(points)-1)
    ]
    print(f"Slicing ranges: {slice_ranges}")

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
        tflite_models.append(save_models(args.output_dir, model_name, sub_model, i))

        
if __name__ == "__main__":
    main()