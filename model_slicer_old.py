#!/usr/bin/env python
#  -*- coding: utf-8 -*-
"""
Filename: model_slicer.py

@Author: Woobean Seo
@Affiliation: Real-Time Operating System Laboratory, Seoul National University
@Created: 07/23/25
@Original Work: Based on DNNPipe repository (https://github.com/SNU-RTOS/DNNPipe)
@Modified by: Taehyun Kim on 08/11/25
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

# Function to partition a model into submodels based on specified layer indices
def slice_dnn(model, start, end, input_tensors):
    """
    Parameters:
        model (tf.keras.Model): Full Keras model to be partitioned.
        start (int): Index of the first layer to include in the submodel.
        end (int): Index of the layer immediately after the last layer in the submodel. (end) is the last layer included.
        input_tensors (dict): Mapping from layer name to its output tensor from the preceding submodel.

    Key data structures:
        input_layers (dict): Tensors from the previous submodel, which are inputs.
        intra_slice_skips (dict): Tensors reused within the same submodels (e.g., for skip connections).
        inter_slice_skips (dict): Tensors needed as input for the next submodels.
        tensors_to_current_layer (tensor or list of tensors): Intermediate tensor(s) passed through operations.
    """
    
    # Initialize data structures
    input_layers = {}
    intra_slice_skips = {}
    inter_slice_skips = {}

    for inp in input_tensors.keys():
        input_shape =  input_tensors[inp].shape[1:]
        input_layers[inp] = tf.keras.layers.Input(shape=input_shape, name=inp)
        intra_slice_skips[inp] = input_layers[inp]

    # Initialize tensors_to_current_layer
    # When model.layers[start] has multiple inputs
    if isinstance(model.layers[start].input, list):
        temp = []
        for layer_start in model.layers[start].input:
            temp.append(input_layers[layer_start.name.split('/')[0]])
            inter_slice_skips[layer_start.name.split('/')[0]] \
                    = input_layers[layer_start.name.split('/')[0]]
        tensors_to_current_layer = temp
    # When model.layers[start] has a single input
    else:
        if len(input_layers) == 1:
            tensors_to_current_layer = next(iter(input_layers.values())) 
        else:
            tensors_to_current_layer = input_layers[model.layers[start].input.name.split('/')[0]]

    # Iterate over layers in the specified range
    for i in range(start, end+1):
        layer = model.layers[i]
        inbound_node = layer._inbound_nodes[0]
        inbound_layers = inbound_node.inbound_layers
        print(f"Start index {i}")

        # Multiple inbound layers
        if isinstance(inbound_layers, list):
            for inbound_layer in layer._inbound_nodes[0].inbound_layers:
                origin_layer = model.get_layer(inbound_layer.name)
                origin_idx = model.layers.index(origin_layer)
                if origin_idx != i-1:
                    if origin_layer.name in intra_slice_skips.keys():
                        try:
                            print(f"Multi 1 {tensors_to_current_layer}")
                            tensors_to_current_layer = layer(tensors_to_current_layer)
                        except (ValueError, TypeError):
                            try:
                                print(f"Multi 2 {tensors_to_current_layer}, {origin_layer.name}, {intra_slice_skips[origin_layer.name]}")
                                tensors_to_current_layer = layer([tensors_to_current_layer, intra_slice_skips[origin_layer.name]])
                            except:
                                print(f"Multi 3 {tensors_to_current_layer}, {origin_layer.name}, {intra_slice_skips[origin_layer.name]}")
                                tensors_to_current_layer = layer(tensors_to_current_layer, intra_slice_skips[origin_layer.name])
                    # Why is this part not in the code?
                    elif origin_layer.name in inter_slice_skips.keys():
                        print(f"Multi 4 {tensors_to_current_layer}, {origin_layer.name}, {inter_slice_skips[origin_layer.name]}")
                        tensors_to_current_layer = layer([tensors_to_current_layer, inter_slice_skips[origin_layer.name]])
                    else:
                        print(f"Multi 5 {tensors_to_current_layer}")
                        tensors_to_current_layer = layer(tensors_to_current_layer)
        # Single inbound layer
        else:
            try:
                print("Single 1 called")
                # What are the cases that this raises either a TypeError or a KeyError?
                origin_layer = model.get_layer(layer._inbound_nodes[0].inbound_layers.name)
                print(f"Single 1 {origin_layer.name}, {intra_slice_skips[origin_layer.name]}")
                tensors_to_current_layer = layer(intra_slice_skips[origin_layer.name])
            except (TypeError, KeyError):
                try:
                    print(f"Single 2 {tensors_to_current_layer}")
                    tensors_to_current_layer = layer(tensors_to_current_layer)
                except (TypeError, ValueError):
                    print(f"Single 3 {tensors_to_current_layer}")
                    tensors_to_current_layer = layer(tensors_to_current_layer[0])

        # Multiple outbound connections
        if len(layer._outbound_nodes)>1:
            dest_idx = model.layers.index(layer._outbound_nodes[1].outbound_layer)
            if dest_idx < end+1:
                intra_slice_skips[layer.name] = tensors_to_current_layer
            else:
                inter_slice_skips[layer.name] = tensors_to_current_layer
        # Single outbound connection
        else:
            if i != len(model.layers)-1:
                dest_idx = model.layers.index(layer._outbound_nodes[0].outbound_layer)
                if dest_idx != i+1:
                    if dest_idx < end+1:
                        intra_slice_skips[layer.name] = tensors_to_current_layer
                    else:
                        inter_slice_skips[layer.name] = tensors_to_current_layer

    # If there are outputs needed for the next submodel (e.g., skip connections)
    if inter_slice_skips:
        tensors_to_current_layer=[tensors_to_current_layer]+list(inter_slice_skips.values())
    else:
        try:
            tensors_to_current_layer = list(tensors_to_current_layer)[0]
        except TypeError:
            pass

    # Create and return the submodel
    slice = tf.keras.models.Model(inputs=list(input_layers.values()), outputs=tensors_to_current_layer)
    return slice

# Prepare inputs for the next submodel based on the outputs of the current submodel
def prepare_next_slice_inputs(sub_model, submodel_outputs):
    next_inputs = {}
    for model_output, actual_output in zip(sub_model.outputs, submodel_outputs):
        layer_name = model_output._keras_history[0].name
        next_inputs[layer_name] = actual_output
    return next_inputs

# Save the sliced submodel as a TFLite model
def save_models(output_dir, model_name, sub_model, submodel_num):
    # Convert the sliced submodel to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(sub_model)
    tflite_model = converter.convert()

    # Save the TFLite model to the specified output directory
    os.makedirs(output_dir, exist_ok=True)
    filename = f"sub_model_{submodel_num}.tflite"
    tflite_path = os.path.join(output_dir, filename)
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved sliced tflite model to: {tflite_path}")

    return tflite_model

# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Pipeline submodel')
    parser.add_argument('--model-path', type=str, required=True, help='Path to original h5 model')
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
    
    return partitioning_points


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Load the model from the given path without compiliation (for inference/slicing only)
    model = load_model(args.model_path, compile=False)
    model_file = os.path.basename(args.model_path)
    model_name = os.path.splitext(model_file)[0]
    
    # Ask the user how many submodels they want to split into
    num_layers = len(model.layers)
    partitioning_points = get_slicing_points_from_user(num_layers)
    num_submodels = len(partitioning_points) - 1

    # Create a dummy input tensor for the first slice
    input_shape = model.layers[0].input_shape[0][1:]
    dummy_input = np.random.rand(1, *input_shape)

    # Perform slicing and model conversion per submodel
    sub_models = []
    litert_submodels = []
    for i in range(num_submodels):
        # Prepare inputs for current submodel: either dummy input or previous submodel's output
        if i == 0:
            slice_inputs = {model.layers[0].name: dummy_input}
        else:
            slice_inputs = prepare_next_slice_inputs(sub_models[i-1], sub_models[i-1].outputs)

        # Slice the model using slice_dnn
        sub_model = slice_dnn(model, 
                                    partitioning_points[i], 
                                    partitioning_points[i+1] - 1, 
                                    slice_inputs)
        sub_models.append(sub_model)

        # Convert and save the sliced submodel to TFLite format
        litert_submodels.append(save_models(args.output_dir, model_name, sub_model, i))

if __name__ == "__main__":
    main()