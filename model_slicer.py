#!/usr/bin/env python
#  -*- coding: utf-8 -*-
"""
Filename: model_slicer.py

@Author: Woobean Seo
@Affiliation: Real-Time Operating System Laboratory, Seoul National University
@Created: 07/23/25
@Original Work: Based on DNNPipe repository (https://github.com/SNU-RTOS/DNNPipe)
@Modified by: Taehyun Kim on 08/06/25
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
        start (int): Index of the first layer to include in the submodel.
        end (int): Index of the layer immediately after the last layer in the submodel. (end-1) is the last layer included.
        prev_outputs (dict): Mapping from layer name to its output tensor from the preceding submodel.

    Key data structures:
        submodel_inputs (dict): Tensors to the current submodel.
        intra_submodel_skips (dict): Tensors reused within the same submodels (e.g., for skip connections).
        inter_submodel_skips (dict): Tensors needed as input for the next submodels.
        x (tensor or list of tensors): Intermediate tensor(s) propagated through current submodel.
    """
    
    # Initialize data structures
    submodel_inputs = {}
    intra_submodel_skips = {}
    inter_submodel_skips = {}

    for inp in prev_outputs.keys():
        input_shape =  prev_outputs[inp].shape[1:]
        submodel_inputs[inp] = tf.keras.layers.Input(shape=input_shape, name=inp)

    for submodel_input in submodel_inputs.keys():
        intra_submodel_skips[submodel_input] = submodel_inputs[submodel_input]

    # Initialize x
    # Case 1: When model.layers[start] has multiple inputs
    if isinstance(model.layers[start].input, list):
        temp = []
        for layer_start in model.layers[start].input:
            temp.append(submodel_inputs[layer_start.name.split('/')[0]])
            inter_submodel_skips[layer_start.name.split('/')[0]] = submodel_inputs[layer_start.name.split('/')[0]]
        x = temp
    # Case 2: When model.layers[start] has a single input
    else:
        if len(submodel_inputs) == 1:
            x = next(iter(submodel_inputs.values())) 
        else:
            x = submodel_inputs[model.layers[start].input.name.split('/')[0]]

    # Iterate over layers in the specified range
    for i in range(start, end):
        layer = model.layers[i]
        inbound_node = layer._inbound_nodes[0]
        inbound_layers = inbound_node.inbound_layers

        # Multiple inbound layers
        if isinstance(inbound_layers, list) and len(inbound_layers) > 1:
            for inbound_layer in layer._inbound_nodes[0].inbound_layers:
                origin_layer = model.get_layer(inbound_layer.name)
                origin_idx = model.layers.index(origin_layer)
                if origin_idx != i-1:
                    if origin_layer.name in intra_submodel_skips.keys():
                        try:
                            x = layer(x)
                        except (ValueError, TypeError):
                            try:
                                x = layer([x, intra_submodel_skips[origin_layer.name]])
                            except:
                                x = layer(x, intra_submodel_skips[origin_layer.name])
                    elif origin_layer.name in inter_submodel_skips.keys():
                        x = layer([x, inter_submodel_skips[origin_layer.name]])
                    else:
                        x = layer(x)
        # Single inbound layer
        else:
            try:
                origin_layer = model.get_layer(layer._inbound_nodes[0].inbound_layers.name)
                x = layer(intra_submodel_skips[origin_layer.name])
            except (TypeError, KeyError):
                try:
                    x = layer(x)
                except (TypeError, ValueError):
                    x = layer(x[0])

        # Multiple outbound connections
        if len(layer._outbound_nodes)>1:
            dest_idx = model.layers.index(layer._outbound_nodes[1].outbound_layer)
            if dest_idx < end:
                intra_submodel_skips[layer.name] = x
            else:
                inter_submodel_skips[layer.name] = x
        # Single outbound connection
        else:
            if i != len(model.layers)-1:
                dest_idx = model.layers.index(layer._outbound_nodes[0].outbound_layer)
                if dest_idx != i+1:
                    if dest_idx < end:
                        intra_submodel_skips[layer.name] = x
                    else:
                        inter_submodel_skips[layer.name] = x

    # If there are outputs needed for the next submodel (e.g., skip connections)
    if inter_submodel_skips:
        x=[x]+list(inter_submodel_skips.values())
    else:
        try:
            x = list(x)[0]
        except TypeError:
            pass

    # Create and return the submodel
    submodel = tf.keras.models.Model(inputs=list(submodel_inputs.values()), outputs=x)
    return submodel

# Function to create a dummy input tensor for the model
def create_dummy_input(shape=(1, 224, 224, 3)):
    return np.random.rand(*shape)

# Prepare inputs for the next submodel based on the outputs of the current submodel
def prepare_next_submodel_inputs(sub_model, submodel_outputs):
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
    
    return n, partitioning_points


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Load the model from the given path without compiliation (for inference/slicing only)
    model = load_model(args.model_path, compile=False)
    model_file = os.path.basename(args.model_path)
    model_name = os.path.splitext(model_file)[0]
    
    # Get total number of layers in the model
    num_layers = len(model.layers)

    # Ask the user how many submodels they want to split into
    n, partitioning_points = get_slicing_points_from_user(num_layers)

    # Create a sample dummy input tensor for the first submodel
    sample_input = create_dummy_input()
    num_submodels = len(partitioning_points) - 1

    # Lists to store intermediate submodels and their corresponding TFLite models
    sub_models = []
    tflite_models = []

    # Perform slicing and model conversion per submodel
    for i in range(num_submodels):
        # Prepare inputs for current submodel: either dummy input or previous submodel's output
        if i == 0:
            submodel_inputs = {model.layers[0].name: sample_input}
        else:
            submodel_inputs = prepare_next_submodel_inputs(sub_models[i-1], sub_models[i-1].outputs)

        # Slice the model using DNNPartitioning
        sub_model = DNNPartitioning(model, partitioning_points[i], partitioning_points[i+1], submodel_inputs)
        sub_models.append(sub_model)

        # Convert and save the sliced submodel to TFLite
        tflite_models.append(save_models(args.output_dir, model_name, sub_model, i))

        
if __name__ == "__main__":
    main()
