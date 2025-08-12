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
def DNNPartitioning(model, start, end, prev_outputs):
    """
    Parameters:
        model (tf.keras.Model): Full Keras model to be partitioned.
        start (int): Index of the first layer to include in the submodel.
        end (int): Index of the layer immediately after the last layer in the submodel. (end-1) is the last layer included.
        prev_outputs (dict): Mapping from layer name to its output tensor from the preceding submodel.

    Key data structures:
        submodel_inputs (dict): Tensors from the previous submodel, which are submodel_inputs.
        intra_submodel_skips (dict): Tensors reused within the same submodels (e.g., for skip connections).
        inter_submodel_skips (dict): Tensors needed as input for the next submodels.
        current_tensors (tensor or list of tensors): Intermediate tensor(s) passed through operations.
    """
    
    # 6-(1) Initialize data structures
    submodel_inputs = {}
    intra_submodel_skips = {}
    inter_submodel_skips = {}

    # 6-(2) Creates Keras Input layers from prev_outputs and stores them in both 
    #    submodel_inputs and intra_submodel_skips for reuse in intra-submodel skip connections
    for inp, tensor in prev_outputs.items():
        input_shape = tensor.shape[1:]
        input_layer = tf.keras.layers.Input(shape=input_shape, name=inp)
        submodel_inputs[inp] = input_layer
        intra_submodel_skips[inp] = input_layer

    # 6-(3) Initialize intermediate tensor(s) sequentially propagated within the submodel
    # Case 1: When model.layers[start] has multiple inputs
    if isinstance(model.layers[start].input, list):
        current_tensors = [] 
        for input_source in model.layers[start].input:
            key = input_source.name.split('/')[0]
            val = submodel_inputs[key]
            current_tensors.append(val)                    
            inter_submodel_skips[key] = val 
    # Case 2: When model.layers[start] has a single input
    else:
        if len(submodel_inputs) == 1:
            current_tensors = list(submodel_inputs.values())[0]
        else:
            key = model.layers[start].input.name.split('/')[0]
            val = submodel_inputs[key]
            current_tensors = val

    # 6-(4) Iterate over layers in the specified range of indices
    for i in range(start, end):
        layer = model.layers[i]
        inbound_node = layer._inbound_nodes[0]
        inbound_layers = inbound_node.inbound_layers

        # Multiple inbound layers
        if isinstance(inbound_layers, list) and len(inbound_layers) > 1:
            for inbound_layer in inbound_layers:
                origin_layer = model.get_layer(inbound_layer.name)
                origin_idx = model.layers.index(origin_layer)
                if origin_idx != i-1:
                    current_tensors = layer([current_tensors, intra_submodel_skips[origin_layer.name]])
        # Single inbound layer
        else:
            origin_layer = model.get_layer(inbound_layers.name)
            if origin_layer.name in intra_submodel_skips:
                current_tensors = layer(intra_submodel_skips[origin_layer.name])
            else:
                current_tensors = layer(current_tensors)

        # Multiple outbound connections
        if len(layer._outbound_nodes)>1:
            dest_idx = model.layers.index(layer._outbound_nodes[1].outbound_layer)
            if dest_idx < end:
                intra_submodel_skips[layer.name] = current_tensors
            else:
                inter_submodel_skips[layer.name] = current_tensors
        # Single outbound connection
        else:
            if i != len(model.layers)-1:
                dest_idx = model.layers.index(layer._outbound_nodes[0].outbound_layer)
                if dest_idx != i+1:
                    if dest_idx < end:
                        intra_submodel_skips[layer.name] = current_tensors
                    else:
                        inter_submodel_skips[layer.name] = current_tensors

    # 6-(5) If there are outputs needed for the next submodel (e.g., skip connections)
    if inter_submodel_skips:
        current_tensors=[current_tensors]+list(inter_submodel_skips.values())

    # 6-(6) Create and return the submodel
    submodel = tf.keras.models.Model(inputs=list(submodel_inputs.values()), outputs=current_tensors)
    return submodel

# Prepare submodel_inputs for the next submodel based on the outputs of the current submodel
def prepare_next_submodel_inputs(sub_model, submodel_outputs):
    next_inputs = {}
    for model_output, actual_output in zip(sub_model.outputs, submodel_outputs):
        layer_name = model_output._keras_history[0].name
        next_inputs[layer_name] = actual_output
    return next_inputs

# Save the submodel as a LiteRT model
def save_submodel(output_dir, sub_model, submodel_num):
    # Convert the submodel to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(sub_model)
    tflite_model = converter.convert()

    # Save the TFLite model to the specified output directory
    os.makedirs(output_dir, exist_ok=True)
    filename = f"sub_model_{submodel_num}.tflite"
    tflite_path = os.path.join(output_dir, filename)
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved LiteRT model to: {tflite_path}")

    return tflite_model

# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to original h5 model')
    parser.add_argument('--output-dir', type=str, default='./models')
    return parser.parse_args()

# Function to get slice indices from the user
def get_slice_indices_from_user(num_layers):
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
        print(f"Enter {n-1} slice indices for ranges: {range_str}")
        user_input = input(f"Enter {x_list}: ").strip()
        cuts = sorted(int(current_tensors) for current_tensors in user_input.split())

        points = [0] + cuts + [num_layers - 1]
    
    # Create partitioning points to match the internal model slicing logic
    slice_pairs = [(points[i], points[i+1]) for i in range(len(points)-1)]
    slice_indices = [1] + [end + 1 for _, end in slice_pairs]

    # For display: show actual slicing ranges in terms of layers
    slice_ranges = [
        (points[i] + (0 if i == 0 else 1), points[i+1])
        for i in range(len(points)-1)
    ]

    if n==1:
        print("No slicing needed. Just converting the model.")
    else:
        print(f"Slicing ranges: {slice_ranges}")
    
    return n, slice_indices


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Load the model from the given path without compiliation (for inference/slicing only)
    model = load_model(args.model_path, compile=False)
    
    # Ask the user how many submodels they want to split into
    num_layers = len(model.layers)
    num_slice, slice_indices = get_slice_indices_from_user(num_layers)

    # Create a dummy input tensor for the first submodel
    dummy_input = np.random.rand(*(1,224,224,3))

    # Perform slicing and model conversion per submodel
    sub_models = []
    tflite_models = []
    for i in range(num_slice):
        # Prepare submodel_inputs for current submodel: either dummy input or previous submodel's output
        if i == 0:
            submodel_inputs = {model.layers[0].name: dummy_input}
        else:
            submodel_inputs = prepare_next_submodel_inputs(sub_models[i-1], sub_models[i-1].outputs)

        # Slice the model using DNNPartitioning
        sub_model = DNNPartitioning(model, 
                                    slice_indices[i], 
                                    slice_indices[i+1], 
                                    submodel_inputs)
        sub_models.append(sub_model)

        # Convert and save the sliced submodel to TFLite format
        tflite_models.append(save_submodel(args.output_dir, sub_model, i))

if __name__ == "__main__":
    main()
