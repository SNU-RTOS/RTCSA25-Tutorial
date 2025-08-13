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

# Partition a Keras model object into multiple slices
def slice_dnn(model, start, end, input_tensors):
    """
    Parameters:
        model (tf.keras.Model): Original TensorFlow model to be partitioned
        start (int): Index of the first layer of the slice
        end (int): Index of the last layer of the slice
        input_tensors (dict): Inputs to the slice {generating layer name: tensor}

    Key data structures:
        inputs (dict): Tensors from the previous submodel, which are inputs
        intra_slice_skips (dict): Tensors reused within the same submodels (e.g., for skip connections)
        inter_slice_skips (dict): Tensors needed as input for the next submodels
        current_tensors (tensor or list of tensors): Intermediate tensor(s) passed through operations
    """
    
    # 6-(1) Initialize data structures
    input_layers = {}          # Newly created layers for the slice: one layer per input tensor
    intra_slice_skips = {}     # Input layers used for processing intra-slice skip connections
    inter_slice_skips = {}     # !!!! used for processing inter-slice skip connections

    # 6-(2) Create input layers 
    for name, tensor in input_tensors.items():
        input_layers[name] = tf.keras.layers.Input(shape=tensor.shape[1:], name=name)
        intra_slice_skips[name] = input_layers[name]

    # 6-(3) Initialize intermediate tensor(s) 
    # Case 1: When model.layers[start] has multiple inputs
    if isinstance(model.layers[start].input, list):
        tensors_to_current_layer = [] 
        for tensor in model.layers[start].input:
            key = tensor.name.split('/')[0]
            val = input_layers[key]
            tensors_to_current_layer.append(val)                    
            inter_slice_skips[key] = val 
    # Case 2: When model.layers[start] has a single input
    else:
        if len(input_layers) == 1:
            tensors_to_current_layer = list(input_layers.values())[0]
        else:
            key = model.layers[start].input.name.split('/')[0]
            val = input_layers[key]
            tensors_to_current_layer = val

    # 6-(4) Iterate over layers in the specified range of indices
    for i in range(start, end+1):
        layer = model.layers[i]
        inbound_layers = layer._inbound_nodes[0].inbound_layers

        # Multiple inbound layers
        if isinstance(inbound_layers, list) and len(inbound_layers) > 1:
            for inbound_layer in inbound_layers:
                origin_layer = model.get_layer(inbound_layer.name)
                origin_idx = model.layers.index(origin_layer)
                if origin_idx != i-1:
                    tensors_from_current_layer = \ 
                        layer([tensors_to_current_layer, intra_slice_skips[origin_layer.name]])
        # Single inbound layer
        else:
            origin_layer = model.get_layer(inbound_layers.name)
            if origin_layer.name in intra_slice_skips:
                tensors_from_current_layer = layer(intra_slice_skips[origin_layer.name])
            else:
                tensors_from_current_layer = layer(tensors_to_current_layer)

        # Multiple outbound connections
        if len(layer._outbound_nodes)>1:
            dest_idx = model.layers.index(layer._outbound_nodes[1].outbound_layer)
            if dest_idx < end+1:
                intra_slice_skips[layer.name] = tensors_from_current_layer
            else:
                inter_slice_skips[layer.name] = tensors_from_current_layer
        # Single outbound connection
        else:
            if i != len(model.layers)-1:
                dest_idx = model.layers.index(layer._outbound_nodes[0].outbound_layer)
                if dest_idx != i+1:
                    if dest_idx < end+1:
                        intra_slice_skips[layer.name] = tensors_from_current_layer
                    else:
                        inter_slice_skips[layer.name] = tensors_from_current_layer
        
        tensors_to_current_layer = tensors_from_current_layer

    # 6-(5) If there are outputs needed for the next submodel (e.g., skip connections)
    if inter_slice_skips:
        tensors_from_current_layer = [tensors_from_current_layer] + list(inter_slice_skips.values())

    # 6-(6) Create and return the submodel
    sliced_model = tf.keras.models.Model(inputs=list(input_layers.values()), outputs=tensors_from_current_layer)
    return sliced_model

# Prepare inputs for the next submodel based on the outputs of the current submodel
def prepare_next_slice_inputs(submodel, submodel_outputs):
    next_inputs = {}
    for model_output, actual_output in zip(submodel.outputs, submodel_outputs):
        layer_name = model_output._keras_history[0].name
        next_inputs[layer_name] = actual_output
    return next_inputs

# Save the submodel as a LiteRT model
def save_sliced_model(output_dir, submodel, submodel_num):
    # Convert the submodel to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(submodel)
    litert_model = converter.convert()

    # Save the LiteRT model to the specified output directory
    os.makedirs(output_dir, exist_ok=True)
    filename = f"submodel_{submodel_num}.tflite"
    litert_path = os.path.join(output_dir, filename)
    with open(litert_path, 'wb') as f:
        f.write(litert_model)
    print(f"Saved LiteRT model to: {litert_path}")

    return litert_model

# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to original h5 model')
    parser.add_argument('--output-dir', type=str, default='./models')
    return parser.parse_args()

# Function to get slice indices from the user
def get_start_idx_slice(num_layers):
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
        print(f"Generated submodels look like: {range_str}")

        user_input = input(f"Enter {x_list}: ").strip()
        cuts = sorted(int(current_tensors) for current_tensors in user_input.split())

        points = [0] + cuts + [num_layers - 1]
    
    # Create partitioning points to match the internal model slicing logic
    slice_pairs = [(points[i], points[i+1]) for i in range(len(points)-1)]
    start_idx_slice = [1] + [end + 1 for _, end in slice_pairs]

    # For display: show actual slicing ranges in terms of layers
    slice_ranges = [
        (points[i] + (0 if i == 0 else 1), points[i+1])
        for i in range(len(points)-1)
    ]

    if n==1:
        print("Just converting the model.")
    else:
        print(f"Submodel layer ranges: {slice_ranges}")
    
    return n, start_idx_slice


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Load the model from the given path without compilation (for inference/slicing only)
    model = load_model(args.model_path, compile=False)
    
    # Ask the user for the number of submodels and the start index of each
    num_layers = len(model.layers)
    num_slices, start_idx_slice = get_start_idx_slice(num_layers)

    # Create a dummy input tensor for the first submodel
    input_shape = model.layers[0].input_shape[0][1:]
    dummy_input = np.random.rand(1, *input_shape)

    # Perform slicing and model conversion per submodel
    sliced_models = []
    for i in range(num_slices):
        # Prepare inputs for current submodel
        if i == 0:
            slice_inputs = {model.layers[0].name: dummy_input}
        else:
            slice_inputs = \
                prepare_next_slice_inputs(sliced_models[i-1], sliced_models[i-1].outputs)

        # Slice the model using slice_dnn
        sliced_model = slice_dnn(model, 
                                  start_idx_slice[i], 
                                  start_idx_slice[i+1]-1, 
                                  slice_inputs)
        sliced_models.append(sliced_model)

        # Convert and save the sliced model to TFLite format
        save_sliced_model(args.output_dir, sliced_model, i)

if __name__ == "__main__":
    main()
