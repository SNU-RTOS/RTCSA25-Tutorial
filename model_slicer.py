#!/usr/bin/env python
#  -*- coding: utf-8 -*-
"""
Filename: model_slicer.py

@Author: Woobean Seo
@Affiliation: Real-Time Operating System Laboratory, Seoul National University
@Created: 07/23/25
@Original Work: Based on DNNPipe repository (https://github.com/SNU-RTOS/DNNPipe)
@Modified by: Taehyun Kim on 08/13/25
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
        input_layers (dict): Newly created layers for the slice: one layer per input tensor
        intra_slice_skips (dict): Tensor(s) used for processing intra-slice skip connections
        inter_slice_skips (dict): Tensor(s) used for processing inter-slice skip connections
        tensors_to_current_layer (list): Tensor(s) to be fed into the current layer 
        tensors_from_current_layer (list): Tensor(s) produced by the current layer
    """
    
    # 6-(1) Initialize data structures
    input_layers = {}          # Newly created layers for the slice: one layer per input tensor
    intra_slice_skips = {}     # Tensor(s) used for processing intra-slice skip connections
    inter_slice_skips = {}     # Tensor(s) used for processing inter-slice skip connections

    # 6-(2) Create input layers 
    for name, tensor in input_tensors.items():
        input_layers[name] = tf.keras.layers.Input(shape=tensor.shape[1:], name=name)
        intra_slice_skips[name] = input_layers[name]

    # 6-(3) Initialize the input tensors for the first layer in the slice
    # Multiple inputs to the first layer in the slice
    tensors_to_current_layer = [] 
    if isinstance(model.layers[start].input, list):
        for tensor in model.layers[start].input:
            key = tensor.name.split('/')[0]
            print(tensor.name)
            val = input_layers[key]
            tensors_to_current_layer.append(val)                    
            inter_slice_skips[key] = val 
    # Single input to the first layer in the slice
    else:
        if len(input_layers) == 1:
            tensors_to_current_layer = list(input_layers.values())[0]
        else:
            key = model.layers[start].input.name.split('/')[0]
            print(model.layers[start].input.name)
            val = input_layers[key]
            tensors_to_current_layer = val

    # 6-(4) Iterate over layers within the specified layer index range
    for i in range(start, end+1):
        layer = model.layers[i]
        
        inbound_layers = layer._inbound_nodes[0].inbound_layers
        # Multiple inputs for current layer
        if isinstance(inbound_layers, list) and len(inbound_layers) > 1:
            for inbound_layer in inbound_layers:
                origin_layer = model.get_layer(inbound_layer.name)
                origin_idx = model.layers.index(origin_layer)
                if origin_idx != i-1:
                    tensors_from_current_layer = \
                        layer([tensors_to_current_layer, intra_slice_skips[origin_layer.name]])
        # Single inputs for current layer
        else:
            origin_layer = model.get_layer(inbound_layers.name)
            if origin_layer.name in intra_slice_skips:
                tensors_from_current_layer = layer(intra_slice_skips[origin_layer.name])
            else:
                tensors_from_current_layer = layer(tensors_to_current_layer)

        # Multiple outputs from the current layer
        if len(layer._outbound_nodes)>1:
            destination_idx = model.layers.index(layer._outbound_nodes[1].outbound_layer)
            if destination_idx < end+1:
                intra_slice_skips[layer.name] = tensors_from_current_layer
            else:
                inter_slice_skips[layer.name] = tensors_from_current_layer
        # Single output from the current layer
        else:
            if i != len(model.layers)-1:
                destination_idx = model.layers.index(layer._outbound_nodes[0].outbound_layer)
                if destination_idx != i+1:
                    if destination_idx < end+1:
                        intra_slice_skips[layer.name] = tensors_from_current_layer
                    else:
                        inter_slice_skips[layer.name] = tensors_from_current_layer

        tensors_to_current_layer = tensors_from_current_layer

    # 6-(5) Construct the output tensors of the sliced model
    if inter_slice_skips:
        tensors_from_current_layer = [tensors_from_current_layer] + list(inter_slice_skips.values())

    # 6-(6) Create and return the sliced model
    sliced_model = \
        tf.keras.models.Model(inputs=list(input_layers.values()), outputs=tensors_from_current_layer)
    return sliced_model

# Prepare inputs for a slice
def prepare_slice_inputs(sliced_model):
    next_inputs = {}
    for output, output_tensor in zip(sliced_model.outputs, sliced_model.outputs):
        layer_name = output._keras_history[0].name
        next_inputs[layer_name] = output_tensor
    return next_inputs

# Save the sliced model as a LiteRT model
def save_sliced_model(output_dir, sliced_model, sliced_model_num):
    # Convert the sliced model to a LiteRT model
    converter = tf.lite.TFLiteConverter.from_keras_model(sliced_model)
    litert_model = converter.convert()
    # Save the LiteRT model to the specified output directory
    os.makedirs(output_dir, exist_ok=True)
    filename = f"submodel_{sliced_model_num}.tflite"
    litert_path = os.path.join(output_dir, filename)
    with open(litert_path, 'wb') as f:
        f.write(litert_model)
    print(f"Saved LiteRT model to: {litert_path}")

    return litert_model

# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to a model file (.h5)')
    parser.add_argument('--output-dir', type=str, default='./models')
    return parser.parse_args()

# Get slice indices from the user
def get_slice_indices(num_layers):
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
        cuts = sorted(int(x) for x in user_input.split())

        points = [0] + cuts + [num_layers - 1]
    
    slice_pairs = [(points[i], points[i+1]) for i in range(len(points)-1)]
    slice_indices = [1] + [end + 1 for _, end in slice_pairs]
    slice_ranges = [
        (points[i] + (0 if i == 0 else 1), points[i+1])
        for i in range(len(points)-1)
    ]

    if n==1:
        print("Just converting the model.")
    else:
        print(f"Layer index ranges for each submodel: {slice_ranges}")
    
    return n, slice_indices


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Load the model from the given path without compilation (for inference/slicing only)
    model = load_model(args.model_path, compile=False)
    
    # Ask the user for the number of slices and the index of the last layer in each submodel
    num_layers = len(model.layers)
    num_slices, slice_indices = get_slice_indices(num_layers)

    # Create a dummy input tensor for the first slice
    input_shape = model.layers[0].input_shape[0][1:]
    dummy_input = np.random.rand(1, *input_shape)

    # Perform slicing and model conversion
    sliced_models = []
    for i in range(num_slices):
        # Prepare inputs
        if i == 0:
            slice_inputs = {model.layers[0].name: dummy_input}
            print(model.layers[0].name)
        else:
            slice_inputs = prepare_slice_inputs(sliced_models[i-1])

        # Slice the model using slice_dnn
        sliced_model = slice_dnn(model, slice_indices[i], slice_indices[i+1]-1, slice_inputs)
        sliced_models.append(sliced_model)

        # Convert and save the sliced model to a LiteRT model
        save_sliced_model(args.output_dir, sliced_model, i)

if __name__ == "__main__":
    main()
