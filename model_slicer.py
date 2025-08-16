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

# Slice a model from start layer to end layer
def slice_dnn(model, start, end, input_tensors): 
    """
    Parameters:
        model (tf.keras.Model): A Keras model object to be sliced
        start (int): Index of the first layer of the slice
        end (int): Index of the last layer of the slice
        input_tensors (dict): Inputs to the slice {input layer name: input tensor}

    Key data structures:
        input_layers (dict): Newly created layers for the slice: one layer per input tensor
        intra_slice_skips (dict): Tensor(s) used for processing intra-slice skip connections
        inter_slice_skips (dict): Tensor(s) used for processing inter-slice skip connections
        tensors_to_current_layer (list): Tensor(s) to be fed into the current layer 
        tensors_from_current_layer (list): Tensor(s) produced by the current layer
    """
    
    input_layers = {}
    tensors_to_start_layer = []    
    intra_slice_skips = {}     
    inter_slice_skips = {} 

    # We assume that the layers of the model are only used once
    # Which means there is an one-to-one correspondence between a layer and its node

    # 6-(1) Create input layers 
    # We need to figure out how the input layer(s) are used in the slice
    for name, tensor in input_tensors.items():
        input_layers[name] = tf.keras.layers.Input(shape=tensor.shape[1:], name=name)
        # Among the input layers, we need to find out the usage of each input layer
        input_layer = model.get_layer(name)
        # If an input layer has multiple outbound layers it means it is used for multiple layers
        if len(input_layer._outbound_nodes) > 1:
            for outbound_node in input_layer._outbound_nodes:
                # If the input layer is an inter-slice skip, we need to add it to inter_slice_skips
                outbound_layer = outbound_node.outbound_layer
                target_idx = model.layers.index(outbound_layer)
                if target_idx > end:
                    inter_slice_skips[name] = input_layers[name]
                # We precisely check if the outbound layer is not the next layer
                elif target_idx <= end and target_idx > start:
                    intra_slice_skips[name] = input_layers[name]
                elif target_idx == start: # target_idx == start
                    print("Start 1")
                    tensors_to_start_layer.append(input_layers[name])
        # If an input layer has a single outbound layer
        else:
            outbound_layer = input_layer._outbound_nodes[0].outbound_layer
            target_idx = model.layers.index(outbound_layer)
            if target_idx > end:
                inter_slice_skips[name] = input_layers[name]
            elif target_idx <= end and target_idx > start:
                intra_slice_skips[name] = input_layers[name]
            elif target_idx == start: # target_idx == start
                print("Start 2")
                tensors_to_start_layer.append(input_layers[name])

    # 6-(3) Build hidden layers
    # Input to a built-in Keras layer is always either a single tensor or a list of tensors
    # Unless the layer is a custom layer
    if(len(tensors_to_start_layer) == 1):
        tensors_to_current_layer = tensors_to_start_layer[0]
    else:
        tensors_to_current_layer = tensors_to_start_layer
        
    for i in range(start, end+1): 
        layer = model.layers[i]
        inbound_layers = layer._inbound_nodes[0].inbound_layers
        # If a layer has a single inbound layer, we can simply get its name
        # Get the tensor object we have to feed into the current layer
        # If a layer has multiple inbound layers, we need to find all the tensors
        # that are used as inputs to the current layer
        # This further divided into two cases:
        # 1. The current layer is the start layer, in this case we already have the tensors_to_current_layer
        # 2. The current layer is not the start layer, in this case we need to find the tensors from the inbound layers
        # Multiple inputs for current layer
        if isinstance(inbound_layers, list):            
            # If the current layer is the start layer, we can use the tensors_to_current_layer directly
            if(i == start):
                tensors_from_current_layer = layer(tensors_to_current_layer)
            else:
                # First we need to get all the necessary tensors from the inbound layers
                # There must exists an intra-slice skip connection
                # try as the natural Keras way
                # raise error if it fails which it means the call function is overriden in a custom way that does not comply with the Keras standard
                # If tensors_to_current_layer is a single tensor, we need to convert it to a list
                tensors_to_current_layer = [tensors_to_current_layer] if not isinstance(tensors_to_current_layer, list) else tensors_to_current_layer
                for inbound_layer in inbound_layers:
                    intra_slice_skip = model.get_layer(inbound_layer.name) # This must be an intra-slice skip
                    # Some layers that directly come out are not treated as intra-slice skips
                    # Before appending we need to check if the tensor is already in the tensors_to_current_layer
                    # If not we need to append it, if yes we can skip it
                    if intra_slice_skip.name not in [t.name.split('/')[0] for t in tensors_to_current_layer]:
                        tensors_to_current_layer.append(intra_slice_skips[intra_slice_skip.name])
                # Now we have all the tensors that are needed for the current layer
                # We can call the current layer with the tensors_to_current_layer
                try:
                    tensors_from_current_layer = \
                        layer(tensors_to_current_layer)
                except:
                    raise ValueError(f"Failed to call layer {layer.name} with tensors {tensors_to_current_layer}. "
                                     "Please check the layer's call function and the input tensors.")
            
        # Single input for current layer
        else:
            origin_layer = model.get_layer(inbound_layers.name)
            if origin_layer.name in intra_slice_skips:
                tensors_from_current_layer = layer(intra_slice_skips[origin_layer.name])
                # print(f"Output: {tensors_from_current_layer}")
            else:
                # print(f"Input: {tensors_to_current_layer}")
                tensors_from_current_layer = layer(tensors_to_current_layer)
                # print(f"Output: {tensors_from_current_layer}")

        # Multiple outputs from current layer
        if len(layer._outbound_nodes) > 1:
            for outbound_node in layer._outbound_nodes:
                skip_target_idx = model.layers.index(outbound_node.outbound_layer)
                print(f"Layer {skip_target_idx}")
                if skip_target_idx > end:
                    print("SAVED")
                    inter_slice_skips[layer.name] = tensors_from_current_layer
                elif skip_target_idx <= end and skip_target_idx > i + 1:
                    intra_slice_skips[layer.name] = tensors_from_current_layer
        # Single output from current layer
        else:
            # required to check if the current layer is not the last layer of the slice
            # if not, duplicated inter_slice_skips might be created
            if i != end: 
                skip_target_idx = model.layers.index(layer._outbound_nodes[0].outbound_layer)
                if skip_target_idx > end:
                    inter_slice_skips[layer.name] = tensors_from_current_layer
                elif skip_target_idx <= end and skip_target_idx > i + 1:
                    intra_slice_skips[layer.name] = tensors_from_current_layer

        tensors_to_current_layer = tensors_from_current_layer

    # 6-(4) Construct output tensors
    if inter_slice_skips:
        print(tensors_from_current_layer)
        print(list(inter_slice_skips.values()))
        tensors_from_current_layer = [tensors_from_current_layer] + list(inter_slice_skips.values())
        print(tensors_from_current_layer)
    # 6-(5) Create and return the slice
    slice = tf.keras.models.Model(inputs=list(input_layers.values()), 
                                  outputs=tensors_from_current_layer)
    return slice

# Prepare inputs for a slice
def prepare_slice_inputs(slice):
    next_inputs = {}
    for output, output_tensor in zip(slice.outputs, slice.outputs):
        layer_name = output._keras_history[0].name
        next_inputs[layer_name] = output_tensor
    return next_inputs

# Save the slice as a LiteRT model
def convert_save_slice(output_dir, slice, slice_num):
    # Convert the slice to a LiteRT model
    converter = tf.lite.TFLiteConverter.from_keras_model(slice)
    litert_model = converter.convert()
    # Save the LiteRT model to the specified output directory
    os.makedirs(output_dir, exist_ok=True)
    filename = f"submodel_{slice_num}.tflite"
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

# Get slicing settings from user
def get_slice_indices(num_layers):
    n = int(input("How many submodels? ").strip())
    if n < 1:
        raise ValueError("The number of submodels must be >= 1")

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
        print("Only converting the model")
    else:
        print(f"Layer index ranges for each submodel: {slice_ranges}")
    
    return n, slice_indices


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Load the model from the given path without compilation (for inference/slicing only)
    model = load_model(args.model_path, compile=False)
    
    # Ask the user for the number of slices and the index of the last layer in each slice
    num_layers = len(model.layers)
    num_slices, slice_indices = get_slice_indices(num_layers)

    # Create a dummy input tensor for the first slice
    input_shape = model.layers[0].input_shape[0][1:]
    dummy_input = np.random.rand(1, *input_shape)

    # Perform slicing and conversion
    slices = []
    for i in range(num_slices):
        # Prepare inputs for the slice
        if i == 0:
            slice_inputs = {model.layers[0].name: dummy_input}
        else:
            slice_inputs = prepare_slice_inputs(slices[i-1])

        # Slice the model using slice_dnn
        slice = slice_dnn(model, slice_indices[i], slice_indices[i+1]-1, slice_inputs)
        slices.append(slice)

        # Convert and save the slice to a LiteRT model
        convert_save_slice(args.output_dir, slice, i)

if __name__ == "__main__":
    main()