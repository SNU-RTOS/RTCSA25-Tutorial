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
        inside_ending_skips (dict): Tensor(s) used for processing inside-ending skip connections
        outside_ending_skips (dict): Tensor(s) used for processing outside-ending skip connections
        tensors_to_start_layer (list): Tensor(s) to be fed into the start layer of the slice
        tensors_to_current_layer (list or KerasTensor): Tensor(s) to be fed into the current layer 
        tensors_from_current_layer (list or KerasTensor): Tensor(s) produced by the current layer
    """
    
    input_layers = {}
    inside_ending_skips = {}     
    outside_ending_skips = {} 
    tensors_to_start_layer = []    

    # 6-(1) Create input layers and figure out the usage of each input layer
    # Case 1, 2, 3, and 4
    for name, tensor in input_tensors.items():
        # Input layers are created
        input_layers[name] = tf.keras.layers.Input(shape=tensor.shape[1:], name=name)
        
        # Inspect an input layer’s outbound nodes to see where the model consumes it
        origin_layer = model.get_layer(name)
        if len(origin_layer._outbound_nodes) > 1: # If an input layer has multiple outbound layers, its output feeds multiple layers
            for origin_outbound_node in origin_layer._outbound_nodes:
                origin_outbound_layer = origin_outbound_node.outbound_layer
                target_idx = model.layers.index(origin_outbound_layer)
                if target_idx == start:
                    tensors_to_start_layer.append(input_layers[name])
                elif target_idx <= end and target_idx > start:
                    inside_ending_skips[name] = input_layers[name]
                elif target_idx > end:
                    # Any outside-ending skip connections that are made here are not used in this slice
                    outside_ending_skips[name] = input_layers[name]
        else:
            origin_outbound_layer = origin_layer._outbound_nodes[0].outbound_layer
            target_idx = model.layers.index(origin_outbound_layer)
            if target_idx == start:
                tensors_to_start_layer.append(input_layers[name])
            elif target_idx <= end and target_idx > start:
                inside_ending_skips[name] = input_layers[name]
            elif target_idx > end:
                # Any outside-ending skip connections that are made here are not used in this slice
                outside_ending_skips[name] = input_layers[name]

    # 6-(2) Build layers and update inside-ending and outside-ending skip connections
    # Case 5, 6, and 7
    # # Set the start layer's inputs from tensors_to_start_layer
    if(len(tensors_to_start_layer) == 1):
        tensors_to_current_layer = tensors_to_start_layer[0]
    else:
        tensors_to_current_layer = tensors_to_start_layer
    
    for i in range(start, end+1): 
        layer = model.layers[i]
        origin_inbound_layers = layer._inbound_nodes[0].inbound_layers
        
        # Build current (i-th) layer 
        if isinstance(origin_inbound_layers, list): # When current layer expects multiple inputs (list of KerasTensors)
            if(i == start):
                tensors_from_current_layer = layer(tensors_to_current_layer)
            else: 
                # NOTE: The model slicer assumes that the output of layer i is always used by layer i+1
                tensors_to_current_layer = [tensors_to_current_layer] if not isinstance(tensors_to_current_layer, list) else tensors_to_current_layer
                
                # From inbound layers, collect the required inside-ending skip tensors
                for origin_inbound_layer in origin_inbound_layers:
                    inside_ending_skip = model.get_layer(origin_inbound_layer.name)
                    if inside_ending_skip.name not in [t.name.split('/')[0] for t in tensors_to_current_layer]:
                        tensors_to_current_layer.append(inside_ending_skips[inside_ending_skip.name])
                
                # Call the functor of the current layer to build a new layer
                try:
                    tensors_from_current_layer = \
                        layer(tensors_to_current_layer)
                except: # When a custom layer's call signature deviates from Keras expectations
                    raise ValueError(f"Failed to call layer {layer.name} with tensors {tensors_to_current_layer}. "
                                     "Please check the layer's call function and the input tensors.")            
        else: # When current layer expects a single input (KerasTensor)
            if origin_inbound_layers.name in inside_ending_skips:
                tensors_from_current_layer = layer(inside_ending_skips[origin_inbound_layers.name])
            else:
                tensors_from_current_layer = layer(tensors_to_current_layer)

        # Update inside_ending_skips and outside_ending_skips based on the current layer's outbound nodes
        if i < end: # Ensure the current layer is not the slice's last layer
            for origin_outbound_node in layer._outbound_nodes: 
                skip_target_idx = model.layers.index(origin_outbound_node.outbound_layer)
                if skip_target_idx <= end and skip_target_idx > i + 1:
                    inside_ending_skips[layer.name] = tensors_from_current_layer
                elif skip_target_idx > end:
                    outside_ending_skips[layer.name] = tensors_from_current_layer

        tensors_to_current_layer = tensors_from_current_layer # End of for i in range(start, end+1)

    # 6-(3) Set output tensors
    if outside_ending_skips:
        tensors_from_current_layer = [tensors_from_current_layer] + list(outside_ending_skips.values())

    # 6-(4) Create and return the slice
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
def get_slice_starts(num_layers):
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
    slice_starts = [1] + [end + 1 for _, end in slice_pairs]
    slice_ranges = [
        (points[i] + (0 if i == 0 else 1), points[i+1])
        for i in range(len(points)-1)
    ]

    if n==1:
        print("Only converting the model")
    else:
        print(f"Layer index ranges for each submodel: {slice_ranges}")
    
    return n, slice_starts


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Load the model from the given path without compilation (for inference/slicing only)
    model = load_model(args.model_path, compile=False)
    
    # Ask the user for the number of slices and the index of the last layer in each slice
    num_layers = len(model.layers)
    num_slices, slice_starts = get_slice_starts(num_layers)

    # Create a dummy input tensor for the first slice
    input_shape = model.layers[0].input_shape[0][1:]
    dummy_input = np.random.rand(1, *input_shape)

    # Perform slicing and conversion
    slices = []
    for i in range(num_slices):
        # Prepare inputs for each slice
        if i == 0:
            slice_inputs = {model.layers[0].name: dummy_input}
        else:
            slice_inputs = prepare_slice_inputs(slices[i-1])

        # Slice the model using slice_dnn
        slice = slice_dnn(model, slice_starts[i], slice_starts[i+1]-1, slice_inputs)
        slices.append(slice)

        # Convert and save the slice to a LiteRT model
        convert_save_slice(args.output_dir, slice, i)

if __name__ == "__main__":
    main()
