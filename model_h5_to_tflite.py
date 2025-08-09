#!/usr/bin/env python
#  -*- coding: utf-8 -*-
"""
Filename: model_h5_to_tflite.py

@Author: Taehyun Kim
@Created: 07/23/25
@Affiliation: Real-Time Operating System Laboratory, Seoul National University
@Contact: thkim@redwood.snu.ac.kr

@Description: Convert .h5 Keras model to .tflite format

"""

import tensorflow as tf
import argparse
import os

# Suppress unessary TensorFlow warnings
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def convert_h5_to_tflite(h5_path, output_dir):
    # Load .h5 Keras model
    model = tf.keras.models.load_model(h5_path, compile=False)

    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Convert the model to TFLite format
    tflite_model = converter.convert()

    # Set output directory and model name
    model_name = os.path.splitext(os.path.basename(h5_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    tflite_path = os.path.join(output_dir, f"{model_name}.tflite")

    # Save the TFLite model
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print(f"TFLite model saved to: {tflite_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .h5 Keras model to .tflite")
    parser.add_argument('--h5-path', type=str, default='./models/resnet50.h5', help="Path to the input .h5 model")
    parser.add_argument('--output-dir', type=str, default='./models', help="Directory to save .tflite model")
    args = parser.parse_args()

    convert_h5_to_tflite(args.h5_path, args.output_dir)
