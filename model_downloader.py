#!/usr/bin/env python
#  -*- coding: utf-8 -*-
"""
Filename: model_downloader.py

@Author: Taehyun Kim
@Created: 07/23/25
@Affiliation: Real-Time Operating System Laboratory, Seoul National University
@Contact: thkim@redwood.snu.ac.kr

@Description: Model downloader for RTCSA25 tutorial

"""

from tensorflow.keras.applications import ResNet50

# Load pretrained ResNet50 model with ImageNet weights
model = ResNet50(weights='imagenet')

# Save the model in .h5 format
model.save("./models/resnet50.h5")

print("Model saved as resnet50.h5")
