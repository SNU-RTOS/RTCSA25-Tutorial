"""
Filename: model_downloader.py

@Author: Taehyun Kim
@Affiliation: Real-Time Operating System Laboratory, Seoul National University
@Modified by: Taehyun Kim on 07/22/25
@Contact: thkim@redwood.snu.ac.kr

@Description: Model downloader for RTCSA25 tutorial

"""

from tensorflow.keras.applications import ResNet50

# Load pretrained ResNet50 model with ImageNet weights
model = ResNet50(weights='imagenet')

# Save the model in .h5 format
model.save("./models/resnet50.h5")

print("Model saved as resnet50.h5")
