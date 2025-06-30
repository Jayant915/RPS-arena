from keras.layers import TFSMLayer
from keras import Model, Input
import numpy as np
import cv2

# Load the SavedModel as a TFSMLayer
layer = TFSMLayer("model.savedmodel", call_endpoint="serving_default")

# Wrap it in a Keras model (input shape must match 224x224x3)
inp = Input(shape=(224, 224, 3))
model = Model(inputs=inp, outputs=layer(inp))

# Test inference
img = np.random.rand(1, 224, 224, 3).astype(np.float32)
output = model.predict(img)
print("Model output:", output)
