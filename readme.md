# 🧠 Low-Code Quantized InceptionNet Inference for ECG Classification

This repository provides a **NumPy-only implementation** of the inference pipeline for a pre-trained **Quantized InceptionNet** model used for **ECG classification**. The model is quantized using **8-bit integer precision (int8)** and exported as a TensorFlow Lite (`.tflite`) file.

We replicate the core behavior of TFLite's interpreter, including quantization, convolution, and activation functions — all implemented manually with NumPy.

---

## 🔍 What’s Included

- Manual inference of a quantized InceptionNet using only NumPy  
- Reference implementation that mirrors `tf.lite.Interpreter` behavior  
- Example input and weight loading from `.npy` and `.json`  
- Compatible with 1D ECG signals of shape **(1, 320, 1)**  

---

## 🧪 Inference Using TFLite Interpreter (Reference)

To verify correctness, here’s how you can run inference using the original `.tflite` model with TensorFlow Lite:

```python
import tensorflow as tf
import numpy as np

# Load model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get tensor indices
input_index = input_details[0]['index']
output_index = output_details[0]['index']

# Load and quantize input if required
input_data = np.load("ecg_sample.npy").astype(np.float32)
if input_details[0]['dtype'] == np.int8:
    scale, zero_point = input_details[0]['quantization']
    input_data = (input_data / scale + zero_point).astype(np.int8)

# Run inference
interpreter.set_tensor(input_index, input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_index)

# Dequantize output if needed
if output_details[0]['dtype'] == np.int8:
    scale, zero_point = output_details[0]['quantization']
    output = scale * (output.astype(np.float32) - zero_point)

# Get predicted class
pred = np.argmax(output, axis=-1)  # shape: (1, 5)
