# 🧠 Low-Code Quantized InceptionNet Inference for ECG Classification

This repository provides a **NumPy-only implementation** of the inference pipeline for a pre-trained **Quantized InceptionNet** model used for **ECG classification**. The model is quantized using **8-bit integer precision (int8)** and follows the structure and behavior of a TensorFlow Lite model, but without requiring TensorFlow at all.

We manually replicate TFLite's quantized inference process using NumPy, including quantization, convolution, pooling, and activation operations.

---

## 🔍 What’s Included

- Manual forward-pass inference for a quantized InceptionNet using only NumPy  
- Layer-wise emulation of TFLite interpreter behavior  
- JSON-based model structure and weights  
- Example ECG signal data for testing  
- No TensorFlow or TFLite dependency required  

---

## 📁 Files

- `quantized_model.py` – NumPy-based inference implementation for Quantized InceptionNet  
- `model_layer_names.py` – Structured layer names grouped into three Inception blocks  
- `model_weights_scales.json` – Dictionary containing pre-trained quantized weights, scales, and zero points for each layer  
- `test_data.npy` – Example pre-processed ECG input data (shape: `(1, 320, 1)`)  
- `test_gts.npy` – Ground-truth labels for test data  

---

## ▶️ Running the Model

To run the NumPy-only inference:

```
python quantized_model.py
```

Accuracy on the test set should be 98.11%

## ⚖️ Optional: TensorFlow Lite Inference (for Verification)
If you'd like to compare results with the original .tflite model, you can use the TensorFlow Lite interpreter as a reference:

```[python3]
import tensorflow as tf
import numpy as np

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = input_details[0]['index']
output_index = output_details[0]['index']

# Load and quantize input
input_data = np.load("ecg_sample.npy").astype(np.float32)
if input_details[0]['dtype'] == np.int8:
    scale, zero_point = input_details[0]['quantization']
    input_data = (input_data / scale + zero_point).astype(np.int8)

# Run inference
interpreter.set_tensor(input_index, input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_index)

# Dequantize output
if output_details[0]['dtype'] == np.int8:
    scale, zero_point = output_details[0]['quantization']
    output = scale * (output.astype(np.float32) - zero_point)

# Get prediction
pred = np.argmax(output, axis=-1)  # shape: (1, 5)
print("Predicted class:", pred)
```

⚠️ Note: TensorFlow is not required for the core model. This section is optional for verifying your NumPy-based results.

## 📌 Requirements
- Python 3.8+
- NumPy
- (Optional) TensorFlow (only if running the verification script above)

## 💡 Motivation
TFLite provides highly optimized inference on edge devices, but hides internal mechanics. This project reveals and replicates TFLite’s quantized operations using only NumPy, which is useful for:

- Custom deployments on hardware without TensorFlow/TFLite
- Debugging inference results or verifying layer-by-layer behavior
