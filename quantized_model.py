import numpy as np
import json
import ast
import math
from model_layer_names import dict_all_layer_names
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def quantize_float_to_int8(input_array, scale, zero_point):
    """
    Quantizes a floating-point array to int8 using the provided scale and zero point.

    This function applies affine quantization to convert floating-point values to
    8-bit integers. It scales the input, shifts by the zero point, rounds to the
    nearest integer, and clips the result to the range [-128, 127] to fit int8.

    Parameters:
    - input_array (np.ndarray): The input array of floating-point values to quantize.
    - scale (float or np.ndarray): The scaling factor(s) for quantization. Can be a scalar or per-channel.
    - zero_point (float or np.ndarray): The zero point(s) for quantization. Can be a scalar or per-channel.

    Returns:
    - np.ndarray: The quantized int8 array with the same shape as input_array.
    """
    return np.clip(np.round(input_array / scale + zero_point), -128, 127).astype(np.int8)

def requantize_int32_to_int8(accumulator_int32, input_scale, weight_scale, output_scale, output_zero_point):
    """
    Requantizes an int32 accumulator to int8 for the next layer using combined scales and zero point.

    This function is used after operations like convolution to adjust the scale and zero point
    for the output. It computes a combined scale from input, weight, and output scales,
    applies it to the int32 values, adds the output zero point, rounds, and clips to int8 range.

    Parameters:
    - accumulator_int32 (np.ndarray): The input int32 array (typically accumulator from convolution).
    - input_scale (float): The scale of the input activations.
    - weight_scale (float): The scale of the weights.
    - output_scale (float): The desired output scale.
    - output_zero_point (int): The desired output zero point.

    Returns:
    - np.ndarray: The requantized int8 array.
    """
    combined_scale = (input_scale * weight_scale) / output_scale
    return np.clip(np.round(accumulator_int32 * combined_scale) + output_zero_point, -128, 127).astype(np.int8)

def perform_quantized_conv1d(input_int8, weights_int8, biases_int32, 
                             input_scale, input_zero_point, 
                             weight_scales, weight_zero_points):
    """
    Performs a quantized 1D convolution using int8 inputs and weights, accumulating in int32.

    This function simulates a depthwise or pointwise convolution in quantized form. It pads the input,
    performs the convolution in a nested loop, subtracting zero points for inputs and weights per-channel
    if applicable, adds bias, and returns the int32 accumulator without ReLU or requantization.

    Parameters:
    - input_int8 (np.ndarray): Quantized int8 input activations with shape (B, W, C_in).
    - weights_int8 (np.ndarray): Quantized int8 weights with shape (K, C_in, C_out).
    - biases_int32 (np.ndarray): Quantized biases with shape (C_out,).
    - input_scale (float or np.ndarray): Input activation scale(s).
    - input_zero_point (int or np.ndarray): Input activation zero point(s), can be per-channel.
    - weight_scales (float or np.ndarray): Weight scale(s), per-output-channel.
    - weight_zero_points (int or np.ndarray): Weight zero point(s), per-output-channel.

    Returns:
    - np.ndarray: Int32 accumulator output with shape (B, W, C_out).
    """
    batch_size, width, channels_in = input_int8.shape
    kernel_size, channels_in_weights, channels_out = weights_int8.shape
    assert channels_in == channels_in_weights

    padding_size = kernel_size // 2
    padded_input = np.pad(input_int8, ((0, 0), (padding_size, padding_size), (0, 0)), 'constant', constant_values=input_zero_point)

    output_accumulator = np.zeros((batch_size, width, channels_out), dtype=np.int32)

    for b in range(batch_size):
        for w in range(width):
            for o in range(channels_out):
                acc = 0
                for k in range(kernel_size):
                    for c in range(channels_in):
                        if len(input_zero_point) > 1:
                            input_val = padded_input[b, w + k, c].astype(np.int32) - input_zero_point[c]  # handles per-channel zp
                        else:
                            input_val = padded_input[b, w + k, c].astype(np.int32) - input_zero_point[0]
                        weight_val = weights_int8[k, c, o].astype(np.int32) - weight_zero_points[o]
                        acc += input_val * weight_val
                acc += biases_int32[o]  # scalar bias
                output_accumulator[b, w, o] = acc

    # Do not apply ReLU or requantization here — return int32
    return output_accumulator

def perform_quantized_add(input_a_int8, input_b_int8, output_scale, output_zero_point):
    # Promote to int32 to avoid overflow during addition
    sum_adjusted = input_a_int8.astype(np.int32) + input_b_int8.astype(np.int32) - output_zero_point
    # Clip to int8 range [-128, 127]
    output_int8 = np.clip(sum_adjusted, -128, 127).astype(np.int8)
    return output_int8

def requantize_int32_with_relu(accumulator_int32, input_scale, input_zero_point, weight_scales, output_scales, output_zero_points):
    """
    Requantizes an int32 accumulator to int8 with per-channel scales, applying ReLU after scaling.

    This function handles requantization with potential per-channel parameters. It expands scalar
    scales/zps to per-channel if needed, computes per-channel scales, applies them to the input
    (subtracting input zp first), rounds, adds output zp, applies ReLU (clipping below zp), and
    clips to int8 range.

    Parameters:
    - accumulator_int32 (np.ndarray): Int32 accumulator with shape (B, W, C_out).
    - input_scale (np.ndarray): Input scale(s), typically scalar as array.
    - input_zero_point (np.ndarray): Input zero point(s), typically scalar as array.
    - weight_scales (np.ndarray): Weight scales, can be scalar or per-channel.
    - output_scales (np.ndarray): Output scales, can be scalar or per-channel.
    - output_zero_points (np.ndarray): Output zero points, can be scalar or per-channel.

    Returns:
    - np.ndarray: Requantized int8 output with ReLU applied, shape (B, W, C_out).
    """
    batch_size, width, channels_out = accumulator_int32.shape
    output_array = np.zeros((batch_size, width, channels_out), dtype=np.int8)

    # Expand scalar scale/zp to per-channel if needed
    if len(weight_scales) == 1:
        weight_scales = np.full(channels_out, weight_scales[0])
    if len(output_scales) == 1:
        output_scales = np.full(channels_out, output_scales[0])
    if len(output_zero_points) == 1:
        output_zero_points = np.full(channels_out, output_zero_points[0])

    for o in range(channels_out):
        scale = input_scale[0] * weight_scales[o] / output_scales[o]
        zp = output_zero_points[o]

        scaled_values = (accumulator_int32[..., o] - input_zero_point[0]) * scale
        rounded_values = np.round(scaled_values) + zp
        relu_values = np.maximum(rounded_values, zp)
        output_array[..., o] = np.clip(relu_values, -128, 127).astype(np.int8)

    return output_array

def perform_max_pooling1d(input_array, pool_size=2, stride=2, padding='valid'):
    """
    Performs 1D max pooling on the input array.

    This function applies max pooling along the width dimension. It supports 'valid' (no padding)
    and 'same' padding modes. For 'same', it pads with the minimum value of the dtype to avoid
    affecting max values. The pooling is done per-batch and per-channel in nested loops.

    Parameters:
    - input_array (np.ndarray): Input array with shape (batch_size, width, channels).
    - pool_size (int): Size of the pooling window (default 2).
    - stride (int): Stride for the pooling window (default 2).
    - padding (str): Padding mode, 'valid' or 'same' (default 'valid').

    Returns:
    - np.ndarray: Pooled output with shape (batch_size, out_width, channels).
    """
    batch_size, width, channels = input_array.shape

    if padding == 'same':
        out_width = int(np.ceil(width / stride))
        pad_needed = max((out_width - 1) * stride + pool_size - width, 0)
        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left

        pad_val = np.iinfo(input_array.dtype).min if np.issubdtype(input_array.dtype, np.integer) else -np.inf
        input_array = np.pad(input_array, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=pad_val)
    else:
        out_width = (width - pool_size) // stride + 1

    output_array = np.zeros((batch_size, out_width, channels), dtype=input_array.dtype)
    for b in range(batch_size):
        for c in range(channels):
            for w in range(out_width):
                start = w * stride
                end = start + pool_size
                output_array[b, w, c] = np.max(input_array[b, start:end, c])

    return output_array

def perform_quantized_add(input_a_int8, scale_a, zero_point_a, input_b_int8, scale_b, zero_point_b, output_scale, output_zero_point):
    # Step 1: Dequantize both to float32
    dequant_a = scale_a * (input_a_int8.astype(np.float32) - zero_point_a)
    dequant_b = scale_b * (input_b_int8.astype(np.float32) - zero_point_b)

    # Step 2: Add in float
    added_float = dequant_a + dequant_b

    # Step 3: Requantize to int8 with output scale and zero point
    added_quant = np.round(added_float / output_scale + output_zero_point)
    return np.clip(added_quant, -128, 127).astype(np.int8)

def perform_quantized_global_avg_pool1d(input_int8, input_scale, input_zero_point, output_scale, output_zero_point):
    # Step 1: dequantize
    dequant_input = input_scale * (input_int8.astype(np.float32) - input_zero_point)
    
    # Step 2: global average pool across width (axis=1)
    pooled_float = np.mean(dequant_input, axis=1, keepdims=True)
    
    # Step 3: quantize result
    quantized_output = np.clip(
        np.round(pooled_float / output_scale + output_zero_point),
        -128, 127
    ).astype(np.int8)
    
    return quantized_output

def perform_quantized_dense(input_int8, weights_int8, biases_int32,
                            input_scale, input_zero_point,
                            weight_scales, weight_zero_points,
                            output_scale, output_zero_point):
    """
    input_int8:      (1, input_dim)
    weights_int8: (input_dim, output_dim)
    biases_int32:  (output_dim,)
    """
    input_dim, output_dim = weights_int8.shape
    input_int32 = (input_int8.astype(np.int32) - input_zero_point).reshape(1, input_dim)

    accumulator_int32 = np.zeros((1, output_dim), dtype=np.int32)

    for j in range(output_dim):
        weights_col = weights_int8[:, j].astype(np.int32) - weight_zero_points[0]
        acc = np.sum(input_int32 * weights_col) + biases_int32[j]
        accumulator_int32[0, j] = acc

    # Requantize
    output_int8 = np.zeros_like(accumulator_int32, dtype=np.int8)
    for j in range(output_dim):
        combined_scale = (input_scale * weight_scales[0]) / output_scale
        val = np.round(accumulator_int32[0, j] * combined_scale) + output_zero_point
        output_int8[0, j] = np.clip(val, -128, 127)

    return output_int8

def extract_and_reshape_conv_weights(layer_data, weight_key):
    """
    Processes and reshapes weights from a dictionary (e.g., from JSON).

    This helper function extracts weights by name, squeezes an axis (likely batch or height for 1D conv),
    and transposes to the expected shape (C_in, C_out, K) for convolution.

    Parameters:
    - layer_data (dict): Dictionary containing model weights.
    - weight_key (str): Key for the weights in the data dict.

    Returns:
    - np.ndarray: Reshaped weights array.
    """
    return np.squeeze(np.array(layer_data[weight_key]["weights"]), axis=1).transpose(1, 2, 0)

def extract_biases(layer_data, bias_key):
    """
    Extracts biases from a dictionary (e.g., from JSON).

    This helper function simply converts the biases list to a numpy array.

    Parameters:
    - layer_data (dict): Dictionary containing model biases.
    - bias_key (str): Key for the biases in the data dict.

    Returns:
    - np.ndarray: Biases array.
    """
    return np.array(layer_data[bias_key]["weights"])

def extract_and_reshape_dense_weights(layer_data, weight_key):
    """
    Processes and reshapes weights from a dictionary (e.g., from JSON) for dense layers.

    This helper function extracts weights by name and transposes to the expected shape (input_dim, output_dim).

    Parameters:
    - layer_data (dict): Dictionary containing model weights.
    - weight_key (str): Key for the weights in the data dict.

    Returns:
    - np.ndarray: Reshaped weights array.
    """
    return np.array(layer_data[weight_key]["weights"]).transpose(1, 0)

def extract_dense_biases(layer_data, bias_key):
    """
    Extracts biases from a dictionary (e.g., from JSON) for dense layers.

    This helper function simply converts the biases list to a numpy array.

    Parameters:
    - layer_data (dict): Dictionary containing model biases.
    - bias_key (str): Key for the biases in the data dict.

    Returns:
    - np.ndarray: Biases array.
    """
    return np.array(layer_data[bias_key]["weights"])

def parse_string_to_float_array(string_repr):
    """
    Converts a string representation of a list to a numpy array of floats.

    This function uses ast.literal_eval to safely parse the string as a Python list,
    then converts it to a numpy array.

    Parameters:
    - string_repr (str): String like '[0.1, 0.2]' to parse.

    Returns:
    - np.ndarray: Array of floats.
    """
    return np.array(ast.literal_eval(string_repr))

def extract_scale_and_zero_point(layer_data, key):
    """
    Retrieves scale and zero point arrays from a dictionary by name.

    This helper function extracts 'scale' and 'zp' strings from the data,
    parses them using parse_string_to_float_array, and returns them as numpy arrays.

    Parameters:
    - layer_data (dict): Dictionary containing scales and zero points.
    - key (str): Key for the entry in the data dict.

    Returns:
    - tuple: (scales np.ndarray, zero_points np.ndarray)
    """
    return parse_string_to_float_array(layer_data[key]["scale"]), parse_string_to_float_array(layer_data[key]["zp"])

def process_quantized_inception_block(input_int8, input_scale, input_zero_point, pretrained_weights, block_id, layer_names_dict):
    # Access weights and parameters for the current block
    block_weights = pretrained_weights[block_id]

    # Initial convolution (conv11)
    conv11_weights = extract_and_reshape_conv_weights(block_weights, layer_names_dict["conv11"]["weight"])
    conv11_biases = extract_biases(block_weights, layer_names_dict["conv11"]["bias"])
    conv11_weight_scales, conv11_weight_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["conv11"]["weight"])
    conv11_output_scales, conv11_output_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["relu1"])

    conv11_accumulator = perform_quantized_conv1d(input_int8, conv11_weights, conv11_biases,
                                                  input_scale, input_zero_point,
                                                  conv11_weight_scales, conv11_weight_zps)
    
    conv11_output = requantize_int32_with_relu(conv11_accumulator, 
                                               input_scale,
                                               input_zero_point,
                                               conv11_weight_scales,
                                               conv11_output_scales, conv11_output_zps)

    conv11_pooled = perform_max_pooling1d(conv11_output, pool_size=2, stride=2)
    
    # Inception sub-block 1 (conv12 and branches)
    conv12_weights = extract_and_reshape_conv_weights(block_weights, layer_names_dict["conv12"]["weight"])
    conv12_biases = extract_biases(block_weights, layer_names_dict["conv12"]["bias"])
    conv12_input_scales, conv12_input_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["maxpool1"])
    conv12_weight_scales, conv12_weight_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["conv12"]["weight"])

    conv12_accumulator = perform_quantized_conv1d(conv11_pooled, conv12_weights, conv12_biases,
                                                  conv12_input_scales, conv12_input_zps,
                                                  conv12_weight_scales, conv12_weight_zps)
    
    conv12_out_scales, conv12_out_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["conv12"]["output_scale"])
    conv12_requantized = requantize_int32_to_int8(conv12_accumulator, conv12_input_scales, conv12_weight_scales, conv12_out_scales, conv12_out_zps)
    
    # Branch 1.1
    conv12_1_weights = extract_and_reshape_conv_weights(block_weights, layer_names_dict["conv12_1"]["weight"])
    conv12_1_biases = extract_biases(block_weights, layer_names_dict["conv12_1"]["bias"])
    conv12_1_weight_scales, conv12_1_weight_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["conv12_1"]["weight"])
    conv12_1_output_scales, conv12_1_output_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["conv12_1"]["output_scale"])

    conv12_1_accumulator = perform_quantized_conv1d(conv12_requantized, conv12_1_weights, conv12_1_biases,
                                                    conv12_out_scales, conv12_out_zps, conv12_1_weight_scales, conv12_1_weight_zps)
    branch1_1_output = requantize_int32_to_int8(conv12_1_accumulator, conv12_out_scales, conv12_1_weight_scales, conv12_1_output_scales, conv12_1_output_zps)

    # Branch 1.2
    conv12_2_weights = extract_and_reshape_conv_weights(block_weights, layer_names_dict["conv12_2"]["weight"])
    conv12_2_biases = extract_biases(block_weights, layer_names_dict["conv12_2"]["bias"])
    conv12_2_weight_scales, conv12_2_weight_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["conv12_2"]["weight"])
    conv12_2_output_scales, conv12_2_output_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["conv12_2"]["output_scale"])

    conv12_2_accumulator = perform_quantized_conv1d(conv12_requantized, conv12_2_weights, conv12_2_biases,
                                                    conv12_out_scales, conv12_out_zps, conv12_2_weight_scales, conv12_2_weight_zps)
    branch1_2_output = requantize_int32_to_int8(conv12_2_accumulator, conv12_out_scales, conv12_2_weight_scales, conv12_2_output_scales, conv12_2_output_zps)

    # Branch 1.3
    conv12_3_weights = extract_and_reshape_conv_weights(block_weights, layer_names_dict["conv12_3"]["weight"])
    conv12_3_biases = extract_biases(block_weights, layer_names_dict["conv12_3"]["bias"])
    conv12_3_weight_scales, conv12_3_weight_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["conv12_3"]["weight"])
    conv12_3_output_scales, conv12_3_output_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["conv12_3"]["output_scale"])

    conv12_3_accumulator = perform_quantized_conv1d(conv12_requantized, conv12_3_weights, conv12_3_biases,
                                                    conv12_out_scales, conv12_out_zps, conv12_3_weight_scales, conv12_3_weight_zps)
    branch1_3_output = requantize_int32_to_int8(conv12_3_accumulator, conv12_out_scales, conv12_3_weight_scales, conv12_3_output_scales, conv12_3_output_zps)

    # Branch 1.4 (with max pooling)
    branch1_4_pooled = perform_max_pooling1d(conv11_pooled, pool_size=3, stride=1, padding="same")

    conv12_4_weights = extract_and_reshape_conv_weights(block_weights, layer_names_dict["conv12_4"]["weight"])
    conv12_4_biases = extract_biases(block_weights, layer_names_dict["conv12_4"]["bias"])
    conv12_4_weight_scales, conv12_4_weight_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["conv12_4"]["weight"])
    conv12_4_input_scales, conv12_4_input_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["maxpool2"])
    concat1_scales, concat1_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["conv12_4"]["output_scale"])

    conv12_4_accumulator = perform_quantized_conv1d(branch1_4_pooled, conv12_4_weights, conv12_4_biases,
                                                    conv12_4_input_scales, conv12_4_input_zps, conv12_4_weight_scales, conv12_4_weight_zps)
    branch1_4_output = requantize_int32_to_int8(conv12_4_accumulator, conv12_4_input_scales, conv12_4_weight_scales, concat1_scales, concat1_zps)

    # Concatenate branches for sub-block 1
    concat1_output = np.concatenate([branch1_4_output, branch1_1_output, branch1_2_output, branch1_3_output], axis=-1)

    relu1_scales, relu1_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["relu2"])
    concat1_with_relu = requantize_int32_with_relu(concat1_output, concat1_scales, concat1_zps, [1.0], relu1_scales, relu1_zps)
    
    # Inception sub-block 2 (conv13 and branches)
    conv13_weights = extract_and_reshape_conv_weights(block_weights, layer_names_dict["conv13"]["weight"])
    conv13_biases = extract_biases(block_weights, layer_names_dict["conv13"]["bias"])
    conv13_input_scales, conv13_input_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["relu2"])
    conv13_weight_scales, conv13_weight_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["conv13"]["weight"])
    
    conv13_accumulator = perform_quantized_conv1d(concat1_with_relu, conv13_weights, conv13_biases,
                                                  conv13_input_scales, conv13_input_zps,
                                                  conv13_weight_scales, conv13_weight_zps)
    
    conv13_out_scales, conv13_out_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["conv13"]["output_scale"])
    conv13_requantized = requantize_int32_to_int8(conv13_accumulator, conv13_input_scales, conv13_weight_scales, conv13_out_scales, conv13_out_zps)

    # Branch 2.1
    conv13_1_weights = extract_and_reshape_conv_weights(block_weights, layer_names_dict["conv13_1"]["weight"])
    conv13_1_biases = extract_biases(block_weights, layer_names_dict["conv13_1"]["bias"])
    conv13_1_weight_scales, conv13_1_weight_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["conv13_1"]["weight"])
    conv13_1_output_scales, conv13_1_output_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["conv13_1"]["output_scale"])

    conv13_1_accumulator = perform_quantized_conv1d(conv13_requantized, conv13_1_weights, conv13_1_biases,
                                                    conv13_out_scales, conv13_out_zps, conv13_1_weight_scales, conv13_1_weight_zps)
    branch2_1_output = requantize_int32_to_int8(conv13_1_accumulator, conv13_out_scales, conv13_1_weight_scales, conv13_1_output_scales, conv13_1_output_zps)

    # Branch 2.2
    conv13_2_weights = extract_and_reshape_conv_weights(block_weights, layer_names_dict["conv13_2"]["weight"])
    conv13_2_biases = extract_biases(block_weights, layer_names_dict["conv13_2"]["bias"])
    conv13_2_weight_scales, conv13_2_weight_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["conv13_2"]["weight"])
    conv13_2_output_scales, conv13_2_output_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["conv13_2"]["output_scale"])

    conv13_2_accumulator = perform_quantized_conv1d(conv13_requantized, conv13_2_weights, conv13_2_biases,
                                                    conv13_out_scales, conv13_out_zps, conv13_2_weight_scales, conv13_2_weight_zps)
    branch2_2_output = requantize_int32_to_int8(conv13_2_accumulator, conv13_out_scales, conv13_2_weight_scales, conv13_2_output_scales, conv13_2_output_zps)

    # Branch 2.3
    conv13_3_weights = extract_and_reshape_conv_weights(block_weights, layer_names_dict["conv13_3"]["weight"])
    conv13_3_biases = extract_biases(block_weights, layer_names_dict["conv13_3"]["bias"])
    conv13_3_weight_scales, conv13_3_weight_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["conv13_3"]["weight"])
    conv13_3_output_scales, conv13_3_output_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["conv13_3"]["output_scale"])

    conv13_3_accumulator = perform_quantized_conv1d(conv13_requantized, conv13_3_weights, conv13_3_biases,
                                                    conv13_out_scales, conv13_out_zps, conv13_3_weight_scales, conv13_3_weight_zps)
    branch2_3_output = requantize_int32_to_int8(conv13_3_accumulator, conv13_out_scales, conv13_3_weight_scales, conv13_3_output_scales, conv13_3_output_zps)
    
    # Branch 2.4 (with max pooling)
    branch2_4_pooled = perform_max_pooling1d(concat1_with_relu, pool_size=3, stride=1, padding="same")

    conv13_4_weights = extract_and_reshape_conv_weights(block_weights, layer_names_dict["conv13_4"]["weight"])
    conv13_4_biases = extract_biases(block_weights, layer_names_dict["conv13_4"]["bias"])
    conv13_4_weight_scales, conv13_4_weight_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["conv13_4"]["weight"])
    conv13_4_input_scales, conv13_4_input_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["maxpool3"])
    concat2_scales, concat2_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["conv13_4"]["output_scale"])

    conv13_4_accumulator = perform_quantized_conv1d(branch2_4_pooled, conv13_4_weights, conv13_4_biases,
                                                    conv13_4_input_scales, conv13_4_input_zps, conv13_4_weight_scales, conv13_4_weight_zps)
    branch2_4_output = requantize_int32_to_int8(conv13_4_accumulator, conv13_4_input_scales, conv13_4_weight_scales, concat2_scales, concat2_zps)

    # Concatenate branches for sub-block 2
    concat2_output = np.concatenate([branch2_4_output, branch2_1_output, branch2_2_output, branch2_3_output], axis=-1)

    relu2_scales, relu2_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["relu3"])
    concat2_with_relu = requantize_int32_with_relu(concat2_output, concat2_scales, concat2_zps, [1.0], relu2_scales, relu2_zps)
    
    # Skip connection (add)
    add_output_scales, add_output_zps = extract_scale_and_zero_point(block_weights, layer_names_dict["add1"])    
    block_output = perform_quantized_add(conv11_pooled, conv12_input_scales, conv12_input_zps, concat2_with_relu, relu2_scales, relu2_zps, add_output_scales, add_output_zps)
    return block_output, add_output_scales, add_output_zps

def run_quantized_model_on_input(input_int8, pretrained_weights, input_scale, input_zero_point):
    """
    Simulates a forward pass through a quantized neural network model.

    This function implements a forward pass for a quantized convolutional neural network, 
    focusing on the first block and an inception-like block. It loads pretrained quantized 
    weights and quantization parameters (scales and zero points) from a JSON file, 
    quantizes a toy input, and processes it through convolution, ReLU, max pooling, and 
    an inception-style block with four parallel branches. The function handles per-layer 
    and per-branch quantization, ensuring compatibility with int8 arithmetic for efficient 
    inference on resource-constrained devices. The outputs of the inception branches are 
    concatenated and passed through a final ReLU activation.

    Key insights:
    - The model is designed for int8 quantization, which reduces memory usage and 
      computational cost, critical for edge devices like mobile phones or embedded systems.
    - The inception-like block employs multiple branches with different convolutional 
      operations, enhancing feature extraction by capturing diverse patterns.
    - Per-channel quantization is supported, allowing finer control over numerical 
      precision for weights and activations, which can improve accuracy compared to 
      per-tensor quantization.
    - The function assumes the existence of 'model_weights_scales.json' (containing 
      weights and quantization parameters) and 'toy_input.npy' (input data), which 
      simplifies testing but requires these files to be correctly formatted.

    No parameters or returns; it executes the model simulation directly and prints the 
    final output for inspection.
    """
    output_int8_1, output_scale_1, output_zp_1 = process_quantized_inception_block(input_int8, input_scale, input_zero_point, pretrained_weights, "first_block", dict_all_layer_names["first_block"])
    output_int8_2, output_scale_2, output_zp_2 = process_quantized_inception_block(output_int8_1, output_scale_1, output_zp_1, pretrained_weights, "second_block", dict_all_layer_names["second_block"])
    output_int8_3, output_scale_3, output_zp_3 = process_quantized_inception_block(output_int8_2, output_scale_2, output_zp_2, pretrained_weights, "third_block", dict_all_layer_names["third_block"])

    # Global average pooling
    global_pool_scales, global_pool_zps = extract_scale_and_zero_point(pretrained_weights, "model/quant_global_average_pooling2d/Mean")
    pooled_output = perform_quantized_global_avg_pool1d(output_int8_3, output_scale_3, output_zp_3, global_pool_scales, global_pool_zps)

    # Dense layer
    dense_weights = extract_and_reshape_dense_weights(pretrained_weights["dense_layer"], "model/quant_dense/MatMul;model/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars")
    dense_biases = extract_dense_biases(pretrained_weights["dense_layer"], "model/quant_dense/LastValueQuant_1/FakeQuantWithMinMaxVars")

    dense_input_scales, dense_input_zps = global_pool_scales, global_pool_zps
    dense_weight_scales, dense_weight_zps = extract_scale_and_zero_point(pretrained_weights["dense_layer"], "model/quant_dense/MatMul;model/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars")
    dense_output_scales, dense_output_zps = extract_scale_and_zero_point(pretrained_weights, "output")

    logits = perform_quantized_dense(np.squeeze(pooled_output, axis=1), dense_weights, dense_biases,
                                     dense_input_scales, dense_input_zps,
                                     dense_weight_scales, dense_weight_zps,
                                     dense_output_scales[0], dense_output_zps[0])

    return logits

def main():
    # Load pretrained quantized weights and quantization parameters from JSON.
    # Insight: JSON is used for portability and human-readability, but parsing large 
    # JSON files can be a bottleneck in production systems; binary formats like 
    # TensorFlow's TFLite might be preferred for efficiency.
    pretrained_weights = json.load(open("model_weights_scales.json", "r"))

    # Extract input scale and zero point for quantization.
    # Insight: The use of parse_string_to_float_array suggests quantization parameters are 
    # stored as strings in JSON, requiring parsing. This could introduce errors if 
    # the string format is inconsistent.
    input_scale, input_zp = parse_string_to_float_array(pretrained_weights["input"]["scale"]), parse_string_to_float_array(pretrained_weights["input"]["zp"])    
    output_scale, output_zp = parse_string_to_float_array(pretrained_weights["output"]["scale"]), parse_string_to_float_array(pretrained_weights["output"]["zp"])
    
    all_predictions = []
    all_ground_truths = []

    test_inputs, test_labels = np.load("test_data.npy"), np.load("test_gts.npy")
    progress_bar = tqdm(total=None)  # Unknown length

    N = 33
    for i, (input_data, label) in enumerate(zip(test_inputs, test_labels)):
        quantized_input = quantize_float_to_int8(input_data, input_scale, input_zp)

        model_output = run_quantized_model_on_input(quantized_input, pretrained_weights, input_scale, input_zp)

        # Dequantize
        dequant_output = output_scale * (model_output.astype(np.float32) - output_zp)

        prediction = np.argmax(dequant_output, axis=-1)
        all_predictions.append(prediction[0])
        all_ground_truths.append(label)

        progress_bar.update(1)  # Manually advance progress bar
        if i > N:
            break
            
    progress_bar.close()

    print(len(all_ground_truths))
    accuracy = accuracy_score(all_ground_truths, all_predictions)
    f1 = f1_score(all_ground_truths, all_predictions, average='weighted')
    precision = precision_score(all_ground_truths, all_predictions, average='weighted')
    recall = recall_score(all_ground_truths, all_predictions, average='weighted')
    conf_matrix = confusion_matrix(all_ground_truths, all_predictions)

    print("Accuracy:", accuracy)
    print("F1:", f1)
    print("Precision:", precision)
    print("Recall:", recall)

    print(all_predictions)
    print(all_ground_truths)

main()
