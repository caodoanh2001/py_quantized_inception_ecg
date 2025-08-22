import numpy as np
import json
import ast
import math
from model_layer_names import dict_all_layer_names
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    
def quantize(x, scale, zp):
    """
    Quantizes a floating-point array to int8 using the provided scale and zero point.

    This function applies affine quantization to convert floating-point values to
    8-bit integers. It scales the input, shifts by the zero point, rounds to the
    nearest integer, and clips the result to the range [-128, 127] to fit int8.

    Parameters:
    - x (np.ndarray): The input array of floating-point values to quantize.
    - scale (float or np.ndarray): The scaling factor(s) for quantization. Can be a scalar or per-channel.
    - zp (float or np.ndarray): The zero point(s) for quantization. Can be a scalar or per-channel.

    Returns:
    - np.ndarray: The quantized int8 array with the same shape as x.
    """
    return np.clip(np.round(x / scale + zp), -128, 127).astype(np.int8)

def requantize(x_int32, input_scale, weight_scale, output_scale, output_zp):
    """
    Requantizes an int32 accumulator to int8 for the next layer using combined scales and zero point.

    This function is used after operations like convolution to adjust the scale and zero point
    for the output. It computes a combined scale from input, weight, and output scales,
    applies it to the int32 values, adds the output zero point, rounds, and clips to int8 range.

    Parameters:
    - x_int32 (np.ndarray): The input int32 array (typically accumulator from convolution).
    - input_scale (float): The scale of the input activations.
    - weight_scale (float): The scale of the weights.
    - output_scale (float): The desired output scale.
    - output_zp (int): The desired output zero point.

    Returns:
    - np.ndarray: The requantized int8 array.
    """
    scale = (input_scale * weight_scale) / output_scale
    return np.clip(np.round(x_int32 * scale) + output_zp, -128, 127).astype(np.int8)

def quantized_conv1d(x_int8, w_q, b_q, 
                          input_scale, input_zp, 
                          weight_scales, weight_zps,
                          stride=1):
    """
    Quantized 1D convolution (int8 inputs & weights, int32 accumulators) with
    TF-like "same" padding and configurable stride.

    Shapes:
      x_int8: (B, W, C_in)          int8
      w_q:    (K, C_in, C_out)      int8
      b_q:    (C_out,)              int32 (or castable)
      input_zp:  scalar or (C_in,)  int
      weight_zps: (C_out,)          int

    No ReLU or requantization is applied; returns raw int32 accumulators.

    Output:
      out: (B, ceil(W/stride), C_out)
    """
    # Basic checks
    assert stride >= 1, "stride must be >= 1"
    B, W, C_in = x_int8.shape
    K, C_in_w, C_out = w_q.shape
    assert C_in == C_in_w, "Input channels must match weight's C_in."

    # Normalize ZPs
    input_zp_arr = np.array(input_zp, dtype=np.int32).reshape(-1)
    if input_zp_arr.size not in (1, C_in):
        raise ValueError("input_zp must be a scalar or length C_in.")
    weight_zps_arr = np.array(weight_zps, dtype=np.int32).reshape(-1)
    assert weight_zps_arr.size == C_out, "weight_zps must be per-output-channel."

    # --- SAME padding math (1D) ---
    # Desired output width
    W_out = math.ceil(W / stride)
    # Total padding required to achieve W_out with kernel K and stride
    total_pad = max(0, (W_out - 1) * stride + K - W)
    pad_left = total_pad // 2
    pad_right = total_pad - pad_left

    # Create padded input, filling padding with input_zp so (x - zp) = 0 there.
    x_padded = np.zeros((B, W + pad_left + pad_right, C_in), dtype=np.int8)
    # Fill the middle with the actual input
    x_padded[:, pad_left:pad_left + W, :] = x_int8

    # Fill left/right pads with input zero-points (scalar or per-channel)
    if input_zp_arr.size == 1:
        val = np.int8(input_zp_arr[0])
        if pad_left > 0:
            x_padded[:, :pad_left, :] = val
        if pad_right > 0:
            x_padded[:, -pad_right:, :] = val
    else:
        # per-channel
        if pad_left > 0:
            x_padded[:, :pad_left, :] = np.int8(input_zp_arr[np.newaxis, np.newaxis, :])
        if pad_right > 0:
            x_padded[:, -pad_right:, :] = np.int8(input_zp_arr[np.newaxis, np.newaxis, :])

    out = np.zeros((B, W_out, C_out), dtype=np.int32)

    # Convolution
    for b in range(B):
        for w_out_idx in range(W_out):
            in_w_start = w_out_idx * stride
            # window covers indices [in_w_start, in_w_start + K)
            for o in range(C_out):
                acc = 0
                w_zp_o = weight_zps_arr[o]
                for k in range(K):
                    iw = in_w_start + k
                    if input_zp_arr.size == 1:
                        izp = input_zp_arr[0]
                        for c in range(C_in):
                            x_val = np.int32(x_padded[b, iw, c]) - izp
                            w_val = np.int32(w_q[k, c, o]) - w_zp_o
                            acc += x_val * w_val
                    else:
                        # per-channel input zp
                        for c in range(C_in):
                            x_val = np.int32(x_padded[b, iw, c]) - np.int32(input_zp_arr[c])
                            w_val = np.int32(w_q[k, c, o]) - w_zp_o
                            acc += x_val * w_val
                acc += np.int32(b_q[o])
                out[b, w_out_idx, o] = acc

    return out

def quantized_add(a_int8, b_int8, out_scale, out_zp):
    # Promote to int32 to avoid overflow during addition
    sum_adjusted = a_int8.astype(np.int32) + b_int8.astype(np.int32) - out_zp
    # Clip to int8 range [-128, 127]
    out_int8 = np.clip(sum_adjusted, -128, 127).astype(np.int8)
    return out_int8

def requantize_with_relu(x_int32, input_scale, input_zp, weight_scales, output_scales, output_zps):
    """
    Requantizes an int32 accumulator to int8 with per-channel scales, applying ReLU after scaling.

    This function handles requantization with potential per-channel parameters. It expands scalar
    scales/zps to per-channel if needed, computes per-channel scales, applies them to the input
    (subtracting input zp first), rounds, adds output zp, applies ReLU (clipping below zp), and
    clips to int8 range.

    Parameters:
    - x_int32 (np.ndarray): Int32 accumulator with shape (B, W, C_out).
    - input_scale (np.ndarray): Input scale(s), typically scalar as array.
    - input_zp (np.ndarray): Input zero point(s), typically scalar as array.
    - weight_scales (np.ndarray): Weight scales, can be scalar or per-channel.
    - output_scales (np.ndarray): Output scales, can be scalar or per-channel.
    - output_zps (np.ndarray): Output zero points, can be scalar or per-channel.

    Returns:
    - np.ndarray: Requantized int8 output with ReLU applied, shape (B, W, C_out).
    """
    B, W, C_out = x_int32.shape
    output = np.zeros((B, W, C_out), dtype=np.int8)

    # Expand scalar scale/zp to per-channel if needed
    if len(weight_scales) == 1:
        weight_scales = np.full(C_out, weight_scales[0])
    if len(output_scales) == 1:
        output_scales = np.full(C_out, output_scales[0])
    if len(output_zps) == 1:
        output_zps = np.full(C_out, output_zps[0])

    for o in range(C_out):
        scale = input_scale[0] * weight_scales[o] / output_scales[o]
        zp = output_zps[o]

        x_scaled = (x_int32[..., o] - input_zp[0]) * scale
        x_rounded = np.round(x_scaled) + zp
        x_relu = np.maximum(x_rounded, zp)
        output[..., o] = np.clip(x_relu, -128, 127).astype(np.int8)

    return output
    
def max_pooling1d(x, pool_size=2, stride=2, padding='valid'):
    """
    Performs 1D max pooling on the input array.

    This function applies max pooling along the width dimension. It supports 'valid' (no padding)
    and 'same' padding modes. For 'same', it pads with the minimum value of the dtype to avoid
    affecting max values. The pooling is done per-batch and per-channel in nested loops.

    Parameters:
    - x (np.ndarray): Input array with shape (batch_size, width, channels).
    - pool_size (int): Size of the pooling window (default 2).
    - stride (int): Stride for the pooling window (default 2).
    - padding (str): Padding mode, 'valid' or 'same' (default 'valid').

    Returns:
    - np.ndarray: Pooled output with shape (batch_size, out_width, channels).
    """
    batch_size, width, channels = x.shape

    if padding == 'same':
        out_width = int(np.ceil(width / stride))
        pad_needed = max((out_width - 1) * stride + pool_size - width, 0)
        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left

        pad_val = np.iinfo(x.dtype).min if np.issubdtype(x.dtype, np.integer) else -np.inf
        x = np.pad(x, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=pad_val)
    else:
        out_width = (width - pool_size) // stride + 1

    out = np.zeros((batch_size, out_width, channels), dtype=x.dtype)
    for b in range(batch_size):
        for c in range(channels):
            for w in range(out_width):
                start = w * stride
                end = start + pool_size
                out[b, w, c] = np.max(x[b, start:end, c])

    return out

def quantized_add(x1_int8, s1, z1, x2_int8, s2, z2, s_out, z_out):
    # Step 1: Dequantize both to float32
    x1_float = s1 * (x1_int8.astype(np.float32) - z1)
    x2_float = s2 * (x2_int8.astype(np.float32) - z2)

    # Step 2: Add in float
    added_float = x1_float + x2_float

    # Step 3: Requantize to int8 with output scale and zero point
    added_quant = np.round(added_float / s_out + z_out)
    return np.clip(added_quant, -128, 127).astype(np.int8)

def quantized_global_avg_pool1d(x_int8, input_scale, input_zp, output_scale, output_zp):
    # Step 1: dequantize
    x_float = input_scale * (x_int8.astype(np.float32) - input_zp)
    
    # Step 2: global average pool across width (axis=1)
    pooled_float = np.mean(x_float, axis=1, keepdims=True)
    
    # Step 3: quantize result
    x_q = np.clip(
        np.round(pooled_float / output_scale + output_zp),
        -128, 127
    ).astype(np.int8)
    
    return x_q

def quantized_dense(x_int8, weight_int8, bias_int32,
                    input_scale, input_zp,
                    weight_scales, weight_zps,
                    output_scale, output_zp):
    """
    x_int8:      (1, input_dim)
    weight_int8: (input_dim, output_dim)
    bias_int32:  (output_dim,)
    """
    input_dim, output_dim = weight_int8.shape
    x_int32 = (x_int8.astype(np.int32) - input_zp).reshape(1, input_dim)

    out_int32 = np.zeros((1, output_dim), dtype=np.int32)

    for j in range(output_dim):
        w = weight_int8[:, j].astype(np.int32) - weight_zps[0]
        acc = np.sum(x_int32 * w) + bias_int32[j]
        out_int32[0, j] = acc

    # Requantize
    out_int8 = np.zeros_like(out_int32, dtype=np.int8)
    for j in range(output_dim):
        scale = (input_scale * weight_scales[0]) / output_scale
        val = np.round(out_int32[0, j] * scale) + output_zp
        out_int8[0, j] = np.clip(val, -128, 127)

    return out_int8
    
def process_weights(data, name):
    """
    Processes and reshapes weights from a dictionary (e.g., from JSON).

    This helper function extracts weights by name, squeezes an axis (likely batch or height for 1D conv),
    and transposes to the expected shape (C_in, C_out, K) for convolution.

    Parameters:
    - data (dict): Dictionary containing model weights.
    - name (str): Key for the weights in the data dict.

    Returns:
    - np.ndarray: Reshaped weights array.
    """
    return np.squeeze(np.array(data[name]["weights"]), axis=1).transpose(1, 2, 0)

def process_biases(data, name):
    """
    Extracts biases from a dictionary (e.g., from JSON).

    This helper function simply converts the biases list to a numpy array.

    Parameters:
    - data (dict): Dictionary containing model biases.
    - name (str): Key for the biases in the data dict.

    Returns:
    - np.ndarray: Biases array.
    """
    return np.array(data[name]["weights"])

def process_dense_weights(data, name):
    """
    Processes and reshapes weights from a dictionary (e.g., from JSON).

    This helper function extracts weights by name, squeezes an axis (likely batch or height for 1D conv),
    and transposes to the expected shape (C_in, C_out, K) for convolution.

    Parameters:
    - data (dict): Dictionary containing model weights.
    - name (str): Key for the weights in the data dict.

    Returns:
    - np.ndarray: Reshaped weights array.
    """
    return np.array(data[name]["weights"]).transpose(1, 0)

def process_dense_biases(data, name):
    """
    Extracts biases from a dictionary (e.g., from JSON).

    This helper function simply converts the biases list to a numpy array.

    Parameters:
    - data (dict): Dictionary containing model biases.
    - name (str): Key for the biases in the data dict.

    Returns:
    - np.ndarray: Biases array.
    """
    return np.array(data[name]["weights"])

def str_to_float_list(str):
    """
    Converts a string representation of a list to a numpy array of floats.

    This function uses ast.literal_eval to safely parse the string as a Python list,
    then converts it to a numpy array.

    Parameters:
    - str (str): String like '[0.1, 0.2]' to parse.

    Returns:
    - np.ndarray: Array of floats.
    """
    return np.array(ast.literal_eval(str))

def get_scale_and_zero_points(data, name):
    """
    Retrieves scale and zero point arrays from a dictionary by name.

    This helper function extracts 'scale' and 'zp' strings from the data,
    parses them using str_to_float_list, and returns them as numpy arrays.

    Parameters:
    - data (dict): Dictionary containing scales and zero points.
    - name (str): Key for the entry in the data dict.

    Returns:
    - tuple: (scales np.ndarray, zero_points np.ndarray)
    """
    return np.array(str_to_float_list(data[name]["scale"])), np.array(str_to_float_list(data[name]["zp"]))

def quantized_inception_block(x_int8, input_scale, input_zp, weights_for_this_block, dict_layer_names):
    ### First Block (quantized conv2d)
    #############################################################################################################################################################################################
    # Process weights and biases for the first convolution (conv11).
    # Insight: The process_weights function reshapes weights to (C_in, C_out, K), 
    # which is critical for 1D convolution compatibility. The long key names suggest 
    # the JSON is exported from a framework like TensorFlow with quantization-aware training.

    conv11_weights = process_weights(weights_for_this_block, dict_layer_names["conv11"]["weight"])
    conv11_biases = process_biases(weights_for_this_block, dict_layer_names["conv11"]["bias"])

    # Retrieve quantization parameters for weights and outputs.
    # Insight: Per-channel quantization for weights (conv11_weight_scales, conv11_weight_zps) 
    # allows each output channel to have its own scale and zero point, potentially reducing 
    # quantization error compared to a single scale for all channels.
    conv11_weight_scales, conv11_weight_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["conv11"]["weight"])
    conv11_output_scales, conv11_output_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["relu1"])

    # Perform first convolution, ReLU, and max pooling.
    # Insight: The quantized_conv1d function accumulates in int32 to avoid overflow, 
    # a common practice in quantized neural networks since int8 multiplication results 
    # can exceed the int8 range.
    out_conv11 = quantized_conv1d(x_int8, conv11_weights, conv11_biases,
                                  input_scale, input_zp,
                                  conv11_weight_scales, conv11_weight_zps, stride=2)
    
    # Requantize with ReLU to prepare for the next layer.
    # Insight: requantize_with_relu applies ReLU after scaling, ensuring non-negative 
    # outputs where required, and clips to int8, maintaining compatibility with 
    # subsequent layers that expect int8 inputs.
    out_conv11 = requantize_with_relu(out_conv11, 
                                      input_scale,
                                      input_zp,
                                      conv11_weight_scales,
                                      conv11_output_scales, conv11_output_zps)

    # Apply max pooling to reduce spatial dimensions.
    # Insight: Max pooling with pool_size=2 and stride=2 halves the width, reducing 
    # computational load for subsequent layers while preserving important features.
    # out_conv11 = max_pooling1d(out_conv11, pool_size=2, stride=2)
    
    ## Inception Block 1
    # Insight: The inception block uses multiple parallel branches to capture features 
    # at different scales, a design inspired by Inception architectures (e.g., GoogLeNet), 
    # which improves model expressiveness without significantly increasing computation.
    
    # Process weights and biases for the first convolution in the inception block.
    conv12_weights = process_weights(weights_for_this_block, dict_layer_names["conv12"]["weight"])
    conv12_biases = process_biases(weights_for_this_block, dict_layer_names["conv12"]["bias"])
    
    # Retrieve quantization parameters for the convolution input and weights.
    # Insight: The input scales/zps come from the previous max pooling layer, 
    # ensuring the quantization parameters are correctly propagated through the network.
    conv12_input_scales, conv12_input_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["relu1"])
    conv12_weight_scales, conv12_weight_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["conv12"]["weight"])

    # Apply convolution for the inception block's initial layer.
    out_conv12 = quantized_conv1d(out_conv11, conv12_weights, conv12_biases,
                                  conv12_input_scales, conv12_input_zps,
                                  conv12_weight_scales, conv12_weight_zps)
    
    # Retrieve output quantization parameters and requantize.
    # Insight: Requantization here adjusts the output to match the scale expected by 
    # the subsequent branches, ensuring numerical consistency across the inception block.
    conv12_out_scales, conv12_out_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["conv12"]["output_scale"])
    requantize_conv12 = requantize(out_conv12, conv12_input_scales, conv12_weight_scales, conv12_out_scales, conv12_out_zps)
    
    # Requantize for three branches (1.1, 1.2, 1.3) and process branch 1.4 separately.
    # Insight: Each branch applies a different convolution to capture varied features, 
    # and requantization ensures outputs are aligned for concatenation later.

    # BRANCH 1.1
    # Process weights, biases, and quantization parameters for the first branch.
    conv12_1_weights = process_weights(weights_for_this_block, dict_layer_names["conv12_1"]["weight"])
    conv12_1_biases = process_biases(weights_for_this_block, dict_layer_names["conv12_1"]["bias"])
    conv12_1_weight_scales, conv12_1_weight_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["conv12_1"]["weight"])
    conv12_1_output_scales, conv12_1_output_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["conv12_1"]["output_scale"])

    # Apply convolution and requantize for branch 1.1.
    # Insight: Each branch processes the same input (requantize_conv12), allowing 
    # parallel feature extraction with different filter sizes or configurations.
    out_conv12_1 = quantized_conv1d(requantize_conv12, conv12_1_weights, conv12_1_biases,
                                    conv12_out_scales, conv12_out_zps, conv12_1_weight_scales, conv12_1_weight_zps)
    out_conv12_1 = requantize(out_conv12_1, conv12_out_scales, conv12_1_weight_scales, conv12_1_output_scales, conv12_1_output_zps)

    # BRANCH 1.2
    # Process weights, biases, and quantization parameters for the second branch.
    conv12_2_weights = process_weights(weights_for_this_block, dict_layer_names["conv12_2"]["weight"])
    conv12_2_biases = process_biases(weights_for_this_block, dict_layer_names["conv12_2"]["bias"])
    conv12_2_weight_scales, conv12_2_weight_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["conv12_2"]["weight"])
    conv12_2_output_scales, conv12_2_output_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["conv12_2"]["output_scale"])

    # Apply convolution and requantize for branch 1.2.
    out_conv12_2 = quantized_conv1d(requantize_conv12, conv12_2_weights, conv12_2_biases,
                                    conv12_out_scales, conv12_out_zps, conv12_2_weight_scales, conv12_2_weight_zps)
    out_conv12_2 = requantize(out_conv12_2, conv12_out_scales, conv12_2_weight_scales, conv12_2_output_scales, conv12_2_output_zps)

    # BRANCH 1.3
    # Process weights, biases, and quantization parameters for the third branch.
    conv12_3_weights = process_weights(weights_for_this_block, dict_layer_names["conv12_3"]["weight"])
    conv12_3_biases = process_biases(weights_for_this_block, dict_layer_names["conv12_3"]["bias"])
    conv12_3_weight_scales, conv12_3_weight_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["conv12_3"]["weight"])
    conv12_3_output_scales, conv12_3_output_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["conv12_3"]["output_scale"])

    # Apply convolution and requantize for branch 1.3.
    out_conv12_3 = quantized_conv1d(requantize_conv12, conv12_3_weights, conv12_3_biases,
                                    conv12_out_scales, conv12_out_zps, conv12_3_weight_scales, conv12_3_weight_zps)
    out_conv12_3 = requantize(out_conv12_3, conv12_out_scales, conv12_3_weight_scales, conv12_3_output_scales, conv12_3_output_zps)

    # BRANCH 1.4
    # Apply max pooling followed by convolution for the fourth branch.
    # Insight: This branch uses max pooling before convolution, likely to reduce 
    # spatial dimensions and computational cost, a common strategy in inception modules.
    out_conv12_4 = max_pooling1d(out_conv11, pool_size=3, stride=1, padding="same")

    # Process weights, biases, and quantization parameters for the fourth branch.
    conv12_4_weights = process_weights(weights_for_this_block, dict_layer_names["conv12_4"]["weight"])
    conv12_4_biases = process_biases(weights_for_this_block, dict_layer_names["conv12_4"]["bias"])
    conv12_4_scales, conv12_4_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["conv12_4"]["weight"])
    conv12_4_input_scales, conv12_4_input_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["maxpool2"])
    concat_1_output_scales, concat_1_output_zps = get_scale_and_zero_points(weights_for_this_block,  dict_layer_names["conv12_4"]["output_scale"])

    # Apply convolution and requantize for branch 1.4.
    out_conv12_4 = quantized_conv1d(out_conv12_4, conv12_4_weights, conv12_4_biases,
                                    conv12_4_input_scales, conv12_4_input_zps, conv12_4_scales, conv12_4_zps)
    out_conv12_4 = requantize(out_conv12_4, conv12_4_input_scales, conv12_4_scales, concat_1_output_scales, concat_1_output_zps)

    # Concatenate outputs from all four branches along the channel axis.
    # Insight: Concatenation combines features from different branches, increasing 
    # the channel dimension and enabling the model to leverage diverse feature representations.
    out_conv12_1234 = np.concatenate([out_conv12_4, out_conv12_1, out_conv12_2, out_conv12_3], axis=-1)

    # Retrieve quantization parameters for the final ReLU and apply requantization.
    # Insight: The final ReLU with a dummy weight scale of [1.0] suggests a pass-through 
    # scale for the concatenated output, ensuring the ReLU operates correctly in the 
    # quantized domain before downstream layers.
    relu1_scales, relu1_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["relu2"])

    out_conv12_1234 = requantize_with_relu(out_conv12_1234, concat_1_output_scales, concat_1_output_zps, [1.0], relu1_scales, relu1_zps)

    conv13_weights = process_weights(weights_for_this_block, dict_layer_names["conv13"]["weight"])
    conv13_biases = process_biases(weights_for_this_block, dict_layer_names["conv13"]["bias"])
    conv13_input_scales, conv13_input_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["relu2"])
    conv13_weight_scales, conv13_weight_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["conv13"]["weight"])
    
    out_conv13 = quantized_conv1d(out_conv12_1234, conv13_weights, conv13_biases,
                                  conv13_input_scales, conv13_input_zps,
                                  conv13_weight_scales, conv13_weight_zps)
    
    conv13_out_scales, conv13_out_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["conv13"]["output_scale"])
    requantize_conv13 = requantize(out_conv13, conv13_input_scales, conv13_weight_scales, conv13_out_scales, conv13_out_zps)

    # BRANCH 2.1
    # Process weights, biases, and quantization parameters for the first branch.
    conv13_1_weights = process_weights(weights_for_this_block, dict_layer_names["conv13_1"]["weight"])
    conv13_1_biases = process_biases(weights_for_this_block, dict_layer_names["conv13_1"]["bias"])
    conv13_1_weight_scales, conv13_1_weight_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["conv13_1"]["weight"])
    conv13_1_output_scales, conv13_1_output_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["conv13_1"]["output_scale"])

    out_conv13_1 = quantized_conv1d(requantize_conv13, conv13_1_weights, conv13_1_biases,
                                    conv13_out_scales, conv13_out_zps, conv13_1_weight_scales, conv13_1_weight_zps)
    out_conv13_1 = requantize(out_conv13_1, conv13_out_scales, conv13_1_weight_scales, conv13_1_output_scales, conv13_1_output_zps)

    # BRANCH 2.2
    # Process weights, biases, and quantization parameters for the second branch.
    conv13_2_weights = process_weights(weights_for_this_block, dict_layer_names["conv13_2"]["weight"])
    conv13_2_biases = process_biases(weights_for_this_block, dict_layer_names["conv13_2"]["bias"])
    conv13_2_weight_scales, conv13_2_weight_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["conv13_2"]["weight"])
    conv13_2_output_scales, conv13_2_output_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["conv13_2"]["output_scale"])

    out_conv13_2 = quantized_conv1d(requantize_conv13, conv13_2_weights, conv13_2_biases,
                                    conv13_out_scales, conv13_out_zps, conv13_2_weight_scales, conv13_2_weight_zps)
    out_conv13_2 = requantize(out_conv13_2, conv13_out_scales, conv13_2_weight_scales, conv13_2_output_scales, conv13_2_output_zps)

    # BRANCH 2.3
    # Process weights, biases, and quantization parameters for the third branch.
    conv13_3_weights = process_weights(weights_for_this_block, dict_layer_names["conv13_3"]["weight"])
    conv13_3_biases = process_biases(weights_for_this_block, dict_layer_names["conv13_3"]["bias"])
    conv13_3_weight_scales, conv13_3_weight_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["conv13_3"]["weight"])
    conv13_3_output_scales, conv13_3_output_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["conv13_3"]["output_scale"])

    out_conv13_3 = quantized_conv1d(requantize_conv13, conv13_3_weights, conv13_3_biases,
                                    conv13_out_scales, conv13_out_zps, conv13_3_weight_scales, conv13_3_weight_zps)
    out_conv13_3 = requantize(out_conv13_3, conv13_out_scales, conv13_3_weight_scales, conv13_3_output_scales, conv13_3_output_zps)
    
    # BRANCH 2.4
    out_conv13_4 = max_pooling1d(out_conv12_1234, pool_size=3, stride=1, padding="same")

    # Process weights, biases, and quantization parameters for the fourth branch.
    conv13_4_weights = process_weights(weights_for_this_block, dict_layer_names["conv13_4"]["weight"])
    conv13_4_biases = process_biases(weights_for_this_block, dict_layer_names["conv13_4"]["bias"])
    conv13_4_scales, conv13_4_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["conv13_4"]["weight"])
    conv13_4_input_scales, conv13_4_input_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["maxpool3"])
    concat_2_output_scales, concat_2_output_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["conv13_4"]["output_scale"])

    # Apply convolution and requantize for branch 2.4.
    out_conv13_4 = quantized_conv1d(out_conv13_4, conv13_4_weights, conv13_4_biases,
                                    conv13_4_input_scales, conv13_4_input_zps, conv13_4_scales, conv13_4_zps)
    out_conv13_4 = requantize(out_conv13_4, conv13_4_input_scales, conv13_4_scales, concat_2_output_scales, concat_2_output_zps)

    out_conv13_1234 = np.concatenate([out_conv13_4, out_conv13_1, out_conv13_2, out_conv13_3], axis=-1)

    relu2_scales, relu2_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["relu3"])

    out_conv13_1234 = requantize_with_relu(out_conv13_1234, concat_2_output_scales, concat_2_output_zps, [1.0], relu2_scales, relu2_zps)
    
    skip_connection1_output_scales, skip_connection1_output_zps = get_scale_and_zero_points(weights_for_this_block, dict_layer_names["add1"])    
    skip_connection1 = quantized_add(out_conv11, conv12_input_scales, conv12_input_zps, out_conv13_1234, relu2_scales, relu2_zps, skip_connection1_output_scales, skip_connection1_output_zps)
    return skip_connection1, skip_connection1_output_scales, skip_connection1_output_zps

def run_quantized_model_on_single_input(x_int8, pretrained_int_weights, input_scale, input_zp):
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
    x_int8, output_scale1, output_zero1 = quantized_inception_block(x_int8, input_scale, input_zp, pretrained_int_weights, dict_all_layer_names["first_block"])
    x_int8, output_scale2, output_zero2 = quantized_inception_block(x_int8, output_scale1, output_zero1, pretrained_int_weights, dict_all_layer_names["second_block"])
    x_int8, output_scale3, output_zero3 = quantized_inception_block(x_int8, output_scale2, output_zero2, pretrained_int_weights, dict_all_layer_names["third_block"])
    
    # Global averaing pooling
    global_pooling_scales, global_pooling_zps = get_scale_and_zero_points(pretrained_int_weights, "model/quant_global_average_pooling2d/Mean")
    x_int8 = quantized_global_avg_pool1d(x_int8, output_scale3, output_zero3, global_pooling_scales, global_pooling_zps)

    # Dense layer
    dense_weights = process_dense_weights(pretrained_int_weights, "model/quant_dense/MatMul;model/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars")
    dense_biases = process_dense_biases(pretrained_int_weights, "model/quant_dense/LastValueQuant_1/FakeQuantWithMinMaxVars")

    dense_input_scales, dense_input_zps = global_pooling_scales, global_pooling_zps
    dense_weight_scales, dense_weight_zps = get_scale_and_zero_points(pretrained_int_weights, "model/quant_dense/MatMul;model/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars")
    dense_output_scales, dense_output_zps = get_scale_and_zero_points(pretrained_int_weights, "output")

    logits = quantized_dense(np.squeeze(x_int8, axis=1), dense_weights, dense_biases,
                    dense_input_scales, dense_input_zps,
                    dense_weight_scales, dense_weight_zps,
                    dense_output_scales[0], dense_output_zps[0])

    return logits

def main():
    # Load dataset
    # Load pretrained quantized weights and quantization parameters from JSON.
    # Insight: JSON is used for portability and human-readability, but parsing large 
    # JSON files can be a bottleneck in production systems; binary formats like 
    # TensorFlow's TFLite might be preferred for efficiency.
    pretrained_int_weights = json.load(open("model_weights_scales.json", "r"))

    # Extract input scale and zero point for quantization.
    # Insight: The use of str_to_float_list suggests quantization parameters are 
    # stored as strings in JSON, requiring parsing. This could introduce errors if 
    # the string format is inconsistent.
    input_scale, input_zp = np.array(str_to_float_list(pretrained_int_weights["input"]["scale"])), np.array(str_to_float_list(pretrained_int_weights["input"]["zp"]))    
    output_scale, output_zp = np.array(str_to_float_list(pretrained_int_weights["output"]["scale"])), np.array(str_to_float_list(pretrained_int_weights["output"]["zp"]))

    all_preds = []
    all_labels = []

    test_dataset, test_gts = np.load("test_data.npy"), np.load("test_gts.npy")
    progress = tqdm(total=None)  # Unknown length

    N = 33
    for i, (data, gt) in enumerate(zip(test_dataset, test_gts)):
        x_int8 = quantize(data, input_scale, input_zp)

        output = run_quantized_model_on_single_input(x_int8, pretrained_int_weights, input_scale, input_zp)

        # Dequantize
        output = output_scale * (output.astype(np.float32) - output_zp)

        pred = np.argmax(output, axis=-1)
        all_preds.append(pred[0])
        all_labels.append(gt)

        progress.update(1)  # Manually advance progress bar
        if i > N:
            break

    progress.close()

    print(len(all_labels))
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    print("Accuracy:", acc)
    print("F1:", f1)
    print("Precision:", precision)
    print("Recall:", recall)

    print(all_preds)
    print(all_labels)

main()
