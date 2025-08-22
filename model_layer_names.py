dict_all_layer_names = {
    "first_block":
    {
        "conv11": {
            "weight": "model/quant_conv2d/Conv2D",
            "bias": "model/quant_conv2d/BiasAdd/ReadVariableOp"
        },
        "relu1": "model/quant_re_lu/Relu;model/quant_conv2d/BiasAdd;model/quant_conv2d_29/Conv2D;model/quant_conv2d/Conv2D;model/quant_conv2d/BiasAdd/ReadVariableOp",
        "conv12": {
            "weight": "model/quant_conv2d_1/Conv2D;model/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_1/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_1/BiasAdd;model/quant_conv2d_7/Conv2D;model/quant_conv2d_1/Conv2D;model/quant_conv2d_1/LastValueQuant_1/FakeQuantWithMinMaxVars"
        },
        "conv12_1": {
            "weight": "model/quant_conv2d_3/Conv2D;model/quant_conv2d_3/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_3/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_3/BiasAdd;model/quant_conv2d_7/Conv2D;model/quant_conv2d_3/Conv2D;model/quant_conv2d_3/LastValueQuant_1/FakeQuantWithMinMaxVars",
        },
        "conv12_2": {
            "weight": "model/quant_conv2d_4/Conv2D;model/quant_conv2d_4/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_4/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_4/BiasAdd;model/quant_conv2d_7/Conv2D;model/quant_conv2d_4/Conv2D;model/quant_conv2d_4/LastValueQuant_1/FakeQuantWithMinMaxVars",
        },
        "conv12_3": {
            "weight": "model/quant_conv2d_5/Conv2D;model/quant_conv2d_5/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_5/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_5/BiasAdd;model/quant_conv2d_7/Conv2D;model/quant_conv2d_5/Conv2D;model/quant_conv2d_5/LastValueQuant_1/FakeQuantWithMinMaxVars",
        },
        "maxpool2": "model/quant_max_pooling2d/MovingAvgQuantize/FakeQuantWithMinMaxVars;model/quant_add_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;model/quant_max_pooling2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1",
        "conv12_4": {
            "weight": "model/quant_conv2d_2/Conv2D;model/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_2/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_2/BiasAdd;model/quant_conv2d_7/Conv2D;model/quant_conv2d_2/Conv2D;model/quant_conv2d_2/LastValueQuant_1/FakeQuantWithMinMaxVars",
        },
        "relu2": "model/quant_re_lu_1/Relu",
        "conv13": {
            "weight": "model/quant_conv2d_6/Conv2D;model/quant_conv2d_6/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_6/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_6/BiasAdd;model/quant_conv2d_7/Conv2D;model/quant_conv2d_6/Conv2D;model/quant_conv2d_6/LastValueQuant_1/FakeQuantWithMinMaxVars"
        },
        "conv13_1": {
            "weight": "model/quant_conv2d_8/Conv2D;model/quant_conv2d_8/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_8/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_8/BiasAdd;model/quant_conv2d_7/Conv2D;model/quant_conv2d_8/Conv2D;model/quant_conv2d_8/LastValueQuant_1/FakeQuantWithMinMaxVars",
        },
        "conv13_2": {
            "weight": "model/quant_conv2d_9/Conv2D;model/quant_conv2d_9/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_9/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_9/BiasAdd;model/quant_conv2d_7/Conv2D;model/quant_conv2d_9/Conv2D;model/quant_conv2d_9/LastValueQuant_1/FakeQuantWithMinMaxVars",
        },
        "conv13_3": {
            "weight": "model/quant_conv2d_10/Conv2D;model/quant_conv2d_10/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_10/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_10/BiasAdd;model/quant_conv2d_7/Conv2D;model/quant_conv2d_10/Conv2D;model/quant_conv2d_10/LastValueQuant_1/FakeQuantWithMinMaxVars",
        },
        "maxpool3": "model/quant_max_pooling2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars;model/quant_add_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;model/quant_max_pooling2d_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1",
        "conv13_4": {
            "weight": "model/quant_conv2d_7/Conv2D;model/quant_conv2d_7/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_7/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_7/BiasAdd;model/quant_conv2d_7/Conv2D;model/quant_conv2d_7/LastValueQuant_1/FakeQuantWithMinMaxVars",
        },
        "relu3": "model/quant_re_lu_2/Relu",
        "add1": "model/quant_add/add"
    },
    "second_block":
    {
        "conv11": {
            "weight": "model/quant_conv2d_11/Conv2D",
            "bias": "model/quant_conv2d_11/BiasAdd/ReadVariableOp"
        },
        "relu1": "model/quant_re_lu_3/Relu;model/quant_conv2d_11/BiasAdd;model/quant_conv2d_11/Conv2D;model/quant_conv2d_11/BiasAdd/ReadVariableOp",
        "conv12": {
            "weight": "model/quant_conv2d_12/Conv2D;model/quant_conv2d_12/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_12/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_12/BiasAdd;model/quant_conv2d_18/Conv2D;model/quant_conv2d_12/Conv2D;model/quant_conv2d_12/LastValueQuant_1/FakeQuantWithMinMaxVars"
        },
        "conv12_1": {
            "weight": "model/quant_conv2d_14/Conv2D;model/quant_conv2d_14/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_14/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_14/BiasAdd;model/quant_conv2d_18/Conv2D;model/quant_conv2d_14/Conv2D;model/quant_conv2d_14/LastValueQuant_1/FakeQuantWithMinMaxVars",
        },
        "conv12_2": {
            "weight": "model/quant_conv2d_15/Conv2D;model/quant_conv2d_15/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_15/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_15/BiasAdd;model/quant_conv2d_18/Conv2D;model/quant_conv2d_15/Conv2D;model/quant_conv2d_15/LastValueQuant_1/FakeQuantWithMinMaxVars",
        },
        "conv12_3": {
            "weight": "model/quant_conv2d_16/Conv2D;model/quant_conv2d_16/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_16/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_16/BiasAdd;model/quant_conv2d_18/Conv2D;model/quant_conv2d_16/Conv2D;model/quant_conv2d_16/LastValueQuant_1/FakeQuantWithMinMaxVars",
        },
        "maxpool2": "model/quant_max_pooling2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars;model/quant_add_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;model/quant_max_pooling2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1",
        "conv12_4": {
            "weight": "model/quant_conv2d_13/Conv2D;model/quant_conv2d_13/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_13/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_13/BiasAdd;model/quant_conv2d_18/Conv2D;model/quant_conv2d_13/Conv2D;model/quant_conv2d_13/LastValueQuant_1/FakeQuantWithMinMaxVars",
        },
        "relu2": "model/quant_re_lu_4/Relu",
        "conv13": {
            "weight": "model/quant_conv2d_17/Conv2D;model/quant_conv2d_17/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_17/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_17/BiasAdd;model/quant_conv2d_18/Conv2D;model/quant_conv2d_17/Conv2D;model/quant_conv2d_17/LastValueQuant_1/FakeQuantWithMinMaxVars"
        },
        "conv13_1": {
            "weight": "model/quant_conv2d_19/Conv2D;model/quant_conv2d_19/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_19/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_19/BiasAdd;model/quant_conv2d_18/Conv2D;model/quant_conv2d_19/Conv2D;model/quant_conv2d_19/LastValueQuant_1/FakeQuantWithMinMaxVars",
        },
        "conv13_2": {
            "weight": "model/quant_conv2d_20/Conv2D;model/quant_conv2d_20/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_20/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_20/BiasAdd;model/quant_conv2d_18/Conv2D;model/quant_conv2d_20/Conv2D;model/quant_conv2d_20/LastValueQuant_1/FakeQuantWithMinMaxVars",
        },
        "conv13_3": {
            "weight": "model/quant_conv2d_21/Conv2D;model/quant_conv2d_21/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_21/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_21/BiasAdd;model/quant_conv2d_18/Conv2D;model/quant_conv2d_21/Conv2D;model/quant_conv2d_21/LastValueQuant_1/FakeQuantWithMinMaxVars",
        },
        "maxpool3": "model/quant_max_pooling2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars;model/quant_add_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;model/quant_max_pooling2d_3/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1",
        "conv13_4": {
            "weight": "model/quant_conv2d_18/Conv2D;model/quant_conv2d_18/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_18/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_18/BiasAdd;model/quant_conv2d_18/Conv2D;model/quant_conv2d_18/LastValueQuant_1/FakeQuantWithMinMaxVars",
        },
        "relu3": "model/quant_re_lu_5/Relu",
        "add1": "model/quant_add_1/add"
    },
    "third_block":
    {
        "conv11": {
            "weight": "model/quant_conv2d_22/Conv2D",
            "bias": "model/quant_conv2d_22/BiasAdd/ReadVariableOp"
        },
        "relu1": "model/quant_re_lu_6/Relu;model/quant_conv2d_22/BiasAdd;model/quant_conv2d_22/Conv2D;model/quant_conv2d_22/BiasAdd/ReadVariableOp",
        "conv12": {
            "weight": "model/quant_conv2d_23/Conv2D;model/quant_conv2d_23/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_23/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_23/BiasAdd;model/quant_conv2d_29/Conv2D;model/quant_conv2d_23/Conv2D;model/quant_conv2d_23/LastValueQuant_1/FakeQuantWithMinMaxVars"
        },
        "conv12_1": {
            "weight": "model/quant_conv2d_25/Conv2D;model/quant_conv2d_25/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_25/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_25/BiasAdd;model/quant_conv2d_29/Conv2D;model/quant_conv2d_25/Conv2D;model/quant_conv2d_25/LastValueQuant_1/FakeQuantWithMinMaxVars",
        },
        "conv12_2": {
            "weight": "model/quant_conv2d_26/Conv2D;model/quant_conv2d_26/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_26/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_26/BiasAdd;model/quant_conv2d_29/Conv2D;model/quant_conv2d_26/Conv2D;model/quant_conv2d_26/LastValueQuant_1/FakeQuantWithMinMaxVars",
        },
        "conv12_3": {
            "weight": "model/quant_conv2d_27/Conv2D;model/quant_conv2d_27/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_27/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_27/BiasAdd;model/quant_conv2d_29/Conv2D;model/quant_conv2d_27/Conv2D;model/quant_conv2d_27/LastValueQuant_1/FakeQuantWithMinMaxVars",
        },
        "maxpool2": "model/quant_max_pooling2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars;model/quant_add_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;model/quant_max_pooling2d_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1",
        "conv12_4": {
            "weight": "model/quant_conv2d_24/Conv2D;model/quant_conv2d_24/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_24/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_24/BiasAdd;model/quant_conv2d_29/Conv2D;model/quant_conv2d_24/Conv2D;model/quant_conv2d_24/LastValueQuant_1/FakeQuantWithMinMaxVars",
        },
        "relu2": "model/quant_re_lu_7/Relu",
        "conv13": {
            "weight": "model/quant_conv2d_28/Conv2D;model/quant_conv2d_28/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_28/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_28/BiasAdd;model/quant_conv2d_29/Conv2D;model/quant_conv2d_28/Conv2D;model/quant_conv2d_28/LastValueQuant_1/FakeQuantWithMinMaxVars"
        },
        "conv13_1": {
            "weight": "model/quant_conv2d_30/Conv2D;model/quant_conv2d_30/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_30/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_30/BiasAdd;model/quant_conv2d_29/Conv2D;model/quant_conv2d_30/Conv2D;model/quant_conv2d_30/LastValueQuant_1/FakeQuantWithMinMaxVars",
        },
        "conv13_2": {
            "weight": "model/quant_conv2d_31/Conv2D;model/quant_conv2d_31/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_31/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_31/BiasAdd;model/quant_conv2d_29/Conv2D;model/quant_conv2d_31/Conv2D;model/quant_conv2d_31/LastValueQuant_1/FakeQuantWithMinMaxVars",
        },
        "conv13_3": {
            "weight": "model/quant_conv2d_32/Conv2D;model/quant_conv2d_32/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_32/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_32/BiasAdd;model/quant_conv2d_29/Conv2D;model/quant_conv2d_32/Conv2D;model/quant_conv2d_32/LastValueQuant_1/FakeQuantWithMinMaxVars",
        },
        "maxpool3": "model/quant_max_pooling2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars;model/quant_add_1/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;model/quant_max_pooling2d_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1",
        "conv13_4": {
            "weight": "model/quant_conv2d_29/Conv2D;model/quant_conv2d_29/LastValueQuant/FakeQuantWithMinMaxVars",
            "bias": "model/quant_conv2d_29/LastValueQuant_1/FakeQuantWithMinMaxVars",
            "output_scale": "model/quant_conv2d_29/BiasAdd;model/quant_conv2d_29/Conv2D;model/quant_conv2d_29/LastValueQuant_1/FakeQuantWithMinMaxVars",
        },
        "relu3": "model/quant_re_lu_8/Relu",
        "add1": "model/quant_add_2/add"
    }
}
