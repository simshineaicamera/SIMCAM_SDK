# Copyright 2017 Intel Corporation.
# The source code, information and material ("Material") contained herein is
# owned by Intel Corporation or its suppliers or licensors, and title to such
# Material remains with Intel Corporation or its suppliers or licensors.
# The Material contains proprietary information of Intel or its suppliers and
# licensors. The Material is protected by worldwide copyright laws and treaty
# provisions.
# No part of the Material may be used, copied, reproduced, modified, published,
# uploaded, posted, transmitted, distributed or disclosed in any way without
# Intel's prior express written permission. No license under any patent,
# copyright or other intellectual property rights in the Material is granted to
# or conferred upon you, either expressly, by implication, inducement, estoppel
# or otherwise.
# Any license under such intellectual property rights must be express and
# approved by Intel in writing.

import numpy as np
from Models.NetworkStage import *
from Models.Network import *
from Controllers.CaffeEnumController import *
from Controllers.MiscIO import *
from google.protobuf import message
from google.protobuf import text_format
import os
import ctypes

try:
    os.environ['GLOG_minloglevel'] = '2'  # Supress Caffe Output
    import caffe
except ImportError:
    print("Error importing caffe")
    quit()

try:
    from caffe.proto import caffe_pb2
except ImportError:
    print("Error importing caffe module caffe_pb2")
    quit()

concat_tracker = []
slice_tracker = []
data_type = np.float16


def caffe_search_pre_op(msg, name):
    if False:
        pass
    return None


def get_caffe_kernel_size(layer):
    if isConvolution(layer.type):
        if layer.convolution_param.kernel_w:
            return layer.convolution_param.kernel_h, layer.convolution_param.kernel_w
        else:
            return layer.convolution_param.kernel_size[0], layer.convolution_param.kernel_size[0]
    if isPooling(layer.type):
        if layer.pooling_param.kernel_w:
            return layer.pooling_param.kernel_h, layer.pooling_param.kernel_w
        else:
            return layer.pooling_param.kernel_size, layer.pooling_param.kernel_size

    if isDeconvolution(layer.type):
        if layer.convolution_param.kernel_w:
            return layer.convolution_param.kernel_h, layer.convolution_param.kernel_w
        else:
            return layer.convolution_param.kernel_size[0], layer.convolution_param.kernel_size[0]

    return 0, 0  # Default


def get_caffe_group(layer):
    if isConvolution(layer.type):
        return layer.convolution_param.group

    if isDeconvolution(layer.type):
        return layer.convolution_param.group

    return 1


def get_caffe_output_channels(layer, prev_output_shape, top, network):
    if isConvolution(layer.type):
        return layer.convolution_param.num_output
    if isFCL(layer.type):
        return layer.inner_product_param.num_output
    if isReshape(layer.type):
        return layer.reshape_param.shape[0]
    if (isPooling(layer.type) or isSoftMax(layer.type) or isLRN(layer.type) or
            isSigmoid(layer.type) or isTanH(layer.type) or isPower(layer.type) or
            isNormalize(layer.type)):
        if top is not None:
            if (len(top) > 1):
                sum_of_k_from_parents = 0
                for parent in top:
                    prev_node = network.search(parent)
                    sum_of_k_from_parents += prev_node.output.shape[0]
                return sum_of_k_from_parents

        return prev_output_shape[0]
    if (isEltwise(layer.type) or isBatchNorm(layer.type) or
            isScale(layer.type) or isPReLU(layer.type)):
        return prev_output_shape[0]

    if isDeconvolution(layer.type):
        return layer.convolution_param.num_output

    if isFlatten(layer.type):
        return prev_output_shape[0] * \
            prev_output_shape[1] * prev_output_shape[2]

    if isPermute(layer.type):
        return prev_output_shape[layer.permute_param.order[1]-1]

    if isPriorBox(layer.type):
        return 1

    # print("Parse Warning: Default OutputChannels being used.")
    return 1  # default case


def get_caffe_op_radix(layer):
    """
    Get the radix of the operation from this layer (single kernel dimensions).
    Currently assumes square matrices due to no example vector kernels in caffe.
    TODO: Find some examples
    :param layer:
    :return:
    """
    if isConvolution(layer.type):
        if layer.convolution_param.kernel_w:
            return layer.convolution_param.kernel_h, layer.convolution_param.kernel_w
        else:
            return layer.convolution_param.kernel_size[0], layer.convolution_param.kernel_size[0]
    elif isLRN(layer.type):
        return 0, layer.lrn_param.local_size
    elif isPooling(layer.type):
        if layer.pooling_param.kernel_size == 0:
            if(layer.pooling_param.global_pooling):
                return -1, -1
        if layer.pooling_param.kernel_w:
            return layer.pooling_param.kernel_h, layer.pooling_param.kernel_w
        else:
            return layer.pooling_param.kernel_size, layer.pooling_param.kernel_size
    elif isPooling(layer.type):

        return layer.pooling_param.kernel_size, layer.pooling_param.kernel_size

    elif isDeconvolution(layer.type):
        if layer.convolution_param.kernel_w:
            return layer.convolution_param.kernel_h, layer.convolution_param.kernel_w
        else:
            return layer.convolution_param.kernel_size[0], layer.convolution_param.kernel_size[0]
    else:
        return 1, 1


def get_caffe_op_padding(layer):
    if isConvolution(layer.type) or isDeconvolution(layer.type):
        if layer.convolution_param.pad_w or layer.convolution_param.pad_h:
            return layer.convolution_param.pad_h, layer.convolution_param.pad_w
        elif layer.convolution_param.pad:
            return layer.convolution_param.pad[0], layer.convolution_param.pad[0]
    if isPooling(layer.type):
        if layer.pooling_param.pad_w or layer.pooling_param.pad_h:
            return layer.pooling_param.pad_h, layer.pooling_param.pad_w
        elif layer.pooling_param.pad:
            return layer.pooling_param.pad, layer.pooling_param.pad

    return 0, 0  # Default Case


def get_caffe_op_stride(layer):
    """
    Gets a layer's stride for the operation. Looks like (for now) there's only one stride dimension supported in caffe?
    :param layer:
    :return: a tuple for stride dimensions X,Y
    """
    if isConvolution(layer.type):
        if layer.convolution_param.stride_w:
            return layer.convolution_param.stride_h, layer.convolution_param.stride_w
        elif layer.convolution_param.stride:
            return layer.convolution_param.stride[0], layer.convolution_param.stride[0]
    if isPooling(layer.type):
        if layer.pooling_param.stride_w:
            return layer.pooling_param.stride_h, layer.pooling_param.stride_w
        elif layer.pooling_param.stride:
            return layer.pooling_param.stride, layer.pooling_param.stride
    if isDeconvolution(layer.type):
        if layer.convolution_param.stride_w:
            return layer.convolution_param.stride_h, layer.convolution_param.stride_w
        elif layer.convolution_param.stride:
            return layer.convolution_param.stride[0], layer.convolution_param.stride[0]

    return 1, 1  # Default Case


def get_caffe_params(layer, blobs):
    global data_type

    if isLRN(layer.type):
        # The latest 0 is just for alignment, it can be removed
        # after UsbLink has been fixed
        return None, np.array(
            [layer.lrn_param.k, layer.lrn_param.alpha, layer.lrn_param.beta, 0], dtype=data_type)
    elif isConvolution(layer.type) or isDeconvolution(layer.type):
        if layer.convolution_param.bias_term:
            return blobs[layer.name][0].data.astype(
                dtype=data_type), blobs[layer.name][1].data.astype(dtype=data_type)
        else:
            return blobs[layer.name][0].data.astype(dtype=data_type), None
    elif isFCL(layer.type):
        if layer.inner_product_param.bias_term:
            return blobs[layer.name][0].data.astype(
                dtype=data_type), blobs[layer.name][1].data.astype(dtype=data_type)
        else:
            return blobs[layer.name][0].data.astype(dtype=data_type), None
    elif isBatchNorm(layer.type):
        if blobs[layer.name][2].data[0] == 0:
            mean = np.zeros(blobs[layer.name][0].data.shape)
            var = np.zeros(blobs[layer.name][1].data.shape) + \
                layer.batch_norm_param.eps
        else:
            mean = blobs[layer.name][0].data * \
                (1 / blobs[layer.name][2].data[0])
            var = blobs[layer.name][1].data * \
                (1 / blobs[layer.name][2].data[0]) + layer.batch_norm_param.eps
        mult = np.reciprocal(np.sqrt(var))
        bias = -mean * mult
        return mult.astype(dtype=data_type), bias.astype(dtype=data_type)
    elif isScale(layer.type):
        if layer.scale_param.bias_term:
            return blobs[layer.name][0].data.astype(
                dtype=data_type), blobs[layer.name][1].data.astype(dtype=data_type)
        else:
            return blobs[layer.name][0].data.astype(dtype=data_type), None
    elif isPReLU(layer.type):
        return None, blobs[layer.name][0].data
    elif isNormalize(layer.type):
        return blobs[layer.name][0].data.astype(dtype=data_type)
    else:
        return None, None


def caffe_apply_minor_op(network, layer, top):
    """
    Searches through the network for the applicable node that we want to attach a minor op to and attaches it.
    :param network:
    :param layer:
    :return:
    """
    if isReLU(layer.type) or isELU(layer.type):
        if top is not None and not isinstance(top[0], str):
            top = top[0]
        for prevlayer in top:
            if isReLU(layer.type):
                applicable_node = network.search(prevlayer)
                if layer.relu_param.negative_slope == 0.0:
                    applicable_node.postOp = StageType.relu
                else:
                    applicable_node.postOp = StageType.leaky_relu
                applicable_node.post_param1 = layer.relu_param.negative_slope
            if isELU(layer.type):
                applicable_node = network.search(prevlayer)
                applicable_node.postOp = StageType.elu
                applicable_node.post_param1 = layer.elu_param.alpha
            if len(top) == 1:
                # This should be always here, but for now we don't support this
                # when applying ReLU to a concat
                applicable_node.unprocessed_name = layer.top[0]
                applicable_node.name = set_string_range(
                    layer.top[0], 100).encode('ascii')
    elif isConcat(layer.type):
        global concat_tracker
        concat_tracker.append((layer.top[0], layer.bottom))
    elif isSlice(layer.type):
        global slice_tracker
        slice_tracker.append(
            (layer.top, layer.bottom, layer.slice_param.slice_point))
    else:
        throw_error(ErrorTable.StageTypeNotSupported, layer.type)

def create_input_layer(myriad_conf, arguments, network, input_shape, input_name):
    node = NetworkStage(input_name,
                      None,  # top
                      StorageOrder.orderZYX,
                      0,
                      0,
                      PadStyle.caffe,
                      DataType.fp16,
                      DataType.fp16,
                      StageType.none,
                      # Radix and stride
                      1,
                      1,
                      1,
                      1,
                      # X, Y, Z
                      input_shape[3],
                      input_shape[2],
                      input_shape[1],
                      # fh, fw
                      1,
                      1,
                      # Output Channels (K)
                      input_shape[1],
                      # Taps, Bias,
                      None,
                      TapsOrder.orderKCHW,
                      None,
                      # Pre and post ops
                      None,
                      StageType.none,
                      None,
                      0,
                      0,
                      None,
                      myriad_conf,
                      arguments,
                      new_x=0,
                      new_y=0,
                      new_c=0)
    network.attach(node)
    return node   

def parse_caffe(arguments, myriad_conf, debug=False, file_gen=False):
    path = arguments.net_description
    weights = arguments.net_weights
    input_image = arguments.image
    outputNodeName = arguments.output_node_name
    inputNodeName = arguments.input_node_name
    raw_scale = arguments.raw_scale
    filename = arguments.outputs_name
    mean = arguments.mean
    channel_swap = arguments.channel_swap

    caffe.set_mode_cpu()
    description = path
    if weights is None:
        open("zero_weights.caffemodel", "wb").close()
        weights = "zero_weights.caffemodel"
        print("\033[91m****** WARNING: using empty weights ******\033[0m")
    if not os.path.isfile(weights):
        throw_error(ErrorTable.ArgumentErrorWeights)
    try:
        net = caffe.Net(description, weights, caffe.TEST)
    except MemoryError:
        throw_error(ErrorTable.CaffeMemoryError)
    try:
        f = open(description)
        file_contents = f.read()
        f.close()
    except BaseException:
        throw_error(ErrorTable.ArgumentErrorDescription)
    msg = caffe_pb2.NetParameter()  # Parse via Caffe's NetParameter
    text_format.Merge(str(file_contents), msg)

    # If out inputNodeName is after a split, we have to start one step before in order
    # to run the split and fill all the inputs for all the paths before concat
    layers = msg.layer
    if len(layers) == 0:
        layers = msg.layers
    startNodeName = inputNodeName

    if layers[0].type == 'Input':
        try:
            input_shape = layers[0].input_param.shape[0].dim
            input_bottom = layers[0].top[0]  # Input name, normally "data"
        except BaseException:
            throw_error(ErrorTable.InputSyntaxNotSupported)
    else:
        try:
            input_shape = msg.input_shape[0].dim
            input_bottom = layers[0].bottom[0]  # Input name, normally "data"
        except BaseException:
            throw_error(ErrorTable.InputSyntaxNotSupported)

    if(input_shape[0] != 1):
        throw_error(ErrorTable.AttemptedBatchMode)

    if inputNodeName:
        input_bottom = net.bottom_names[inputNodeName][0]
        for i, layername in enumerate(net._layer_names):
            if input_bottom in net.top_names[layername]:
                if net.layers[i].type == 'Split':
                    input_bottom = net.bottom_names[layername][0]
                    startNodeName = layername
        input_shape = [
            net.blobs[input_bottom].shape[0],
            net.blobs[input_bottom].shape[1],
            net.blobs[input_bottom].shape[2],
            net.blobs[input_bottom].shape[3]]
    # Network
    if input_image is None or input_image == "Debug":
        try:
            input_data = np.ones(input_shape).astype(data_type)
        except BaseException:
            throw_error(ErrorTable.InputSyntaxNotSupported)
        input_data = np.random.uniform(-1, 1,
                                       input_shape).astype(dtype=data_type)

    else:
        input_data = parse_img(
            input_image,
            input_shape,
            raw_scale=raw_scale,
            mean=mean,
            channel_swap=channel_swap)
    if outputNodeName is None:
        outputNodeName = net.outputs[0]

    network = Network(msg.name, input_data)
    arguments.network = network

    prev_output_shape = [input_data[0].shape]
    global concat_tracker
    last_layer = None
    first_layer = None
    nlayers = len(layers)
    
    # Create input layer 
    create_input_layer(myriad_conf, arguments, network, input_shape, input_bottom)
    inshape = input_shape
    prev_output_shape = [input_data[0].shape]
    top = None

    # Check if last layer's "name" & "top" have same name
    for idx, layer in enumerate(layers):
        if(net.outputs[0] in layer.top):
            if(layer.name != layer.top[0]):
                throw_warning(ErrorTable.OutputNodeNameTopMismatch,
                              (layer.name, layer.top[0]))

    for idx, layer in enumerate(layers):
        # debug = True
        if debug:
            print("------------")
            print(layer)
        if layer.type == 'Input':
            continue

        if inputNodeName:
            if inputNodeName == layer.name:
                first_layer = layer
            elif first_layer is None:
                continue

        if isEltwise(layer.type) and len(
                layer.bottom) == 2 and layer.bottom[0] == input_bottom:
            tmp = layer.bottom[0]
            layer.bottom[0] = layer.bottom[1]
            layer.bottom[1] = tmp

        # First Node's top has to be None, but also other layers that
        # have the same bottom of the first layer
        curslicing = []
        if layer.bottom[0] == input_bottom:
            top = None
            prev_output_shape = [input_data[0].shape]
        else:
            # Check if our bottoms come from Slice
            for slice in slice_tracker:
                for i in range(len(slice[0])):
                    for j in range(len(layer.bottom)):
                        if layer.bottom[j] == slice[0][i]:
                            # It's one of Slice outputs, take its input
                            # 4 dims, second is nplanes
                            inputplanes = net.blobs[slice[1][0]].shape[1]
                            start = 0 if i == 0 else slice[2][i - 1]
                            end = slice[2][i] if i < len(
                                slice[2]) else inputplanes
                            if slice[1][0] == input_bottom:
                                curslicing.append([None, start, end])
                            else:
                                curslicing.append([slice[1][0], start, end])
                            layer.bottom[j] = slice[1][0]
                            break

            # Convert layer.bottom, which is a protobuf object, into a list in
            # order to be editable
            top = []
            for obj in layer.bottom:
                top.append(obj)
            # Concat check
            if len(concat_tracker) != 0:
                # We track all previous concats, as we may have a many-to-many
                # connection
                for concat in concat_tracker:
                    for i in range(len(top)):
                        if concat[0] == top[i]:
                            # If the next layer will try to attach to the
                            # non-existant concat intermediary node.
                            top[i] = concat[1]
            if top[0] == input_bottom:
                top = None
                prev_output_shape = [input_data[0].shape]
            else:
                prev_output_shape = []
                nodes = network.search_several(top)
                if isConcat(layer.type):
                    for node_i, node in enumerate(nodes):
                        node.concat_axis = layer.concat_param.axis

                # nodes is a list, which can contain nodes or list of nodes, in
                # case of concat
                if len(nodes) == 0:
                    throw_error(ErrorTable.GraphConstructionFailure, top)

                for i, node in enumerate(nodes):
                    if node == 0:
                        throw_error(
                            ErrorTable.GraphConstructionFailure, top[i])
                    if hasattr(node, '__iter__'):
                        # node is a list of nodes, i.e. nodes that make a
                        # concat
                        shape = node[0].output.shape
                        for i in range(len(node)):
                            if i > 0:
                                if node[i].concat_axis == 1:
                                    if (shape[1] != node[i].output.shape[1] or
                                        shape[2] != node[i].output.shape[2]):
                                        throw_error(ErrorTable.StageDetailsNotSupported, layer.name)
                                    shape = (shape[0] + node[i].output.shape[0], shape[1], shape[2])
                                elif node[i].concat_axis == 2:
                                    if (shape[0] != node[i].output.shape[0] or
                                        shape[2] != node[i].output.shape[2]):
                                        throw_error(ErrorTable.StageDetailsNotSupported, layer.name)
                                    shape = (shape[0], shape[1]+ node[i].output.shape[1], shape[2])
                                else:
                                    throw_error(ErrorTable.StageDetailsNotSupported, layer.name)

                        prev_output_shape.append(shape)
                    else:
                        prev_output_shape.append(node.output.shape)
        inshape = prev_output_shape[0]
        # Only eltwise and concat supports multiple inputs now
        if isEltwise(layer.type) or isConcat(layer.type):
            for i in range(len(prev_output_shape)):
                if i > 0:
                    if isEltwise(layer.type) or (isConcat(layer.type) and layer.concat_param.axis == 1):
                        if (inshape[1] != prev_output_shape[i][1] or
                            inshape[2] != prev_output_shape[i][2]):
                                throw_error(ErrorTable.StageDetailsNotSupported, layer.name)
                        inshape = (max(inshape[0], prev_output_shape[i][0]), inshape[1], inshape[2])
                    else:
                        # We have a concat on axis != 1. Most probably axis 2.
                        if (inshape[0] != prev_output_shape[i][0] or
                            inshape[2] != prev_output_shape[i][2]):
                                throw_error(ErrorTable.StageDetailsNotSupported, layer.name)
                        inshape = (inshape[0], max(inshape[1], prev_output_shape[i][1]), inshape[2])

        if isDropout(layer.type):
            continue
        if isBatchNorm(layer.type) or isScale(layer.type):
            # Check if absorption is possible into a convolution
            node = network.search(layer.bottom[0])
            if node != 0 and (
                    node.op == StageType.convolution or node.op == StageType.depthwise_convolution):
                w, b = get_caffe_params(layer, net.params)
                # Transpose in order to be able to use dimension broadcasting
                node.taps = (node.taps.T * w).T
                if node.bias is not None:
                    if b is not None:
                        node.bias = node.bias * w + b
                    else:
                        node.bias = node.bias * w
                else:
                    if b is not None:
                        node.addBias(np.array(b).astype(np.float16))
                node.name = node.unprocessed_name + '/' + layer.name
                node.changeName(node.name)
                node.alias.append(node.unprocessed_name)
                node.alias.append(layer.name)
                if layer.name == outputNodeName:
                    break
                continue
        if isInnerLRN(layer):
            # Square the inputs
            network.attach(
                NetworkStage(layer.name + "_Square",
                             top,
                             StorageOrder.orderZYX,
                             0,
                             0,
                             PadStyle.caffe,
                             DataType.fp16,
                             DataType.fp16,
                             StageType.square,
                             # Radix and stride
                             1,
                             1,
                             1,
                             1,
                             # X, Y, Z
                             inshape[2],
                             inshape[1],
                             inshape[0],
                             # fh, fw
                             0,
                             0,
                             # Output Channels (K)
                             inshape[0],
                             # Taps, Bias
                             None,
                             TapsOrder.orderKCHW,
                             None,
                             # Pre and post ops
                             None,
                             StageType.none,
                             None,
                             0,
                             0,
                             None,
                             myriad_conf,
                             args=arguments)
            )
            # Average pooling of squares
            network.attach(
                NetworkStage(layer.name + "_AvgPool",
                             [layer.name + "_Square"],
                             StorageOrder.orderZYX,
                             # Padding
                             (layer.lrn_param.local_size - 1) // 2,
                             (layer.lrn_param.local_size - 1) // 2,
                             PadStyle.caffe,
                             DataType.fp16,
                             DataType.fp16,
                             StageType.average_pooling,
                             # Radix and stride
                             layer.lrn_param.local_size,
                             layer.lrn_param.local_size,
                             1,
                             1,
                             # X, Y, Z
                             inshape[2],
                             inshape[1],
                             inshape[0],
                             # fh, fw
                             layer.lrn_param.local_size,
                             layer.lrn_param.local_size,
                             # Output Channels (K)
                             inshape[0],
                             # Taps, Bias
                             None,
                             TapsOrder.orderKCHW,
                             None,
                             # Pre and post ops
                             None,
                             StageType.none,
                             None,
                             0,
                             0,
                             None,
                             myriad_conf,
                             args=arguments)
            )
            # (1 + alpha * prev) ^ -beta
            network.attach(
                NetworkStage(layer.name + "_InnerLRN",
                             [layer.name + "_AvgPool"],
                             StorageOrder.orderZYX,
                             0,
                             0,
                             PadStyle.caffe,
                             DataType.fp16,
                             DataType.fp16,
                             StageType.innerlrn,
                             # Radix and stride
                             1,
                             1,
                             1,
                             1,
                             # X, Y, Z
                             inshape[2],
                             inshape[1],
                             inshape[0],
                             # fh, fw
                             1,
                             1,
                             # Output Channels (K)
                             inshape[0],
                             # Taps
                             None,
                             TapsOrder.orderKCHW,
                             # Biases (lrn parameters here)
                             np.array([layer.lrn_param.k, layer.lrn_param.alpha,
                                       layer.lrn_param.beta, 0], dtype=data_type),
                             # Pre and post ops
                             None,
                             StageType.none,
                             None,
                             0,
                             0,
                             None,
                             myriad_conf,
                             args=arguments)
            )
            # Multiply input with previous stage output
            if top is None:
                top = [layer.name + "_InnerLRN", None]
            else:
                top = [top[0], layer.name + "_InnerLRN"]
            network.attach(
                NetworkStage(layer.name,
                             top,
                             StorageOrder.orderZYX,
                             0,
                             0,
                             PadStyle.caffe,
                             DataType.fp16,
                             DataType.fp16,
                             StageType.eltwise_prod,
                             # Radix and stride
                             1,
                             1,
                             1,
                             1,
                             # X, Y, Z
                             inshape[2],
                             inshape[1],
                             inshape[0],
                             # fh, fw
                             1,
                             1,
                             # Output Channels (K)
                             inshape[0],
                             # Taps, Bias,
                             None,
                             TapsOrder.orderKCHW,
                             None,
                             # Pre and post ops
                             None,
                             StageType.none,
                             None,
                             0,
                             0,
                             None,
                             myriad_conf,
                             args=arguments)
            )
            last_layer = layer
            if layer.name == outputNodeName:
                break
            continue

        if isReshape(layer.type):
            if(len(layer.reshape_param.shape.dim) == 3):
                new_shape_X = 1
                new_shape_Y = layer.reshape_param.shape.dim[2]
                new_shape_C = layer.reshape_param.shape.dim[1]
            else:
                new_shape_X = layer.reshape_param.shape.dim[3]
                new_shape_Y = layer.reshape_param.shape.dim[2]
                new_shape_C = layer.reshape_param.shape.dim[1]

            network.attach(
                NetworkStage(layer.name,
                             top,
                             StorageOrder.orderZYX,
                             0,
                             0,
                             PadStyle.caffe,
                             DataType.fp16,
                             DataType.fp16,
                             StageType.reshape,
                             # Radix and stride
                             1,
                             1,
                             1,
                             1,
                             # X, Y, Z
                             inshape[2],
                             inshape[1],
                             inshape[0],
                             # fh, fw
                             1,
                             1,
                             # Output Channels (K)
                             inshape[0],
                             # Taps, Bias,
                             None,
                             TapsOrder.orderKCHW,
                             None,
                             # Pre and post ops
                             None,
                             StageType.none,
                             None,
                             0,
                             0,
                             None,
                             myriad_conf,
                             arguments,
                             new_x = new_shape_X,
                             new_y = new_shape_Y,
                             new_c = new_shape_C)
            )
            last_layer = layer
            if layer.name == outputNodeName:
                break
            continue

        if isPriorBox(layer.type):
            params = np.array((prev_output_shape[1][1],     # img H
                               prev_output_shape[1][2],     # img W
                               len(layer.prior_box_param.min_size), 
                               len(layer.prior_box_param.max_size),
                               len(layer.prior_box_param.aspect_ratio),
                               len(layer.prior_box_param.variance), 
                               layer.prior_box_param.flip,
                               layer.prior_box_param.clip),
                               dtype=np.dtype("<f4"))
            params = np.append(params, layer.prior_box_param.min_size[0:])
            params = np.append(params, layer.prior_box_param.max_size[0:])
            params = np.append(params, layer.prior_box_param.aspect_ratio[0:])
            params = np.append(params, layer.prior_box_param.variance[0:])
            if (layer.prior_box_param.HasField("step_w") and
                    layer.prior_box_param.HasField("step_h")):
                # We don't check for both step and step_h/step_h being set
                # because caffe should yeld an error before this.
                params = np.append(params, layer.prior_box_param.step_w)
                params = np.append(params, layer.prior_box_param.step_h)
            elif (layer.prior_box_param.HasField("step")):
                params = np.append(params, layer.prior_box_param.step)
                params = np.append(params, layer.prior_box_param.step)
            else:
                params = np.append(params, 0)
                params = np.append(params, 0)
            params = np.append(params, layer.prior_box_param.offset)
            params = params.astype(dtype = np.dtype("<f4"))

            node = NetworkStage(layer.name, top, StorageOrder.orderZYX,
                    0, 0, PadStyle.none,
                    DataType.fp16, DataType.fp16,
                    get_caffe_op_type(layer),            # op_type
                    1, 1,                                # op_x, op_y, 
                    1, 1,                                # sx, sy,
                    inshape[2], inshape[1], inshape[0],  # X, Y, Z
                    0, 0,                                # fw, fh
                    get_caffe_output_channels(layer, inshape, top, network),
                    None, None, None, # taps, taps_order, bias,
                    None,             # Pre Op
                    StageType.none,   # Post Op
                    None,             # Post Op Param 1
                    0,                # Post Op StrideX
                    0,                # Post Op StrideX
                    myriad_config=myriad_conf, args=arguments,
                    opParams=params)

            network.attach(node)

            last_layer = layer
            if layer.name == outputNodeName:
                break
            continue

        if (isDetectionOutput(layer.type)):
            detection_param = layer.detection_output_param

            share_location = 1 if detection_param.share_location else 0

            det_out_dtype = np.dtype("<i4, <i4, <i4, <f4, <i4, <i4, <i4, <f4, <i4, <f4")

            op_params = np.array((detection_param.num_classes,
                share_location,
                detection_param.background_label_id,
                detection_param.nms_param.nms_threshold,
                detection_param.nms_param.top_k,
                detection_param.code_type,
                detection_param.keep_top_k,
                detection_param.confidence_threshold,
                detection_param.variance_encoded_in_target,
                detection_param.nms_param.eta), det_out_dtype)

            op_params = op_params.flatten()
            op_params = op_params.view("<f4")

            node = NetworkStage(layer.name, top, StorageOrder.orderZYX,
                    0, 0, PadStyle.none,
                    DataType.fp16, DataType.fp16,
                    get_caffe_op_type(layer),
                    1, 1,
                    1, 1,
                    inshape[2], inshape[1], inshape[0],  # X, Y, Z
                    0, 0,
                    get_caffe_output_channels(layer, inshape, top, network),
                    taps, TapsOrder.orderKCHW, None,
                    None,  # Pre Op
                    StageType.none,  # Post Op
                    None,  # Post Op Param 1
                    0,  # Post Op StrideX
                    0,  # Post Op StrideX
                    0, myriad_conf, args=arguments,
                    opParams=op_params, 
                    new_y=detection_param.keep_top_k)

            network.attach(node)

            last_layer = layer
            if layer.name == outputNodeName:
                break
            continue

        if (isPower(layer.type) or 
           isNormalize(layer.type) or 
           isPermute(layer.type)):
            taps = None
            (new_x, new_y, new_c) = (0, 0, 0)
            if isPower(layer.type):
                op_params = np.array((layer.power_param.shift,
                                   layer.power_param.scale,
                                   layer.power_param.power),
                                   dtype = np.dtype("<f4"))
            elif isNormalize(layer.type):
                op_params = np.array((layer.norm_param.across_spatial,
                                   layer.norm_param.channel_shared),
                                   dtype = np.dtype("int32"))
                taps = get_caffe_params(layer, net.params)
            elif isPermute(layer.type):
                caffe_perm_ord = np.array(layer.permute_param.order[1:], dtype = "i4")
                if(np.count_nonzero(caffe_perm_ord) != len(caffe_perm_ord)):
                    raise Exception("Permute on batch axis is not supported. \
                            Layer = {0}".format(layer.name))

                perm_ord = np.arange(3)
                # Caffe axis are NCHW(0,1,2,3). Myriad axis are CHW(0,1,2).
                # Hence substract 1.
                perm_ord[0 : len(caffe_perm_ord)] = caffe_perm_ord - 1
                new_c, new_y, new_x = np.array(inshape)[perm_ord]

                # Decode the caffe permute order to myriad permute order.
                ord_decoder = np.array([2, 0, 1], dtype = "i4")
                myriad_perm_ord = ord_decoder[np.roll(perm_ord, -1)]
                op_params = np.array(myriad_perm_ord, dtype = "i4")

            node = NetworkStage(layer.name, top, StorageOrder.orderZYX,
                    0, 0, PadStyle.none,
                    DataType.fp16, DataType.fp16,
                    get_caffe_op_type(layer),
                    1, 1,
                    1, 1,
                    inshape[2], inshape[1], inshape[0],  # X, Y, Z
                    0, 0,
                    get_caffe_output_channels(layer, inshape, top, network),
                    taps, TapsOrder.orderKCHW, None,
                    None,  # Pre Op
                    StageType.none,  # Post Op
                    None,  # Post Op Param 1
                    0,  # Post Op StrideX
                    0,  # Post Op StrideX
                    0, myriad_conf, args=arguments, 
                    new_x = new_x, new_y = new_y, new_c = new_c,
                    opParams = op_params)

            network.attach(node)

            last_layer = layer
            if layer.name == outputNodeName:
                break
            continue

        if isSoftMax(layer.type):
            softmax_param = np.array([(layer.softmax_param.axis)],
                    dtype=np.dtype("<i4"))
            if(softmax_param[0] not in (1, 2)):
                throw_error(ErrorTable.StageTypeNotSupported,
                        "Axis parameter value {0} for layer {1} of type {2}".format(
                            softmax_param[0], layer.name, layer.type))

            softmax_node = NetworkStage(layer.name,
                                      top,
                                      StorageOrder.orderZYX,
                                      0,
                                      0,
                                      PadStyle.none,
                                      DataType.fp16,
                                      DataType.fp16,
                                      get_caffe_op_type(layer),
                                      1,
                                      1,
                                      1,
                                      1,
                                      # X, Y, Z
                                      inshape[2],
                                      inshape[1],
                                      inshape[0],
                                      0,
                                      0,
                                      get_caffe_output_channels(
                                          layer, inshape, top, network),
                                      None,
                                      TapsOrder.orderKCHW,
                                      None,
                                      None,  # Pre Op
                                      StageType.none,  # Post Op
                                      None,  # Post Op Param 1
                                      0,  # Post Op StrideX
                                      0,  # Post Op StrideX
                                      0,
                                      myriad_conf,
                                      args=arguments,
                                      opParams = softmax_param)

            network.attach(softmax_node)

            last_layer = layer
            if layer.name == outputNodeName:
                break
            continue

        if isCrop(layer.type):
            # Caffe axis storage order N, C, H, W is assumed. Where W is the
            # fastest growing dimension.
            crop_axis = layer.crop_param.axis
            if crop_axis < 0:
                crop_axis += 4

            if crop_axis == 0:
                throw_error(ErrorTable.AttemptedBatchMode)

            crop_offset = np.array([0, 0, 0], np.dtype("<u4"))
            for offset_i in range(0, 3):
                if offset_i >= crop_axis - 1:
                    if len(layer.crop_param.offset) == 1:
                        crop_offset[offset_i] = layer.crop_param.offset[0]
                    else:
                        crop_offset[offset_i] = \
                            layer.crop_param.offset[offset_i - (crop_axis - 1)]

            # MvTensor crops a 3D volume with dimensions XYZ and storage
            # order S.
            # Toolkit/MvTensor axis storage order is assumed to be (N)HWC.
            # Where the fastes groing dimension is C. Hence X = H, Y = W,
            # Z = C and N = 1 always.
            # The offset structure has the order: offset_X, offset_Y,
            # offset_Z hence the parameters array has to be:
            # [offset_H, oofset_W, offset_Z]
            crop_offset = np.array([crop_offset[2], crop_offset[1],
                                    crop_offset[0]], dtype=np.dtype("<u4"))

            ref_bottom = network.search_several(layer.bottom[1])

            ref_bottom_dimX = ref_bottom.outputDimX
            ref_bottom_dimY = ref_bottom.outputDimY
            ref_bottom_dimZ = ref_bottom.outputDimZ

            # Using the new_x, new_y, new_c as in reshape to set the
            # output dimensions and pass the parameters to the crop
            # function.
            ref_dims = {
                0: (ref_bottom_dimX, ref_bottom_dimY, ref_bottom_dimZ),
                1: (ref_bottom_dimX, ref_bottom_dimY, inshape[0]),
                2: (ref_bottom_dimX, inshape[1], inshape[0])
            }

            # Call with crop_axis - 1 becaue in caffe first axis is
            # batch size.
            (new_x, new_y, new_c) = ref_dims.get(crop_axis - 1)

            crop_node = NetworkStage(layer.name,
                                     top, StorageOrder.orderZYX,
                                     0,
                                     0,
                                     PadStyle.none,
                                     DataType.fp16,
                                     DataType.fp16,
                                     get_caffe_op_type(layer),
                                     1,
                                     1,
                                     1,
                                     1,
                                     # X, Y, Z
                                     inshape[2],
                                     inshape[1],
                                     inshape[0],
                                     0,
                                     0,
                                     get_caffe_output_channels(
                                         layer, inshape, top, network),
                                     None,
                                     TapsOrder.orderKCHW,
                                     None,
                                     None,  # Pre Op
                                     StageType.none,  # Post Op
                                     None,  # Post Op Param 1
                                     0,  # Post Op StrideX
                                     0,  # Post Op StrideX
                                     0,
                                     myriad_conf,
                                     args=arguments,
                                     new_x=new_x,
                                     new_y=new_y,
                                     new_c=new_c,
                                     opParams=crop_offset)

            network.attach(crop_node)

            last_layer = layer
            if layer.name == outputNodeName:
                break
            continue

        if (isConcat(layer.type) or (isConvolution(layer.type) and get_caffe_kernel_size(layer)[0] > 1) or (
                isDeconvolution(layer.type) and get_caffe_kernel_size(layer)[0] > 1)) and len(curslicing) > 0:
            # Concat of slicing, cannot work, we have to add the slice layer
            # Convolution also does not support input strides
            # Convolution dilation is not supported for slicing.
            conv_dilation = 1
            layer_params = np.array([conv_dilation], dtype=np.dtype("<i4"))
            for slice in curslicing:
                for i in range(len(top)):
                    if top[i] == slice[0]:
                        slicename = layer.name + '_Slice' + \
                            str(slice[1]) + '_' + str(slice[2])
                        network.attach(
                            NetworkStage(slicename,
                                         [slice[0]],
                                         StorageOrder.orderZYX,
                                         0,
                                         0,
                                         PadStyle.caffe,
                                         DataType.fp16,
                                         DataType.fp16,
                                         StageType.copy,
                                         1,
                                         1,
                                         1,
                                         1,
                                         inshape[2],
                                         inshape[1],
                                         inshape[0],
                                         1,
                                         1,
                                         slice[2] - slice[1],
                                         None,
                                         TapsOrder.orderKCHW,
                                         None,
                                         None,
                                         StageType.none,
                                         None,
                                         0,
                                         0,
                                         curslicing,
                                         myriad_conf,
                                         args=arguments,
                                         opParams=layer_params)
                        )
                        top[i] = slicename

        if arguments.explicit_concat and isConcat(layer.type):
            outstride = 2 * sum(prev_output_shape[idx][0]
                                for idx in range(len(top)))
            for idx, prev in enumerate(top):
                if idx == 0:
                    substagename = layer.name
                else:
                    substagename = layer.name + '_' + \
                        ('input' if prev is None else prev)
                node = NetworkStage(
                    substagename,
                    top if idx == 0 else [
                        top[idx]],
                    StorageOrder.orderZYX,
                    0,
                    0,
                    PadStyle.caffe,
                    DataType.fp16,
                    DataType.fp16,
                    StageType.copy,
                    1,
                    1,
                    1,
                    1,
                    prev_output_shape[idx][2],
                    prev_output_shape[idx][1],
                    prev_output_shape[idx][0],
                    1,
                    1,
                    prev_output_shape[idx][0],
                    None,
                    TapsOrder.orderKCHW,
                    None,
                    None,
                    StageType.none,
                    None,
                    0,
                    0,
                    curslicing,
                    myriad_conf,
                    args=arguments)
                if idx == 0:
                    firstnode = node
                network.attach(node)
                if idx == 0:
                    if layer.name == outputNodeName:
                        outputPointer, outputIndex = node.setoutput(
                            outstride, 0, MemoryIndex.output.value)
                    else:
                        outputPointer, outputIndex = node.setoutput(outstride)
                else:
                    node.setoutput(outstride, outputPointer, outputIndex)
                outputPointer = outputPointer + 2 * prev_output_shape[idx][0]
            if layer.name == outputNodeName:
                firstnode.isoutput = True
                break
            continue

        if isDepthwiseConvolution(layer, get_caffe_output_channels(
                layer, inshape, top, network), inshape[0]):
            depthwise_node = NetworkStage(layer.name, top,
                                          StorageOrder.orderZYX,  # s_order,
                                          # pad_x, pad_y, pad_type,
                                          get_caffe_op_padding(layer)[0],
                                          get_caffe_op_padding(layer)[1],
                                          PadStyle.caffe,
                                          # dtype,  precision,
                                          DataType.fp16,
                                          DataType.fp16,
                                          # op_type
                                          StageType.depthwise_convolution,
                                          # op_x, op_y
                                          get_caffe_op_radix(layer)[0],
                                          get_caffe_op_radix(layer)[1],
                                          # sx, sy
                                          get_caffe_op_stride(layer)[0],
                                          get_caffe_op_stride(layer)[1],
                                          # x, y, c
                                          inshape[2],
                                          inshape[1],
                                          inshape[0],
                                          # fh, fw
                                          get_caffe_kernel_size(layer)[0],
                                          get_caffe_kernel_size(layer)[1],
                                          # Output Channels (K)
                                          get_caffe_output_channels(
                                              layer, inshape, top, network),
                                          # taps, taps_order
                                          get_caffe_params(
                                              layer, net.params)[0],
                                          TapsOrder.orderKCHW,
                                          # bias, pre_op_type, post_op_type,
                                          get_caffe_params(
                                              layer, net.params)[1],
                                          None,
                                          None,
                                          # post_1, post_sx, post_sy, slicing = None
                                          0,
                                          0,
                                          0,
                                          None,
                                          myriad_conf,
                                          args=arguments)
            network.attach(depthwise_node)
            last_layer = layer
            if layer.name == outputNodeName:
                break
            continue

        if (
            not isReLU(layer.type) and not
            isConcat(layer.type) and not
            isSlice(layer.type) and not
            isELU(layer.type) and not
            isDepthwiseConvolution(
                layer,
                get_caffe_output_channels(
                    layer,
                    inshape,
                    top,
                    network),
                inshape[0])):

            layer_params = None
            if(isConvolution(layer.type) or isDeconvolution(layer.type)):
                # Currently only equal dilation on all axes is supported.
                conv_dilation = 1
                if(len(layer.convolution_param.dilation) > 0):
                    conv_dilation = layer.convolution_param.dilation[0]

                layer_params = np.array([conv_dilation], dtype=np.dtype("<i4"))

            ngroups = get_caffe_group(layer)
            addednodes = []
            addednames = []
            for group in range(ngroups):
                taps = get_caffe_params(layer, net.params)[0]
                if(isDeconvolution(layer.type)):
                    # For Deconv the wheights are in CKHW format.
                    # Transform to KCWH
                    taps = np.swapaxes(taps, 0, 1)
                    # Taps need to be roated in the HW plane because caffe
                    # implements the deconvolution via convolution backward pass
                    # which does an 180deg rotation.
                    taps = np.rot90(taps, 2, (2, 3))
                bias = get_caffe_params(layer, net.params)[1]
                prev = top
                layername = layer.name
                if ngroups > 1:  # Warning: group convolution cannot follow slice
                    curslicing = []
                    curslicing.append([(top[0] if top is not None else None),
                                       inshape[0] // ngroups * group,
                                       inshape[0] // ngroups * (group + 1)])
                    if(isDeconvolution(layer.type)):
                        taps = taps[:, taps.shape[1] // ngroups *
                                    group: taps.shape[1] // ngroups * (group + 1), ]
                    else:
                        taps = taps[taps.shape[0] // ngroups *
                                    group: taps.shape[0] // ngroups * (group + 1), ]
                    if bias is not None:
                        bias = bias[bias.shape[0] // ngroups *
                                    group: bias.shape[0] // ngroups * (group + 1), ]
                    if get_caffe_kernel_size(layer)[0] > 1:
                        if top is None:
                            slicename = 'input'
                        else:
                            slicename = top[0] if isinstance(
                                top[0], str) else top[0][0]
                        slicename = slicename + '_s' + str(group)
                        network.attach(
                            NetworkStage(
                                slicename,
                                top,
                                StorageOrder.orderZYX,
                                0,
                                0,
                                PadStyle.caffe,
                                DataType.fp16,
                                DataType.fp16,
                                StageType.copy,
                                1,
                                1,
                                1,
                                1,
                                inshape[2],
                                inshape[1],
                                inshape[0],
                                1,
                                1,
                                inshape[0] // ngroups,
                                None,
                                TapsOrder.orderKCHW,
                                None,
                                None,
                                StageType.none,
                                None,
                                0,
                                0,
                                curslicing,
                                myriad_conf,
                                args=arguments,
                                opParams=layer_params))
                        prev = [slicename]
                    addednames.append(layer.name + '_p' + str(group))
                    layername = layer.name + '_p' + str(group)
                node = NetworkStage(
                    # Name, Top, Order
                    layername,
                    prev,
                    StorageOrder.orderZYX,
                    # Padding
                    get_caffe_op_padding(layer)[0],
                    get_caffe_op_padding(layer)[1],
                    PadStyle.caffe,
                    DataType.fp16, DataType.fp16,
                    # Op, StrideX, StrideY
                    get_caffe_op_type(layer),
                    get_caffe_op_radix(layer)[0],
                    get_caffe_op_radix(layer)[1],
                    get_caffe_op_stride(layer)[0],
                    get_caffe_op_stride(layer)[1],
                    # X, Y, Z
                    inshape[2],
                    inshape[1],
                    inshape[0],
                    # fh, fw
                    get_caffe_kernel_size(layer)[0],
                    get_caffe_kernel_size(layer)[1],
                    # Output Channels (K)
                    get_caffe_output_channels(
                        layer, inshape, top, network) // ngroups,
                    taps,
                    TapsOrder.orderKCHW,
                    bias,
                    None,  # Pre Op
                    StageType.none,  # Post Op
                    None,  # Post Op Param 1
                    0,  # Post Op StrideX
                    0,  # Post Op StrideX
                    curslicing,
                    myriad_conf,
                    args=arguments,
                    opParams=layer_params
                )

                network.attach(node)
                addednodes.append(node)
            if ngroups > 1:
                if idx == nlayers - 1:
                    NetworkStage.concat(addednodes)
                else:
                    concat_tracker.append((layer.name, addednames))

        else:
            caffe_apply_minor_op(network, layer, top)
        last_layer = layer
        if layer.name == outputNodeName:
            break

    if last_layer.type == 'Concat':
        nodes = network.search_several(last_layer.bottom)
        NetworkStage.concat(nodes)

    if(isDetectionOutput(last_layer.type)):
        network.outputIsSsdDetOut = True

    if outputNodeName is not None:
        if inputNodeName is not None:
            # Ensure we have the same inputs for each method
            net.blobs[input_bottom].data[...] = input_data
            try:
                net.forward(start=startNodeName, end=outputNodeName)
            except BaseException:
                throw_error(ErrorTable.NoOutputNode,
                            outputNodeName + "/" + startNodeName)
        else:
            # Ensure we have the same inputs for each method
            net.blobs['data'].data[...] = input_data
            try:
                net.forward(end=outputNodeName)
            except BaseException:
                throw_error(ErrorTable.NoOutputNode, outputNodeName)
    else:
        if inputNodeName is not None:
            # Ensure we have the same inputs for each method
            net.blobs[input_bottom].data[...] = input_data
            net.forward(start=startNodeName)
        else:
            # Ensure we have the same inputs for each method
            net.blobs['data'].data[...] = input_data
            net.forward()

    if file_gen:
        try:
            np.save(filename + "_expected.npy",
                    net.blobs[outputNodeName].data[0].astype(dtype=np.float16))

        except BaseException:
            throw_error(ErrorTable.NoOutputNode, extra=net.blobs.keys())

    caffe_output_shape = net.blobs[outputNodeName].data.shape
    output_shape       = np.ones(3, dtype = "i4")
    # Substract 1 because caffe output will (almost)always have the batch dimension.
    output_shape_len   = len(caffe_output_shape) - 1
    output_shape[0 : output_shape_len] = caffe_output_shape[1:]
    network.outputTensor = zyx_to_yxz_dimension_only(output_shape)

    return network
