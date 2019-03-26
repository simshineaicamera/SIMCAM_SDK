
from Models.CaffeEnumDeclarations import *
from Controllers.EnumController import *

def isConvolution(layer):
    return layer in ["Convolution", CaffeStage.CONVOLUTION.value]


def isReLU(layer):
    return layer in ["ReLU", CaffeStage.RELU.value]


def isSigmoid(layer):
    return layer in ["Sigmoid"]


def isTanH(layer):
    return layer in ["TanH"]


def isPReLU(layer):
    return layer in ["PReLU"]


def isPooling(layer):
    return layer in ["Pooling", CaffeStage.POOLING.value]


def isSoftMax(layer):
    return layer in ["Softmax", CaffeStage.SOFTMAX.value]


def isFCL(layer):
    return layer in ["InnerProduct", CaffeStage.INNER_PRODUCT.value]


def isLRN(layer):
    return layer in ["LRN", CaffeStage.LRN.value]


def isInnerLRN(layer):
    return layer.type in [
        "LRN", CaffeStage.LRN.value] and layer.lrn_param.norm_region == 1


def isConcat(layer):
    return layer in ["Concat", CaffeStage.CONCAT.value]


def isDropout(layer):
    return layer in ["Dropout", CaffeStage.DROPOUT.value]


def isEltwise(layer):
    return layer in ["Eltwise", "Bias", CaffeStage.ELTWISE.value]


def isSlice(layer):
    return layer in ["Slice", CaffeStage.SLICE.value]


def isBatchNorm(layer):
    return layer in ["BatchNorm"]


def isScale(layer):
    return layer in ["Scale"]


def isDeconvolution(layer):
    return layer in ["Deconvolution", CaffeStage.DECONVOLUTION.value]


def isPower(layer):
    return layer in ["Power", CaffeStage.POWER.value]


def isReshape(layer):
    return layer in ["Reshape", CaffeStage.RESHAPE.value]


def isELU(layer):
    return layer in ["ELU"]


def isFlatten(layer):
    return layer in ["Flatten", CaffeStage.FLATTEN.value]


def isCrop(layer):
    return layer in ["Crop"]


def isDepthwiseConvolution(layer, output_channels, input_channels):
    return layer.type in ["Convolution"] and \
        layer.convolution_param.group > 1 and \
        layer.convolution_param.group == input_channels and \
        output_channels % input_channels == 0


def isPermute(layer):
    return layer in ["Permute"]


def isNormalize(layer):
    return layer in ["Normalize"]


def isPriorBox(layer):
    return layer in ["PriorBox"]


def isDetectionOutput(layer):
    return layer in ["DetectionOutput"]


def get_caffe_op_type(layer, input_channels=1, output_channels=1):
    """
    Gets the relevant Toolkit Enum for the corresponding Caffe layer stage type.
    :param layer:
        The particular layer field of the caffe Net msg that we want to discover the type.
    :return: StageType Enum
    """
    if isConvolution(layer.type):
        return StageType.convolution
    if isFCL(layer.type):
        return StageType.fully_connected_layer
    if isSoftMax(layer.type):
        return StageType.soft_max
    if isPooling(layer.type):
        pooling_type = layer.pooling_param.pool
        if pooling_type == 0:  # Max
            return StageType.max_pooling
        if pooling_type == 1:  # Average
            return StageType.average_pooling
        if pooling_type == 2:  # Stochastic
            throw_error(ErrorTable.StageTypeNotSupported, "Stochastic Pooling")
            return StageType.stochastic_pooling
    if isLRN(layer.type):
        return StageType.LRN
    if isEltwise(layer.type):
        if layer.type == 'Bias':
            return StageType.eltwise_sum
        elif layer.eltwise_param.operation == 0:
            return StageType.eltwise_prod
        elif layer.eltwise_param.operation == 2:
            return StageType.eltwise_max
        else:
            return StageType.eltwise_sum
    if isBatchNorm(layer.type) or isScale(layer.type):
        return StageType.scale
    if isPReLU(layer.type):
        return StageType.prelu
    if isSigmoid(layer.type):
        return StageType.sigmoid
    if isTanH(layer.type):
        return StageType.tanh
    if isDeconvolution(layer.type):
        return StageType.deconvolution
    if isReshape(layer.type):
        return StageType.reshape
    if isFlatten(layer.type):
        return StageType.toplanemajor
    if isPower(layer.type):
        return StageType.power
    if isCrop(layer.type):
        return StageType.crop
    if isDepthwiseConvolution(layer, output_channels, input_channels):
        return StageType.depthwise_convolution
    if isPermute(layer.type):
        return StageType.permute
    if isNormalize(layer.type):
        return StageType.normalize
    if isPriorBox(layer.type):
        return StageType.prior_box
    if isDetectionOutput(layer.type):
        return StageType.detection_output

    throw_error(ErrorTable.StageTypeNotSupported, layer.type)
