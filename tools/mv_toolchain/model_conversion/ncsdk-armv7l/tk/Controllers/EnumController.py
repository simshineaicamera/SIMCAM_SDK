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

import sys
from Models.EnumDeclarations import *


def stage_as_label(stage):
    d = {
        StageType.convolution: "conv",
        StageType.max_pooling: "maxpool",
        StageType.average_pooling: "avpool",
        StageType.deconvolution: "deconv",
        StageType.reshape: "reshape",
        StageType.power: "power",
        StageType.permute: "permute",
        StageType.normalize: "normalize",
        StageType.prior_box: "prior_box",
        StageType.detection_output: "detection_output",
    }
    if stage in d:
        return d[stage]
    return "no_optimization"


def parser_as_enum(string):
    if string in ["caffe", "Caffe", "CAFFE"]:
        parser = Parser.Caffe
    elif string in ["TF", "TensorFlow", "tensorflow", "TENSORFLOW", "tf"]:
        parser = Parser.TensorFlow
    else:
        throw_error(ErrorTable.ParserNotSupported, string)
        parser = None
    return parser


def validation_as_enum(string):
    if string in [
        "top5",
        "TOP5",
        "top-5",
            "TOP-5"]:
        val = ValidationStatistic.top5
    elif string in ["top1", "top", "top-1", "TOP", "TOP1", "TOP-1"]:
        val = ValidationStatistic.top1
    elif string in ["test", "debug", "metrics", "accuracy"]:
        val = ValidationStatistic.accuracy_metrics
    elif string in ["class-check-exact", "class-accuracy-exact", "class-verification-exact", "class-test-exact"]:
        val = ValidationStatistic.class_check_exact
    elif string in ["class-check-broad", "class-accuracy-broad", "class-verification-broad", "class-test-broad"]:
        val = ValidationStatistic.class_check_broad
    else:
        val = ValidationStatistic.invalid

    return val


def parse_mode(string):
    if string == "generate":
        mode = OperationMode.generation
    elif string == "profile":
        mode = OperationMode.profile
    elif string == "debug_validate_test":
        mode = OperationMode.test_validation
        print("Warning: This mode has no Guarantee of Functionality / Maintenance")
    elif string == "debug_generate_test":
        mode = OperationMode.test_generation
        print("Warning: This mode has no Guarantee of Functionality / Maintenance")
    elif string in ["debug_validate", "validate"]:
        mode = OperationMode.validation
    elif string == "debug_demo":
        mode = OperationMode.demo
        print("Warning: This mode has no Guarantee of Functionality / Maintenance")
    elif string == "TF":
        mode = OperationMode.testTensorFlow
        print("Warning: This mode has no Guarantee of Functionality / Maintenance")
    elif string in ["temp", "stress"]:
        mode = OperationMode.temperature_profile
    elif string == "optlist":
        mode = OperationMode.optimization_list
    else:
        mode = OperationMode.invalid
        throw_error(ErrorTable.ModeSelectionError, string)
    return mode


def completion_msg(mode):
    if mode == "generate":
        msg = "...Blob File Generated [OK]"
    elif mode == "profile":
        msg = "...PDF Report Created [OK]"
    else:
        msg = "...Complete: [OK]"

    return msg


def get_class_of_op(op):
    """
    Get type of operation
    :param op:
    :return:
    """
    if op in [StageType.convolution, StageType.depthwise_convolution]:
        return "Convolution"
    elif op in [StageType.fully_connected_layer]:
        return "FCL"
    elif op in [StageType.max_pooling, StageType.average_pooling]:
        return "Pooling"
    elif op in [StageType.deconvolution]:
        return "Deconvolution"
    elif op in [StageType.reshape]:
        return "Reshape"
    elif op in [StageType.permute]:
        return "Permute"
    elif op in [StageType.normalize]:
        return "Normalize"
    elif op in [StageType.prior_box]:
        return "PriorBox"
    elif op in [StageType.detection_output]:
        return "DetectionOutput"
    else:
        # print(op)
        return "Unknown"


def dtype_size(e):
    """
    Size in bytes
    :param e:
    :return:
    """
    if e == DataType.fp64:
        return 8
    if e == DataType.fp32:
        return 4
    if e == DataType.fp16 or e == 2:
        return 2
    if e == DataType.fp8:
        return 1
    if e == DataType.int64:
        return 8
    if e == DataType.int32:
        return 4
    if e == DataType.int16:
        return 2
    if e == DataType.int8:
        return 1
    if e == DataType.chr:
        return 1
    if e == DataType.int4:
        return 0.5
    if e == DataType.int2:
        return 0.25
    if e == DataType.bit:
        return 0.125


def enum_as_dtype(e):
    """
    Return a numpy type instead of our enums
    :param e:
    :return:
    """
    if e == DataType.fp64:
        return np.floatfp64
    if e == DataType.fp32:
        return np.float32
    if e == DataType.fp16 or e == 2:
        return np.float16
    if e == DataType.fp8:
        return np.float8
    if e == DataType.int64:
        return np.int64
    if e == DataType.int32:
        return np.int32
    if e == DataType.int16:
        return np.int16
    if e == DataType.int8:
        return np.int8
    if e == DataType.int4:
        return np.int4
    if e == DataType.int2:
        return np.int2
    if e == DataType.bit:
        return np.bit

def dtype_as_enum(dtype):
    """
    Return the enum corresponding to the numpy dtype
    :param dtype:
    :return e:
    """
    dt_np2enum = {
            np.dtype('float64') : DataType.fp64,
            np.dtype('float32') : DataType.fp32,
            np.dtype('float16') : DataType.fp16,
            np.dtype('int64') : DataType.int64,
            np.dtype('int32') : DataType.int32,
            np.dtype('int16') : DataType.int16,
            np.dtype('int8') : DataType.int8
        }

    return dt_np2enum[dtype]

def throw_warning(e, extra=None):
    msg = "[Warning: " + str(e.value) + "] "
    if e == ErrorTable.OptimizationParseError:
        msg += " Problem parsing optimization file. Using Defaults"
    if e == ErrorTable.OutputNodeNameTopMismatch:
        (name, top) = extra
        msg += " Output layer\'s name (" + str(name) + \
            ") must match its top (" + str(top) + ")"
    print("\033[93m" + str(msg) + "\033[0m", file=sys.stderr)


def throw_error(e, extra=None):
    msg = "[Error " + str(e.value) + "] "
    if e == ErrorTable.CaffeImportError:
        msg += "Setup Error: Caffe Import Error."
    if e == ErrorTable.CaffeSyntaxError:
        msg += "Setup Error: Caffe Syntax Error: " + str(extra)
    if e == ErrorTable.PythonVersionError:
        msg += "Setup Error: Using a version of Python that is unsupported."
    if e == ErrorTable.ModeSelectionError:
        msg += "Toolkit Error: No such Mode '" + str(extra) + "'."
    if e == ErrorTable.ArgumentErrorDescription:
        msg += "Argument Error: Network description cannot be found."
    if e == ErrorTable.ArgumentErrorWeights:
        msg += "Argument Error: Network weight cannot be found."
    if e == ErrorTable.ArgumentErrorImage:
        msg += "Argument Error: Image cannot be found."
    if e == ErrorTable.ArgumentErrorExpID:
        msg += "Argument Error: Expected ID not provided."
    if e == ErrorTable.NoUSBBinary:
        msg += "Toolkit Error: No moviUsbBoot Executable detected."
    if e == ErrorTable.USBError:
        msg += "Toolkit Error: USB Failure. Code: " + str(extra)
    if e == ErrorTable.MyriadExeNotPresent:
        msg += "Setup Error: no Myriad Executable detected."
    if e == ErrorTable.NoOutputNode:
        msg += "Toolkit Error: Provided OutputNode/InputNode name does not exist or does not match with one contained in " + \
               "model file Provided: " + str(extra)
    if e == ErrorTable.StageTypeNotSupported:
        msg += "Toolkit Error: Stage Type Not Supported: " + str(extra)
    if e == ErrorTable.StageDetailsNotSupported:
        msg += "Toolkit Error: Stage Details Not Supported: " + str(extra)
    if e == ErrorTable.DataTypeNotSupported:
        msg += "Toolkit Error: Data Type Not Supported: " + str(extra)
    if e == ErrorTable.ParserNotSupported:
        msg += "Toolkit Error: Parser Not Supported: " + str(extra)
    if e == ErrorTable.InputNotFirstLayer:
        msg += "Toolkit Error: Internal Error: Input Stage is not first layer."
    if e == ErrorTable.GraphConstructionFailure:
        msg += "Toolkit Error: Internal Error: Could not build graph. Missing link: " + \
            str(extra)
    if e == ErrorTable.ConversionNotSupported:
        msg += "Toolkit Error: Internal Error: Invalid Conversion Optimization. From: " + \
            str(extra)
    if e == ErrorTable.ArgumentErrorRequired:
        msg += "Toolkit Error: Setup Error: Not all required arguments were passed / Erroneous arguments."
    if e == ErrorTable.InputSyntaxNotSupported:
        msg += "Toolkit Error: Input Layer must be in an input_shape construct."
    if e == ErrorTable.ValidationSelectionError:
        msg += "Argument Error: Validation metric not supported " + str(extra)
    if e == ErrorTable.UnrecognizedFileType:
        msg += "Toolkit Error: Unable to tell what parser is required. Consider overriding with --parser argument."
    if e == ErrorTable.InvalidInputFile:
        msg += "Toolkit Error: Mismatch between input layer of network and given input file."
    if e == ErrorTable.AttemptedBatchMode:
        msg += "Toolkit Error: >1 image inference not supported."
    if e == ErrorTable.MyriadRuntimeIssue:
        msg += 'Myriad Error: "' + str(extra) + '".'
    if e == ErrorTable.InvalidNumberOfShaves:
        msg += "Setup Error: Too Many / Too Few Shave Processors Selected."
    if e == ErrorTable.CaffeMemoryError:
        msg += "Caffe Error: MemoryError. Potential Cause: Available RAM not sufficient for Network to be loaded into Caffe"
    if e == ErrorTable.TupleSyntaxWrong:
        msg += "Setup Error: Tuple Syntax Incorrect, should be in form x,y,z "
    if e == ErrorTable.InputFileUnsupported:
        msg += "Toolkit Error: Filetype not supported as a input."
    if e == ErrorTable.USBDataTransferError:
        msg += "USB Error: Problem Transferring data."
    if e == ErrorTable.OptimizationParseError:
        msg += "Setup Error: Problem parsing configuration File."
    if e == ErrorTable.NoTemperatureRecorded:
        msg += "Toolkit Error: No Temperature Read from device."
    if e == ErrorTable.TFNotEvaluated:
        msg += "Setup Error: Values for input contain placeholder. Pass an absolute value."
    if e == ErrorTable.NoResources:
        msg += "Setup Error: Not enough resources on Myriad to process this network."
    if e == ErrorTable.InvalidNpyFile:
        msg += "Toolkit Error: Unable to load .npy file (" + str(extra) + ")"
    if e == ErrorTable.InvalidTuple:
        msg += "Toolkit Error: Invalid tuple format (" + str(extra) + ")"
    if e == ErrorTable.InvalidMean:
        msg += "Toolkit Error: Invalid mean value (" + str(extra) + ")"

    print("\033[91m" + str(msg) + "\033[0m", file=sys.stderr)
    quit()
    assert 0, msg
