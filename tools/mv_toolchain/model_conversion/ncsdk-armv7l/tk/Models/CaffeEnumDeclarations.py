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

from enum import Enum

"""
This is so we can support Network Descriptions which use internal Caffe enums
instead of strings
"""


class CaffeStage(Enum):
    NONE = 0
    ABSVAL = 35
    ACCURACY = 1
    ARGMAX = 30
    BNLL = 2
    CONCAT = 3
    CONTRASTIVE_LOSS = 37
    CONVOLUTION = 4
    DATA = 5
    DECONVOLUTION = 39
    DROPOUT = 6
    DUMMY_DATA = 32
    EUCLIDEAN_LOSS = 7
    ELTWISE = 25
    EXP = 38
    FLATTEN = 8
    HDF5_DATA = 9
    HDF5_OUTPUT = 10
    HINGE_LOSS = 28
    IM2COL = 11
    IMAGE_DATA = 12
    INFOGAIN_LOSS = 13
    INNER_PRODUCT = 14
    LRN = 15
    MEMORY_DATA = 29
    MULTINOMIAL_LOGISTIC_LOSS = 16
    MVN = 34
    POOLING = 17
    POWER = 26
    RELU = 18
    SIGMOID = 19
    SIGMOID_CROSS_ENTROPY_LOSS = 27
    SILENCE = 36
    SOFTMAX = 20
    SOFTMAX_LOSS = 21
    SPLIT = 22
    SLICE = 33
    TANH = 23
    WINDOW_DATA = 24
    THRESHOLD = 31
    RESHAPE = 40
