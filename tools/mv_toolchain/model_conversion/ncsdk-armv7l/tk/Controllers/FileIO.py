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

import warnings
import ctypes
import numpy as np
from Models.EnumDeclarations import *
from Controllers.EnumController import *

data_offset = 0
zero_data_offset = 0
buffer = []
bss_buffer = []
# It will be incremented on first use
buffer_index = MemoryIndex.workbuffer.value - 1


def file_init():
    global data_offset
    global zero_data_offset
    global buffer
    global bss_buffer
    global buffer_index

    data_offset = 0
    zero_data_offset = 0
    buffer = []
    bss_buffer = []
    # It will be incremented on first use
    buffer_index = MemoryIndex.workbuffer.value - 1


def get_numpy_element_byte_size(array):
    if array.dtype == np.float32 \
            or array.dtype == np.int32 \
            or array.dtype == np.uint32:
        warnings.warn(
            "\033[93mYou are using a large type. " +
            "Consider reducing your data sizes for best performance\033[0m")
        return 4
    if array.dtype == np.float16 \
            or array.dtype == np.int16 \
            or array.dtype == np.uint16:
        return 2
    if array.dtype == np.uint8 \
            or array.dtype == np.int8:
        warnings.warn(
            "\033[93mYou are using an experimental type. May not be fully functional\033[0m")
        return 1

    throw_error(ErrorTable.DataTypeNotSupported, array.dtype)


def align(offset, data, align_to=64):
    """
    Extends data to be a size aligned to a known amount.
    :param offset: count which will need to be adjusted
    :param data: data that will be padded with zeroes
    :param align_to: desired alignment
    :return: (adjusted offset, flattened adjusted data)
    """
    rem = offset % align_to
    new_offset = offset if (rem == 0) else offset + (align_to - rem)

    if data is not None:
        new_data = np.pad(
            data.flatten(),
            (0, int((new_offset - offset) / data.dtype.itemsize)), mode="constant")
    else:
        new_data = None
    return new_offset, new_data


def get_buffer(for_data, datatype):
    """
    Gets an offset for a buffer in the blob file. Relative to the start of the blob file.
    :param for_data: The data to be added to the global write buffer
    :param datatype: Bytesize of the current array.
    :return: offset to data, global buff index
    """
    global data_offset
    global buffer
    buffer_size = len(for_data.flatten()) * dtype_size(datatype)
    (buffer_size, for_data) = align(buffer_size, for_data, 64)
    buffer.append(for_data)

    data_offset += buffer_size
    return data_offset - buffer_size, len(buffer)


def get_zero_buffer(for_data, datatype):
    """
    Gets an offset for a buffer in the work buffer. Relative to the start of the blob file.
                       +----------+
                       |          |
                       |   pad    |
    outputPointer+---> +----------+
                       |          |
                       |  buffer  |
                       |          |
                       +----------+
                       |          |
                       |   pad    |
                       +----------+
    Note: The buffer is overallocated for each stage, because convolutions write
    outside the buffers.
    The maximum pad is computed for RADIX_MAX, stride 1, and SAME padding. (i.e
    the kernel is centered from the first point in the image)
    Let's take radix = 3. For the first output we add up the first two elements
    of the first two rows from the input, of course, multiplied by the kernel weights.
    So for the last kernel (bottom-right), the input has an offset of 1 row and 1 column,
    which is radix/2 * row_width + radix/2. (In our algorithm this means it will write
    starting from -(radix/2 * row_width + radix/2))
    If we put the first kernel (upper-left) to overalp the first point in the image,
    we will get the result for the radix/2 * row_width + radix/2 output element.
    (In our algorithm it will write starting from (radix/2 * row_width + radix/2),
    so it will write outside with the same amount of elements)
    Therefore, the maximum offset has to be radix/2 * row_width + radix/2.

    :param for_data: The data to be added to the global write buffer
    :param datatype: Bytesize of the current array.
    :return: offset to data, global buff index
    """
    global zero_data_offset
    global bss_buffer
    global buffer_index

    RADIX_MAX = 5   # TODO: Make parameterized

    width = for_data.shape[2]
    channels = for_data.shape[0]
    pad = RADIX_MAX // 2 * (width + 1) * (channels) * dtype_size(datatype)
    buffer_size = len(for_data.flatten()) * dtype_size(datatype) + 2 * pad

    (buffer_size, for_data) = align(buffer_size, for_data, 64)

    bss_buffer.append(for_data)
    zero_data_offset += buffer_size
    buffer_index += 1
    if zero_data_offset - buffer_size + pad + buffer_index > 100 * 1024 * 1024:
        throw_error(ErrorTable.NoResources)

    return zero_data_offset - buffer_size + pad, buffer_index


def replace_buffer(new_data, offset, datatype):
    """
    In case a buffer needs to be reshaped, you must also edit the global buffer that will be written to file.
    THE BUFFER MUST REMAIN THE SAME SIZE AS BEFORE or you may experience unexpected results.

    :param new_data: The new buffer
    :param offset: the offset that was returned from the original get_buffer call
    :param datatype: the byte size of the type we are using.
    :return: Nothing.
    """
    offset = offset - 1

    if offset < 0:
        # If our buffer was a zero buffer, we don't need to bother changing.
        return

    global buffer
    buffer_size = len(new_data.flatten()) * dtype_size(datatype)
    (buffer_size, new_data) = align(buffer_size, new_data)

    # print(offset)
    buffer[offset] = new_data


def write_data(f):
    """
    Write the global buffer to a file.
    :param f: file handler
    :return: Nothing.
    """
    global buffer
    for data in buffer:
        f.write(data)


def data_size():
    global buffer
    # Get length of each array & multiply by the byte size of it's type
    byte_count = sum([a.flatten().shape[0] *
                      get_numpy_element_byte_size(a) for a in buffer])
    return byte_count


def byte_size(item):
    """
    :param item: variable to assess
    :return: size of item in bytes when passed to blob file.
    """
    if type(item) is bytes:
        return len(item)
    else:
        return ctypes.sizeof(item)


def get_buffer_start(blob):
    """
    Index VALUE 1+  = In Blob
    Index VALUE 0   = Not Assigned
    Index VALUE -1- = BSS
    :param blob:
    :return:
    """
    file_size = 0
    if blob.VCS_Fix:
        file_size = byte_size(ctypes.c_uint64(0)) * 4

    file_size += byte_size(blob.filesize)
    file_size += byte_size(blob.version)
    file_size += byte_size(blob.name)
    file_size += byte_size(blob.report_dir)
    file_size += byte_size(blob.stage_count)
    file_size += byte_size(ctypes.c_uint32(0))
    assert file_size == 0xf8, "Blob format modified, please change the " +\
                              "FathomRun/tests/per_layer_tests/util/generate_test_data.py file"
    file_size += blob.myriad_params.binary_size()
    file_size += blob.network.head[0].binary_size() * blob.network.count
    file_size += align(file_size, np.zeros(1), align_to=8)[0] - file_size
    return ctypes.c_uint32(file_size)


def estimate_file_size(blob):
    file_size = 0
    if blob.VCS_Fix:
        file_size = byte_size(ctypes.c_uint64(0)) * 4

    file_size += byte_size(blob.filesize)
    file_size += byte_size(blob.version)
    file_size += byte_size(blob.name)
    file_size += byte_size(blob.report_dir)
    file_size += byte_size(blob.stage_count)
    file_size += byte_size(ctypes.c_uint32(0))
    file_size += blob.myriad_params.binary_size()
    file_size += blob.network.head[0].binary_size() * blob.network.count
    file_size += align(file_size, np.zeros(1), align_to=8)[0] - file_size
    file_size += data_size()
    if file_size > 320 * 1024 * 1024:
        throw_error(ErrorTable.NoResources)
    return ctypes.c_uint32(file_size)
