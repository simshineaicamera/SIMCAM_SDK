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
from ctypes import *
from Controllers.MiscIO import *
from Controllers.DataTransforms import *
from Controllers.EnumController import *
from Models.EnumDeclarations import *
from Views.Graphs import *

from linecache import getline


class NetworkStage:

    def __init__(
            self,
            name,
            top,
            s_order,
            pad_y,
            pad_x,
            pad_type,
            dtype,
            precision,
            op_type,
            op_y,
            op_x,
            sy,
            sx,
            x,
            y,
            c,
            fh,
            fw,
            k,
            taps,
            taps_order,
            bias,
            pre_op_type,
            post_op_type,
            post_1,
            post_sx,
            post_sy,
            slicing=None,
            myriad_config=None,
            args=None,
            opParams=None,
            new_x=0,
            new_y=0,
            new_c=0):
        self.changeName(name)

        # Historically cocat layer was working only for axis1 i.e channels.
        # Concat on axis 2 is available only for CaffeParser and concat_axis 
        # needs to default to 1 in order not to break the TensorFlowParser.
        self.concat_axis = 1

        # mvTensor cannot deal with such convolution, which is equivalent to fc
        if (op_type == StageType.convolution and op_x == 1 and op_y == 1
                and x == 1 and y == 1):
            op_type = StageType.fully_connected_layer

        self.network = args.network
        self.top = top
        self.tail = []
        self.op = op_type
        self.radixX = op_x
        self.radixY = op_y
        self.padX = pad_x
        self.padY = pad_y
        self.alias = [name]

        if self.radixX == -1 and self.radixY == -1:
            # Global Operation
            self.radixX = x
            self.radixY = y

        self.strideX = sx
        self.strideY = sy

        self.optMask = readOptimisationMask(name, self, myriad_config, args)

        self.inputStrideX = 2 * c  # Concat Stride
        self.inputStrideY = 2 * c * x
        self.inputStrideZ = 2
        self.inputOffset = 0
        if slicing:
            if top is None:
                for slice in slicing:
                    if slice[0] is None:
                        c = slice[2] - slice[1]
                        self.inputOffset = slice[1] * 2
                        break
            else:
                for input in top:
                    for slice in slicing:
                        if slice[0] == input:
                            c = slice[2] - slice[1]
                            self.inputOffset = slice[1] * 2
                            break
        if (op_type == StageType.eltwise_sum or op_type ==
                StageType.eltwise_prod or op_type == StageType.eltwise_max):
            # Ignore given k, which could be wrong if it ignored slicing
            k = c
        self.inputDimX = x
        self.inputDimY = y
        self.inputDimZ = c

        self.tapDimX = fw * fh
        self.tapDimY = c
        self.tapDimZ = k  # Will be replaced when attached to another stage..

        self.outputDimZ = k
        if self.op in [StageType.fully_connected_layer]:
            self.inputDimX = 1
            self.inputDimY = 1
            self.inputDimZ = x * y * c

            self.tapDimX = 1
            self.tapDimY = x * y * c

            self.outputDimX = 1
            self.outputDimY = 1

        elif self.op in [StageType.convolution, StageType.depthwise_convolution, StageType.max_pooling, StageType.average_pooling]:
            if pad_type == PadStyle.tfsame:
                self.outputDimX = math.ceil(x / self.strideX)
                self.outputDimY = math.ceil(y / self.strideY)
            elif pad_type == PadStyle.tfvalid:
                self.outputDimX = math.ceil((x - self.radixX + 1) / self.strideX)
                self.outputDimY = math.ceil((y - self.radixY + 1) / self.strideY)
            # Caffe, convolution uses floor
            elif self.op in [StageType.convolution, StageType.depthwise_convolution]:
                if self.radixX == 1 and self.radixY == 1 and self.padX == 1 and self.padY == 1:
                    throw_error(
                        ErrorTable.StageDetailsNotSupported,
                        'Padding 1 not supported for 1x1 convolution in ' + name)
                # This code should be executed only for caffe layers.
                radix_x_extent = self.radixX
                radix_y_extent = self.radixY
                dilation = 1
                if opParams is not None:
                    dilation = opParams[0]
                    radix_x_extent = dilation * (self.radixX - 1) + 1
                    radix_y_extent = dilation * (self.radixY - 1) + 1

                self.outputDimX = (x + 2 * self.padX - radix_x_extent) // self.strideX + 1
                self.outputDimY = (y + 2 * self.padY - radix_y_extent) // self.strideY + 1

            else:  # Caffe, pooling uses ceil
                self.outputDimX = math.ceil((x + 2 * self.padX - self.radixX) / self.strideX) + 1
                self.outputDimY = math.ceil((y + 2 * self.padY - self.radixY) / self.strideY) + 1
                self.outputDimX = min(self.outputDimX, math.ceil((x + self.padX) / self.strideX))
                self.outputDimY = min(self.outputDimY, math.ceil((y + self.padY) / self.strideY))

        elif self.op in [StageType.deconvolution]:
            if pad_type == PadStyle.tfsame:
                pad_X = math.floor(self.radixX / 2)
                pad_Y = math.floor(self.radixY / 2)
            elif pad_type == PadStyle.tfvalid:
                pad_X = self.radixX - 1
                pad_Y = self.radixY - 1
            elif pad_type == PadStyle.caffe:
                pad_X = self.padX
                pad_Y = self.padY
            else:
                pad_X = 0
                pad_Y = 0

            self.outputDimX = self.strideX * (x - 1) + self.radixX - 2 * pad_X
            self.outputDimY = self.strideY * (y - 1) + self.radixY - 2 * pad_Y

        elif self.op == StageType.toplanemajor:
            self.outputDimX = 1
            self.outputDimY = 1
            self.outputDimZ = x * y * c

        elif self.op in [StageType.reshape]:
            self.outputDimX = new_x
            self.outputDimY = new_y
            self.outputDimZ = new_c

            if (new_x == 0):
                self.outputDimX = x
            elif (new_x > 0):
                self.outputDimX = new_x

            if (new_y == 0):
                self.outputDimY = y
            elif (new_y > 0):
                self.outputDimY = new_y

            if (new_c == 0):
                self.outputDimZ = c
            elif (new_c > 0):
                self.outputDimZ = new_c

            if (new_x == -1):
                self.outputDimX = x * y * \
                    c // (self.outputDimY * self.outputDimZ)
            if (new_y == -1):
                self.outputDimY = x * y * \
                    c // (self.outputDimX * self.outputDimZ)
            if (new_c == -1):
                self.outputDimZ = x * y * \
                    c // (self.outputDimX * self.outputDimY)

        elif self.op in [StageType.reorg]:
            stride = opParams[0]
            self.outputDimX = int(self.inputDimX / stride)
            self.outputDimY = int(self.inputDimY / stride)
            self.outputDimZ = int(self.inputDimZ * stride * stride)

        elif self.op in [StageType.crop, StageType.permute]:
            self.outputDimX = new_x
            self.outputDimY = new_y
            self.outputDimZ = new_c
        elif self.op in [StageType.prior_box] :
            self.tapDimX = int(opParams[1])
            self.tapDimY = int(opParams[0])

            opParams = opParams[2:]
            min_size_size = opParams[0]
            max_size_size = opParams[1]
            flip = int(opParams[4])
            aspect_ratio_size = opParams[2]
            aspect_ratio_size = aspect_ratio_size*2 if (flip) else aspect_ratio_size

            num_priors = (1 + aspect_ratio_size) * min_size_size + max_size_size

            self.outputDimX = 1
            self.outputDimY = int(x * y * num_priors * 4)
            self.outputDimZ = 2
        elif self.op in [StageType.detection_output]:
            self.outputDimX = 7
            # output[0,0,0] = contains the number of detections.
            # The rest of the elements on the first line are grabage.
            # The detections start at the second line and span maximum new_y lines
            # i.e. top_k lines at max.
            self.outputDimY = new_y + 1
            self.outputDimZ = 1
        else:
            self.outputDimX = x
            self.outputDimY = y

        self.output = np.zeros(
            (int(
                self.outputDimZ), int(
                self.outputDimY), int(
                self.outputDimX))).astype(
                    enum_as_dtype(dtype))

        self.tapStrideX = 2 * self.tapDimZ  # Concat Stride
        self.tapStrideY = 2 * self.tapDimZ
        self.tapStrideZ = 2

        self.outputStrideX = 2 * self.outputDimZ  # Concat Stride
        self.outputStrideY = 2 * self.outputDimZ * self.outputDimX
        self.outputStrideZ = 2

        # Provide accessible backups in case concat or similar stages
        # overwrites them.
        self.unprocessed_w = x
        self.unprocessed_h = y
        self.unprocessed_c = c
        self.unprocessed_k = k
        self.unprocessed_output = self.output  # Used for theoretical graph sizes

        self.datatype = dtype
        self.precision = precision

        self.data = np.zeros((int(self.inputDimZ),
                              int(self.inputDimY),
                              int(self.inputDimX)
                              )).astype(enum_as_dtype(dtype))

        self.taps = taps
        self.tapsPointer = 0  # We will fill them in generate
        self.tapsIndex = 0
        self.tapsOrder = taps_order
        self.bias = bias
        self.biasPointer = 0  # We will fill them in generate
        self.biasIndex = 0
        self.opParams = opParams
        self.opParamsPointer = 0  # We will fill them in generate
        self.opParamsIndex = 0

        self.concatResult = False
        self.storageOrder = s_order
        self.padStyle = pad_type

        self.dataPointer = self.inputOffset
        self.dataIndex = 0

        self.outputPointer = 0
        self.outputIndex = 0

        if pre_op_type:
            self.preOp = pre_op_type
        else:
            self.preOp = StageType.none

        if post_op_type and post_op_type != StageType.none:
            self.postOp = post_op_type
            if post_1:
                self.post_param1 = post_1
            else:
                self.post_param1 = int(0)
            self.post_strideX = post_sx
            self.post_strideY = post_sy
        else:
            if (self.op in [StageType.convolution,
                            StageType.depthwise_convolution,
                            StageType.fully_connected_layer,
                            StageType.scale]) and bias is not None:
                self.postOp = StageType.bias
            else:
                self.postOp = StageType.none
            self.post_param1 = 0
            self.post_strideX = 0
            self.post_strideY = 0

        # in the case of Reshape, the outputDims are used as parameters for reshape;
        # make sure that the parameter values are written to myriad
        if self.op in [StageType.reshape]:
            self.outputDimX = new_x
            self.outputDimY = new_y
            self.outputDimZ = new_c

        # Only to be used after myriad execution
        self.flops = None
        self.ms = None
        self.BWs = None
        self.isoutput = False
        self.isconcat = False

    def addBias(self, bias):
        if bias is not None:
            if self.bias is None:
                self.bias = bias
                self.postOp = StageType.bias
            else:
                self.bias = self.bias + bias


    def putBias(self):
        if self.bias is not None:
            self.biasPointer, self.biasBufferIndex = get_buffer(
                self.bias.astype(np.float16), self.datatype)
            self.biasIndex = MemoryIndex.blob.value

    def putTaps(self):
        if self.taps is not None:
            self.tapsPointer, self.tapsBufferIndex = get_buffer(
                self.taps.astype(np.float16), self.datatype)
            self.tapsIndex = MemoryIndex.blob.value

    def putOpParams(self):
        """ Puts the operation parameters in the blob buffer """
        if self.opParams is not None:
            self.opParamsPointer, self.opParamsBufferIndex = \
                get_buffer(self.opParams, dtype_as_enum(self.opParams.dtype))
            self.opParamsIndex = MemoryIndex.blob.value

    def changeName(self, new_name):
        self.unprocessed_name = new_name
        self.name = set_string_range(new_name, 100).encode('ascii')

    def close(self):
        self.outputPointer = 0
        self.outputIndex = MemoryIndex.output

    def attach(self, stage):
        """
        Attaches a node to this one.
        :param stage:
        :return:
        """
        if (stage.op == StageType.convolution and self.op == StageType.depthwise_convolution and
            stage.radixX == 1 and stage.radixY == 1 and self.postOp == StageType.none):
            print('Fusing depthconv and conv in',self.unprocessed_name,'and',stage.unprocessed_name)
            #Create the weights for a convolution that does deptwhise convolution (inCH, outCH, kH, kW)
            taps = np.zeros([self.inputDimZ, self.tapDimZ, self.radixY, self.radixX], np.float32)
            multiplier = int(self.tapDimZ/self.tapDimY)
            for y in range(self.radixY):
                for x in range(self.radixX):
                    for c in range(self.tapDimY):
                        for i in range(multiplier):
                            taps[c,c*multiplier+i,y,x] = self.taps[y,x,c,i]
            #Turn them to [kH, kW, inCH, outCH) in order to be able to use matmul
            taps = taps.transpose(2,3,0,1)
            #Fuse the weights of the following 1x1 convolution into the just created weights
            stage.taps = np.matmul(taps,stage.taps[0,0])
            #Bring some data from the previous stage (self) to this one (stage) as we are saving this one
            #Saving the previous node would be simpler, but unfortunately the parser keeps track
            #of what's the latest created node (stage), so we must keep it
            stage.inputDimX = self.inputDimX
            stage.inputDimY = self.inputDimY
            stage.inputDimZ = self.inputDimZ
            stage.inputStrideX = self.inputStrideX
            stage.inputStrideY = self.inputStrideY
            stage.inputStrideZ = self.inputStrideZ
            stage.tapDimX = self.tapDimX
            stage.tapDimY = self.tapDimY
            stage.radixX = self.radixX
            stage.radixY = self.radixY
            stage.strideX = self.strideX
            stage.strideY = self.strideY
            stage.padStyle = self.padStyle
            stage.top = self.top
            stage.data = self.data
            stage.dataIndex = self.dataIndex
            stage.dataPointer = self.dataPointer
            #Remove self from network and change references
            self.network.count = self.network.count - 1
            self.network.stageslist.remove(self)
            stage.top = self.top
            if self in self.network.head:
                stage.network.storageOrder = stage.storageOrder
                self.network.head.remove(self)
                self.network.head.append(stage)
            else:
                for parents in self.network.search_several(self.top):
                    newtail = []
                    for p in parents.tail:
                        if p == self:
                            newtail.append(stage)
                    parents.tail = newtail
            return

        # This line is to build a correct graph after renaming due to absorption
        # When scale or batchnorm is absorbed into convolution, its name is appended
        # to the name of the convolution layer, so bottoms (here they are called tops,
        # wrong choice) of attached layers have to be renamed too
        # All the cases (attach, attach_several, attach_eltwise) are present in
        # test t19_UnitTest/NetworkConfig/AbsorptionRenaming.prototxt
        # What needs to be verified is the correct generation of the report
        # diagram
        stage.top = [self.unprocessed_name]
        self.tail.append(stage)
        if self.outputPointer == 0 and self.outputIndex == MemoryIndex.none.value:
            self.outputPointer, self.outputIndex = get_zero_buffer(
                self.output, self.datatype)

        stage.dataPointer = stage.inputOffset + self.outputPointer      # Input Pointer
        stage.dataIndex = self.outputIndex

        if (stage.op != StageType.fully_connected_layer and not self.isconcat
                and self.op != StageType.reshape):
            stage.inputDimX, stage.inputDimY, stage.inputDimZ = self.outputDimX, self.outputDimY, self.outputDimZ
            stage.tapDimY = self.outputDimZ
        if stage.op in [StageType.max_pooling]:
            stage.output = np.zeros(
                (stage.outputDimZ, stage.outputDimY, stage.outputDimX))

    def setoutput(self, outputStride, outputPointer=None, outputIndex=None):
        if self.outputPointer == 0 and self.outputIndex == MemoryIndex.none.value:
            self.output = np.zeros(
                (int(
                    outputStride / 2), int(
                    self.outputDimY), int(
                    self.outputDimX))).astype(
                enum_as_dtype(
                    self.datatype))
            if outputPointer is not None and outputIndex is not None:
                self.outputPointer = outputPointer
                self.outputIndex = outputIndex
            else:
                self.outputPointer, self.outputIndex = get_zero_buffer(
                    self.output, self.datatype)
            self.outputStrideX = outputStride
        self.isconcat = True
        return self.outputPointer, self.outputIndex

    def concat(stages, lastlayer=True):
        """
        Set the output pointers and fix the output strides to
        concatenate the outputs into the same buffer
        """

        # This check is almost irrelevant since there are other cases when the code
        # fails and this is a hack not propper code.
        for stage_i, stage in enumerate(stages):
            if stage.concat_axis != stages[0].concat_axis:
                raise Exception("A layer cannot be part of multiple concats")

        if(stages[0].concat_axis == 1):
            z = sum([int(stage.unprocessed_k) for stage in stages])
            x = int(stages[0].outputDimX)
            y = int(stages[0].outputDimY)

            concat_size = (y, x, z)

            dtype = stages[0].datatype
            if lastlayer:
                for stage in stages:
                    stage.isoutput = True
                buffer = 0
                buffer_index = MemoryIndex.output.value
            elif stages[0].outputPointer == 0 and stages[0].outputIndex == MemoryIndex.none.value:
                buffer, buffer_index = get_zero_buffer(np.zeros(concat_size).astype(enum_as_dtype(dtype)), dtype)
            else:
                buffer = stages[0].outputPointer
                buffer_index = stages[0].outputIndex

            concat_offset = 0

            for s_num, stage in enumerate(stages):
                offset_pointer = buffer

                if(stage.outputPointer == 0):
                    stage.outputPointer = offset_pointer + concat_offset*2  # TODO: REMOVE HARDCODED 2 For FP16 Size

                stage.outputIndex = buffer_index

                stage.concatResult = True
                concat_offset += int(stage.outputDimZ)

                stage.outputStrideX = z*2
                stage.outputStrideY = z*2*stage.outputDimX
                stage.tapStrideY = stage.outputDimZ * 2
        elif stages[0].concat_axis == 2:

            z = int(stages[0].outputDimZ)
            y = sum([int(stage.outputDimY) for stage in stages])
            x = int(stages[0].outputDimX)

            concat_size = (y, x, z)

            dtype = stages[0].datatype
            if lastlayer:
                for stage in stages:
                    stage.isoutput = True
                buffer = 0
                buffer_index = MemoryIndex.output.value
            elif stages[0].outputPointer == 0 and stages[0].outputIndex == MemoryIndex.none.value:
                buffer, buffer_index = get_zero_buffer(np.zeros(concat_size).astype(enum_as_dtype(dtype)), dtype)
            else:
                buffer = stages[0].outputPointer
                buffer_index = stages[0].outputIndex

            concat_offset = 0

            for s_num, stage in enumerate(stages):
                offset_pointer = buffer

                if(stage.outputPointer == 0):
                    stage.outputPointer = offset_pointer + concat_offset*2  # TODO: REMOVE HARDCODED 2 For FP16 Size

                stage.outputIndex = buffer_index

                stage.concatResult = True
                concat_offset += int(stage.outputDimY * stage.outputDimX * stage.outputDimZ)

        else:
            # This check is almost irrelevant since there are other cases when the code
            # fails and this is a hack not propper code.
            raise Exception("Concat on axis {0} not implemented".format(stages[0].concat_axis))

    def attach_eltwise(self, parents):
        # Attach two parents to this elementwise operations layer
        # The second layer will be put in the weights pointer


        if hasattr(parents[0], '__iter__'):
            NetworkStage.concat(parents[0], False)
            parents[0] = parents[0][0]
        # This line is to build a correct graph after renaming due to
        # absorption, see attach
        self.top[0] = parents[0].unprocessed_name
        # We have only two cases: intermediate, input or intermediate,
        # intermediate
        if parents[1] == 0:
            parents[0].outputPointer, parents[0].outputIndex = get_zero_buffer(
                parents[0].output, self.datatype)
            self.dataPointer = self.inputOffset + parents[0].outputPointer
            self.dataIndex = parents[0].outputIndex
            self.tapsPointer = 0
            self.tapsIndex = MemoryIndex.input.value
            parents[0].tail.append(self)
        else:
            if hasattr(parents[1], '__iter__'):
                NetworkStage.concat(parents[1], False)
                parents[1] = parents[1][0]
            # This line is to build a correct graph after renaming due to
            # absorption, see attach
            self.top[1] = parents[1].unprocessed_name
            if parents[0].outputIndex == 0:
                parents[0].outputPointer, parents[0].outputIndex = get_zero_buffer(
                    parents[0].output, self.datatype)
            if parents[1].outputIndex == 0:
                parents[1].outputPointer, parents[1].outputIndex = get_zero_buffer(
                    parents[1].output, self.datatype)
            self.dataPointer = self.inputOffset + parents[0].outputPointer
            self.dataIndex = parents[0].outputIndex
            self.tapsPointer = parents[1].outputPointer
            self.tapsIndex = parents[1].outputIndex
            parents[1].tail.append(self)
        return

    def attach_multiple_bottoms(self, parents):
        # Attach a layer with at most 3 bottoms.
        # 1st bottom -> to input data pointer.
        # 2nd bottom -> to weights data pointer.
        # 3rd bottom -> to biases data pointer.

        if(len(parents) > 3):
            raise Exception("Layer with {0} inputs not supported".format(len(parents)))

        for bottom_idx, bottom in enumerate(parents):
            if hasattr(bottom, '__iter__'):
                NetworkStage.concat(parents[bottom_idx], False)
                parents[bottom_idx] = parents[bottom_idx][0]

            if bottom == 0:
                # This bottom is the input (ussualy named "data") to the network.
                if bottom_idx == 0:
                    self.dataPointer = 0
                    self.dataIndex   = MemoryIndex.input.value
                elif bottom_idx == 1:
                    self.tapsPointer = 0
                    self.tapsIndex   = MemoryIndex.input.value
                else:
                    self.biasPointer = 0
                    self.biasIndex   = MemoryIndex.input.value
            else:
                # This bottom is the output of a layer.
                if(parents[bottom_idx].outputIndex == 0):
                    out_ptr, out_idx = get_zero_buffer(parents[bottom_idx].output, self.datatype)
                    parents[bottom_idx].outputPointer = out_ptr
                    parents[bottom_idx].outputIndex   = out_idx

                if bottom_idx == 0:
                    self.dataPointer = self.inputOffset + parents[bottom_idx].outputPointer
                    self.dataIndex   = parents[bottom_idx].outputIndex
                elif bottom_idx == 1:
                    self.tapsPointer = parents[bottom_idx].outputPointer
                    self.tapsIndex   = parents[bottom_idx].outputIndex
                else:
                    self.biasPointer = parents[bottom_idx].outputPointer
                    self.biasIndex   = parents[bottom_idx].outputIndex

        #parents[1].tail.append(self)
        #return
        for bottom_idx, bottom in reversed(list(enumerate(parents))):
            if bottom != 0:
                parents[bottom_idx].tail.append(self)
                return

    def attach_several(self, parents):
        """
        Attach a node to several parents. Under 'concat' rules, the parents will be combined at the channel level.
        Under yxz this means that we need a writeback offset.

        TODO: This is coded for YXZ only.
        The default mode should be ZYX and we should transform it during the optimize() phase of network setup.

        :param parents:
        :return:
        """

        # attach_several works with both one and more parents, which must be
        # concat inputs

        if not hasattr(parents, '__iter__'):
            parents.attach(self)
            return

        # Next three lines are to build a correct graph after renaming due to
        # absorption, see attach
        self.top = []
        for l in parents:
            self.top.append(l.unprocessed_name)

        NetworkStage.concat(parents, False)
        z = sum([int(p.unprocessed_k) for p in parents])
        parents[len(parents) - 1].tail.append(self)
        self.inputDimZ = z
        self.inputStrideX = z * 2

        self.dataPointer = self.inputOffset + \
            parents[0].outputPointer      # Input Pointer
        self.dataIndex = parents[0].outputIndex

        self.tapDimY = z

        if self.op in [StageType.max_pooling]:
            self.outputDimZ = self.inputDimZ
            self.outputStrideX = self.inputStrideX

    def search(self, seek_name):
        """
        return: 0 if not found. The searched node if found.
        :param seek_name: name of node we're looking for
        """
        if self.name == seek_name or self.unprocessed_name == seek_name or seek_name in self.alias:
            return self

        for t in self.tail:
            if t.name == seek_name or t.unprocessed_name == seek_name or seek_name in t.alias:
                return t                                # Search item == current item
            else:
                # Not found, traverse deeper
                recursive_result = t.search(seek_name)
                if (recursive_result != 0 and recursive_result.name == seek_name) or \
                        (recursive_result != 0 and recursive_result.unprocessed_name == seek_name) or \
                        (recursive_result != 0 and seek_name in recursive_result.alias):
                    # Found in one of the tree nodes, bubble up.
                    return recursive_result
        return 0  # Not found, backtrack

    def set_blob_vars(self):
        """
         Builds the layout of the network stage for use in the blobfile.
         Currently undergoing refactoring so that ctypes are only applied here.
         :return:
        """
        self.write_items = [
            self.name,
            c_char(self.op.value),
            c_uint32(self.optMask),
            c_int8(self.radixX),
            c_int8(self.radixY),
            c_uint8(self.strideX),
            c_uint8(self.strideY),
            c_uint8(self.padX),
            c_uint8(self.padY),
            c_uint8(self.padStyle.value),
            c_uint32(self.inputDimX),
            c_uint32(self.inputDimY),
            c_uint32(self.inputDimZ),
            c_uint32(self.tapDimX),
            c_uint32(self.tapDimY),
            c_uint32(self.tapDimZ),
            c_uint32(self.outputDimX),
            c_uint32(self.outputDimY),
            c_uint32(self.outputDimZ),
            c_uint32(self.inputStrideX),
            c_uint32(self.inputStrideY),
            c_uint32(self.inputStrideZ),
            c_uint32(self.tapStrideX),
            c_uint32(self.tapStrideY),
            c_uint32(self.tapStrideZ),
            c_uint32(self.outputStrideX),
            c_uint32(self.outputStrideY),
            c_uint32(self.outputStrideZ),
            c_uint8(self.datatype.value),
            c_uint8(self.precision.value),
            c_uint8(self.storageOrder.value),
            c_uint32(self.dataPointer),
            c_uint16(self.dataIndex),
            c_uint32(self.tapsPointer),
            c_uint16(self.tapsIndex),
            c_uint32(self.biasPointer),
            c_uint16(self.biasIndex),
            c_uint32(self.opParamsPointer),
            c_uint16(self.opParamsIndex),
            c_uint32(self.outputPointer),
            c_uint16(self.outputIndex),
            c_uint8(self.preOp.value),
            c_uint8(self.postOp.value),
            c_float(self.post_param1) if isinstance(self.post_param1, float) else c_int32(self.post_param1),
            c_ushort(self.post_strideX),
            c_ushort(self.post_strideY)
        ]

        for t in self.tail:
            t.set_blob_vars()

    def generate(self, f):
        """
        Writes this object (Layer description) to provided file.

        :param f: open file handler
        :return: byte_size # TODO: Unverified. Compare with binary_size output.
        """

        # We can probably make this smarter with this:
        # http://stackoverflow.com/questions/11296010/iterate-through-class-members-in-order-of-their-declaration

        sz = 0
        for item in self.write_items:
            f.write(item)
            sz += byte_size(item)

        # Don't recurse, we have a list of all the stages
        # for t in self.tail:
        #    sz += t.generate(f)

        return sz

    def binary_size(self):
        """
        Get byte size of structure when written to blob file
        Doesn't count our data arrays (we only want the pointers to that data on Myriad)
        :return: total size that will be written to blob file for this stage.
        """

        file_sizes = [
            byte_size(t) for t in self.write_items if isinstance(
                t, ctypes._SimpleCData) or isinstance(
                t, bytes)]
        return sum(file_sizes)

    def debug(self, to_file=False, f=None):
        """
        Developers can use this function to print values recursively through every layer.
        :param to_file: A field that could be used to write debug info to a file optionally.
        :param f: The corresponding filename if to_file is True
        :return: Nothing
        """
        if to_file:
            pass
        else:
            pass

        for t in self.tail:
            t.debug(to_file, f)

    def finalize(self):
        # Taps (and maybe bias) can be modified by batch normalization layer
        # that follows, so add them at the end
        self.putTaps()
        self.putBias()
        self.putOpParams()

        for t in self.tail:
            t.finalize()

    def check_algo(self, network):
        # Check if two layers write to the same output and one of them is convolution
        # Force im2col_v2 in this case, where it's normally never used
        if self.op == StageType.convolution and self.inputDimZ >= 200:
            if network.writes_output(self, self.outputIndex):
                if self.optMask & 0x80000000 == 0x80000000:
                    self.optMask = (self.optMask & 0x7fffffff) | 4
                    print(
                        'Layer ',
                        self.unprocessed_name,
                        ' forced to im2col_v2, because its output is used in concat')
        for t in self.tail:
            t.check_algo(network)

    def writes_output(self, exclude_layer, index):
        # Return True if this or a tail layer uses index as input
        for t in self.tail:
            if t.writes_output(exclude_layer, index):
                return True
        if self != exclude_layer and self.outputIndex == index:
            return True
        return False

    def assign_remnant_buffers(self, net):
        sizes = []
        offsets = []
        names = []

        if self.top is not None and isinstance(self.top[0], str):
            # This should ensure that the input strides are correct after
            # applying all concat rules.
            parent = net.search(self.top[0])

            self.inputStrideX = parent.outputStrideX
        if self.isoutput:
            sizes.append(self.output.shape)
            offsets.append(self.outputPointer)
            names.append(self.name)
        elif self.outputPointer == 0 and self.outputIndex == MemoryIndex.none.value and (self.top is None or len(self.top) <= 1 or get_class_of_op(self.op) != "Pooling"):
            self.outputIndex = MemoryIndex.output.value
            sizes.append(self.output.shape)
            offsets.append(self.outputPointer)
            names.append(self.name)
            self.isoutput = True
        elif self.outputPointer == 0 and self.outputIndex == MemoryIndex.none.value and len(self.top) > 1 and get_class_of_op(self.op) == "Pooling":
            node = net.head[0].search(self.top[0])
            self.output = np.zeros(
                (node.outputDimZ,
                 node.outputDimY,
                 node.outputDimX)).astype(np.float16)
            self.outputIndex = MemoryIndex.output.value
            sizes.append(self.output.shape)
            offsets.append(self.outputPointer)
            names.append(self.name)
            self.isoutput = True

        for t in self.tail:
            t_res = t.assign_remnant_buffers(net)
            sizes.extend(t_res[0])
            offsets.extend(t_res[1])
            names.extend(t_res[2])
        return sizes, offsets, names

    def convert_inputs_outputs_to_yxz(self, recurse, debug=False):
        """
        It is necessary to convert the first input because it contains data, but the other inputs
        and outputs are buffers for myriads use only. But, we will need to have the final outputs shaped correctly, so
        we will transform them too.
        :param recurse: Set to false to apply to a single element, true to traverse.
        :param debug: Set to true to enable debug print messages
        :return: Nothing. Side effect of transformed buffers.
        """
        if self.storageOrder == StorageOrder.orderYXZ:
            if debug:
                print("Already in this form")
        elif self.storageOrder == StorageOrder.orderZYX:
            if len(self.data.shape) == 4:
                self.data = np.reshape(
                    self.data, (self.data.shape[1], self.data.shape[2], self.data.shape[3]))
                self.data = zyx_to_yxz(
                    self.data, self.datatype).astype(
                    dtype=np.float16)
                self.storageOrder = StorageOrder.orderYXZ
            else:
                if not self.concatResult:
                    self.output = zyx_to_yxz(
                        self.output, self.datatype).astype(
                        dtype=np.float16)
                    self.storageOrder = StorageOrder.orderYXZ

        elif self.storageOrder == StorageOrder.orderXYZ:
            if len(self.data.shape) == 4:
                self.data = np.reshape(
                    self.data, (self.data.shape[1], self.data.shape[2], self.data.shape[3]))
                self.data = xyz_to_yxz(
                    self.data, self.datatype).astype(
                    dtype=np.float16)
                self.storageOrder = StorageOrder.orderYXZ
            else:
                throw_error(
                    ErrorTable.ConversionNotSupported,
                    self.storageOrder.name)

        else:
            throw_error(
                ErrorTable.ConversionNotSupported,
                self.storageOrder.name)

        if 0:
            self.data.tofile('InputTensor.bin')

        if recurse:
            for node in self.tail:
                node.convert_inputs_outputs_to_yxz(recurse)

    def convert_taps_to_hwck(self, recurse):
        if self.tapsOrder != TapsOrder.orderHWCK:
            if get_class_of_op(
                    self.op) in [
                    "Convolution",
                    "FCL",
                    "Deconvolution"]:
                if self.op in [StageType.fully_connected_layer]:
                    if self.unprocessed_h > 1 or self.unprocessed_w > 1:
                        self.taps = self.taps.reshape(
                            self.unprocessed_k,
                            self.unprocessed_c,
                            self.unprocessed_h,
                            self.unprocessed_w)
                    else:
                        self.taps = self.taps.reshape(
                            self.taps.shape[0], self.taps.shape[1], 1, 1)
                self.taps = kchw_to_hwck(self.taps)
                replace_buffer(self.taps, self.tapsBufferIndex, self.datatype)
            else:
                if (self.taps is None or
                        get_class_of_op(self.op) == "FCL" or
                        self.op == StageType.scale or
                        self.op == StageType.normalize):
                    pass
                else:
                    throw_error(
                        ErrorTable.ConversionNotSupported,
                        self.op.name)

            self.storageOrder = StorageOrder.orderYXZ.value
        if recurse:
            for node in self.tail:
                node.convert_taps_to_hwck(recurse)

    def getBWs(self):
        in_dim = self.data.flatten().shape[0]
        if self.taps is not None:
            tap_dim = self.taps.flatten().shape[0]
        else:
            tap_dim = 0
        out_dim = self.output.shape[0]

        KB = 1024
        MB = KB * KB

        MS = self.ms
        S = MS / 1000

        if self.op == StageType.convolution:
            arrays = in_dim * self.radixX * self.radixY  # Read Data NxN Times
            arrays += tap_dim                        # Taps once (already NxN)
            arrays += out_dim * self.radixX * self.radixY  # Accumulate NxN
        else:
            arrays = in_dim + tap_dim + out_dim
        self.BWs = ((arrays * 2) / MB) / S
        return self.BWs

    def getBW(self):
        in_dim = self.data.flatten().shape[0]
        if self.taps is not None:
            tap_dim = self.taps.flatten().shape[0]
        else:
            tap_dim = 0
        out_dim = self.output.shape[0]

        if self.op == StageType.convolution:
            arrays = in_dim * self.radixX * self.radixY  # Read Data NxN Times
            arrays += tap_dim                        # Taps once (already NxN)
            arrays += out_dim * self.radixX * self.radixY  # Accumulate NxN
        else:
            arrays = in_dim + tap_dim + out_dim

        return (arrays * 2)

    def minmax(self, attr, min, max):
        if min > getattr(self, attr):
            min = getattr(self, attr)
        if max < getattr(self, attr):
            max = getattr(self, attr)

        for t in self.tail:
            min, max = t.minmax(attr, min, max)

        return min, max

    def calculate_metrics(self, timings):
        self.flops = self.getFlops()
        self.ms = timings[0]
        self.BWs = self.getBWs()

    def getFlops(self):
        """

        :return:

        """
        flops = 0
        if self.op == StageType.convolution:
            # Output channels too.
            flops = self.unprocessed_k * self.outputDimX * self.outputDimY * \
                self.inputDimZ * self.radixX * self.radixY * 2

        elif self.op == StageType.max_pooling:
            flops = self.unprocessed_k * self.outputDimX * \
                self.outputDimY * self.radixX * self.radixY

        elif self.op == StageType.average_pooling:
            flops = self.unprocessed_k * self.outputDimX * \
                self.outputDimY * self.radixX * self.radixY * 2

        elif self.op == StageType.fully_connected_layer:
            in_dim = self.data.flatten().shape[0]
            out_channels = self.output.flatten().shape[0]
            flops = in_dim * out_channels * 2

        elif self.op == StageType.depthwise_convolution:
            flops = self.radixX * self.radixY * self.unprocessed_k * \
                self.outputDimX * self.outputDimY * 2

        elif self.op == StageType.soft_max:
            in_dim = self.data.flatten().shape[0]
            flops = in_dim * 3

        return flops / 1000000

    def summaryStats(self):
        totalTime = self.ms
        totalBW = self.getBW()

        for t in self.tail:
            a, b = t.summaryStats()
            totalTime += a
            totalBW += b

        return totalTime, totalBW

    def newick(self, head=False):
        """
        output a file containing a description of the node in Newick format.
        To be called from Network.head and traversed.
        :param head:
        :return:
        """
        nw = str(self.unprocessed_name) + ":" + str(len(self.tail))

        if len(self.tail) != 0:
            nw += ",("
            for idx, t in enumerate(self.tail):
                nw += t.newick()
                if idx + 1 != len(self.tail):
                    nw += ","
            nw += ")"
        else:
            pass
        return nw

    def graphviz(
            self,
            dot,
            ms_min,
            ms_max,
            bws_min,
            bws_max,
            flop_min,
            flop_max):

        table = '''<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="3">
<TR>
    <TD  BGCOLOR = "#A3A3A3" COLSPAN="3">{0}</TD>
</TR>
<TR>
    <TD  BGCOLOR = "#DDDDDD" COLSPAN="3">{7}</TD>
</TR>
<TR>
    <TD BGCOLOR = "{1}"> {2} <br/> (MFLOPs) </TD>
    <TD BGCOLOR = "{3}"> {4} <br/> (MB/s) </TD>
    <TD BGCOLOR = "{5}"> {6} <br/> (ms)</TD>
</TR>
</TABLE>>
'''.format(
            self.unprocessed_name, get_normalized_color(
                "#B1F1EF", "#2ED1C6", flop_min, flop_max, self.flops), self.flops, get_normalized_color(
                "#FFE5FC", "#B2189E", bws_min, bws_max, format(
                    self.BWs, ".2f")), format(
                    self.BWs, ".2f"), get_normalized_color(
                        "#FFFFCC", "#FFFF00", ms_min, ms_max, format(
                            self.ms, ".2f")), format(
                                self.ms, ".2f"), str(
                                    self.unprocessed_output.shape))

        dot.node(self.unprocessed_name, table, shape="plaintext")
        if self.top is not None:
            for t in self.top:
                if not isinstance(t, str):
                    for tt in t:
                        dot.edge(tt, self.unprocessed_name)
                else:
                    dot.edge(t, self.unprocessed_name)

        else:
            dot.edge("Input", self.unprocessed_name)

        last_nodes = []
        for t in self.tail:
            dot, last = t.graphviz(
                dot, ms_min, ms_max, bws_min, bws_max, flop_min, flop_max)
            last_nodes.extend(last)
        if len(self.tail) == 0:
            last_nodes = [self.unprocessed_name]

        return dot, last_nodes
