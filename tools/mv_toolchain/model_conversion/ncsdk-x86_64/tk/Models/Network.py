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

from Controllers.MiscIO import *
import numpy as np
from Models.NetworkStage import *
import ctypes


class Network:
    def __init__(self, name, data):
        self.name = name
        self.head = []
        self.count = 0
        self.stageslist = []

        # self.data = data
        # self.dataPointer = 0

        self.outputInfo = None
        self.outputNeedsTransforming = False
        self.outputTensor = None

        self.outputIsSsdDetOut = False

        self.inputTensor = data
        self.datatype = DataType.fp16

    def attach(self, stage, debug=False):
        """
        Attaches to the top of the network tree, or if already filled, finds
        the appropriate node to attach to.
        Restriction: Names MUST be unique and parents must already be attached.
        :param stage: Stage to attach
        :param debug: enable some debug messages
        :return: 1 if successful, 0 if not
        """
        self.stageslist.append(stage)
        if len(self.head) == 0:
            self.head.append(stage)
            stage.data = self.inputTensor
            stage.dataIndex = MemoryIndex.input.value
            self.storageOrder = stage.storageOrder

            self.count = 1
            if debug:
                print("attached.")
            return 1        # Attached to head
        else:
            if stage.top is None:
                stage.data = self.inputTensor
                stage.dataIndex = MemoryIndex.input.value
                self.head.append(stage)
                self.count += 1
            elif len(stage.top) > 1:
                appropriate_nodes = self.search_several(stage.top)
                #stage.attach_eltwise(appropriate_nodes)
                stage.attach_multiple_bottoms(appropriate_nodes)
                if debug:
                    print("attached.")
                self.count += 1
            else:
                parent = stage.top[0]
                appropriate_nodes = self.search_several(parent)
                if appropriate_nodes == 0:
                    throw_error(ErrorTable.GraphConstructionFailure, parent)
                else:
                    stage.attach_several(appropriate_nodes)
                    self.count += 1
                    if debug:
                        print("attached.")

            return 1    # Attached to appropiate node in tree.

    def search(self, seek_name):
        """
        Forwarder for tree search of the network tree.
        return: 0 if not found. The searched node if found.
        :param seek_name: name of node, without padded characters
        """
        if seek_name == 0:
            throw_error(ErrorTable.GraphConstructionFailure, seek_name)

        for stage in self.head:
            ret = stage.search(seek_name)
            if ret != 0:
                return ret
        return 0

    def search_several(self, seek_names):
        # This will also work with one seek_names as string
        """

        :param seek_names:
        :return:
        """
        if isinstance(seek_names, str):
            return self.search(seek_names)
        nodes = []
        for name in seek_names:
            # name can be a name or a sequence of names, if it's a concat
            if isinstance(name, str):
                nodes.append(self.search(name))
            else:
                nodes.append(self.search_several(name))

        return nodes

    def generate_info(self, f):
        """
        Writes the information section of the blob file.
        :param f:
        :return:
        """
        sz = 0
        # The stages have to be processed in the order they have been
        # created, not in a tree-based order, otherwise we risk not
        # respecting dependencies
        for stage in self.stageslist:
            sz += stage.generate(f)

        for nul in range(align(sz, np.zeros((1)), align_to=8)[0] - sz):
            # Fill in some padding to align the start of the weights
            f.write(c_char(0))

    def generate_data(self, f):
        """
        Writes the data section of the blob file.
        :param f:
        :return:
        """
        write_data(f)

    def debug(self):
        """
        Print layer information from each node in the network
        :return:
        """
        for stage in self.head:
            stage.debug()
        pass

    def finalize(self):
        """
        Run any actions that need to happen after network is constructed.
        :return:
        """
        sizes = []
        pointers = []
        names = []
        # Go through all output pointers and make sure they are assigned.
        for stage in self.head:
            t_res = stage.assign_remnant_buffers(self)
            sizes.extend(t_res[0])
            pointers.extend(t_res[1])
            names.extend(t_res[2])
        # self.debug()
        self.outputInfo = (sizes, pointers, names)
        self.check_algo()
        for stage in self.head:
            stage.finalize()
            stage.set_blob_vars()

    def check_algo(self):
        """
        Force im2col_v2 when convolutions are concatenated, because otherwise
        other versions could be used, which write outside their buffer
        """
        for stage in self.head:
            stage.check_algo(self)

    def writes_output(self, exclude_layer, index):
        """
        Return true if there is at least one layer which writes to an output
        with index; used by check_algo
        """
        for stage in self.head:
            if stage.writes_output(exclude_layer, index):
                return True
        return False

    def optimize(self):
        """
        Convert into our ideal representation for myriad
        :return: Nothing
        """
        self.convert_network_input_to_yxz()
        for stage in self.head:
            stage.convert_inputs_outputs_to_yxz(True)
            stage.convert_taps_to_hwck(True)    # recursively
        for idx, out_node in enumerate(self.outputInfo[0]):
            self.outputInfo[0][idx] = (out_node[2], out_node[1], out_node[0])

        # Horizontally combine any available 1x1s
        # for stage in self.head:
        #     stage.combine1x1()
        #     quit()

        self.outputNeedsTransforming = True

    def gather_metrics(self, timings):
        prev_len = 0
        # The stages have to be processed in the order they have been
        # created, not in a tree-based order, otherwise we risk not
        # respecting dependencies
        for stage in self.stageslist:
            stage.calculate_metrics(timings[prev_len:])
            prev_len = prev_len + 1

    def convert_network_input_to_yxz(self, debug=False):
        """
        It is necessary to convert the first input because it contains data, but the other inputs
        and outputs are buffers for myriads use only. But, we will need to have the final outputs shaped correctly, so
        we will transform them too.
        :param recurse: Set to false to apply to a single element, true to traverse.
        :param debug: Set to true to enable debug print messages
        :return: Nothing. Side effect of transformed buffers.
        """
        if self.storageOrder.value == StorageOrder.orderYXZ.value:
            if debug:
                print("Already in this form")
        elif self.storageOrder.value == StorageOrder.orderZYX.value:
            if len(self.inputTensor.shape) == 4:
                self.inputTensor = np.reshape(
                    self.inputTensor,
                    (self.inputTensor.shape[1],
                     self.inputTensor.shape[2],
                     self.inputTensor.shape[3]))
                self.inputTensor = zyx_to_yxz(
                    self.inputTensor,
                    self.datatype.value).astype(
                    dtype=np.float16)
                self.storageOrder = StorageOrder.orderYXZ
            else:
                self.inputTensor = zyx_to_yxz(
                    self.inputTensor,
                    self.datatype.value).astype(
                    dtype=np.float16)
                self.storageOrder = StorageOrder.orderYXZ
        elif self.storageOrder.value == StorageOrder.orderXYZ.value:
            if len(self.inputTensor.shape) == 4:
                self.inputTensor = np.reshape(
                    self.inputTensor,
                    (self.inputTensor.shape[1],
                     self.inputTensor.shape[2],
                     self.inputTensor.shape[3]))
                self.inputTensor = xyz_to_yxz(
                    self.inputTensor,
                    self.datatype.value).astype(
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

    def verify(self):
        """
        Calls verify() on the top of the network to be recursed down
        :return:
        """
        for stage in self.head:
            stage.verify()

    def newick(self):
        """
        Outer wrapper for newick format.
        :return:
        """
        # To review
        nw = "( "
        for idx, t in enumerate(self.head):
            nw += t.newick(head=True)
            if idx + 1 != len(self.head):
                nw += ","
        nw += " );"
        return nw
