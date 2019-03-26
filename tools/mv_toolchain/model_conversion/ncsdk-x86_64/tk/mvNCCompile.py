#! /usr/bin/env python3

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

import os
import sys
import argparse
import numpy as np
from Controllers.EnumController import *
from Controllers.FileIO import *
from Models.Blob import *
from Models.EnumDeclarations import *
from Models.MyriadParam import *

major_version = np.uint32(2)
release_number = np.uint32(0)


def parse_args():
    parser = argparse.ArgumentParser(description="mvNCCompile.py converts Caffe or Tensorflow networks to graph files\n" +
                                     "that can be used by the Movidius Neural Compute Platform API")
    parser.add_argument('network', type=str, help='Network file (.prototxt, .meta, .pb, .protobuf)')
    parser.add_argument('-w', dest='weights', type=str, help='Weights file (override default same name of .protobuf)')
    parser.add_argument('-in', dest='inputnode', type=str, help='Input node name')
    parser.add_argument('-on', dest='outputnode', type=str, help='Output node name')
    parser.add_argument('-o', dest='outfile', type=str, default='graph', help='Generated graph file (default graph)')
    parser.add_argument('-s', dest='nshaves', type=int, default=1, help='Number of shaves (default 1)')
    parser.add_argument('-is', dest='inputsize', nargs=2, type=int, help='Input size for networks that don\'t provide an input shape, width and height expected')
    args = parser.parse_args()
    return args


class Arguments:
    def __init__(self, network, inputnode, outputnode, outfile, inputsize, nshaves, weights):
        self.net_description = network
        filetype = network.split(".")[-1]
        self.parser = Parser.TensorFlow
        if filetype in ["prototxt"]:
            self.parser = Parser.Caffe
            if weights is None:
                weights = network[:-8] + 'caffemodel'
                if not os.path.isfile(weights):
                    weights = None
        self.conf_file = network[:-len(filetype)] + 'conf'
        if not os.path.isfile(self.conf_file):
            self.conf_file = None
        self.net_weights = weights
        self.input_node_name = inputnode
        self.output_node_name = outputnode
        self.input_size = inputsize
        self.number_of_shaves = nshaves
        self.image = 'Debug'
        self.raw_scale = None
        self.outputs_name = None
        self.mean = None
        self.channel_swap = None
        self.explicit_concat = False
        self.acm = 0
        self.timer = None
        self.number_of_iterations = 2
        self.upper_temperature_limit = -1
        self.lower_temperature_limit = -1
        self.backoff_time_normal = -1
        self.backoff_time_high = -1
        self.backoff_time_critical = -1
        self.temperature_mode = 'Advanced'
        self.network_level_throttling = 1
        self.stress_full_run = 1
        self.stress_usblink_write = 1
        self.stress_usblink_read = 1
        self.debug_readX = 100
        self.mode = 'generation'
        self.outputs_name = 'output'
        self.save_input = None
        self.save_output = None
        self.device_no = None

def create_graph(network, inputnode = None, outputnode = None, outfile = 'graph', nshaves = 1, inputsize = None, weights = None):
    file_init()
    args = Arguments(network, inputnode, outputnode, outfile, inputsize, nshaves, weights)
    myriad_config = MyriadParam(0, nshaves - 1)
    if args.conf_file is not None:
        get_myriad_info(args, myriad_config)
    filetype = network.split(".")[-1]
    if filetype in ["prototxt"]:
        from Controllers.CaffeParser import parse_caffe
        net = parse_caffe(args, myriad_config)
    elif filetype in ["pb", "protobuf", "meta"]:
        from Controllers.TensorFlowParser import parse_tensor
        net = parse_tensor(args, myriad_config)
    else:
        throw_error(ErrorTable.ParserNotSupported)
    net.finalize()
    net.optimize()
    graph_file = Blob(major_version, net.name, '', myriad_config, net, outfile)
    graph_file.generate()


if __name__ == "__main__":
    print("\033[1mmvNCCompile v" + (u"{0:02d}".format(major_version, )) + "." +
          (u"{0:02d}".format(release_number, )) +
          ", Copyright @ Movidius Ltd 2016\033[0m\n")
    args = parse_args()
    create_graph(args.network, args.inputnode, args.outputnode, args.outfile, args.nshaves, args.inputsize, args.weights)
