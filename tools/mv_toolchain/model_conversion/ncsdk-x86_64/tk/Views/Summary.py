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

from Models.EnumDeclarations import *
import math
import shutil

g_total_time = 0
number = 0


def print_summary_of_nodes(
        node,
        w_stages,
        w_terminal,
        w_layer,
        w_float,
        w_name):
    """
    Commandline print of nodes and their statistics.
    :param number:
    :param node:
    :return:
    """
    global g_total_time
    global number

    formatted_name = node.unprocessed_name[slice(-w_name, None)]
    if len(node.unprocessed_name) > w_name:
        formatted_name = '...' + formatted_name[3:]
    print("{}{}{}{}{}".format(
        ("{}".format(str(number)).ljust(w_layer)),
        ("{}".format(formatted_name).ljust(w_name)),
        ("{:6.1f}".format(node.flops).rjust(w_float)),
        ("{:6.1f}".format(node.BWs).rjust(w_float)),
        ("{:4.3f}".format(node.ms).rjust(w_float))))

    number += 1
    g_total_time += node.ms


def print_summary_of_network(blob_file):
    """
    Print timings related to the flattened blob file.
    :param blob_file:
    :return:
    """
    global g_total_time

    # Formatting parameters
    w_stages = math.ceil(math.log10(len(blob_file.network.stageslist)))
    w_terminal = shutil.get_terminal_size()[0]
    w_layer = w_stages + 3
    w_float = 8
    w_name = w_terminal - w_layer - 3 * w_float - 1
    w_total = w_layer + w_name + 3 * w_float

    print("Network Summary")

    if False:  # == NetworkLimitation.DDR_Speed_Bound:   # Compare to something
        print("This network is bound by the speed of DDR. Consider using smaller datatypes or reducing your data size.")

    print("\nDetailed Per Layer Profile")

    # Header
    print(
        "{}{}{}{}{}\n{}{}{}{}{}".format(
            ''.ljust(w_layer),
            ''.ljust(w_name),
            ''.rjust(w_float),
            'Bandwidth'.rjust(w_float),
            'time'.rjust(
                w_float - 1),
            '#'.ljust(w_layer),
            'Name'.ljust(w_name),
            'MFLOPs'.rjust(w_float),
            '(MB/s)'.rjust(w_float),
            '(ms)'.rjust(w_float)))
    print(''.join(['=' for i in range(w_total)]))

    # Layers
    for stage in blob_file.network.stageslist:
        print_summary_of_nodes(
            stage,
            w_stages,
            w_terminal,
            w_layer,
            w_float,
            w_name)

    # Footer
    print(''.join(['-' for i in range(w_total)]))
    print(
        '{}{}'.format(
            'Total inference time'.rjust(
                w_layer + w_name),
            '{:.2f}'.format(g_total_time).rjust(
                3 * w_float)))
    print(''.join(['-' for i in range(w_total)]))
