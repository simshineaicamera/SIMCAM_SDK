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

import datetime
from graphviz import Digraph
import math
import warnings
from Controllers.EnumController import *
import numpy as np


def get_normalized_color(start_color, end_color, start_no, end_no, value):
    """
    Gets a color relative to a numbers position in a range. (Heatmap)
    :param start_color: First color in RGB e.g. #43F3FC
    :param end_color: Last color in range
    :param start_no: Start of range
    :param end_no: End of range
    :param value: value within range
    :return: normalized RGB value in the form #RRGGBB
    """
    a_r, a_g, a_b = int(start_color[1:3], 16), int(
        start_color[3:5], 16), int(start_color[5:], 16)
    b_r, b_g, b_b = int(end_color[1:3], 16), int(
        end_color[3:5], 16), int(end_color[5:], 16)

    # There's some subtle difference in type when truncated.
    value = float(value)
    if end_no - start_no == 0:
        return "#FFFFFF"
    percentage = (value - start_no) / (end_no - start_no)

    r_diff = b_r - a_r
    g_diff = b_g - a_g
    b_diff = b_b - a_b

    adjusted_r = r_diff * percentage + a_r
    adjusted_g = g_diff * percentage + a_g
    adjusted_b = b_diff * percentage + a_b

    invalid_values = [float('NaN'), float('Inf')]
    if math.isnan(adjusted_r) or math.isnan(
            adjusted_b) or math.isnan(adjusted_g):
        warnings.warn("Non-Finite value detected", RuntimeWarning)
        return "#%X%X%X" % (122, 122, 122)

    return "#%X%X%X" % (int(adjusted_r), int(adjusted_g), int(adjusted_b))


def generate_graphviz(net, blob, filename="output"):
    """
    Generate a graphviz representation of the network.
    :param net
    :param blob:
    :param filename: Name of output file
    :return:
    """

    print("Generating Profile Report '" + str(filename) + "_report.html'...")
    dot = Digraph(name=filename, format='svg')

    # Legend
    table = '''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="3">
<TR><TD  BGCOLOR = "#E0E0E0" COLSPAN="3">Layer</TD></TR>
<TR><TD BGCOLOR = "#88FFFF"> Complexity <br/> (MFLOPs) </TD>
<TD BGCOLOR = "#FF88FF"> Bandwidth <br/> (MB/s) </TD>
<TD BGCOLOR = "#FFFF88"> Time <br/> (ms)</TD></TR>
</TABLE>>
'''
    dot.node("Legend", table, shape="plaintext")

    # Input
    table = '''input: {}'''.format(net.inputTensor.shape)
    dot.node('''Input''', table)

    # Nodes
    ms_min, ms_max = net.head[0].minmax("ms", net.head[0].ms, net.head[0].ms)
    for stage in net.head:
        ms_min, ms_max = stage.minmax("ms", ms_min, ms_max)
    bws_min, bws_max = net.head[0].minmax(
        "BWs", net.head[0].BWs, net.head[0].BWs)
    for stage in net.head:
        bws_min, bws_max = stage.minmax("BWs", bws_min, bws_max)
    flop_min, flop_max = net.head[0].minmax(
        "flops", net.head[0].flops, net.head[0].flops)
    for stage in net.head:
        flop_min, flop_max = stage.minmax("flops", flop_min, flop_max)
    last_nodes = []
    for stage in net.head:
        dot, last = stage.graphviz(
            dot, ms_min, ms_max, bws_min, bws_max, flop_min, flop_max)
        last_nodes.extend(last)

    # Output
    channels = 0
    for shape in net.outputInfo[0]:
        channels = channels + shape[2]
    table = '''output: {}'''.format(
        [net.outputInfo[0][0][0], net.outputInfo[0][0][1], channels])
    dot.node("Output", table)
    for node in last_nodes:
        if net.search(node).isoutput:
            dot.edge(node, "Output")

    total_time = 0
    total_bw = 0
    for stage in net.head:
        time, bw = stage.summaryStats()
        total_time += time
        total_bw += bw

    # Summary
    table = '''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="3">
<TR><TD  BGCOLOR = "#C60000" COLSPAN="3">Summary</TD></TR>
<TR><TD  BGCOLOR = "#E2E2E2" COLSPAN="3">{0} SHV Processors</TD></TR>
<TR><TD  BGCOLOR = "#DADADA" COLSPAN="3">Inference time {1} ms</TD></TR>
<TR><TD  BGCOLOR = "#E2E2E2" COLSPAN="3">Bandwidth {2} MB/sec</TD></TR>
<TR><TD  BGCOLOR = "#DADADA" COLSPAN="3">This network is Compute bound</TD></TR>
</TABLE>>
'''.format((blob.myriad_params.lastShave.value - blob.myriad_params.firstShave.value) + 1, format(total_time, ".2f"), format((((total_bw / (1024 * 1024)) / (total_time / 1000))), ".2f"))

    dot.node("Summary", table, shape="plaintext")
    dot.render()

    generate_html_report(filename + ".gv.svg", net.name, filename=filename)


def generate_ete(blob):
    print("Currently does not work alongside caffe integration due to GTK conflicts")
    """
from ete3 import *
f = open("Graph.txt")
graph = f.read()
f.close()
print(graph)
t = Tree(graph)
t.show()
    """


def dataurl(file):
    """
    Converts image to a base64 encoded address which can be used inline in html.
    (page wont change if you delete the image file)
    :param file:
    :return:
    """
    import base64
    encoded = base64.b64encode(open(file, "rb").read())
    return "data:image/svg+xml;base64," + str(encoded)[2:-1]


def generate_html_report(graph_filename, network_name, filename="output"):
    html_start = """
<html>
<head>
"""
    css = """
<style>
.container{
   text-align: center;
}
h3{
    font-weight: 100;
    font-size: x-large;
}
#mvNCLogo, #ReportImage{
   margin: auto;
   display: block;
}
#mvNCLogo{
   width: 300px;
   padding-left: 50px;
}
#ReportImage{
   width: 60%;
}
.infobox{
    text-align: left;
    margin-left: 2%;
    font-family: monospace;
}
</style>
"""
    html_end = """
</head>
<body>

<div class="container">
    <img id="MovidiusLogo" src="MovidiusLogo.png" />
    <hr />
    <h3> Network Analysis </h3>
    <div class="infobox">
        <div> Network Model: <b> """ + network_name + """ </b> </div>
        <div> Generated on: <b>  """ + datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + """ </b> </div>
    </div>
    <img id="ReportImage" src=" """ + dataurl(graph_filename) + """ " />
</div>

</body>
</html>
    """
    document = html_start + css + html_end
    f = open(filename + '_report.html', 'w')
    f.write(document)
    f.close()


def generate_temperature_report(data, filename="output"):
    tempBuffer = np.trim_zeros(data)

    if tempBuffer.size == 0:
        throw_error(ErrorTable.NoTemperatureRecorded)

    print(tempBuffer)
    print("Average Temp", np.mean(tempBuffer))
    print("Peak Temp", np.amax(tempBuffer))

    try:
        import matplotlib.pyplot as plt
        plt.plot(tempBuffer)
        plt.ylabel('Temp')
        # plt.show()
        plt.savefig(filename + "_temperature.png")
    except BaseException:
        pass
