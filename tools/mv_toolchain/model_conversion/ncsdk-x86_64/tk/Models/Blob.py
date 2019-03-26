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
from Controllers.FileIO import *
import sys


class Blob:
    def __init__(
            self,
            version,
            name,
            report_dir,
            myriad_params,
            network,
            blob_name):
        """
        This object contains all the information required for a blob file + some additional info for processing.
        :param version: The version of the toolkit used to generate this blob. Useful for the potential of
         having backwards compatibility - although it's easier to regenerate your file.
        :param name: Name of the network represented in the blob file
        :param report_dir: Where to output our reports (Note: TODO)
        :param myriad_params: Myriad configurations (Note: TODO)
        :param network: A Network object to attach to the blob.
        :return:
        """
        self.version = c_uint32(version)
        self.filesize = c_uint32(16)
        self.name = set_string_range(name, 100).encode('ascii')
        self.report_dir = set_string_range(report_dir, 100).encode('ascii')
        self.myriad_params = myriad_params
        self.network = network
        self.stage_count = c_uint32(self.network.count)
        self.VCS_Fix = True
        self.blob_name = blob_name

    def generate(self):
        """
        Generates the actual blob file.
        :return:
        """
        with open(self.blob_name, 'wb') as f:
            if self.VCS_Fix:
                f.write(c_uint64(0))    # VCS incorrectly reads first 16 Bytes
                f.write(c_uint64(0))    # VCS incorrectly reads first 16 Bytes
                f.write(c_uint64(0))    # VCS incorrectly reads first 16 Bytes
                f.write(c_uint64(0))    # VCS incorrectly reads first 16 Bytes
            f.write(estimate_file_size(self))
            f.write(self.version)

            f.write(self.name)
            f.write(self.report_dir)
            f.write(self.stage_count)
            f.write(get_buffer_start(self))    # Size of a network element.
            self.myriad_params.generate(f)
            self.network.generate_info(f)
            self.network.generate_data(f)
