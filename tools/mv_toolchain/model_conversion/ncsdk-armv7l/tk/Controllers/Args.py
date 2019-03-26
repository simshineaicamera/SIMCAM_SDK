import os
import argparse

from Controllers.EnumController import *
from Controllers.MiscIO import *


class FathomArguments:
    """
    Container for Fathom Arguments.
    """

    def ensure_arg_compatibility(self):
        # General Checks
        if self.mode not in [OperationMode.optimization_list]:  # Don't care about this argument for just print modes
            path_check(self.net_description, ErrorTable.ArgumentErrorDescription)

        if self.image == "Debug":
            self.image = None
            return

        if self.mode in [OperationMode.validation, OperationMode.test_validation]:
            if self.expected_index is None and self.validation_type in [ValidationStatistic.top1, ValidationStatistic.top5]:
                throw_error(ErrorTable.ArgumentErrorExpID)

            if self.image is None or self.image is "None" or not os.path.isfile(self.image):
                throw_error(ErrorTable.ArgumentErrorImage)
            if self.validation_type == ValidationStatistic.invalid:
                throw_error(ErrorTable.ValidationSelectionError, self.validation_type)
        if self.parser in [Parser.TensorFlow]:
            if self.image == "None":
                self.image = None
                return
            if self.image is None or not os.path.isfile(self.image):
                throw_error(ErrorTable.ArgumentErrorImage)

        if self.number_of_shaves < 1 or self.number_of_shaves > 12:
            throw_error(ErrorTable.InvalidNumberOfShaves)

        return


def coords(s):
    try:
        x, y, z = map(int, s.split(','))
        return x, y, z
    except:
        throw_error(ErrorTable.TupleSyntaxWrong)


def usage_msg(name=None):
    return '''Fathom
         Show this help:
            Fathom help
         Version:
            Fathom version

         Fathom [generate|validate|profile] ...

         Generate:
            * --network-description [...]
            ? --network-weights [...]
            ? --output-node-name [...]
            ? --input-node-name [...]
            ? --output-location [...]
            ? --output-name [...]
            ? --parser [caffe|tensorflow]
            ? --raw-scale [...]
            ? --mean [...]
            ? --channel-swap [x,y,z]
         Validate:
            * --network-description [...]
            * --image [...]
            * --output-validation-type [top1|top5]
            ? --network-weights [...]
            ? --output-node-name [...]
            ? --input-node-name [...]
            ? --output-expected-id [...]
            ? --output-location [...]
            ? --output-name [...]
            ? --parser [caffe|tensorflow]
            ? --num-shaves [1-12]
            ? --raw-scale [...]
            ? --mean [...]
            ? --channel-swap [x,y,z]
         Profile:
            * --network-description [...]
            ? --network-weights [...]
            ? --output-node-name [...]
            ? --input-node-name [...]
            ? --output-location [...]
            ? --output-name [...]
            ? --parser [caffe|tensorflow]
            ? --num-shaves [1-12]
        '''


def define_and_parse_args():
    # Argument Checking
    parser = argparse.ArgumentParser(description="""Fathom is Movidius\'s machine learning software framework. \n
Fathom converts trained offline neural networks into embedded neural networks running on the \
ultra-low power Myriad 2 VPU.\nBy targeting Myriad 2, \
Fathom makes it easy to profile, tune and optimize your standard TensorFlow or Caffe neural network. """,
                                     formatter_class=argparse.RawTextHelpFormatter, add_help=False, usage=usage_msg())

    # Mandatory
    parser.add_argument('mode', metavar='', type=str, nargs='?',
                        help='[0] Fathom Operational Mode')

    # General Parameters
    parser.add_argument('--output-name', metavar='', type=str, nargs='?',
                        help='Name to be used for any output files your stage may produce')
    parser.add_argument('--output-location', metavar='', type=str, nargs='?',
                        help='Path to the location to store any output files your stage may produce')
    parser.add_argument('--num-shaves', metavar='', type=str, nargs='?',
                        help='Number of SHV processors to use. Default: 1, Max: 12')

    # Source of Network Configurations
    parser.add_argument('--network-description', metavar='', type=str, nargs='?',
                        help='Relative path to network description file. Typical usage is for caffe prototxt file' +
                        ' or TensorFlow .pb file')
    parser.add_argument('--network-weights', metavar='', type=str, nargs='?',
                        help='Relative path to network weights file. Typical usage is for caffe caffemodel file' +
                        ' is not used for TensorFlow')
    parser.add_argument('--blob-file', metavar='', type=str, nargs='?',
                        help='Path to existing blob file to use for processing.')
    # TODO: Integrate into network-description
    parser.add_argument('--output-node-name', metavar='', type=str, nargs='?',
                        help='Name of final output node in the graph')
    parser.add_argument('--input-node-name', metavar='', type=str, nargs='?',
                        help='Name of first input node in the graph')

    # Additional arguments for each mode
    parser.add_argument('--parser', metavar='', type=str, nargs='?',
                        help="Choose Type of Network Parser (Caffe/TensorFlow)")

    parser.add_argument('--ACM', metavar='', type=str, nargs='?',
                        help="change ACM value")
    parser.add_argument('--device-identifier', metavar='', type=str, nargs='?',
                        help="Experimental flag to run on a specified stick.")

    # Experimental Options (USE AT YOUR OWN RISK. COULD DAMAGE YOUR STICK)
    parser.add_argument('--run-several', metavar='', type=str, nargs='?',
                        help="Run FathomRun X times. For experimental use only. Use at your own risk.")
    parser.add_argument('--tmp-upper-lim', metavar='', type=str, nargs='?',
                        help="The temperature at which to mark Fathom as 'High' Temperature . For experimental use only. Use at your own risk.")
    parser.add_argument('--tmp-lower-lim', metavar='', type=str, nargs='?',
                        help="The temperature at which to mark Fathom as 'Critical' Temperature. For experimental use only. Use at your own risk.")
    parser.add_argument('--backoff-normal', metavar='', type=str, nargs='?',
                        help="The duration in MS at which FathomRun will wait during normal temperatures. For experimental use only. Use at your own risk.")
    parser.add_argument('--backoff-high', metavar='', type=str, nargs='?',
                        help="The duration in MS at which FathomRun will wait during high temperatures. For experimental use only. Use at your own risk.")
    parser.add_argument('--backoff-crit', metavar='', type=str, nargs='?',
                        help="The duration in MS at which FathomRun will wait during critical temperatures. For experimental use only. Use at your own risk.")
    parser.add_argument('--debug-readX', metavar='', type=str, nargs='?',
                        help="The amount of values we read in temp mode. For experimental use only. Use at your own risk.")
    parser.add_argument('--temperature-mode', metavar='', type=str, nargs='?',
                        help="Whether Simple or Advanced Temperature Mode. For experimental use only. Use at your own risk.")
    parser.add_argument('--stress-usblink-read', metavar='', type=str, nargs='?',
                        help="Run Multiple iterations of reading with USBLink. For experimental use only. Use at your own risk.")
    parser.add_argument('--stress-usblink-write', metavar='', type=str, nargs='?',
                        help="Run Multiple iterations of writing with USBLink. For experimental use only. Use at your own risk.")
    parser.add_argument('--stress-full-run', metavar='', type=str, nargs='?',
                        help="Run Multiple iterations of a typical read-process-write scenario. For experimental use only. Use at your own risk.")
    parser.add_argument('--stress-boot-run', metavar='', type=str, nargs='?',
                        help="Run Multiple iterations of a typical boot-read-process-write scenario. For experimental use only. Use at your own risk.")
    parser.add_argument('--stress-boot-init', metavar='', type=str, nargs='?',
                        help="Boot and init N Times. For experimental use only. Use at your own risk.")
    parser.add_argument('--timer', action="store_true",
                        help="Time USBLink.")
    parser.add_argument('--disable-query', action="store_true",
                        help="Uses defaults for MvTensor queries, and so allows the use of USBLinkDebug")
    parser.add_argument('--set-network-level-throttling', action="store_true",
                        help="Change to Network Level throttling.")
    parser.add_argument('--set-no-throttling', action="store_true",
                        help="Change to Network Level throttling.")
    parser.add_argument('--explicit-concat', action="store_true",
                        help="Enable explicit concat layer.")

    # Validation
    parser.add_argument('--output-expected-id', metavar='', type=str, nargs='?',
                        help='Expected Classification Result for Validation comparison.')
    parser.add_argument('--output-validation-type', metavar='', type=str, nargs='?',
                        help='Pass/Fail Criteria (Top-1)/(Top-5)')
    parser.add_argument('--class-test-threshold', metavar='', type=str, nargs='?',
                        help='Change the threshold amount for significant classes.')

    parser.add_argument('--image', metavar='', type=str, nargs='?',
                        help='Image to use in operation')
    parser.add_argument('--raw-scale', metavar='', type=str, nargs='?',
                        help='Raw Scale Operation - multiplied by your image input. Default 1')
    parser.add_argument('--mean', metavar='', type=str, nargs='?',
                        help='Path to Mean file - Subtract from image input channels. Default None')

    parser.add_argument('--conf-file', metavar='', type=str, nargs='?',
                        help='Configuration File to use. Default: optimisation.conf')

    parser.add_argument('--ma2480', action="store_true",
                        help="Dev flag to enable MXHWGen")
    parser.add_argument('--save-input', metavar='', type=str, nargs='?', const="InputTensor.bin",
                        default=None, help='Save input tensor to file. Default: InputTensor.bin')
    parser.add_argument('--save-output', metavar='', type=str, nargs='?', const="OutputTensor.bin",
                        default=None, help='Save output tensor to file. Default: OutputTensor.bin')
    parser.add_argument('--channel-swap', type=coords,
                        help="Coordinate", nargs='?')
    parser.add_argument('--input-size', type=int,
                        help="Rescale input image to this size if network does not provide an input size", nargs='?')

    # Parse Arguments into variables that we can use.
    try:
        args = parser.parse_args()
    except:
        throw_error(ErrorTable.ArgumentErrorRequired)

    if args.mode is None:
        throw_error(ErrorTable.ArgumentErrorRequired)

    if args.mode == "version" or args.mode[0] == "version":
        quit()
    if args.mode == "help" or args.mode[0] == "help":
        print(usage_msg())
        quit()


    fa = FathomArguments()


    fa.mode = parse_mode(args.mode)

    # Details for Location of Output Files
    fa.outputs_location = path_arg(args.output_location)
    fa.outputs_name = args.output_name if args.output_name is not None else "Fathom"

    fa.device_no = args.device_identifier

    fa.number_of_shaves = int(args.num_shaves) if args.num_shaves is not None else 1
    fa.number_of_iterations = int(args.run_several) if args.run_several is not None else 2

    fa.upper_temperature_limit = int(args.tmp_upper_lim) if args.tmp_upper_lim is not None else -1
    fa.lower_temperature_limit = int(args.tmp_lower_lim) if args.tmp_lower_lim is not None else -1
    fa.backoff_time_normal = int(args.backoff_normal) if args.backoff_normal is not None else -1
    fa.backoff_time_high = int(args.backoff_high)  if args.backoff_high is not None else -1
    fa.backoff_time_critical = int(args.backoff_crit) if args.backoff_crit is not None else -1
    fa.debug_readX = int(args.debug_readX) if args.debug_readX is not None else 100
    fa.temperature_mode = "Advanced" if hasattr(args, 'temperature_mode') and args.temperature_mode in [None, "adv", "advanced"] else "Simple"
    fa.stress_usblink_write = int(args.stress_usblink_write) if args.stress_usblink_write is not None else 1
    fa.stress_usblink_read = int(args.stress_usblink_read)  if args.stress_usblink_read is not None else 1
    fa.stress_full_run = int(args.stress_full_run) if args.stress_full_run is not None else 1
    fa.stress_boot_run = int(args.stress_boot_run) if args.stress_boot_run is not None else 1
    fa.stress_boot_init = int(args.stress_boot_init) if args.stress_boot_init is not None else 1
    fa.input_size = int(args.input_size) if args.input_size else None
    fa.timer = args.timer
    fa.disable_query = args.disable_query
    fa.network_level_throttling = args.set_network_level_throttling
    fa.no_throttling = args.set_no_throttling
    fa.explicit_concat = args.explicit_concat
    fa.ma2480 = args.ma2480

    fa.acm = int(args.ACM) if args.ACM is not None else 0

    fa.net_description = path_arg(args.network_description)
    fa.net_weights = path_arg(args.network_weights)

    fa.output_node_name = args.output_node_name
    fa.input_node_name = args.input_node_name

    if fa.net_description is not None:
        if args.parser is None:
            fa.parser = predict_parser(fa.net_description)
        else:
            fa.parser = parser_as_enum(args.parser)
    else:
        if fa.mode not in [OperationMode.optimization_list]:
            throw_error(ErrorTable.ArgumentErrorRequired)
        else:
            fa.parser = Parser.Debug

    fa.expected_index = args.output_expected_id
    fa.class_test_threshold = float(args.class_test_threshold) if args.class_test_threshold is not None else 0.20  # Default to 20%
    fa.validation_type = validation_as_enum(args.output_validation_type)
    fa.image = path_arg(args.image)
    fa.raw_scale = float(args.raw_scale) if args.raw_scale is not None else 1
    fa.mean = path_arg(args.mean)
    fa.channel_swap = args.channel_swap

    fa.conf_file = args.conf_file if args.conf_file is not None else "optimisation.conf"
    fa.save_input = args.save_input
    fa.save_output = args.save_output

    fa.ensure_arg_compatibility()

    return fa


def path_arg(path):
    if path is not None:
        return os.path.normpath(path)
    else:
        return None


def path_check(path, error):
    if path is not None and os.path.isfile(path):
        return True
    else:
        throw_error(error)
