from enum import Enum
from Models.NetworkStage import *


class PatternType(Enum):
    Unknown = 0
    Completed = 1
    LeakyReLU = 2


class Pattern:

        _pattern_type = PatternType.Unknown
        _name = ''
        _ops_names = []
        _input_shape = []
        _output_shape = []
        _prev_node_name = ''
        _params = []

        def __init__(self, pattern_type=PatternType.Unknown, name="", ops_names=[], inputs=[], outputs=[],
                     prev_node_name="", params=[]):
            self._pattern_type = pattern_type
            self._name = name
            self._ops_names = ops_names
            self._input_shape = inputs
            self._output_shape = outputs
            self._prev_node_name = prev_node_name
            self._params = params

        def get_type(self):
            return self._pattern_type

        def get_name(self):
            return self._name

        def get_input_shape(self):
            return self._input_shape

        def get_output_shape(self):
            return self._output_shape

        def get_prev_name(self):
            return self._prev_node_name

        def get_param(self, param_idx):
            return self._params[param_idx]


class TFPreprocessor:

    _found_patterns = []
    _pattern_checked = []
    _handled_ops = []
    _graph = None

    def _inside_pattern(self, op):
        for pattern_ops in self._handled_ops:
            if op.name in pattern_ops:
                return self._handled_ops.index(pattern_ops)
        return -1

    def _is_checked(self, pattern_idx):
        if self._pattern_checked[pattern_idx]:
            return True
        else:
            self._pattern_checked[pattern_idx] = True
            return False

    def preprocess(self, graph):
        """
        Find known patterns in the given TensorFlow graph.
        :param graph: Input TensorFlow graph
        """

        self._found_patterns.clear()
        self._handled_ops.clear()
        self._pattern_checked.clear()
        self._graph = graph

        for idx, node in enumerate(graph.get_operations()):
            if node.name not in self._handled_ops:
                # LeakyReLU
                if node.type == 'Maximum':
                    # Get inputs
                    input_nodes = [graph.get_operation_by_name(input_tensor.op.name) for input_tensor in node.inputs]
                    ops_names = [node.name]

                    # Check if Mul in inputs
                    for input_node in input_nodes:
                        if input_node.type == 'Mul':

                            scalar_cond = False
                            common_cond = False

                            input_shape = None

                            for mul_input in input_node.inputs:

                                # Check if Mul has a scalar input
                                if not mul_input.get_shape().as_list():
                                    scalar_cond = True
                                    continue

                                # Check if Mul and Maximum have common input
                                for max_input in node.inputs:
                                    if max_input == mul_input:
                                        common_cond = True
                                        ops_names.append(input_node.name)
                                    else:
                                        input_shape = max_input.get_shape()

                            if scalar_cond and common_cond:
                                new_pattern = Pattern(PatternType.LeakyReLU, node.name, ops_names, input_shape,
                                                      node.outputs[0].get_shape(), node.outputs[0].name, [0.1])
                                self._found_patterns.append(new_pattern)
                                self._pattern_checked.append(False)
                                self._handled_ops.append(ops_names)

    def pattern_found(self, node):
        """
        Check if the given node is a part of previously found pattern
        :param node: Input TensorFlow node
        :return: True and a pattern object if a given node is a part of any pattern. If a pattern has been processed
        already, a returned pattern object has a pattern type of PatternType.Completed
        :raises: RuntimeError when no graph has been processed earlier or a node from a different graph was passed
        """

        if not self._graph:
            raise RuntimeError("The graph has to be processed first")

        if self._graph != node.graph:
            raise RuntimeError("Cannot process node from unknown graph")

        pattern_idx = self._inside_pattern(node)
        if pattern_idx >= 0:
            if not self._is_checked(pattern_idx):
                return True, self._found_patterns[pattern_idx]
            else:
                return True, Pattern(PatternType.Completed)
        return False, None
