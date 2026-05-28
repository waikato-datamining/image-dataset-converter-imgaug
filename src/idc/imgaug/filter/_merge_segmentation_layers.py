import argparse
import logging
import traceback

import numpy as np
from typing import List

from wai.logging import LOGGING_WARNING

from idc.api import ImageSegmentationData
from kasperl.api import make_list, flatten_list, safe_deepcopy
from seppl.io import BatchFilter

MERGE_OPERATION_ADD = "add"
MERGE_OPERATION_SUBTRACT = "subtract"
MERGE_OPERATIONS = [
    MERGE_OPERATION_ADD,
    MERGE_OPERATION_SUBTRACT,
]


class MergeSegmentationLayers(BatchFilter):
    """
    Merges the specified source layer from an image segmentation item in internal storage with the target layer of the item passing through the filter.
    """

    def __init__(self, source_name: str = None, source_layer: str = None, target_layer: str = None,
                 merge_operation: str = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param source_name: the name of the item in storage
        :type source_name: str
        :param source_layer: the source layer to use (from item in storage)
        :type source_layer: str
        :param target_layer: the target layer to apply to (current item)
        :type target_layer: str
        :param merge_operation: how to merge the two layers
        :type merge_operation: str
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.source_name = source_name
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.merge_operation = merge_operation

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "merge-segmentation-layers"

    def description(self) -> str:
        """
        Returns a description of the filter.

        :return: the description
        :rtype: str
        """
        return "Merges the specified source layer from an image segmentation item in internal storage with the target layer of the item passing through the filter."

    def accepts(self) -> List:
        """
        Returns the list of classes that are accepted.

        :return: the list of classes
        :rtype: list
        """
        return [ImageSegmentationData]

    def generates(self) -> List:
        """
        Returns the list of classes that get produced.

        :return: the list of classes
        :rtype: list
        """
        return [ImageSegmentationData]

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        parser.add_argument("-n", "--source_name", type=str, default=None, help="The name of the item in internal storage.", required=True)
        parser.add_argument("-s", "--source_layer", type=str, default=None, help="The source layer/label to use from the item in internal storage.", required=True)
        parser.add_argument("-t", "--target_layer", type=str, default=None, help="The target layer/label to use from the item passing through.", required=True)
        parser.add_argument("-o", "--merge_operation", choices=MERGE_OPERATIONS, default=MERGE_OPERATION_ADD, help="How to merge the two layers.", required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.source_name = ns.source_name
        self.source_layer = ns.source_layer
        self.target_layer = ns.target_layer
        self.merge_operation = ns.merge_operation

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()

        if self.source_name is None:
            raise Exception("No name for storage item specified!")
        if self.source_layer is None:
            raise Exception("No source layer/label specified!")
        if self.target_layer is None:
            raise Exception("No target layer/label specified!")
        if self.merge_operation is None:
            self.merge_operation = MERGE_OPERATION_ADD

    def _do_process(self, data):
        """
        Processes the data record(s).

        :param data: the record(s) to process
        :return: the potentially updated record(s)
        """
        result = []

        # get source item
        if self.source_name not in self.session.storage:
            self.logger().error("Storage item '%s' not present!" % self.source_name)
            return data
        source = self.session.storage[self.source_name]
        if not isinstance(source, ImageSegmentationData):
            self.logger().error("Storage item '%s' is not of type %s: %s" % (self.source_name, str(ImageSegmentationData), type(source)))
            return data
        if not source.has_layer(self.source_layer):
            self.logger().error("Storage item '%s' doesn't have layer: %s" % (self.source_name, self.source_layer))
            return data

        for item in make_list(data):
            if isinstance(item, ImageSegmentationData):
                if item.has_layer(self.target_layer):
                    target = safe_deepcopy(item)
                    source_layer = np.where(source.annotation.layers[self.source_layer] > 0, 1, 0).astype(np.int16)
                    target_layer = np.where(target.annotation.layers[self.target_layer] > 0, 1, 0).astype(np.int16)
                    if self.logger().isEnabledFor(logging.DEBUG):
                        self.logger().debug("source: %s" % str(np.unique(source_layer, return_counts=True)))
                        self.logger().debug("target: %s" % str(np.unique(target_layer, return_counts=True)))
                    if source_layer.shape != target_layer.shape:
                        self.logger().error("Layers have different dimensions: source=%s vs target=%s" % (str(source_layer.shape), str(target_layer.shape)))
                    else:
                        if self.merge_operation == MERGE_OPERATION_ADD:
                            target_layer += source_layer
                            if self.logger().isEnabledFor(logging.DEBUG):
                                self.logger().debug("add: %s" % str(np.unique(target_layer, return_counts=True)))
                        elif self.merge_operation == MERGE_OPERATION_SUBTRACT:
                            target_layer -= source_layer
                            if self.logger().isEnabledFor(logging.DEBUG):
                                self.logger().debug("subtract: %s" % str(np.unique(target_layer, return_counts=True)))
                        else:
                            self.logger().error("Unhandled merge operation: %s" % self.merge_operation)
                        target_layer = np.where(target_layer < 0, 0, target_layer).astype(np.int16)
                        target_layer = np.where(target_layer > 0, 255, target_layer).astype(np.uint8)
                        if self.logger().isEnabledFor(logging.DEBUG):
                            self.logger().debug("final: %s" % str(np.unique(target_layer, return_counts=True)))
                        target.annotation.layers[self.target_layer] = target_layer
                        item = target
            result.append(item)

        return flatten_list(result)
