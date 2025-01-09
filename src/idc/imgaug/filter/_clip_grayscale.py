import argparse
from typing import List

import numpy as np
from seppl.io import Filter
from wai.logging import LOGGING_WARNING

from idc.api import ImageClassificationData, ObjectDetectionData, ImageSegmentationData, flatten_list, make_list, \
    array_to_image


class ClipGrayscale(Filter):
    """
    Changes the pixel values of grayscale images either by a factor or by a fixed value.
    """

    def __init__(self, min_value: int = None, min_replacement: int = None, 
                 max_value: int = None, max_replacement: int = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param min_value: the smallest allowed pixel value
        :type min_value: int
        :param min_replacement: the pixel value to replace values with that fall below minimum
        :type min_replacement: int
        :param max_value: the largest allowed pixel value
        :type max_value: int
        :param max_replacement: the pixel value to replace values with that go above maximum
        :type max_replacement: int
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.min_value = min_value
        self.min_replacement = min_replacement
        self.max_value = max_value
        self.max_replacement = max_replacement

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "clip-grayscale"

    def description(self) -> str:
        """
        Returns a description of the filter.

        :return: the description
        :rtype: str
        """
        return "Changes the grayscale values that fall below the minimum or go above the maximum to the specified replacement values."

    def accepts(self) -> List:
        """
        Returns the list of classes that are accepted.

        :return: the list of classes
        :rtype: list
        """
        return [ImageClassificationData, ObjectDetectionData, ImageSegmentationData]

    def generates(self) -> List:
        """
        Returns the list of classes that get produced.

        :return: the list of classes
        :rtype: list
        """
        return [ImageClassificationData, ObjectDetectionData, ImageSegmentationData]

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        parser.add_argument("-m", "--min_value", type=int, help="The smallest allowed grayscale pixel value.", default=0, required=False)
        parser.add_argument("-r", "--min_replacement", type=int, help="The replacement grayscale pixel value for values that fall below the minimum.", default=0, required=False)
        parser.add_argument("-M", "--max_value", type=int, help="The largest allowed grayscale pixel value.", default=255, required=False)
        parser.add_argument("-R", "--max_replacement", type=int, help="The replacement grayscale pixel value for values that go above the minimum.", default=255, required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.min_value = ns.min_value
        self.min_replacement = ns.min_replacement
        self.max_value = ns.max_value
        self.max_replacement = ns.max_replacement

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if self.min_value is None:
            self.min_value = 0
        if self.min_replacement is None:
            self.min_replacement = 0
        if self.max_value is None:
            self.max_value = 255
        if self.max_replacement is None:
            self.max_replacement = 255
        if self.min_value >= self.max_value:
            raise Exception("Min value must be smaller than max one: min=%d, max=%d" % (self.min_value, self.max_value))
        if (self.min_value == 0) and (self.max_value == 255):
            self.logger().warning("No clipping occurring when using: min=%d, max=%d" % (self.min_value, self.max_value))

    def _do_process(self, data):
        """
        Processes the data record(s).

        :param data: the record(s) to process
        :return: the potentially updated record(s)
        """
        # nothing to do?
        if (self.min_value == 0) and (self.max_value == 255):
            return flatten_list(make_list(data))

        result = []
        for item in make_list(data):
            img_pil = item.image
            if img_pil.mode == "L":
                img_gray = np.array(img_pil)
                if self.min_value > 0:
                    img_gray = np.where(img_gray < self.min_value, self.min_replacement, img_gray)
                if self.max_value < 255:
                    img_gray = np.where(img_gray > self.max_value, self.max_replacement, img_gray)
                img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)
                _, img_pil_bytes = array_to_image(img_gray, item.image_format)
                item_new = type(item)(image_name=item.image_name, data=img_pil_bytes.getvalue(),
                                      metadata=item.get_metadata(), annotation=item.annotation)
                result.append(item_new)
            else:
                self.logger().warning("Not a grayscale image: %s" % item.image_name)
                result.append(item)

        return flatten_list(result)
