import argparse
import io
from typing import List

import numpy as np
from PIL import Image
from seppl.io import Filter
from wai.logging import LOGGING_WARNING

from idc.api import ImageClassificationData, ObjectDetectionData, ImageSegmentationData, flatten_list, make_list, array_to_image


class ChangeGrayscale(Filter):
    """
    Changes the pixel values of grayscale images either by a factor or by a fixed value.
    """

    def __init__(self, factor: float = None, increment:int = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param factor: the factor with which to scale the pixel values
        :type factor: float
        :param increment: the int value to change the pixel values by
        :type increment: int
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.factor = factor
        self.increment = increment

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "change-grayscale"

    def description(self) -> str:
        """
        Returns a description of the filter.

        :return: the description
        :rtype: str
        """
        return "Changes the pixel values of grayscale images either by a factor or by a fixed value."

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
        parser.add_argument("--factor", type=float, help="The factor with which to scale the pixel values.", default=None, required=False)
        parser.add_argument("--increment", type=str, help="The value to change the pixel values by.", default=None, required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.factor = ns.factor
        self.increment = ns.increment

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if (self.factor is None) and (self.increment is None):
            raise Exception("Neither factor nor increment provided!")

    def _do_process(self, data):
        """
        Processes the data record(s).

        :param data: the record(s) to process
        :return: the potentially updated record(s)
        """
        result = []
        for item in make_list(data):
            img_pil = item.image
            if img_pil.mode == "L":
                img_gray = np.array(img_pil)
                if self.factor is not None:
                    img_gray = img_gray * self.factor
                else:
                    img_gray = img_gray + self.increment
                img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)
                _, img_pil_bytes = array_to_image(img_gray, item.image_format)
                item_new = type(item)(image_name=item.image_name, data=img_pil_bytes.getvalue(),
                                      metadata=item.get_metadata(), annotation=item.annotation)
                result.append(item_new)
            else:
                self.logger().warning("Not a grayscale image: %s" % item.image_name)
                result.append(item)

        return flatten_list(result)
