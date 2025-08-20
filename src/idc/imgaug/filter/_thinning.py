import argparse
from typing import List

import numpy as np
from PIL import Image
from wai.logging import LOGGING_WARNING

from idc.api import ImageClassificationData, ObjectDetectionData, ImageSegmentationData, image_to_bytesio
from kasperl.api import make_list, flatten_list
from seppl.io import Filter
from ._thinning_utils import thinning


class Thinning(Filter):
    """
    Thinning algorithm developed by Lingdong Huang.
    https://github.com/LingDong-/skeleton-tracing/blob/master/py/trace_skeleton.py
    """

    def __init__(self, factor: float = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param factor: the factor with which to scale the image before applying thinning algorithm
        :type factor: float
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.factor = factor

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "thinning"

    def description(self) -> str:
        """
        Returns a description of the filter.

        :return: the description
        :rtype: str
        """
        return "Thinning algorithm developed by Lingdong Huang: https://github.com/LingDong-/skeleton-tracing/blob/master/py/trace_skeleton.py"

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
        parser.add_argument("-f", "--factor", type=float, help="The factor with which to scale the image before applying the thinning algorithm.", default=1.0, required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.factor = ns.factor

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if self.factor is None:
            self.factor = 1.0

    def _do_process(self, data):
        """
        Processes the data record(s).

        :param data: the record(s) to process
        :return: the potentially updated record(s)
        """
        result = []
        for item in make_list(data):
            img_pil = item.image
            if img_pil.mode == "1":
                # downscale?
                if self.factor != 1.0:
                    w, h = img_pil.size
                    w_new = int(w * self.factor)
                    h_new = int(h * self.factor)
                    img_pil = img_pil.resize((w_new, h_new))
                    self.logger().info("Downscaling to: %s" % str((w_new, h_new)))

                # perform thinning
                array_thin = thinning(np.array(img_pil).astype(np.uint8))

                array_thin = np.where(array_thin > 0, 255, 0)
                img_out = Image.fromarray(np.uint8(array_thin))

                # upscale?
                if self.factor != 1.0:
                    img_out = img_out.resize(item.image.size)
                    self.logger().info("Upscaling to: %s" % str(item.image.size))

                # create output
                img_out_bytes = image_to_bytesio(img_out, item.image_format).getvalue()
                item_new = type(item)(image_name=item.image_name, data=img_out_bytes,
                                      metadata=item.get_metadata(), annotation=item.annotation)
                result.append(item_new)
            else:
                self.logger().warning("Not a binary image: %s" % item.image_name)
                result.append(item)

        return flatten_list(result)
