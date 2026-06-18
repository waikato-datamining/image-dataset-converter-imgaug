import argparse
import numpy as np
from typing import List

from PIL import Image
from simple_palette_utils import parse_rgb
from wai.logging import LOGGING_WARNING

from idc.api import ImageClassificationData, ImageSegmentationData, ObjectDetectionData, DepthData, array_to_image
from kasperl.api import make_list, flatten_list, safe_deepcopy
from seppl.io import BatchFilter


class Pad(BatchFilter):
    """
    Pads the images to have at least the specified width/height.
    """

    def __init__(self, width: int = None, height: int = None, background: str = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param width: the minimum width to use
        :type width: int
        :param height: the minimum height to use
        :type height: int
        :param background: the background color to use
        :type background: str
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.width = width
        self.height = height
        self.background = background
        self._background = None

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "pad"

    def description(self) -> str:
        """
        Returns a description of the handler.

        :return: the description
        :rtype: str
        """
        return "Pads the images to have at least the specified width/height."

    def accepts(self) -> List:
        """
        Returns the list of classes that are accepted.

        :return: the list of classes
        :rtype: list
        """
        return [ImageClassificationData, ImageSegmentationData, ObjectDetectionData, DepthData]

    def generates(self) -> List:
        """
        Returns the list of classes that get produced.

        :return: the list of classes
        :rtype: list
        """
        return [ImageClassificationData, ImageSegmentationData, ObjectDetectionData, DepthData]

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        parser.add_argument("-W", "--width", type=int, default=None, help="The minimum width to pad to, ignored if not specified.", required=False)
        parser.add_argument("-H", "--height", type=int, default=None, help="The minimum height to pad to, ignored if not specified.", required=False)
        parser.add_argument("-b", "--background", type=str, metavar="R,G,B", default="0,0,0", help="The RGB triplet (R,G,B) to use for the background color", required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.width = ns.width
        self.height = ns.height
        self.background = ns.background

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if self.background is None:
            self.background = "0,0,0"
        self._background = parse_rgb([self.background])
        if len(self._background) != 1:
            raise Exception("Invalid color specification: %s" % self.background)
        self._background = self._background[0]

    def _adjust_matrix(self, matrix: np.ndarray, width_old: int, height_old: int, width_new: int, height_new: int) -> np.ndarray:
        matrix_new = np.zeros((height_new, width_new), dtype=matrix.dtype)
        matrix_new[0:height_old, 0:width_old] = matrix
        return matrix_new

    def _do_process(self, data):
        """
        Processes the data record(s).

        :param data: the record(s) to process
        :return: the potentially updated record(s)
        """
        if (self.width is None) and (self.height is None):
            self.logger().warning("Neither width nor height specified!")
            return data

        result = []

        for item in make_list(data):
            modify = False
            width_new = item.image_width
            height_new = item.image_height
            if (self.width is not None) and (item.image_width < self.width):
                width_new = self.width
                self.logger().info("Updating width to: %d" % width_new)
                modify = True
            if (self.height is not None) and (item.image_height < self.height):
                height_new = self.height
                self.logger().info("Updating height to: %d" % height_new)
                modify = True
            if modify:
                img_new = Image.new(item.image.mode, (width_new, height_new), self._background)
                img_new.paste(item.image, (0, 0))
                item_new = type(item)(image_name=item.image_name, image=img_new, image_format=item.image_format,
                                      annotation=safe_deepcopy(item.annotation),
                                      metadata=safe_deepcopy(item.get_metadata()))
                # adjust annotations as well
                if item.has_annotation():
                    if isinstance(item, ImageClassificationData):
                        # nothing to do
                        pass
                    elif isinstance(item, ImageSegmentationData):
                        # adjust the layers
                        for layer in item_new.annotation.layers:
                            item_new.annotation.layers[layer] = self._adjust_matrix(item_new.annotation.layers[layer], item.image_width, item.image_height, width_new, height_new)
                    elif isinstance(item, ObjectDetectionData):
                        # make sure to have absolute coordinates
                        if item.is_normalized():
                            item_new.annotation = item.get_absolute()
                    elif isinstance(item, DepthData):
                        # adjust depth information matrix
                        item_new.annotation.data = self._adjust_matrix(item_new.annotation.data, item.image_width, item.image_height, width_new, height_new)
                    else:
                        self.logger().warning("Unhandled data type: %s" % str(item))
                result.append(item_new)
            else:
                self.logger().info("Nothing to do: %s" % item.image_name)
                result.append(item)

        return flatten_list(result)
