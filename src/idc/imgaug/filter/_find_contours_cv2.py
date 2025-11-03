import argparse
from typing import List

import cv2
import numpy as np
from wai.common.adams.imaging.locateobjects import LocatedObjects
from wai.logging import LOGGING_WARNING

from idc.api import ImageData, ObjectDetectionData, ImageSegmentationData, ensure_binary, contours_to_objdet
from kasperl.api import make_list, flatten_list, safe_deepcopy
from seppl.io import BatchFilter

MIN_RECT_WIDTH = "min_rect_width"
MIN_RECT_HEIGHT = "min_rect_height"


class FindContoursCV2(BatchFilter):
    """
    Finds the contours in the binary image and stores them as polygons in the annotations.
    """

    def __init__(self, label: str = None, min_size: int = None, max_size: int = None, calculate_min_rect: bool = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param label: the label to use for the detected contours
        :type label: str
        :param min_size: the minimum width or height contours must have
        :type min_size: int
        :param max_size: the maximum width or height contours can have
        :type max_size: int
        :param calculate_min_rect: whether to calculate the minimal rectangle for each contour
        :type calculate_min_rect: bool
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.label = label
        self.min_size = min_size
        self.max_size = max_size
        self.calculate_min_rect = calculate_min_rect

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "find-contours-cv2"

    def description(self) -> str:
        """
        Returns a description of the handler.

        :return: the description
        :rtype: str
        """
        return "Finds the contours in the binary image using OpenCV's findContours method and stores "\
            + "them as polygons in the annotations. "\
            + "In case of image segmentation data, the annotations are analyzed, otherwise the base image."\
            + "When calculating the minimal rectangles, the following fields get added to the meta-data "\
            + "of the objects: " + MIN_RECT_WIDTH + ", " + MIN_RECT_HEIGHT + ". "\
            + "The minimal rectangle width/height also get checked against the specified min/max sizes."

    def accepts(self) -> List:
        """
        Returns the list of classes that are accepted.

        :return: the list of classes
        :rtype: list
        """
        return [ImageData]

    def generates(self) -> List:
        """
        Returns the list of classes that get produced.

        :return: the list of classes
        :rtype: list
        """
        return [ObjectDetectionData]

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        parser.add_argument("--label", type=str, help="The label to use when processing images other than image segmentation ones.", default="object", required=False)
        parser.add_argument("-m", "--min_size", type=int, default=None, help="The minimum width or height that detected contours must have.", required=False)
        parser.add_argument("-M", "--max_size", type=int, default=None, help="The maximum width or height that detected contours can have.", required=False)
        parser.add_argument("-r", "--calculate_min_rect", action="store_true", help="Whether to calculate the minimal rectangle for each contour.", required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.label = ns.label
        self.min_size = ns.min_size
        self.max_size = ns.max_size
        self.calculate_min_rect = ns.calculate_min_rect

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if self.calculate_min_rect is None:
            self.calculate_min_rect = False

    def _do_process(self, data):
        """
        Processes the data record(s).

        :param data: the record(s) to process
        :return: the potentially updated record(s)
        """
        result = []

        for item in make_list(data):
            ann = LocatedObjects()

            if isinstance(item, ImageSegmentationData):
                for i, label in enumerate(item.annotation.labels):
                    if label not in item.annotation.layers:
                        continue
                    layer = item.annotation.layers[label]
                    layer = np.where(layer > 0, 1, 0)
                    contours, _ = cv2.findContours(np.array(layer).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    self.logger().info("%s - # of contours: %s" % (label, str(len(contours))))
                    contours_to_objdet(contours, ann, label, min_size=self.min_size, max_size=self.max_size,
                                       calculate_min_rect=self.calculate_min_rect)
            else:
                binary = ensure_binary(item.image, self.logger())
                contours, _ = cv2.findContours(np.array(binary).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                self.logger().info("# of contours: %s" % str(len(contours)))
                contours_to_objdet(contours, ann, self.label, min_size=self.min_size, max_size=self.max_size,
                                   calculate_min_rect=self.calculate_min_rect)

            self.logger().info("# of polygons added: %s" % str(len(ann)))
            item_new = ObjectDetectionData(source=item.source, image_name=item.image_name,
                                           image=safe_deepcopy(item.image), data=safe_deepcopy(item.data),
                                           annotation=ann, metadata=item.get_metadata())
            result.append(item_new)

        return flatten_list(result)
