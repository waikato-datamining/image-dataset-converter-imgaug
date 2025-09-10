import argparse
import math
from typing import List

import cv2
import numpy as np
from shapely import Polygon
from wai.common.adams.imaging.locateobjects import LocatedObjects, LocatedObject
from wai.common.geometry import Polygon, Point
from wai.logging import LOGGING_WARNING

from seppl.io import BatchFilter
from kasperl.api import make_list, flatten_list, safe_deepcopy
from idc.api import ImageData, ObjectDetectionData, ImageSegmentationData, LABEL_KEY, ensure_grayscale


class HoughLinesProbabilistic(BatchFilter):
    """
    Detects lines in the image and stores them as polygons.
    """

    def __init__(self, label: str = None, rho: float = None, theta: float = None,
                 threshold: int = None, min_line_length: int = None, max_line_gap: int = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param label: the label to use for the detected contours
        :type label: str
        :param rho: Distance resolution of the accumulator in pixels
        :type rho: float
        :param theta: Angle resolution of the accumulator in radians.
        :type theta: float
        :param threshold: Accumulator threshold parameter. Only those lines are returned that get enough votes (> threshold).
        :type threshold: int
        :param min_line_length: Second threshold for hysteresis procedure in Canny()
        :type min_line_length: float
        :param max_line_gap: Maximum allowed gap between points on the same line to link them.
        :type max_line_gap: int
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.label = label
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "hough-lines-prob"

    def description(self) -> str:
        """
        Returns a description of the handler.

        :return: the description
        :rtype: str
        """
        return "Finds line segments in a binary image using the probabilistic Hough transform. "

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
        parser.add_argument("--rho", type=float, default=1.0, help="Distance resolution of the accumulator in pixels.", required=False)
        parser.add_argument("--theta", type=float, default=math.pi/180, help="Angle resolution of the accumulator in radians.", required=False)
        parser.add_argument("--threshold", type=int, default=50, help="Accumulator threshold parameter. Only those lines are returned that get enough votes (>threshold).", required=False)
        parser.add_argument("--min_line_length", type=int, default=0, help="Minimum line length. Line segments shorter than that are rejected.", required=False)
        parser.add_argument("--max_line_gap", type=int, default=0, help="Maximum allowed gap between points on the same line to link them.", required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.label = ns.label
        self.rho = ns.rho
        self.theta = ns.theta
        self.threshold = ns.threshold
        self.min_line_length = ns.min_line_length
        self.max_line_gap = ns.max_line_gap

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if self.rho is None:
            self.rho = 1.0
        if self.theta is None:
            self.theta = math.pi/180
        if self.threshold is None:
            self.threshold = 50
        if self.min_line_length is None:
            self.min_line_length = 0
        if self.max_line_gap is None:
            self.max_line_gap = 0

    def _detect_lines(self, image: np.ndarray, ann: LocatedObjects, label: str):
        """
        Processes the contours and adds the polygons to the annotations.

        :param image: the image to process
        :param ann: the annotations to append
        :type ann: LocatedObjects
        :param label: the label to use
        :type label: str
        """
        lines = cv2.HoughLinesP(image.astype(np.uint8),
                                self.rho,
                                self.theta,
                                threshold=self.threshold,
                                minLineLength=self.min_line_length,
                                maxLineGap=self.max_line_gap)
        if lines is not None:
            for line in lines:
                x0, y0, x1, y1 = line[0]
                points = [Point(x0, y0), Point(x1, y1), Point(x0, y0)]
                polygon = Polygon(*points)
                obj = LocatedObject(x0, y0, x1 - x0 + 1, y1 - y0 + 1)
                obj.metadata[LABEL_KEY] = label
                obj.set_polygon(polygon)
                ann.append(obj)

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
                    layer = np.where(layer > 0, 255, 0)
                    self._detect_lines(layer, ann, label)
            else:
                grayscale = ensure_grayscale(item.image, self.logger())
                self._detect_lines(np.asarray(grayscale), ann, self.label)

            self.logger().info("# of lines added: %s" % str(len(ann)))
            item_new = ObjectDetectionData(source=item.source, image_name=item.image_name,
                                           image=safe_deepcopy(item.image), data=safe_deepcopy(item.data),
                                           annotation=ann, metadata=item.get_metadata())
            result.append(item_new)

        return flatten_list(result)
