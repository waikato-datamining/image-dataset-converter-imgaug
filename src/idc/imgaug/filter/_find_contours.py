import argparse
import numpy as np
from typing import List

from seppl.io import Filter
from wai.logging import LOGGING_WARNING
from wai.common.geometry import Polygon, Point
from wai.common.adams.imaging.locateobjects import LocatedObject, LocatedObjects
from idc.api import ObjectDetectionData, ImageSegmentationData, make_list, flatten_list, LABEL_KEY
from smu import mask_to_polygon, polygon_to_lists


CONNECTIVITY_LOW = "low"
CONNECTIVITY_HIGH = "high"
CONNECTIVITY = [
    CONNECTIVITY_LOW,
    CONNECTIVITY_HIGH,
]


class FindContours(Filter):
    """
    Detects blobs in the annotations of the image segmentation data and turns them into object detection polygons.
    """

    def __init__(self, mask_threshold: float = 0.1, mask_nth: int = 1,
                 view_margin: int = 5, fully_connected: str = "low",
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param mask_threshold: the (lower) probability threshold for mask values in order to be considered part of the object (0-1)
        :type mask_threshold: float
        :param mask_nth: the contour tracing can be slow for large masks, by using only every nth row/col, this can be sped up dramatically
        :type mask_nth: int
        :param view_margin: the margin in pixels to enlarge the view with in each direction
        :type view_margin: int
        :param fully_connected: whether regions of high or low values should be fully-connected at isthmuses
        :type fully_connected: str
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.mask_threshold = mask_threshold
        self.mask_nth = mask_nth
        self.view_margin = view_margin
        self.fully_connected = fully_connected

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "find-contours"

    def description(self) -> str:
        """
        Returns a description of the filter.

        :return: the description
        :rtype: str
        """
        return "Detects blobs in the annotations of the image segmentation data and turns them into object detection polygons."

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
        return [ObjectDetectionData]

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        parser.add_argument("-t", "--mask_threshold", type=float, help="The (lower) probability threshold for mask values in order to be considered part of the object (0-1).", default=0.1, required=False)
        parser.add_argument("-n", "--mask_nth", type=float, help="The contour tracing can be slow for large masks, by using only every nth row/col, this can be sped up dramatically.", default=1, required=False)
        parser.add_argument("-m", "--view_margin", type=int, help="The margin in pixels to enlarge the view with in each direction.", default=5, required=False)
        parser.add_argument("-f", "--fully_connected", choices=CONNECTIVITY, help="Whether regions of high or low values should be fully-connected at isthmuses.", default=CONNECTIVITY_LOW, required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.mask_threshold = ns.mask_threshold
        self.mask_nth = ns.mask_nth
        self.view_margin = ns.view_margin
        self.fully_connected = ns.fully_connected

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if self.mask_threshold is None:
            self.mask_threshold = 0.1
        if self.mask_nth is None:
            self.mask_nth = 1
        if self.view_margin is None:
            self.view_margin = 5
        if self.fully_connected is None:
            self.fully_connected = CONNECTIVITY_LOW

    def _do_process(self, data):
        """
        Processes the data record(s).

        :param data: the record(s) to process
        :return: the potentially updated record(s)
        """
        result = []
        for item in make_list(data):
            objs = LocatedObjects()
            for i, label in enumerate(item.annotation.labels):
                if label not in item.annotation.layers:
                    continue
                layer = item.annotation.layers[label]
                layer = np.where(layer == 255, 1, 0)
                polys = mask_to_polygon(layer, mask_threshold=self.mask_threshold, mask_nth=self.mask_nth,
                                        fully_connected=self.fully_connected)
                for poly in polys:
                    px, py = polygon_to_lists(poly, swap_x_y=True, as_type="int")
                    left = min(px)
                    right = max(px)
                    top = min(py)
                    bottom = max(py)
                    points = [Point(x, y) for x, y in zip(px, py)]
                    polygon = Polygon(*points)
                    obj = LocatedObject(left, top, right - left + 1, bottom - top + 1)
                    obj.metadata[LABEL_KEY] = label
                    obj.set_polygon(polygon)
                    objs.append(obj)
            item_new = ObjectDetectionData(source=item.source, image_name=item.image_name, data=item._data,
                                           annotation=objs, metadata=item.get_metadata())
            result.append(item_new)

        return flatten_list(result)
