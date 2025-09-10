import argparse
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


class FastLineDetection(BatchFilter):
    """
    Detects lines in the image and stores them as polygons.
    """

    def __init__(self, label: str = None, length_threshold: int = None, distance_threshold: float = None,
                 canny_th1: float = None, canny_th2: float = None, canny_aperture_size: int = None, do_merge: bool = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param label: the label to use for the detected contours
        :type label: str
        :param length_threshold: Segment shorter than this will be discarded
        :type length_threshold: int
        :param distance_threshold: A point placed from a hypothesis line segment farther than this will be regarded as an outlier
        :type distance_threshold: float
        :param canny_th1: First threshold for hysteresis procedure in Canny()
        :type canny_th1: float
        :param canny_th2: Second threshold for hysteresis procedure in Canny()
        :type canny_th2: float
        :param canny_aperture_size: Aperture size for the sobel operator in Canny(). If zero, Canny() is not applied and the input image is taken as an edge image.
        :type canny_aperture_size: int
        :param do_merge: If true, incremental merging of segments will be performed
        :type do_merge: bool
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.label = label
        self.length_threshold = length_threshold
        self.distance_threshold = distance_threshold
        self.canny_th1 = canny_th1
        self.canny_th2 = canny_th2
        self.canny_aperture_size = canny_aperture_size
        self.do_merge = do_merge

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "fast-line-detection"

    def description(self) -> str:
        """
        Returns a description of the handler.

        :return: the description
        :rtype: str
        """
        return "Detects lines in the image and stores them as polygons."

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
        parser.add_argument("--length_threshold", type=int, default=10, help="Segment shorter than this will be discarded.", required=False)
        parser.add_argument("--distance_threshold", type=float, default=1.414213562, help="A point placed from a hypothesis line segment farther than this will be regarded as an outlier.", required=False)
        parser.add_argument("--canny_th1", type=float, default=50.0, help="First threshold for hysteresis procedure in Canny().", required=False)
        parser.add_argument("--canny_th2", type=float, default=50.0, help="Second threshold for hysteresis procedure in Canny().", required=False)
        parser.add_argument("--canny_aperture_size", type=int, default=3, help="Aperture size for the sobel operator in Canny(). If zero, Canny() is not applied and the input image is taken as an edge image.", required=False)
        parser.add_argument("--do_merge", action="store_true", help="If true, incremental merging of segments will be performed.", required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.label = ns.label
        self.length_threshold = ns.length_threshold
        self.distance_threshold = ns.distance_threshold
        self.canny_th1 = ns.canny_th1
        self.canny_th2 = ns.canny_th2
        self.canny_aperture_size = ns.canny_aperture_size
        self.do_merge = ns.do_merge

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if self.length_threshold is None:
            self.length_threshold = 10
        if self.distance_threshold is None:
            self.distance_threshold = 1.414213562
        if self.canny_th1 is None:
            self.canny_th1 = 50.0
        if self.canny_th2 is None:
            self.canny_th2 = 50.0
        if self.canny_aperture_size is None:
            self.canny_aperture_size = 3
        if self.do_merge is None:
            self.do_merge = False

    def _detect_lines(self, image: np.ndarray, ann: LocatedObjects, label: str):
        """
        Processes the contours and adds the polygons to the annotations.

        :param image: the image to process
        :param ann: the annotations to append
        :type ann: LocatedObjects
        :param label: the label to use
        :type label: str
        """
        fld = cv2.ximgproc.createFastLineDetector(length_threshold=self.length_threshold,
                                                  distance_threshold=self.distance_threshold,
                                                  canny_th1=self.canny_th1,
                                                  canny_th2=self.canny_th2,
                                                  canny_aperture_size=self.canny_aperture_size,
                                                  do_merge=self.do_merge)
        lines = fld.detect(image.astype(np.uint8))
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
