import argparse
import math
import sys
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from wai.common.adams.imaging.locateobjects import LocatedObject
from wai.logging import LOGGING_WARNING

from idc.api import ImageData, REQUIRED_FORMAT_GRAYSCALE, ensure_grayscale
from idc.filter import RequiredFormatFilter
from idc.imgaug.filter._sub_images_utils import extract_regions
from kasperl.api import make_list, flatten_list
from ._aruco import ARUCO_TYPES, DEFAULT_ARUCO_TYPE

ARUCO_CROP_TYPE_OUTSIDE = "outside"
ARUCO_CROP_TYPE_INSIDE = "inside"
ARUCO_CROP_TYPES = [
    ARUCO_CROP_TYPE_OUTSIDE,
    ARUCO_CROP_TYPE_INSIDE,
]

DEFAULT_CROP_TYPE = ARUCO_CROP_TYPE_OUTSIDE


@dataclass
class Rectangle:
    label: str = None
    top: int = None
    left: int = None
    bottom: int = None
    right: int = None

    @property
    def center_x(self) -> int:
        """
        Returns the x value for the center of the rectangle.

        :return: the center x
        :rtype: int
        """
        return self.left + (self.right - self.left + 1) // 2

    @property
    def center_y(self) -> int:
        """
        Returns the y value for the center of the rectangle.

        :return: the center y
        :rtype: int
        """
        return self.top + (self.bottom - self.top + 1) // 2

    @property
    def width(self) -> int:
        """
        Returns the width of the rectangle.

        :return: the width
        :rtype: int
        """
        return self.right - self.left + 1

    @property
    def height(self) -> int:
        """
        Returns the height of the rectangle.

        :return: the height
        :rtype: int
        """
        return self.bottom - self.top + 1

    def distance_to_center(self, x: int, y: int) -> float:
        """
        Computes the distance of the rectangle's center to the coordinates.

        :param x: the x to compute distance to
        :type x: int
        :param y: the y to compute distance to
        :type y: int
        :return: the distance from the center to this point
        :rtype: float
        """
        return math.sqrt((self.center_x - x)**2 + (self.center_y - y)**2)

    def __str__(self):
        """
        String representation of the rectangle.

        :return: short description of the rectangle (for debugging mainly)
        """
        return "t=%s, l=%s, b=%s, r=%s, label=%s" % (str(self.top), str(self.left), str(self.bottom), str(self.right), str(self.label))


class ArucoCrop(RequiredFormatFilter):
    """
    Crops the image according to the ArUco markers.
    """

    def __init__(self, incorrect_format_action: str = None,
                 aruco_type: str = None, min_num_markers: int = None, crop_type: str = None, crop_success_key: str = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param incorrect_format_action: how to react to incorrect input format
        :type incorrect_format_action: str
        :param aruco_type: the type of aruco to detect
        :type aruco_type: str
        :param min_num_markers: the minimum number of markers that require detecting
        :type min_num_markers: int
        :param crop_type: how to perform the crop in relation to the markers
        :type crop_type: str
        :param crop_success_key: the (optional) key in the meta-data to store the crop success in (true/false)
        :type crop_success_key: str
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(incorrect_format_action=incorrect_format_action,
                         logger_name=logger_name, logging_level=logging_level)
        self.aruco_type = aruco_type
        self.min_num_markers = min_num_markers
        self.crop_type = crop_type
        self.crop_success_key = crop_success_key

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "aruco-crop"

    def description(self) -> str:
        """
        Returns a description of the handler.

        :return: the description
        :rtype: str
        """
        return "Crops the image according to the ArUco markers."

    def _required_format(self) -> str:
        """
        Returns what input format is required for applying the filter.

        :return: the type of image
        :rtype: str
        """
        return REQUIRED_FORMAT_GRAYSCALE

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
        return [ImageData]

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        parser.add_argument("-t", "--aruco_type", choices=ARUCO_TYPES.keys(), default=DEFAULT_ARUCO_TYPE, help="The type of markers to detect.", required=False)
        parser.add_argument("-m", "--min_num_markers", type=int, default=3, help="The minimum number of markers that need to be detected in order to proceed with cropping.", required=False)
        parser.add_argument("-c", "--crop_type", choices=ARUCO_CROP_TYPES, default=DEFAULT_CROP_TYPE, help="How to crop in relation to the markers.", required=False)
        parser.add_argument("--crop_success_key", type=str, default=None, help="The meta-data key to store the crop success under (true/false).", required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.aruco_type = ns.aruco_type
        self.min_num_markers = ns.min_num_markers
        self.crop_type = ns.crop_type
        self.crop_success_key = ns.crop_success_key

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if self.aruco_type is None:
            self.aruco_type = DEFAULT_ARUCO_TYPE
        if self.min_num_markers is None:
            self.min_num_markers = 3
        if self.crop_type is None:
            self.crop_type = DEFAULT_CROP_TYPE

    def _determine_corner(self, marker_corners, left_top: bool) -> Tuple[int, int]:
        """
        Determines the specified corner from the marker coordinates.
        Though ArUco markers "know" their top/left, bottom/right corners, we won't use that information.

        :param marker_corners: the marker coordinates
        :param left_top: whether left/top or right/bottom corner
        :type left_top: bool
        :return: the tuple of x/y coordinates of the corner
        :rtype: tuple
        """
        if left_top:
            x = sys.maxsize
            y = sys.maxsize
            for i in range(4):
                x = min(x, marker_corners[i][0])
                y = min(y, marker_corners[i][1])
        else:
            x = 0
            y = 0
            for i in range(4):
                x = max(x, marker_corners[i][0])
                y = max(y, marker_corners[i][1])
        return int(x), int(y)

    def _determine_outer_rect(self, item: ImageData, markers: List[Rectangle]) -> Rectangle:
        """
        Determines the outer rectangle.

        :param markers: the markers to enclose in the rectangle
        :type markers: list
        :return: the rectangle that encompasses the markers
        :rtype: Rectangle
        """
        result = Rectangle(left=item.image_width - 1, top=item.image_height - 1, right=0, bottom=0)
        for marker in markers:
            result.left = min(result.left, marker.left)
            result.top = min(result.top, marker.top)
            result.right = max(result.right, marker.right)
            result.bottom = max(result.bottom, marker.bottom)
        return result

    def _determine_inner_rect(self, item: ImageData, markers: List[Rectangle]) -> Rectangle:
        """
        Determines the inner rectangle.

        :param markers: the markers to use for determining the rectangle
        :type markers: list
        :return: the rectangle that sits within the markers
        :rtype: Rectangle
        """
        outer = self._determine_outer_rect(item, markers)

        # determine locations of the markers in relation to the outer rectangle corners
        tl = None
        tr = None
        bl = None
        br = None
        tld = max(item.image_width, item.image_height)
        trd = max(item.image_width, item.image_height)
        bld = max(item.image_width, item.image_height)
        brd = max(item.image_width, item.image_height)
        max_dist = math.sqrt((outer.width // 2)**2 + (outer.height // 2)**2)
        for marker in markers:
            # tl
            d = marker.distance_to_center(outer.left, outer.top)
            if (d < tld) and (d < max_dist):
                tld = d
                tl = marker
            # tr
            d = marker.distance_to_center(outer.right, outer.top)
            if (d < trd) and (d < max_dist):
                trd = d
                tr = marker
            # bl
            d = marker.distance_to_center(outer.left, outer.bottom)
            if (d < bld) and (d < max_dist):
                bld = d
                bl = marker
            # br
            d = marker.distance_to_center(outer.right, outer.bottom)
            if (d < brd) and (d < max_dist):
                brd = d
                br = marker

        # determine corners of inner rectangle
        x0 = 0
        y0 = 0
        x1 = item.image_width
        y1 = item.image_height
        if tl is not None:
            x0 = max(x0, tl.right)
            y0 = max(y0, tl.bottom)
        if tr is not None:
            x1 = min(x1, tr.left)
            y0 = max(y0, tr.bottom)
        if bl is not None:
            x0 = max(x0, bl.right)
            y1 = min(y1, bl.top)
        if br is not None:
            x1 = min(x1, br.left)
            y1 = min(y1, br.top)

        result = Rectangle(left=x0, top=y0, right=x1, bottom=y1)

        return result

    def _add_crop_success(self, item, success: bool):
        """
        Stores the crop success/failure in the meta-data, if a key was specified.

        :param item: the item to update
        :param success: whether successful or not
        :type success: bool
        """
        if self.crop_success_key is None:
            return
        if not item.has_metadata():
            item.set_metadata(dict())
        item.get_metadata()[self.crop_success_key] = str(success)

    def _do_process(self, data):
        """
        Processes the data record(s).

        :param data: the record(s) to process
        :return: the potentially updated record(s)
        """
        result = []

        for item in make_list(data):
            # prepare image
            gray = ensure_grayscale(item.image, logger=self.logger())
            gray = np.array(gray).astype(np.uint8)

            # set up detector
            aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_TYPES[self.aruco_type])
            parameters = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

            # detect markers
            all_corners, ids, rejected = detector.detectMarkers(gray)
            if ids is None:
                self.logger().warning("No markers detected, skipping: %s" % item.image_name)
                self._add_crop_success(item, False)
                result.append(item)
                continue
            elif len(ids) < self.min_num_markers:
                self.logger().warning("Insufficient # of markers detected (%d < %d), skipping: %s" % (len(ids), self.min_num_markers, item.image_name))
                self._add_crop_success(item, False)
                result.append(item)
                continue
            else:
                self.logger().info("# markers detected: %d" % len(ids))

            # determine marker positions
            markers = []
            for marker_corners, marker_id in zip(all_corners, ids):
                marker_corners = marker_corners.squeeze()
                marker_id = marker_id.squeeze()
                lt = self._determine_corner(marker_corners, True)
                rb = self._determine_corner(marker_corners, False)
                marker = Rectangle(left=lt[0], top=lt[1], right=rb[0], bottom=rb[1], label=str(marker_id))
                markers.append(marker)
                self.logger().info("marker: %s" % str(marker))

            # determine crop area
            if self.crop_type == ARUCO_CROP_TYPE_OUTSIDE:
                crop_rect = self._determine_outer_rect(item, markers)
            elif self.crop_type == ARUCO_CROP_TYPE_INSIDE:
                crop_rect = self._determine_inner_rect(item, markers)
            else:
                self._add_crop_success(item, False)
                result.append(item)
                self.logger().warning("Unhandled crop type '%s', skipping: %s" % (self.crop_type, item.image_name))
                continue

            # crop incl annotations
            self.logger().info("crop rect: %s" % str(crop_rect))
            bbox = (crop_rect.left, crop_rect.top, crop_rect.right, crop_rect.bottom)
            obj = LocatedObject(crop_rect.left, crop_rect.top, crop_rect.width, crop_rect.height)
            try:
                new_items = extract_regions(item, regions_lobj=[obj], regions_xyxy=[bbox], suppress_empty=False, suffix="",
                                            include_partial=True, logger=self.logger())
            except:
                self.logger().exception("Failed to extract regions!")
                self._add_crop_success(item, False)
                result.append(item)

            if len(new_items) == 1:
                self._add_crop_success(new_items[0][1], True)
                result.append(new_items[0][1])
            else:
                self.logger().warning("Crop failed?")
                self._add_crop_success(item, False)
                result.append(item)

        return flatten_list(result)
