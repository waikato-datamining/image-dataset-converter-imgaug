import argparse
from typing import List

import cv2
import numpy as np
from wai.logging import LOGGING_WARNING

from idc.api import ImageData, ensure_grayscale
from kasperl.api import make_list, flatten_list, BatchFilter
from ._aruco import ARUCO_TYPES, DEFAULT_ARUCO_TYPE
from ._rotate import Rotate, IMGAUG_MODE_REPLACE


class ArucoAutoRotate(BatchFilter):
    """
    Automatically rotates the image according to the orientation of the ArUco marker(s) in 90degree increments.
    """

    def __init__(self, aruco_type: str = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param aruco_type: the type of aruco to detect
        :type aruco_type: str
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.aruco_type = aruco_type

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "aruco-autorotate"

    def description(self) -> str:
        """
        Returns a description of the handler.

        :return: the description
        :rtype: str
        """
        return "Automatically rotates the image according to the orientation of the ArUco marker(s) in 90degree increments."

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
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.aruco_type = ns.aruco_type

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if self.aruco_type is None:
            self.aruco_type = DEFAULT_ARUCO_TYPE

    def _do_process(self, data):
        """
        Processes the data record(s).

        :param data: the record(s) to process
        :return: the potentially updated record(s)
        """
        result = []

        for item in make_list(data):
            # prepare image
            gray = ensure_grayscale(item.image, logger=None)
            gray = np.array(gray).astype(np.uint8)

            # set up detector
            aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_TYPES[self.aruco_type])
            parameters = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

            # detect markers
            all_corners, ids, rejected = detector.detectMarkers(gray)
            if ids is None:
                self.logger().warning("No markers detected, skipping: %s" % item.image_name)
                result.append(item)
                continue
            else:
                self.logger().info("# markers detected: %d" % len(ids))

            # only use first marker to determine rotation
            marker_corners = all_corners[0]
            marker_corners = marker_corners.squeeze()
            tl_x = int(marker_corners[0][0])
            tl_y = int(marker_corners[0][1])
            br_x = int(marker_corners[2][0])
            br_y = int(marker_corners[2][1])

            if (tl_x < br_x) and (tl_y < br_y):
                deg = 0
            elif (tl_x > br_x) and (tl_y < br_y):
                deg = -90
            elif (tl_x > br_x) and (tl_y > br_y):
                deg = -180
            else:
                deg = -270

            if deg != 0:
                self.logger().info("Rotating image by: %d" % deg)
                rotate = Rotate(mode=IMGAUG_MODE_REPLACE, from_degree=deg, to_degree=deg, suffix="", logging_level=self.logging_level)
                rotate.session = self.session
                rotate.initialize()
                new_item = rotate.process(item)
                result.append(new_item)
            else:
                result.append(item)

        return flatten_list(result)
