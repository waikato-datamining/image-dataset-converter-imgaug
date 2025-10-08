import argparse
import logging
from typing import List

import cv2
import numpy as np
from wai.logging import LOGGING_WARNING

from idc.api import ImageData, REQUIRED_FORMAT_GRAYSCALE, ensure_grayscale
from idc.filter import RequiredFormatFilter
from kasperl.api import make_list, flatten_list

DEFAULT_PREFIX = "aruco-"

DEFAULT_ARUCO_TYPE = "DICT_6X6_250"

ARUCO_TYPES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
}


def generate_aruco(size: int, aruco_id: int, aruco_type: str, output_file: str, logger: logging.Logger = None) -> bool:
    """
    Generates an AruCo code marker image.

    :param size: the size of the image
    :type size: int
    :param aruco_id: the ID to encode
    :type aruco_id: int
    :param aruco_type: the type of marker to generate
    :type aruco_type: str
    :param output_file: the file to store the generated marker in
    :type output_file: str
    :param logger: the optional logger instance to use
    :type logger: logging.Logger
    :return: whether image was successfully written
    :rtype: bool
    """
    if logger is not None:
        logger.info("Getting dictionary: %s" % aruco_type)
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_TYPES[aruco_type])
    if logger is not None:
        logger.info("Generating marker for ID: %d" % aruco_id)
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, aruco_id, size)
    if logger is not None:
        logger.info("Writing marker to: %s" % output_file)
    return cv2.imwrite(output_file, marker_image)


class ArucoDetector(RequiredFormatFilter):
    """
    Detects ArUco markers and adds them to the meta-data.
    """

    def __init__(self, incorrect_format_action: str = None,
                 prefix: str = None, aruco_type: str = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param incorrect_format_action: how to react to incorrect input format
        :type incorrect_format_action: str
        :param prefix: the prefix to use in the meta-data
        :type prefix: str
        :param aruco_type: the type of aruco to detect
        :type aruco_type: str
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(incorrect_format_action=incorrect_format_action,
                         logger_name=logger_name, logging_level=logging_level)
        self.prefix = prefix
        self.aruco_type = aruco_type

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "aruco-detector"

    def description(self) -> str:
        """
        Returns a description of the handler.

        :return: the description
        :rtype: str
        """
        return "Detects ArUco markers and adds them to the meta-data."

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
        parser.add_argument("-p", "--prefix", type=str, default=DEFAULT_PREFIX, help="The prefix to use for the detected markers in the meta-data.", required=False)
        parser.add_argument("-t", "--aruco_type", choices=ARUCO_TYPES.keys(), default=DEFAULT_ARUCO_TYPE, help="The type of markers to detect.", required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.prefix = ns.prefix
        self.aruco_type = ns.aruco_type

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if self.prefix is None:
            self.prefix = DEFAULT_PREFIX
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
            gray = ensure_grayscale(item.image, logger=self.logger())
            gray = np.array(gray).astype(np.uint8)

            # set up detector
            aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_TYPES[self.aruco_type])
            parameters = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

            # detect markers
            all_corners, ids, rejected = detector.detectMarkers(gray)
            self.logger().info("# markers detected: %d" % len(ids))

            # store results
            meta = None
            if len(ids) > 0:
                meta = dict()
                for marker_corners, marker_id in zip(all_corners, ids):
                    marker_corners = marker_corners.squeeze()
                    marker_id = marker_id.squeeze()
                    self.logger().info("marker id: %s" % str(marker_id))
                    meta[self.prefix + str(marker_id) + "-id"] = str(marker_id)
                    meta[self.prefix + str(marker_id) + "-topleft.x"] = int(marker_corners[0][0])
                    meta[self.prefix + str(marker_id) + "-topleft.y"] = int(marker_corners[0][1])
                    meta[self.prefix + str(marker_id) + "-topright.x"] = int(marker_corners[1][0])
                    meta[self.prefix + str(marker_id) + "-topright.y"] = int(marker_corners[1][1])
                    meta[self.prefix + str(marker_id) + "-bottomright.x"] = int(marker_corners[2][0])
                    meta[self.prefix + str(marker_id) + "-bottomright.y"] = int(marker_corners[2][1])
                    meta[self.prefix + str(marker_id) + "-bottomleft.x"] = int(marker_corners[3][0])
                    meta[self.prefix + str(marker_id) + "-bottomleft.y"] = int(marker_corners[3][1])

            item_new = item.duplicate()
            if meta is not None:
                if not item_new.has_metadata():
                    item_new.set_metadata(meta)
                else:
                    item_new.get_metadata().extend(meta)

            result.append(item_new)

        return flatten_list(result)
