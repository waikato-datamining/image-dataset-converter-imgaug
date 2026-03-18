import cv2
import logging


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

DEFAULT_ARUCO_TYPE = "DICT_6X6_250"


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
