import argparse
import math
from typing import List

import cv2
import numpy as np
from wai.common.adams.imaging.locateobjects import LocatedObjects, LocatedObject
from wai.logging import LOGGING_WARNING

from idc.filter import RequiredFormatFilter, REQUIRED_FORMAT_GRAYSCALE
from kasperl.api import make_list, flatten_list, safe_deepcopy
from idc.api import ImageData, ObjectDetectionData, ImageSegmentationData, add_apply_to_param, APPLY_TO_IMAGE, \
    APPLY_TO_BOTH, APPLY_TO_ANNOTATIONS, LABEL_KEY, DEFAULT_LABEL


class SimpleBlobDetector(RequiredFormatFilter):
    """
    Finds blobs in grayscale images/annotations and stores them as rectangle annotations.
    """

    def __init__(self, apply_to: str = None, incorrect_format_action: str = None,
                 label: str = None, logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param apply_to: where to apply the filter to
        :type apply_to: str
        :param incorrect_format_action: how to react to incorrect input format
        :type incorrect_format_action: str
        :param label: the label to use for the detected blobs
        :type label: str
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(incorrect_format_action=incorrect_format_action,
                         logger_name=logger_name, logging_level=logging_level)
        self.apply_to = apply_to
        self.label = label
        self.blob_color = None
        self.filter_by_area = None
        self.filter_by_circularity = None
        self.filter_by_color = None
        self.filter_by_convexity = None
        self.filter_by_inertia = None
        self.max_area = None
        self.max_circularity = None
        self.max_convexity = None
        self.max_inertia_ratio = None
        self.max_threshold = None
        self.min_area = None
        self.min_circularity = None
        self.min_convexity = None
        self.min_dist_between_blobs = None
        self.min_inertia_ratio = None
        self.min_threshold = None
        self.threshold_step = None

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "simple-blob-detector"

    def description(self) -> str:
        """
        Returns a description of the handler.

        :return: the description
        :rtype: str
        """
        return "Finds blobs in grayscale images/annotations and stores them as rectangle annotations."

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
        return [ObjectDetectionData]

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        add_apply_to_param(parser)
        parser.add_argument("--label", type=str, default=DEFAULT_LABEL, help="The label to use for the blob annotations.", required=False)
        parser.add_argument("--min_threshold", type=str, default=None, help="The minimum threshold (inclusive) for converting to binary.", required=False)
        parser.add_argument("--max_threshold", type=str, default=None, help="The maximum threshold (exclusive) for converting to binary.", required=False)
        parser.add_argument("--threshold_step", type=str, default=None, help="The distance thresholdStep between neighboring thresholds.", required=False)
        parser.add_argument("--filter_by_color", action="store_true", help="This filter compares the intensity of a binary image at the center of a blob to blobColor. If they differ, the blob is filtered out. Use blobColor = 0 to extract dark blobs and blobColor = 255 to extract light blobs.", required=False)
        parser.add_argument("--blob_color", type=int, default=None, help="The blob color to use.", required=False)
        parser.add_argument("--filter_by_area", action="store_true", help="Extracted blobs have an area between minArea (inclusive) and maxArea (exclusive).", required=False)
        parser.add_argument("--min_area", type=str, default=None, help="The minimum area to use.", required=False)
        parser.add_argument("--max_area", type=float, default=None, help="The maximum area.", required=False)
        parser.add_argument("--filter_by_circularity", action="store_true", help="Extracted blobs have circularity ((4∗π∗Area)/(perimeter∗perimeter)) between minCircularity (inclusive) and maxCircularity (exclusive).", required=False)
        parser.add_argument("--min_circularity", type=str, default=None, help="The minimum circularity.", required=False)
        parser.add_argument("--max_circularity", type=str, default=None, help="The maximum circularity.", required=False)
        parser.add_argument("--filter_by_convexity", action="store_true", help="Extracted blobs have convexity (area / area of blob convex hull) between minConvexity (inclusive) and maxConvexity (exclusive).", required=False)
        parser.add_argument("--min_convexity", type=str, default=None, help="The minimum convexity.", required=False)
        parser.add_argument("--max_convexity", type=str, default=None, help="The maximum convexity.", required=False)
        parser.add_argument("--filter_by_inertia", action="store_true", help="Extracted blobs have this ratio between minInertiaRatio (inclusive) and maxInertiaRatio (exclusive).", required=False)
        parser.add_argument("--min_inertia_ratio", type=str, default=None, help="The minimum inertia ratio.", required=False)
        parser.add_argument("--max_inertia_ratio", type=str, default=None, help="The maximum inertia ratio.", required=False)
        parser.add_argument("--min_dist_between_blobs", type=str, default=None, help="The minimum distance between detected blobs.", required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.apply_to = ns.apply_to
        self.label = ns.label
        self.blob_color = ns.blob_color
        self.filter_by_area = ns.filter_by_area
        self.filter_by_circularity = ns.filter_by_circularity
        self.filter_by_color = ns.filter_by_color
        self.filter_by_convexity = ns.filter_by_convexity
        self.filter_by_inertia = ns.filter_by_inertia
        self.max_area = ns.max_area
        self.max_circularity = ns.max_circularity
        self.max_convexity = ns.max_convexity
        self.max_inertia_ratio = ns.max_inertia_ratio
        self.max_threshold = ns.max_threshold
        self.min_area = ns.min_area
        self.min_circularity = ns.min_circularity
        self.min_convexity = ns.min_convexity
        self.min_dist_between_blobs = ns.min_dist_between_blobs
        self.min_inertia_ratio = ns.min_inertia_ratio
        self.min_threshold = ns.min_threshold
        self.threshold_step = ns.threshold_step

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if self.apply_to is None:
            self.apply_to = APPLY_TO_IMAGE
        if self.label is None:
            self.label = DEFAULT_LABEL

    def _add_blobs(self, keypoints, ann: LocatedObjects, label: str):
        """
        Processes the keypoints and adds the polygons to the annotations.

        :param keypoints: the keypoints to process
        :param ann: the annotations to append
        :type ann: LocatedObjects
        :param label: the label to use
        :type label: str
        """
        for i in range(len(keypoints)):
            keypoint = keypoints[i]
            cx = keypoint.pt[0]
            cy = keypoint.pt[1]
            diameter = keypoint.size
            obj = LocatedObject(int(math.ceil(cx - diameter/2)), int(math.ceil(cy - diameter/2)), math.ceil(diameter), math.ceil(diameter))
            obj.metadata[LABEL_KEY] = label
            obj.metadata["cx"] = cx
            obj.metadata["cy"] = cy
            obj.metadata["diameter"] = diameter
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
            params = cv2.SimpleBlobDetector_Params()
            print(dir(params))
            if self.blob_color is not None:
                params.blobColor = self.blob_color
            if self.filter_by_area is not None:
                params.filterByArea = self.filter_by_area
            if self.filter_by_circularity is not None:
                params.filterByCircularity = self.filter_by_circularity
            if self.filter_by_color is not None:
                params.filterByColor = self.filter_by_color
            if self.filter_by_convexity is not None:
                params.filterByConvexity = self.filter_by_convexity
            if self.filter_by_inertia is not None:
                params.filterByInertia = self.filter_by_inertia
            if self.max_area is not None:
                params.maxArea = self.max_area
            if self.max_circularity is not None:
                params.maxCircularity = self.max_circularity
            if self.max_convexity is not None:
                params.maxConvexity = self.max_convexity
            if self.max_inertia_ratio is not None:
                params.maxInertiaRatio = self.max_inertia_ratio
            if self.max_threshold is not None:
                params.maxThreshold = self.max_threshold
            if self.min_area is not None:
                params.minArea = self.min_area
            if self.min_circularity is not None:
                params.minCircularity = self.min_circularity
            if self.min_convexity is not None:
                params.minConvexity = self.min_convexity
            if self.min_dist_between_blobs is not None:
                params.minDistBetweenBlobs = self.min_dist_between_blobs
            if self.min_inertia_ratio is not None:
                params.minInertiaRatio = self.min_inertia_ratio
            if self.min_threshold is not None:
                params.minThreshold = self.min_threshold
            if self.threshold_step is not None:
                params.thresholdStep = self.threshold_step
            detector = cv2.SimpleBlobDetector_create(params)

            if self.apply_to in [APPLY_TO_IMAGE, APPLY_TO_BOTH]:
                keypoints = detector.detect(np.array(item.image).astype(np.uint8))
                self.logger().info("# of blobs: %s" % str(len(keypoints)))
                self._add_blobs(keypoints, ann, self.label)
            if self.apply_to in [APPLY_TO_ANNOTATIONS, APPLY_TO_BOTH]:
                if isinstance(item, ImageSegmentationData):
                    for i, label in enumerate(item.annotation.labels):
                        if label not in item.annotation.layers:
                            continue
                        layer = item.annotation.layers[label]
                        keypoints = detector.detect(np.array(layer).astype(np.uint8))
                        self.logger().info("%s - # of blobs: %s" % (label, str(len(keypoints))))
                        self._add_blobs(keypoints, ann, label)

            self.logger().info("# of polygons added: %s" % str(len(ann)))
            item_new = ObjectDetectionData(source=item.source, image_name=item.image_name,
                                           image=safe_deepcopy(item.image), data=safe_deepcopy(item.data),
                                           annotation=ann, metadata=item.get_metadata())
            result.append(item_new)

        return flatten_list(result)
