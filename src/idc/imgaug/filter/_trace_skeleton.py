import argparse
from typing import List, Tuple

import numpy as np

from wai.common.adams.imaging.locateobjects import LocatedObject, LocatedObjects
from wai.common.geometry import Polygon, Point
from wai.logging import LOGGING_WARNING
from kasperl.api import make_list, flatten_list, safe_deepcopy
from idc.api import ObjectDetectionData, ImageSegmentationData, ImageData, APPLY_TO_IMAGE, APPLY_TO_ANNOTATIONS, APPLY_TO_BOTH, add_apply_to_param, image_to_bytesio, LABEL_KEY
from idc.filter import RequiredFormatFilter, REQUIRED_FORMAT_BINARY, OUTPUT_FORMAT_ASIS, add_output_format, array_to_output_format
from ._thinning_utils import thinning, traceSkeleton


class TraceSkeleton(RequiredFormatFilter):
    """
    Thinning and tracing algorithm developed by Lingdong Huang.
    https://github.com/LingDong-/skeleton-tracing/blob/master/py/trace_skeleton.py
    """

    def __init__(self, apply_to: str = None, chunk_size: int = None, max_iter: int = None,
                 incorrect_format_action: str = None, output_format: str = None, label: str = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param apply_to: where to apply the filter to
        :type apply_to: str
        :param chunk_size: the chunk size to use
        :type chunk_size: int
        :param max_iter: maximum number of iterations
        :type max_iter: int
        :param incorrect_format_action: how to react to incorrect input format
        :type incorrect_format_action: str
        :param output_format: the output format to use
        :type output_format: str
        :param label: the label to use when processing images other than image segmentation
        :type label: str
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(incorrect_format_action=incorrect_format_action, logger_name=logger_name, logging_level=logging_level)
        self.apply_to = apply_to
        self.chunk_size = chunk_size
        self.max_iter = max_iter
        self.output_format = output_format
        self.label = label

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "trace-skeleton"

    def description(self) -> str:
        """
        Returns a description of the filter.

        :return: the description
        :rtype: str
        """
        return "Thinning and tracing algorithm developed by Lingdong Huang: https://github.com/LingDong-/skeleton-tracing/blob/master/py/trace_skeleton.py"

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        add_apply_to_param(parser)
        parser.add_argument("-c", "--chunk_size", metavar="SIZE", type=int, help="The chunk size to use.", default=10, required=False)
        parser.add_argument("-m", "--max_iter", metavar="ITER", type=int, help="The maximum number of iterations to perform.", default=999, required=False)
        add_output_format(parser)
        parser.add_argument("--label", type=str, help="The label to use when processing images other than image segmentation ones.", default="object", required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.apply_to = ns.apply_to
        self.chunk_size = ns.chunk_size
        self.max_iter = ns.max_iter
        self.output_format = ns.output_format
        self.label = ns.label

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

    def _required_format(self) -> str:
        """
        Returns what input format is required for applying the filter.

        :return: the type of image
        :rtype: str
        """
        return REQUIRED_FORMAT_BINARY

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if self.apply_to is None:
            self.apply_to = APPLY_TO_IMAGE
        if self.output_format is None:
            self.output_format = OUTPUT_FORMAT_ASIS
        if self.label is None:
            self.label = "object"

    def _apply_filter(self, array: np.ndarray, ann: LocatedObjects, label: str) -> np.ndarray:
        """
        Applies the filter to the image and returns the numpy array.

        :param array: the image the filter to apply to
        :type array: np.ndarray
        :param ann: the annotations to append
        :type ann: LocatedObjects
        :param label: the label to use
        :type label: str
        :return: the filtered image
        :rtype: np.ndarray
        """
        # perform thinning
        array_new = array.astype(np.uint64)
        array_new = np.where(array_new > 0, 1, 0)
        array_thin = thinning(array_new)
        array_thin = np.where(array_thin > 0, 255, 0).astype(np.uint8)

        # trace skeleton
        rects = []
        polys = traceSkeleton(array_thin, 0, 0, array_thin.shape[1], array_thin.shape[0], self.chunk_size, self.max_iter, rects)
        for poly in polys:
            if len(poly) < 2:
                continue
            else:
                for i in range(len(poly) - 1):
                    x0, y0 = poly[i]
                    x1, y1 = poly[i+1]
                    points = [Point(x0, y0), Point(x1, y1), Point(x0, y0)]
                    left = min([p.x for p in points])
                    top = min([p.y for p in points])
                    right = max([p.x for p in points])
                    bottom = max([p.y for p in points])
                    polygon = Polygon(*points)
                    obj = LocatedObject(left, top, right - left + 1, bottom - top + 1)
                    obj.set_polygon(polygon)
                    obj.metadata[LABEL_KEY] = label
                    ann.append(obj)

        return array_thin

    def _do_process(self, data):
        """
        Processes the data record(s).

        :param data: the record(s) to process
        :return: the potentially updated record(s)
        """
        result = []
        for item in make_list(data):
            ann = LocatedObjects()
            # apply to image
            if self.apply_to in [APPLY_TO_IMAGE, APPLY_TO_BOTH]:
                # incorrect format?
                if not self._can_process(item.image):
                    result.append(item)
                    continue
                # process
                image = self._ensure_correct_format(item.image)
                array = np.asarray(image).astype(np.uint8)
                array_new = self._apply_filter(array, ann, self.label)
            # apply to annotations, nothing to do for image
            else:
                array_new = np.asarray(item.image).astype(np.uint8)

            # generate image/bytes
            img_new = array_to_output_format(array_new, self.output_format, self.logger())
            bytes_new = image_to_bytesio(img_new, item.image_format).getvalue()

            # apply to annotations?
            annotation_new = safe_deepcopy(item.annotation)
            if isinstance(item, ImageSegmentationData) and item.has_annotation():
                if self.apply_to in [APPLY_TO_ANNOTATIONS, APPLY_TO_BOTH]:
                    for label in annotation_new.layers:
                        self._apply_filter(annotation_new.layers[label], ann, label)

            item_new = ObjectDetectionData(image_name=item.image_name,
                                           data=bytes_new,
                                           metadata=safe_deepcopy(item.get_metadata()),
                                           annotation=ann)
            result.append(item_new)

        return flatten_list(result)
