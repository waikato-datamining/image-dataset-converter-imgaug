from typing import List

import numpy as np

from idc.api import ImageClassificationData, ObjectDetectionData, ImageSegmentationData, REQUIRED_FORMAT_BINARY
from idc.filter import ImageAndAnnotationFilter
from ._thinning_utils import thinning


class Thinning(ImageAndAnnotationFilter):
    """
    Thinning algorithm developed by Lingdong Huang.
    https://github.com/LingDong-/skeleton-tracing/blob/master/py/trace_skeleton.py
    """

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "thinning"

    def description(self) -> str:
        """
        Returns a description of the filter.

        :return: the description
        :rtype: str
        """
        return "Thinning algorithm developed by Lingdong Huang: https://github.com/LingDong-/skeleton-tracing/blob/master/py/trace_skeleton.py"

    def accepts(self) -> List:
        """
        Returns the list of classes that are accepted.

        :return: the list of classes
        :rtype: list
        """
        return [ImageClassificationData, ObjectDetectionData, ImageSegmentationData]

    def generates(self) -> List:
        """
        Returns the list of classes that get produced.

        :return: the list of classes
        :rtype: list
        """
        return [ImageClassificationData, ObjectDetectionData, ImageSegmentationData]

    def _required_format(self) -> str:
        """
        Returns what input format is required for applying the filter.

        :return: the type of image
        :rtype: str
        """
        return REQUIRED_FORMAT_BINARY

    def _apply_filter(self, source: str, array: np.ndarray) -> np.ndarray:
        """
        Applies the filter to the image and returns the numpy array.

        :param source: whether image or layer
        :type source: str
        :param array: the image the filter to apply to
        :type array: np.ndarray
        :return: the filtered image
        :rtype: np.ndarray
        """
        array_new = array.astype(np.uint8)
        array_new = np.where(array_new > 0, 1, 0)
        array_thin = thinning(array_new)
        array_thin = np.where(array_thin > 0, 255, 0).astype(np.uint8)
        return array_thin
