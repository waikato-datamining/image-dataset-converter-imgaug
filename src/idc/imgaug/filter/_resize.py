import argparse
from typing import List, Union

import imgaug.augmenters as iaa
from seppl.io import Filter
from wai.logging import LOGGING_WARNING

from idc.api import ImageClassificationData, ObjectDetectionData, ImageSegmentationData, flatten_list, make_list
from ._augment_util import augment_image

KEEP_ASPECT_RATIO = "keep-aspect-ratio"


class Resize(Filter):
    """
    Resizes all images according to the specified width/height. When only resizing one dimension, use 'keep-aspect-ratio' for the other one to keep the aspect ratio intact.
    """

    def __init__(self, width: Union[int, str] = None, height: Union[int, str] = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param width: the new width of the image
        :param height: the new height of the image
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.width = width
        self.height = height
        self._width = None
        self._height = None

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "resize"

    def description(self) -> str:
        """
        Returns a description of the filter.

        :return: the description
        :rtype: str
        """
        return "Resizes all images according to the specified width/height. When only resizing one dimension, use '%s' for the other one to keep the aspect ratio intact." % KEEP_ASPECT_RATIO

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

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        parser.add_argument("-W", "--width", type=str, help="The new width for the image; use '%s' when only supplying height and you want to keep the aspect ratio intact." % KEEP_ASPECT_RATIO, default=KEEP_ASPECT_RATIO, required=False)
        parser.add_argument("-H", "--height", type=str, help="The new height for the image; use '%s' when only supplying width and you want to keep the aspect ratio intact." % KEEP_ASPECT_RATIO, default=KEEP_ASPECT_RATIO, required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.width = ns.width
        self.height = ns.height

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if self.width is None:
            self.width = KEEP_ASPECT_RATIO
        if self.height is None:
            self.height = KEEP_ASPECT_RATIO

        try:
            self._width = int(self.width)
        except:
            self._width = self.width
        try:
            self._height = int(self.height)
        except:
            self._height = self.height

    def _do_process(self, data):
        """
        Processes the data record(s).

        :param data: the record(s) to process
        :return: the potentially updated record(s)
        """
        # nothing to do?
        if (self.height == self.width) and (self.height == KEEP_ASPECT_RATIO):
            return data

        result = []
        aug = iaa.Resize({"height": self._height, "width": self._width})
        for item in make_list(data):
            item_new = augment_image(item, aug)
            result.append(item_new)

        return flatten_list(result)
