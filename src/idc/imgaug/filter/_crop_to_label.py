import argparse
from typing import List, Optional

from seppl.io import Filter
from wai.logging import LOGGING_WARNING

from idc.api import ObjectDetectionData, flatten_list, make_list
from idc.imgaug.filter._sub_images_utils import extract_regions


class CropToLabel(Filter):
    """
    Resizes all images according to the specified width/height. When only resizing one dimension, use 'keep-aspect-ratio' for the other one to keep the aspect ratio intact.
    """

    def __init__(self, region_label: str = None, keep_missing: bool = False,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param region_label: the label of the region/bbox to crop to
        :type region_label: str
        :param keep_missing: whether to keep images that don't have the label
        :type keep_missing: bool
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.region_label = region_label
        self.keep_missing = keep_missing

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "crop-to-label"

    def description(self) -> str:
        """
        Returns a description of the filter.

        :return: the description
        :rtype: str
        """
        return "Crops an image to the bbox with the specified label."

    def accepts(self) -> List:
        """
        Returns the list of classes that are accepted.

        :return: the list of classes
        :rtype: list
        """
        return [ObjectDetectionData]

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
        parser.add_argument("-r", "--region_label", type=str, help="The label of the bbox to crop the image to.", required=True)
        parser.add_argument("-k", "--keep_missing", action="store_true", help="For keeping images that don't have the label instead of suppressing them.")
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.region_label = ns.region_label
        self.keep_missing = ns.keep_missing

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if self.keep_missing is None:
            self.keep_missing = False
        if self.region_label is None:
            raise Exception("No region label specified!")

    def _crop(self, item: ObjectDetectionData) -> Optional[ObjectDetectionData]:
        """
        Crops the image.

        :param item: the image to crop
        :type item: ObjectDetectionData
        :return: the cropped image or None if no label present
        :rtype: ObjectDetectionData
        """
        # get annotations
        ann = item.get_absolute()
        obj = None
        if ann is None:
            self.logger().warning("Failed to obtain absolute annotations!")
            if self.keep_missing:
                return item
            else:
                return None
        # locate 1st annotation with specified label
        for o in ann:
            if ("type" in o.metadata) and (o.metadata["type"] == self.region_label):
                obj = o
                break
        if obj is None:
            self.logger().warning("Label '%s' not found: %s" % (self.region_label, item.image_name))
            if self.keep_missing:
                return item
            else:
                return None

        # get sub region
        bbox = (obj.x, obj.y, obj.x + obj.width - 1, obj.y + obj.height - 1)
        new_items = extract_regions(item, regions_lobj=[obj], regions_xyxy=[bbox], suppress_empty=True, suffix="",
                                    include_partial=True, logger=self.logger())
        if len(new_items) == 0:
            if self.keep_missing:
                return item
            else:
                return None
        else:
            return new_items[0][1]

    def _do_process(self, data):
        """
        Processes the data record(s).

        :param data: the record(s) to process
        :return: the potentially updated record(s)
        """
        result = []
        for item in make_list(data):
            if not isinstance(item, ObjectDetectionData):
                self.logger().warning("Not object detection data: %s/%s" % (item.image_name, str(type(item))))
                if self.keep_missing:
                    result.append(item)
                continue
            if not item.has_annotation():
                self.logger().warning("No annotations available: %s" % item.image_name)
                if self.keep_missing:
                    result.append(item)
                continue
            # perform crop
            new_item = self._crop(item)
            if new_item is not None:
                result.append(new_item)

        return flatten_list(result)
