import argparse
from typing import List

from PIL import Image
from wai.common.adams.imaging.locateobjects import LocatedObject, LocatedObjects
from wai.logging import LOGGING_WARNING

from idc.api import ImageClassificationData, ImageSegmentationData, ObjectDetectionData, DepthData, adjust_matrix, \
    fit_located_object
from kasperl.api import make_list, flatten_list, safe_deepcopy
from seppl.io import BatchFilter


class SimpleCrop(BatchFilter):
    """
    Crops the image to the specified width/height.
    """

    def __init__(self, width: int = None, height: int = None, include_partial: bool = False,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param width: the minimum width to use
        :type width: int
        :param height: the minimum height to use
        :type height: int
        :param include_partial: whether to include only annotations that fit fully into a region or also partial ones
        :type include_partial: bool
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.width = width
        self.height = height
        self.include_partial = include_partial

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "simple-crop"

    def description(self) -> str:
        """
        Returns a description of the handler.

        :return: the description
        :rtype: str
        """
        return "Crops the image to the specified width/height."

    def accepts(self) -> List:
        """
        Returns the list of classes that are accepted.

        :return: the list of classes
        :rtype: list
        """
        return [ImageClassificationData, ImageSegmentationData, ObjectDetectionData, DepthData]

    def generates(self) -> List:
        """
        Returns the list of classes that get produced.

        :return: the list of classes
        :rtype: list
        """
        return [ImageClassificationData, ImageSegmentationData, ObjectDetectionData, DepthData]

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        parser.add_argument("-W", "--width", type=int, default=None, help="The width to crop to, ignored if not specified.", required=False)
        parser.add_argument("-H", "--height", type=int, default=None, help="The height to crop to, ignored if not specified.", required=False)
        parser.add_argument("-p", "--include_partial", action="store_true", help="Whether to include only annotations that fit fully into a region or also partial ones", required=False)
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
        self.include_partial = ns.include_partial

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if self.include_partial is None:
            self.include_partial = False

    def _do_process(self, data):
        """
        Processes the data record(s).

        :param data: the record(s) to process
        :return: the potentially updated record(s)
        """
        if (self.width is None) and (self.height is None):
            self.logger().warning("Neither width nor height specified!")
            return data

        result = []

        for item in make_list(data):
            modify = False
            width_new = item.image_width
            height_new = item.image_height
            if (self.width is not None) and (item.image_width > self.width):
                width_new = self.width
                self.logger().info("Updating width to: %d" % width_new)
                modify = True
            if (self.height is not None) and (item.image_height > self.height):
                height_new = self.height
                self.logger().info("Updating height to: %d" % height_new)
                modify = True
            if modify:
                img_new = Image.new(item.image.mode, (width_new, height_new), (0, 0, 0))
                img_new.paste(item.image, (0, 0))
                item_new = type(item)(image_name=item.image_name, image=img_new, image_format=item.image_format,
                                      annotation=safe_deepcopy(item.annotation),
                                      metadata=safe_deepcopy(item.get_metadata()))
                # adjust annotations as well
                if item.has_annotation():
                    if isinstance(item, ImageClassificationData):
                        # nothing to do
                        pass
                    elif isinstance(item, ImageSegmentationData):
                        # adjust the layers
                        for layer in item_new.annotation.layers:
                            item_new.annotation.layers[layer] = adjust_matrix(item_new.annotation.layers[layer], width_new, height_new)
                    elif isinstance(item, ObjectDetectionData):
                        # make sure to have absolute coordinates
                        if item.is_normalized():
                            annotation = item.get_absolute()
                        else:
                            annotation = item_new.annotation
                        # remove annotations that fall outside the crop
                        new_objects = []
                        region_lobj = LocatedObject(0, 0, width_new, height_new)
                        for ann_lobj in annotation:
                            ratio = ann_lobj.overlap_ratio(region_lobj)
                            if ((ratio > 0) and self.include_partial) or (ratio >= 1):
                                new_objects.append(fit_located_object(-1, region_lobj, ann_lobj, self.logger()))
                        item_new.annotation = LocatedObjects(new_objects)
                    elif isinstance(item, DepthData):
                        # adjust depth information matrix
                        item_new.annotation.data = adjust_matrix(item_new.annotation.data, width_new, height_new)
                    else:
                        self.logger().warning("Unhandled data type: %s" % str(item))
                result.append(item_new)
            else:
                self.logger().info("Nothing to do: %s" % item.image_name)
                result.append(item)

        return flatten_list(result)
