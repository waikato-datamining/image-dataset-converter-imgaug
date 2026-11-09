import argparse
from typing import List, Optional

from wai.common.adams.imaging.locateobjects import LocatedObjects
from wai.logging import LOGGING_WARNING

from idc.api import ObjectDetectionData
from idc.imgaug.filter._sub_images_utils import reverse_categories_lookup, locatedobject_to_detection, \
    detection_to_locatedobject, polygon_nms
from kasperl.api import make_list, flatten_list
from seppl.io import BatchFilter


class RemoveOverlappingPolygons(BatchFilter):
    """
    Removes overlapping polygons.
    """

    def __init__(self, ios_threshold: float = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param ios_threshold: the threshold for intersect over small to use
        :type ios_threshold: float
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.ios_threshold = ios_threshold

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "remove-overlapping-polygons"

    def description(self) -> str:
        """
        Returns a description of the filter.

        :return: the description
        :rtype: str
        """
        return "Removes overlapping polygons."

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
        parser.add_argument("-t", "--ios_threshold", metavar="THRESH", type=float, default=0.7, help="The IoS (intersect over small) threshold for removing polygons.")
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.ios_threshold = ns.ios_threshold

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if self.ios_threshold is None:
            self.ios_threshold = 0.7

    def _remove_overlaps(self, item: ObjectDetectionData) -> Optional[ObjectDetectionData]:
        """
        Removes the overlaps.

        :param item: the image to process the annotations for
        :type item: ObjectDetectionData
        :return: the processed image
        :rtype: ObjectDetectionData
        """
        # get annotations
        ann = item.get_absolute()
        if ann is None:
            self.logger().warning("Failed to obtain absolute annotations!")
            return item

        # convert to Detection objects
        categories = dict()
        raw = []
        for obj in ann:
            detection = locatedobject_to_detection(obj, categories)
            raw.append(detection)

        pruned = polygon_nms(raw, ios_threshold=self.ios_threshold)
        self.logger().info("#raw: %d -> #after removal: %d" % (len(raw), len(pruned)))

        # assemble merged annotations
        category_ids = reverse_categories_lookup(categories)
        ann_new = LocatedObjects()
        for detection in pruned:
            ann_new.append(detection_to_locatedobject(detection, category_ids))

        result = item.duplicate(annotation=ann_new)
        return result

    def _do_process(self, data):
        """
        Processes the data record(s).

        :param data: the record(s) to process
        :return: the potentially updated record(s)
        """
        result = []
        for item in make_list(data):
            if isinstance(item, ObjectDetectionData) and item.has_annotation():
                result.append(self._remove_overlaps(item))
            else:
                result.append(item)

        return flatten_list(result)
