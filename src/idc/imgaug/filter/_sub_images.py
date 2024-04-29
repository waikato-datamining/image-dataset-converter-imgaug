import argparse
from typing import List

from seppl.io import Filter
from wai.logging import LOGGING_WARNING

from idc.api import ImageClassificationData, ObjectDetectionData, ImageSegmentationData, flatten_list, make_list
from idc.imgaug.filter._sub_images_utils import REGION_SORTING_NONE, REGION_SORTING, PLACEHOLDERS, DEFAULT_SUFFIX, \
    parse_regions, process_image


class SubImages(Filter):
    """
    Extracts sub-images (incl their annotations) from the images coming through, using the defined regions.
    """

    def __init__(self, regions: List[str] = None, region_sorting: str = REGION_SORTING_NONE,
                 include_partial: bool = False, suppress_empty: bool = False, suffix: str = DEFAULT_SUFFIX,
                 pad_width: int = None, pad_height: int = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param regions: the regions (X,Y,WIDTH,HEIGHT) to crop and forward with their annotations
        :type regions: list
        :param region_sorting: how to sort the supplied region definitions
        :type region_sorting: str
        :param include_partial: whether to include only annotations that fit fully into a region or also partial ones
        :type include_partial: bool
        :param suppress_empty: suppresses sub-images that have no annotations (object detection)
        :type suppress_empty: bool
        :param suffix: the suffix pattern to use for the generated sub-images (with placeholders)
        :type suffix: str
        :param pad_width: the width to pad to, return as is if None
        :type pad_width: int
        :param pad_height: the height to pad to, return as is if None
        :type pad_height: int
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.regions = regions
        self.region_sorting = region_sorting
        self.include_partial = include_partial
        self.suppress_empty = suppress_empty
        self.suffix = suffix
        self.pad_width = pad_width
        self.pad_height = pad_height
        self._regions_xyxy = None
        self._regions_lobj = None

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "sub-images"

    def description(self) -> str:
        """
        Returns a description of the filter.

        :return: the description
        :rtype: str
        """
        return "Extracts sub-images (incl their annotations) from the images coming through, using the defined regions. "\
               "When using x/y in the suffix, these images can be reassembled using the 'idc-combine-sub-images' tool."

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
        parser.add_argument("-r", "--regions", type=str, default=None, help="The regions (X,Y,WIDTH,HEIGHT) to crop and forward with their annotations (0-based coordinates)", required=True, nargs="+")
        parser.add_argument("-s", "--region_sorting", choices=REGION_SORTING, default=REGION_SORTING_NONE, help="How to sort the supplied region definitions", required=False)
        parser.add_argument("-p", "--include_partial", action="store_true", help="Whether to include only annotations that fit fully into a region or also partial ones", required=False)
        parser.add_argument("-e", "--suppress_empty", action="store_true", help="Suppresses sub-images that have no annotations", required=False)
        parser.add_argument("-S", "--suffix", type=str, default=DEFAULT_SUFFIX, help="The suffix pattern to use for the generated sub-images, available placeholders: " + "|".join(
            PLACEHOLDERS), required=False)
        parser.add_argument("--pad_width", type=int, default=None, help="The width to pad the sub-images to (on the right).", required=False)
        parser.add_argument("--pad_height", type=int, default=None, help="The height to pad the sub-images to (at the bottom).", required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.regions = ns.regions
        self.region_sorting = ns.region_sorting
        self.include_partial = ns.include_partial
        self.suppress_empty = ns.suppress_empty
        self.suffix = ns.suffix
        self.pad_width = ns.pad_width
        self.pad_height = ns.pad_height

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()

        if (self.regions is None) or (len(self.regions) == 0):
            raise Exception("No region definitions supplied!")
        if self.region_sorting is None:
            self.region_sorting = REGION_SORTING_NONE
        if self.include_partial is None:
            self.include_partial = False
        if self.suppress_empty is None:
            self.suppress_empty = False
        if self.suffix is None:
            self.suffix = DEFAULT_SUFFIX

        self._regions_lobj, self._regions_xyxy = parse_regions(self.regions, self.region_sorting, self.logger())

    def _do_process(self, data):
        """
        Processes the data record(s).

        :param data: the record(s) to process
        :return: the potentially updated record(s)
        """
        result = []

        for item in make_list(data):
            sub_items = process_image(item, self._regions_lobj, self._regions_xyxy, self.suffix,
                                      self.suppress_empty, self.include_partial, self.logger(),
                                      pad_width=self.pad_width, pad_height=self.pad_height)
            # failed to process?
            if sub_items is None:
                result.append(item)
            else:
                for _, sub_item, _ in sub_items:
                    result.append(sub_item)

        return flatten_list(result)
