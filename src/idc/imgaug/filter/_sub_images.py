import argparse
from typing import List

from seppl.io import Filter
from wai.logging import LOGGING_WARNING

from idc.api import ImageClassificationData, ObjectDetectionData, ImageSegmentationData, flatten_list, make_list
from idc.imgaug.filter._sub_images_utils import REGION_SORTING_NONE, REGION_SORTING, PLACEHOLDERS, DEFAULT_SUFFIX, \
    parse_regions, extract_regions, generate_regions, regions_to_string


class SubImages(Filter):
    """
    Extracts sub-images (incl their annotations) from the images coming through, using the defined regions.
    """

    def __init__(self, regions: List[str] = None, region_sorting: str = REGION_SORTING_NONE,
                 num_rows: int = None, num_cols: int = None, row_height: int = None, col_width: int = None,
                 overlap_right: int = None, overlap_bottom: int = None,
                 include_partial: bool = False, suppress_empty: bool = False, suffix: str = DEFAULT_SUFFIX,
                 pad_width: int = None, pad_height: int = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param regions: the regions (X,Y,WIDTH,HEIGHT) to crop and forward with their annotations
        :type regions: list
        :param region_sorting: how to sort the supplied region definitions
        :type region_sorting: str
        :param num_rows: the number of rows to use, if no regions defined
        :type num_rows: int
        :param num_cols: the number of columns to use, if no regions defined
        :type num_cols: int
        :param row_height: the height of rows
        :type row_height: int
        :param col_width: the width of columns
        :type col_width: int
        :param overlap_right: the overlap in pixels on the right, if no regions defined
        :type overlap_right: int
        :param overlap_bottom: the overlap in pixels on the bottom, if no regions defined
        :type overlap_bottom: int
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
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.row_height = row_height
        self.col_width = col_width
        self.overlap_right = overlap_right
        self.overlap_bottom = overlap_bottom
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
        return "Extracts sub-images (incl their annotations) from the images coming through, using the defined regions or #rows/cols. "\
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
        parser.add_argument("-r", "--regions", type=str, default=None, help="The regions (X,Y,WIDTH,HEIGHT) to crop and forward with their annotations (0-based coordinates)", required=False, nargs="*")
        parser.add_argument("--num_rows", type=int, help="The number of rows, if no regions defined.", default=None, required=False)
        parser.add_argument("--num_cols", type=int, help="The number of columns, if no regions defined.", default=None, required=False)
        parser.add_argument("--row_height", type=int, help="The height of rows.", default=None, required=False)
        parser.add_argument("--col_width", type=int, help="The width of columns.", default=None, required=False)
        parser.add_argument("--overlap_right", type=int, help="The overlap between two images (on the right of the left-most image), if no regions defined.", default=0, required=False)
        parser.add_argument("--overlap_bottom", type=int, help="The overlap between two images (on the bottom of the top-most image), if no regions defined.", default=0, required=False)
        parser.add_argument("-s", "--region_sorting", choices=REGION_SORTING, default=REGION_SORTING_NONE, help="How to sort the supplied region definitions", required=False)
        parser.add_argument("-p", "--include_partial", action="store_true", help="Whether to include only annotations that fit fully into a region or also partial ones", required=False)
        parser.add_argument("-e", "--suppress_empty", action="store_true", help="Suppresses sub-images that have no annotations", required=False)
        parser.add_argument("-S", "--suffix", type=str, default=DEFAULT_SUFFIX, help="The suffix pattern to use for the generated sub-images, available placeholders: " + "|".join(PLACEHOLDERS), required=False)
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
        self.num_rows = ns.num_rows
        self.num_cols = ns.num_cols
        self.row_height = ns.row_height
        self.col_width = ns.col_width
        self.overlap_right = ns.overlap_right
        self.overlap_bottom = ns.overlap_bottom
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

        if (self.regions is not None) and (len(self.regions) == 0):
            self.regions = None
        if (self.num_rows is None) or (self.num_cols is None):
            self.num_rows = None
            self.num_cols = None
        if (self.row_height is None) or (self.col_width is None):
            self.row_height = None
            self.col_width = None
        if (self.regions is None) and (self.num_rows is None) and (self.row_height is None):
            raise Exception("Neither regions nor #rows/cols nor row height/col width specified!")
        if self.region_sorting is None:
            self.region_sorting = REGION_SORTING_NONE
        if self.include_partial is None:
            self.include_partial = False
        if self.suppress_empty is None:
            self.suppress_empty = False
        if self.suffix is None:
            self.suffix = DEFAULT_SUFFIX
        if self.overlap_right is None:
            self.overlap_right = 0
        if self.overlap_bottom is None:
            self.overlap_bottom = 0
            
        self._regions_lobj = None
        self._regions_xyxy = None
        if self.regions is not None:
            self._regions_lobj, self._regions_xyxy = parse_regions(self.regions, self.region_sorting, self.logger())

    def _do_process(self, data):
        """
        Processes the data record(s).

        :param data: the record(s) to process
        :return: the potentially updated record(s)
        """
        result = []

        for item in make_list(data):
            if self._regions_lobj is not None:
                regions_lobj = self._regions_lobj
                regions_xyxy = self._regions_xyxy
            elif (self.num_rows is not None) and (self.num_cols is not None):
                regions = generate_regions(item.image_width, item.image_height,
                                           num_rows=self.num_rows, num_cols=self.num_cols,
                                           overlap_right=self.overlap_right, overlap_bottom=self.overlap_bottom,
                                           logger=self.logger())
                regions_str = regions_to_string(regions, logger=self.logger())
                regions_lobj, regions_xyxy = parse_regions(regions_str.split(" "), self.region_sorting, self.logger())
            else:
                regions = generate_regions(item.image_width, item.image_height,
                                           row_height=self.row_height, col_width=self.col_width,
                                           overlap_right=self.overlap_right, overlap_bottom=self.overlap_bottom,
                                           logger=self.logger())
                regions_str = regions_to_string(regions, logger=self.logger())
                regions_lobj, regions_xyxy = parse_regions(regions_str.split(" "), self.region_sorting, self.logger())
            sub_items = extract_regions(item, regions_lobj, regions_xyxy, self.suffix,
                                        self.suppress_empty, self.include_partial, self.logger(),
                                        pad_width=self.pad_width, pad_height=self.pad_height)
            # failed to process?
            if sub_items is None:
                result.append(item)
            else:
                for _, sub_item, _ in sub_items:
                    result.append(sub_item)

        return flatten_list(result)
