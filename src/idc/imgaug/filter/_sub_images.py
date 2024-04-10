import argparse
import io

from typing import List

from seppl.io import Filter
from wai.logging import LOGGING_WARNING
from wai.common.adams.imaging.locateobjects import LocatedObjects
from idc.api import ImageClassificationData, ObjectDetectionData, ImageSegmentationData, flatten_list, make_list
from idc.imgaug.filter._sub_images_utils import REGION_SORTING_NONE, REGION_SORTING, PLACEHOLDERS, DEFAULT_SUFFIX, \
    parse_regions, region_filename, fit_located_object, fit_layers


class SubImages(Filter):
    """
    Extracts sub-images (incl their annotations) from the images coming through, using the defined regions.
    """

    def __init__(self, regions: List[str] = None, region_sorting: str = REGION_SORTING_NONE,
                 include_partial: bool = False, suppress_empty: bool = False, suffix: str = DEFAULT_SUFFIX,
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
        return "Extracts sub-images (incl their annotations) from the images coming through, using the defined regions."

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
        parser.add_argument("-e", "--suppress_empty", action="store_true", help="Suppresses sub-images that have no annotations (object detection and image segmentation)", required=False)
        parser.add_argument("-S", "--suffix", type=str, default=DEFAULT_SUFFIX, help="The suffix pattern to use for the generated sub-images, available placeholders: " + "|".join(
            PLACEHOLDERS), required=False)
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

        self._regions_xyxy, self._regions_lobj = parse_regions(self.regions, self.region_sorting, self.logger())

    def _do_process(self, data):
        """
        Processes the data record(s).

        :param data: the record(s) to process
        :return: the potentially updated record(s)
        """
        result = []

        for item in make_list(data):
            pil = item.image
            for region_index, region_xyxy in enumerate(self._regions_xyxy):
                self.logger().info("Applying region %d :%s" % (region_index, str(region_xyxy)))

                # crop image
                x0, y0, x1, y1 = region_xyxy
                if x1 > item.image_width:
                    x1 = item.image_width
                if y1 > item.image_height:
                    y1 = item.image_height
                sub_image = pil.crop((x0, y0, x1, y1))
                sub_bytes = io.BytesIO()
                sub_image.save(sub_bytes, format=item.image_format)
                image_name_new = region_filename(item.image_name, self._regions_lobj, self._regions_xyxy, region_index, self.suffix)

                # crop annotations and forward
                region_lobj = self._regions_lobj[region_index]
                if isinstance(item, ImageClassificationData):
                    item_new = ImageClassificationData(image_name=image_name_new, data=sub_bytes.getvalue(),
                                                       annotation=item.annotation, metadata=item.get_metadata())
                    result.append(item_new)
                elif isinstance(item, ObjectDetectionData):
                    new_objects = []
                    for ann_lobj in item.annotation:
                        ratio = region_lobj.overlap_ratio(ann_lobj)
                        if ((ratio > 0) and self.include_partial) or (ratio >= 1):
                            new_objects.append(fit_located_object(region_index, region_lobj, ann_lobj, self.logger()))
                    if not self.suppress_empty or (len(new_objects) > 0):
                        item_new = ObjectDetectionData(image_name=image_name_new, data=sub_bytes.getvalue(),
                                                       annotation=LocatedObjects(new_objects), metadata=item.get_metadata())
                        result.append(item_new)
                elif isinstance(item, ImageSegmentationData):
                    new_annotations = fit_layers(region_lobj, item.annotation, self.suppress_empty)
                    if not self.suppress_empty or (len(new_annotations.layers) > 0):
                        item_new = ImageSegmentationData(image_name=image_name_new, data=sub_bytes.getvalue(),
                                                         annotation=new_annotations, metadata=item.get_metadata())
                        result.append(item_new)
                else:
                    self.logger().warning("Unhandled data (%s), skipping!" % str(type(item)))
                    result.append(item)

        return flatten_list(result)
