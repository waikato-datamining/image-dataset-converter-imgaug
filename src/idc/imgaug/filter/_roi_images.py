import argparse
from typing import List

from PIL import Image, ImageDraw
from wai.logging import LOGGING_WARNING

from seppl.io import BatchFilter
from kasperl.api import make_list, flatten_list
from idc.api import ObjectDetectionData, ImageData
from idc.imgaug.filter._sub_images_utils import PLACEHOLDERS, DEFAULT_SUFFIX, \
    extract_regions, locatedobject_to_xyxy


class RegionOfInterestImages(BatchFilter):
    """
    Extracts sub-images using the bbox of all the annotations that have matching labels.
    For polygons, the background outside the polygon can be blanked.
    """

    def __init__(self, labels: List[str] = None, suffix: str = DEFAULT_SUFFIX,
                 skip_failed: bool = None, blank_background: bool = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param labels: the label(s) to extract, extracts all if None or the list is empty
        :type labels: list
        :param suffix: the suffix pattern to use for the generated sub-images (with placeholders)
        :type suffix: str
        :param skip_failed: whether to skip images with no annotations or when extraction fails
        :type skip_failed: bool
        :param blank_background: whether to blank the background in case of polygons
        :type blank_background: bool
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.labels = labels
        self.suffix = suffix
        self.skip_failed = skip_failed
        self.blank_background = blank_background

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "roi-images"

    def description(self) -> str:
        """
        Returns a description of the filter.

        :return: the description
        :rtype: str
        """
        return "Extracts sub-images using the bbox of all the object detection annotations that have matching labels. If no labels are specified, all annotations are extracted. For polygons, the background outside the polygon can be blanked."

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
        parser.add_argument("--labels", type=str, default=None, help="The label(s) of the annotations to forward as sub-images, uses all annotations if not specified.", required=False, nargs="*")
        parser.add_argument("-S", "--suffix", type=str, default=DEFAULT_SUFFIX, help="The suffix pattern to use for the generated sub-images, available placeholders: " + "|".join(PLACEHOLDERS), required=False)
        parser.add_argument("-f", "--skip_failed", action="store_true", help="Removes images with no annotations or when failing to extract objects.", required=False)
        parser.add_argument("-b", "--blank_background", action="store_true", help="In case of polygon annotations, this flag enables removing the background outside of the polygon.", required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.labels = ns.labels
        self.suffix = ns.suffix
        self.skip_failed = ns.skip_failed
        self.blank_background = ns.blank_background

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if (self.labels is not None) and (len(self.labels) == 0):
            self.labels = None
        if self.suffix is None:
            self.suffix = DEFAULT_SUFFIX
        if self.skip_failed is None:
            self.skip_failed = False
        if self.blank_background is None:
            self.blank_background = False

    def _remove_background(self, item: ImageData):
        """
        Removes the background from the item if a polygon annotation is present.

        :param item: the object detection to remove the background for
        :type item: ImageData
        :return: the potentially updated item
        :rtype: ImageData
        """
        if not isinstance(item, ObjectDetectionData):
            return item

        ann = item.get_absolute()
        for lobj in ann:
            if lobj.has_polygon():
                im = item.image
                # create mask
                mask = Image.new("L", im.size, 0)
                draw = ImageDraw.Draw(mask)
                points = [(x, y) for x, y in zip(lobj.get_polygon_x(), lobj.get_polygon_y())]
                draw.polygon(points, fill=255)
                # create background
                background = Image.new("RGB", im.size, (0, 0, 0))
                # create composite
                im_new = Image.composite(im, background, mask)
                # assemble output
                item_new = item.duplicate(image=im_new)
                return item_new
            break

        return item

    def _do_process(self, data):
        """
        Processes the data record(s).

        :param data: the record(s) to process
        :return: the potentially updated record(s)
        """
        result = []
        labels = None
        if self.labels is not None:
            labels = set(self.labels)

        for item in make_list(data):
            ann = item.get_absolute()
            if ann is None:
                self.logger().warning("Failed to obtain absolute annotations!")
                if not self.skip_failed:
                    result.append(item)
                continue

            # determines bboxes to extract
            lobjs = []
            xyxys = []
            for o in ann:
                extract = (labels is None) or (("type" in o.metadata) and (o.metadata["type"] in labels))
                if not extract:
                    continue
                lobjs.append(o)
                xyxys.append(locatedobject_to_xyxy(o))

            sub_items = extract_regions(item, lobjs, xyxys, self.suffix,
                                        False, False, self.logger())
            # failed to process?
            if sub_items is None:
                if not self.skip_failed:
                    result.append(item)
            else:
                for _, sub_item, _ in sub_items:
                    if self.blank_background:
                        sub_item = self._remove_background(sub_item)
                    result.append(sub_item)

        return flatten_list(result)
