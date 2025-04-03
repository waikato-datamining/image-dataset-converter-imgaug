import argparse
from typing import List

from idc.api import ObjectDetectionData, flatten_list, make_list
from idc.imgaug.filter._sub_images_utils import PLACEHOLDERS, DEFAULT_SUFFIX, \
    extract_regions, locatedobject_to_xyxy
from seppl.io import Filter
from wai.logging import LOGGING_WARNING


class RegionOfInterestImages(Filter):
    """
    Extracts sub-images using the bbox of all the annotations that have matching labels.
    """

    def __init__(self, labels: List[str] = None, suffix: str = DEFAULT_SUFFIX,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param labels: the label(s) to extract, extracts all if None or the list is empty
        :type labels: list
        :param suffix: the suffix pattern to use for the generated sub-images (with placeholders)
        :type suffix: str
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.labels = labels
        self.suffix = suffix

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
        return "Extracts sub-images using the bbox of all the object detection annotations that have matching labels. If no labels are specified, all annotations are extracted."

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

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if (self.labels is not None) and (len(self.labels) == 0):
            self.labels = None
        if self.suffix is None:
            self.suffix = DEFAULT_SUFFIX

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
                result.append(item)
            else:
                for _, sub_item, _ in sub_items:
                    result.append(sub_item)

        return flatten_list(result)
