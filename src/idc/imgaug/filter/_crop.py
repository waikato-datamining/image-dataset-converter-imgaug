import argparse
from typing import List

from ._base_image_augmentation import BaseImageAugmentation, IMGAUG_MODE_REPLACE
from wai.logging import LOGGING_WARNING
from idc.api import ImageClassificationData, ObjectDetectionData, ImageSegmentationData
import imgaug.augmenters as iaa


class Crop(BaseImageAugmentation):
    """
    Crops images.
    """

    def __init__(self, mode: str = IMGAUG_MODE_REPLACE, suffix: str = None,
                 seed: int = None, seed_augmentation: bool = False, threshold: float = 0.0,
                 from_percent: float = None, to_percent: float = None, update_size: bool = False,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param mode: the image augmentation mode to use
        :type mode: str
        :param suffix: the suffix to use for the file names in case of augmentation mode 'add'
        :type suffix: str
        :param seed: the seed value to use for the random number generator; randomly seeded if not provided
        :type seed: int
        :param seed_augmentation: whether to seed the augmentation; if specified, uses the seeded random generator to produce a seed value
        :type seed_augmentation: bool
        :param threshold: the threshold to use for Random.rand(): if equal or above, augmentation gets applied; range: 0-1; default: 0 (= always)
        :type threshold: float
        :param from_percent: the minimum percent to crop from images (0-1)
        :type from_percent: float
        :param to_percent: the maximum percent to crop from images (0-1)
        :type to_percent: float
        :param update_size: whether to update the image size after the crop operation or scale back to original size
        :type update_size: bool
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(mode=mode, suffix=suffix, seed=seed,
                         seed_augmentation=seed_augmentation, threshold=threshold,
                         logger_name=logger_name, logging_level=logging_level)
        self.from_percent = from_percent
        self.to_percent = to_percent
        self.update_size = update_size

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "crop"

    def description(self) -> str:
        """
        Returns a description of the filter.

        :return: the description
        :rtype: str
        """
        return "Crops images."

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
        parser.add_argument("-f", "--from_percent", type=float, help="The minimum percent to crop from images (0-1).", default=None, required=False)
        parser.add_argument("-t", "--to_percent", type=float, help="The maximum percent to crop from images (0-1).", default=None, required=False)
        parser.add_argument("-u", "--update_size", action="store_true", help="Whether to update the image size after the crop operation or scale back to original size.", default=None, required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.from_percent = ns.from_percent
        self.to_percent = ns.to_percent
        self.update_size = ns.update_size

    def _default_suffix(self):
        """
        Returns the default suffix to use for images when using "add" rather than "replace" as mode.

        :return: the default suffix
        :rtype: str
        """
        return "-cropped"

    def _can_augment(self):
        """
        Checks whether augmentation can take place.

        :return: whether we can augment
        :rtype: bool
        """
        return (self.from_percent is not None) and (self.to_percent is not None)

    def _create_pipeline(self, aug_seed):
        """
        Creates and returns the augmentation pipeline.

        :param aug_seed: the seed value to use, can be None
        :type aug_seed: int
        :return: the pipeline
        :rtype: iaa.Sequential
        """
        keep_size = not self.update_size

        if self.from_percent == self.to_percent:
            return iaa.Sequential([
                iaa.Crop(percent=self.from_percent, keep_size=keep_size)
            ])
        else:
            return iaa.Sequential([
                iaa.Crop(percent=(self.from_percent, self.to_percent), keep_size=keep_size)
            ])
