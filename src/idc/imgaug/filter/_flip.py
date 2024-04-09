import argparse
from typing import List

from ._base_image_augmentation import BaseImageAugmentation, IMGAUG_MODE_REPLACE
from wai.logging import LOGGING_WARNING
from idc.api import ImageClassificationData, ObjectDetectionData, ImageSegmentationData
import imgaug.augmenters as iaa

LEFT_TO_RIGHT = "lr"
UP_TO_DOWN = "ud"
LEFT_TO_RIGHT_AND_UP_TO_DOWN = "lrud"

DIRECTIONS = [
    LEFT_TO_RIGHT,
    UP_TO_DOWN,
    LEFT_TO_RIGHT_AND_UP_TO_DOWN
]


class Flip(BaseImageAugmentation):
    """
    Flips images either left-to-right, up-to-down or both.
    """

    def __init__(self, mode: str = IMGAUG_MODE_REPLACE, suffix: str = None,
                 seed: int = None, seed_augmentation: bool = False, threshold: float = 0.0,
                 direction: str = None, logger_name: str = None, logging_level: str = LOGGING_WARNING):
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
        :param direction: the direction to flip
        :type direction: str
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(mode=mode, suffix=suffix, seed=seed,
                         seed_augmentation=seed_augmentation, threshold=threshold,
                         logger_name=logger_name, logging_level=logging_level)
        self.direction = direction

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "flip"

    def description(self) -> str:
        """
        Returns a description of the filter.

        :return: the description
        :rtype: str
        """
        return "Flips images either left-to-right, up-to-down or both."

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
        parser.add_argument("-d", "--direction", choices=DIRECTIONS, help="The direction to flip.", default=None, required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.direction = ns.direction

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if self.direction is None:
            raise Exception("No flip direction defined!")
        if self.direction not in DIRECTIONS:
            raise Exception("Invalid flip direction: %s" % self.direction)

    def _default_suffix(self):
        """
        Returns the default suffix to use for images when using "add" rather than "replace" as mode.

        :return: the default suffix
        :rtype: str
        """
        return "-flipped"

    def _can_augment(self):
        """
        Checks whether augmentation can take place.

        :return: whether we can augment
        :rtype: bool
        """
        if self.direction is None:
            return False
        return True

    def _create_pipeline(self, aug_seed):
        """
        Creates and returns the augmentation pipeline.

        :param aug_seed: the seed value to use, can be None
        :type aug_seed: int
        :return: the pipeline
        :rtype: iaa.Sequential
        """
        if self.direction == LEFT_TO_RIGHT:
            return iaa.Sequential([
                iaa.Fliplr(),
            ])
        elif self.direction == UP_TO_DOWN:
            return iaa.Sequential([
                iaa.Flipud(),
            ])
        elif self.direction == LEFT_TO_RIGHT_AND_UP_TO_DOWN:
            return iaa.Sequential([
                iaa.Fliplr(),
                iaa.Flipud(),
            ])
        else:
            raise Exception("Unsupported direction: %s" % self.direction)
