import argparse
from random import Random
from typing import List

import cv2
import numpy as np
from wai.logging import LOGGING_WARNING

from idc.api import ImageData, ImageClassificationData, ObjectDetectionData, ImageSegmentationData, array_to_image
from ._base_image_augmentation import BaseFilter, IMGAUG_MODE_REPLACE


class HSLGrayScale(BaseFilter):
    """
    Turns RGB images into fake grayscale ones by converting them to HSL and then using the L channel for all channels. The brightness can be influenced and varied even.
    """

    def __init__(self, mode: str = IMGAUG_MODE_REPLACE, suffix: str = None,
                 seed: int = None, seed_augmentation: bool = False, threshold: float = 0.0,
                 from_factor: float = None, to_factor: float = None,
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
        :param from_factor: the start of the factor range to apply to the L channel to darken or lighten the image (<1: darker, >1: lighter)
        :type from_factor: float
        :param to_factor: he end of the factor range to apply to the L channel to darken or lighten the image (<1: darker, >1: lighter)
        :type to_factor: float
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(mode=mode, suffix=suffix, seed=seed,
                         seed_augmentation=seed_augmentation, threshold=threshold,
                         logger_name=logger_name, logging_level=logging_level)
        self.from_factor = from_factor
        self.to_factor = to_factor

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "hsl-grayscale"

    def description(self) -> str:
        """
        Returns a description of the filter.

        :return: the description
        :rtype: str
        """
        return "Turns RGB images into fake grayscale ones by converting them to HSL and then using the L channel for all channels. The brightness can be influenced and varied even."

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
        parser.add_argument("-f", "--from_factor", type=float, help="The start of the factor range to apply to the L channel to darken or lighten the image (<1: darker, >1: lighter).", default=None, required=False)
        parser.add_argument("-t", "--to_factor", type=float, help="The end of the factor range to apply to the L channel to darken or lighten the image (<1: darker, >1: lighter).", default=None, required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.from_factor = ns.from_factor
        self.to_factor = ns.to_factor

    def _default_suffix(self):
        """
        Returns the default suffix to use for images when using "add" rather than "replace" as mode.

        :return: the default suffix
        :rtype: str
        """
        return "-gray"

    def _can_augment(self):
        """
        Checks whether augmentation can take place.

        :return: whether we can augment
        :rtype: bool
        """
        return (self.from_factor is not None) and (self.to_factor is not None)

    def _augment(self, item: ImageData, aug_seed: int, image_name: str) -> ImageData:
        """
        Augments the image.

        :param item: the image to augment
        :type item: ImageData
        :param aug_seed: the seed value to use, can be None
        :type aug_seed: int
        :param image_name: the new image name
        :type image_name: str
        :return: the potentially updated image
        :rtype: ImageData
        """
        # convert to HSL
        img_pil = item.image
        img_rgb = np.array(img_pil)
        img_hls = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HLS)
        img_l = img_hls[:, :, 1]

        # determine factor
        factor = None
        if (self.from_factor is not None) and (self.to_factor is not None):
            if self.from_factor == self.to_factor:
                factor = self.from_factor
            else:
                rnd = Random(aug_seed)
                factor = rnd.random() * (self.to_factor - self.from_factor) + self.from_factor

        # adjust brightness?
        if factor is not None:
            img_l = img_l * factor
            img_l = img_l.astype(np.uint8)

        # convert back to PIL bytes
        _, img_pil_bytes = array_to_image(img_l, item.image_format)
        result = type(item)(image_name=image_name, data=img_pil_bytes.getvalue(),
                            image_format=item.image_format,
                            metadata=item.get_metadata(), annotation=item.annotation)
        return result
