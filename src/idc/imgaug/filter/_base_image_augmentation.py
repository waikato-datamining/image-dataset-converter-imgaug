import abc

import imgaug.augmenters as iaa
from wai.logging import LOGGING_WARNING

from idc.api import ImageData
from ._augment_util import augment_image
from ._base_filter import BaseFilter, IMGAUG_MODE_REPLACE


class BaseImageAugmentation(BaseFilter, abc.ABC):
    """
    Ancestor for image augmentation filters.
    """

    def __init__(self, mode: str = IMGAUG_MODE_REPLACE, suffix: str = None,
                 seed: int = None, seed_augmentation: bool = False, threshold: float = 0.0,
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
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(mode=mode, suffix=suffix, seed=seed,
                         seed_augmentation=seed_augmentation, threshold=threshold,
                         logger_name=logger_name, logging_level=logging_level)
    """
    Base class for stream processors that augment images.
    """

    def _create_pipeline(self, aug_seed):
        """
        Creates and returns the augmentation pipeline.

        :param aug_seed: the seed value to use, can be None
        :type aug_seed: int
        :return: the pipeline
        :rtype: iaa.Sequential
        """
        raise NotImplementedError()

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
        seq = self._create_pipeline(aug_seed)
        return augment_image(item, seq, image_name=image_name)
