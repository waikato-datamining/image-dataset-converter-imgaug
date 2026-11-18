from typing import List

import imgaug.augmenters as iaa

from idc.api import exif_orientation_str, ObjectDetectionData, \
    ImageSegmentationData, EXIF_ORIENTATION_NONE, EXIF_ORIENTATION_ROTATE_180, EXIF_ORIENTATION_ROTATE_270, \
    EXIF_ORIENTATION_ROTATE_90, EXIF_ORIENTATION_FLIP_LEFT_RIGHT, EXIF_ORIENTATION_FLIP_TOP_BOTTOM
from kasperl.api import make_list, flatten_list, BatchFilter
from ._augment_util import augment_image


class ExifAutoRotateAnnotations(BatchFilter):
    """
    Automatically rotates the image according to the EXIF orientation including the annotations.
    """

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "exif-autorotate-annotations"

    def description(self) -> str:
        """
        Returns a description of the handler.

        :return: the description
        :rtype: str
        """
        return "Automatically rotates the image and annotations according to the EXIF orientation."

    def accepts(self) -> List:
        """
        Returns the list of classes that are accepted.

        :return: the list of classes
        :rtype: list
        """
        return [ObjectDetectionData, ImageSegmentationData]

    def generates(self) -> List:
        """
        Returns the list of classes that get produced.

        :return: the list of classes
        :rtype: list
        """
        return [ObjectDetectionData, ImageSegmentationData]

    def _do_process(self, data):
        """
        Processes the data record(s).

        :param data: the record(s) to process
        :return: the potentially updated record(s)
        """
        result = []

        for item in make_list(data):
            orientation_str = exif_orientation_str(item.image)
            pipeline = None
            if orientation_str == EXIF_ORIENTATION_NONE:
                self.logger().info("No rotation necessary: %s" % item.image_name)
                result.append(item)
                continue
            elif orientation_str == EXIF_ORIENTATION_ROTATE_90:
                pipeline = iaa.Sequential([
                    iaa.Affine(
                        rotate=(90, 90),
                        seed=0,
                        fit_output=True,
                    )
                ])
            elif orientation_str == EXIF_ORIENTATION_ROTATE_180:
                pipeline = iaa.Sequential([
                    iaa.Affine(
                        rotate=(180, 180),
                        seed=0,
                        fit_output=True,
                    )
                ])
            elif orientation_str == EXIF_ORIENTATION_ROTATE_270:
                pipeline = iaa.Sequential([
                    iaa.Affine(
                        rotate=(270, 270),
                        seed=0,
                        fit_output=True,
                    )
                ])
            elif orientation_str == EXIF_ORIENTATION_FLIP_LEFT_RIGHT:
                pipeline = iaa.Sequential([
                    iaa.Fliplr(),
                ])
            elif orientation_str == EXIF_ORIENTATION_FLIP_TOP_BOTTOM:
                pipeline = iaa.Sequential([
                    iaa.Flipud(),
                ])
            else:
                self.logger().warning("Cannot rotate this orientation: %s/%s" % (item.image_name, orientation_str))
                result.append(item)
                continue

            if pipeline is not None:
                self.logger().info("Applying rotation: %s/%s" % (item.image_name, orientation_str))
                result.append(augment_image(item, pipeline, logger=self.logger()))

        return flatten_list(result)
