import abc
import io

import imageio.v2 as imageio
import imgaug.augmenters as iaa
import numpy as np
from PIL import Image
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from wai.common.adams.imaging.locateobjects import LocatedObjects, absolute_to_normalized
from wai.common.geometry import Point as WaiPoint
from wai.common.geometry import Polygon as WaiPolygon
from wai.logging import LOGGING_WARNING

from idc.api import ImageData, ObjectDetectionData, ImageSegmentationData, combine_layers, split_layers
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

        image = imageio.imread(item.image_bytes)

        # convert annotations
        bboxesoi = None
        polysoi = None
        imgsegmap = None
        normalized = False
        annotation = item.annotation

        # object detection
        if isinstance(item, ObjectDetectionData):
            normalized = item.is_normalized()
            annotation = item.get_absolute()
            has_polys = False
            for obj in annotation:
                if obj.has_polygon():
                    has_polys = True
                    break
            if has_polys:
                polys = []
                for obj in annotation:
                    x = obj.get_polygon_x()
                    y = obj.get_polygon_y()
                    points = []
                    for i in range(len(x)):
                        points.append((x[i], y[i]))
                    poly = Polygon(points)
                    polys.append(poly)
                    polysoi = PolygonsOnImage(polys, shape=image.shape)
            else:
                bboxes = []
                for obj in annotation:
                    bbox = BoundingBox(x1=obj.x, y1=obj.y, x2=obj.x + obj.width - 1, y2=obj.y + obj.height - 1)
                    bboxes.append(bbox)
                bboxesoi = BoundingBoxesOnImage(bboxes, shape=image.shape)

        # image segmentation
        elif isinstance(item, ImageSegmentationData):
            combined = combine_layers(item)
            imgsegmap = SegmentationMapsOnImage(combined, shape=(item.image_height, item.image_width))

        # augment
        bbs_aug = None
        polys_aug = None
        imgsegmap_aug = None
        if bboxesoi is not None:
            image_aug, bbs_aug = seq(image=image, bounding_boxes=bboxesoi)
        elif polysoi is not None:
            image_aug, polys_aug = seq(image=image, polygons=polysoi)
        elif imgsegmap is not None:
            image_aug, imgsegmap_aug = seq(image=image, segmentation_maps=imgsegmap)
        else:
            image_aug = seq(image=image)

        # update annotations
        objs_aug = None
        annotation_new = annotation
        if bbs_aug is not None:
            objs_aug = []
            for i, bbox in enumerate(bbs_aug):
                # skip ones outside image
                if bbox.is_out_of_image(image_aug):
                    continue
                # clip bboxes to fit into image
                bbox = bbox.clip_out_of_image(image_aug)
                # update located object
                obj_aug = annotation[i].get_clone()
                obj_aug.x = bbox.x1
                obj_aug.y = bbox.y1
                obj_aug.width = bbox.x2 - bbox.x1 + 1
                obj_aug.height = bbox.y2 - bbox.y1 + 1
                objs_aug.append(obj_aug)
                annotation_new = LocatedObjects(objs_aug)
        elif polys_aug is not None:
            objs_aug = []
            for i, poly in enumerate(polys_aug):
                # skip ones outside image
                if poly.is_out_of_image(image_aug):
                    continue
                # clip bboxes to fit into image
                polys = poly.clip_out_of_image(image_aug)
                if len(polys) == 0:
                    continue
                for p in polys:
                    # update located object
                    obj_aug = annotation[i].get_clone()
                    bbox = p.to_bounding_box()
                    obj_aug.x = bbox.x1
                    obj_aug.y = bbox.y1
                    obj_aug.width = bbox.x2 - bbox.x1 + 1
                    obj_aug.height = bbox.y2 - bbox.y1 + 1
                    points = []
                    for row in p.coords:
                        points.append(WaiPoint((int(row[0])), int(row[1])))
                    obj_aug.set_polygon(WaiPolygon(*points))
                    objs_aug.append(obj_aug)
            annotation_new = LocatedObjects(objs_aug)
        elif imgsegmap_aug is not None:
            annotation_new = split_layers(imgsegmap_aug.get_arr(), annotation.labels)

        # convert back to normalized space?
        if (objs_aug is not None) and normalized:
            annotation_new = absolute_to_normalized(annotation_new, item.image_width, item.image_height)

        img_new = Image.fromarray(np.uint8(image_aug))
        img_new_bytes = io.BytesIO()
        img_new.save(img_new_bytes, format=item.image_format)

        result = type(item)(image_name=image_name, data=img_new_bytes.getvalue(),
                            image_format=item.image_format,
                            metadata=item.get_metadata(), annotation=annotation_new)
        return result
