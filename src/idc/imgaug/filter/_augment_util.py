import imageio.v2 as imageio
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from wai.common.adams.imaging.locateobjects import LocatedObjects, absolute_to_normalized
from wai.common.geometry import Point as WaiPoint
from wai.common.geometry import Polygon as WaiPolygon

from idc.api import ImageData, ObjectDetectionData, ImageSegmentationData, combine_layers, split_layers, array_to_image


def augment_image(item: ImageData, pipeline, image_name: str = None) -> ImageData:
    """
    Augments the image by applying the pipeline. The annotations get processed accordingly.

    :param item: the image to augment
    :type item: ImageData
    :param image_name: the new image name, uses the current one when None
    :type image_name: str
    :param pipeline: the augmentation pipeline
    :return: the potentially updated image
    :rtype: ImageData
    """
    if image_name is None:
        image_name = item.image_name

    image = imageio.imread(item.image_bytes)

    # convert annotations
    bboxesoi = None
    polysoi = None
    imgsegmap = None
    normalized = False
    annotation = item.annotation

    if item.has_annotation():
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
        image_aug, bbs_aug = pipeline(image=image, bounding_boxes=bboxesoi)
    elif polysoi is not None:
        image_aug, polys_aug = pipeline(image=image, polygons=polysoi)
    elif imgsegmap is not None:
        image_aug, imgsegmap_aug = pipeline(image=image, segmentation_maps=imgsegmap)
    else:
        image_aug = pipeline(image=image)

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

    _, img_new_bytes = array_to_image(image_aug, item.image_format)

    result = type(item)(image_name=image_name, data=img_new_bytes.getvalue(),
                        image_format=item.image_format,
                        metadata=item.get_metadata(), annotation=annotation_new)
    return result
