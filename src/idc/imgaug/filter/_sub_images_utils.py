import logging
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from wai.common.adams.imaging.locateobjects import LocatedObject, LocatedObjects
from wai.common.geometry import Point as WaiPoint, Polygon as WaiPolygon
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree

from idc.api import ImageSegmentationAnnotations, ImageClassificationData, ImageSegmentationData, ObjectDetectionData, \
    ImageData, crop_image, pad_image, fit_layers, fit_located_object, array_to_image, empty_image, DepthData, \
    fit_matrix, DepthInformation, intersect_over_union, locatedobject_polygon_to_shapely, shapely_to_polygon, \
    get_object_label

REGION_SORTING_NONE = "none"
REGION_SORTING_XY = "x-then-y"
REGION_SORTING_YX = "y-then-x"
REGION_SORTING = [
    REGION_SORTING_NONE,
    REGION_SORTING_XY,
    REGION_SORTING_YX,
]

PH_X = "{X}"
PH_Y = "{Y}"
PH_W = "{W}"
PH_H = "{H}"
PH_X0 = "{X0}"
PH_Y0 = "{Y0}"
PH_X1 = "{X1}"
PH_Y1 = "{Y1}"
PH_INDEX = "{INDEX}"
PLACEHOLDERS = [
    PH_X,
    PH_Y,
    PH_W,
    PH_H,
    PH_X0,
    PH_Y0,
    PH_X1,
    PH_Y1,
    PH_INDEX,
]

DEFAULT_SUFFIX = "-{INDEX}"


def parse_regions(regions: List[str], region_sorting: str, logger: logging.Logger) -> Tuple[List, List]:
    """
    Parses the string regions and returns xyxy and LocatedObject lists.

    :param regions: the regions to parse
    :type regions: list
    :param region_sorting: how to sort the regions
    :type region_sorting: str
    :param logger: for generating some logging info
    :type logger: logging.Logger
    :return: the tuple of LocatedObject list and xyxy tuple list
    :rtype: tuple
    """
    region_lobjs = []
    regions_xyxy = []
    for region in regions:
        coords = [int(x) for x in region.split(",")]
        if len(coords) == 4:
            x, y, w, h = coords
            region_lobjs.append(LocatedObject(x=x, y=y, width=w, height=h))

    logger.info("unsorted regions: %s" % str([str(x) for x in region_lobjs]))

    if region_sorting is not REGION_SORTING_NONE:
        if region_sorting == REGION_SORTING_XY:
            def sorting(obj: LocatedObject):
                return "%06d %06d" % (obj.x, obj.y)
        elif region_sorting == REGION_SORTING_YX:
            def sorting(obj: LocatedObject):
                return "%06d %06d" % (obj.y, obj.x)
        else:
            raise Exception("Unhandled region sorting: %s" % region_sorting)
        region_lobjs.sort(key=sorting)
        logger.info("sorted regions: %s" % str([str(x) for x in region_lobjs]))

    for lobj in region_lobjs:
        regions_xyxy.append((lobj.x, lobj.y, lobj.x + lobj.width - 1, lobj.y + lobj.height - 1))
    logger.info("sorted xyxy: %s" % str(regions_xyxy))

    return region_lobjs, regions_xyxy


def region_filename(path: str, regions_lobj: List[LocatedObject], regions_xyxy: List[Tuple], index: int, suffix_template: str) -> str:
    """
    Generates a new filename based on the original and the index of the region.

    :param path: the base filename
    :type path: str
    :param regions_lobj: the regions as located objects
    :type regions_lobj: list
    :param regions_xyxy: the regions as xyxy tuples
    :type regions_xyxy: list
    :param index: the region index
    :type index: int
    :param suffix_template: the template to use for the suffix
    :type suffix_template: str
    :return: the generated filename
    :rtype: str
    """
    parts = os.path.splitext(path)
    index_pattern = "%0" + str(len(str(len(regions_lobj)))) + "d"
    index_str = index_pattern % index
    suffix = suffix_template
    suffix = suffix.replace(PH_INDEX, index_str)
    suffix = suffix.replace(PH_X0, str(regions_xyxy[index][0]))
    suffix = suffix.replace(PH_Y0, str(regions_xyxy[index][1]))
    suffix = suffix.replace(PH_X1, str(regions_xyxy[index][2]))
    suffix = suffix.replace(PH_Y1, str(regions_xyxy[index][3]))
    suffix = suffix.replace(PH_X, str(regions_lobj[index].x))
    suffix = suffix.replace(PH_Y, str(regions_lobj[index].y))
    suffix = suffix.replace(PH_W, str(regions_lobj[index].width))
    suffix = suffix.replace(PH_H, str(regions_lobj[index].height))
    return parts[0] + suffix + parts[1]


def extract_regions(item: ImageData, regions_lobj: List[LocatedObject], regions_xyxy: List[Tuple], suffix: str,
                    suppress_empty: bool, include_partial: bool, logger: logging.Logger,
                    pad_width: Optional[int] = None, pad_height: Optional[int] = None,
                    min_partial_iou: float = 0.0) -> Optional[List[Tuple[LocatedObject, ImageData, LocatedObject]]]:
    """
    Processes the image according to the defined regions and returns a list of tuples consisting of the located
    object for the region and the new image/annotations.

    :param item: the image to process
    :type item: ImageData
    :param regions_lobj: the regions as list of located object
    :type regions_lobj: list
    :param regions_xyxy: the regions as list of xyxy tuples
    :type regions_xyxy: list
    :param suffix: the suffix template for the new images
    :type suffix: str
    :param suppress_empty: whether to suppress sub-images with no annotations
    :type suppress_empty: bool
    :param include_partial: whether to include partial objdet annotations (ones that get cut off by the region)
    :type include_partial: bool
    :param logger: for logging purposes
    :type logger: logging.Logger
    :param pad_width: the width to pad to, return as is if None
    :type pad_width: int
    :param pad_height: the height to pad to, return as is if None
    :type pad_height: int
    :return: the list of tuples (located object of region, new image, located object of original dims)
    :param min_partial_iou: when including partial objdet annotations, only keep if IoU between partial and full objdet is at least this threshold, ignored if 0
    :type min_partial_iou: float
    :rtype: list
    """
    result = []

    pil = item.image
    for region_index, region_xyxy in enumerate(regions_xyxy):
        logger.info("Applying region %d: %s" % (region_index, str(region_xyxy)))

        # crop image
        x0, y0, x1, y1 = region_xyxy
        if x1 >= item.image_width:
            x1 = item.image_width - 1
        if y1 >= item.image_height:
            y1 = item.image_height - 1
        sub_image = pil.crop((x0, y0, x1+1, y1+1))
        orig_dims = LocatedObject(0, 0, sub_image.size[0], sub_image.size[1])
        sub_image = pad_image(sub_image, pad_width=pad_width, pad_height=pad_height)
        _, sub_bytes = array_to_image(sub_image, item.image_format)
        image_name_new = region_filename(item.image_name, regions_lobj, regions_xyxy, region_index, suffix)

        # crop annotations and forward
        region_lobj = regions_lobj[region_index]
        if isinstance(item, ImageClassificationData):
            annotation = item.annotation
            if not suppress_empty or (annotation is not None):
                item_new = ImageClassificationData(image_name=image_name_new, data=sub_bytes.getvalue(),
                                                   annotation=annotation, metadata=item.get_metadata())
                result.append((region_lobj, item_new, orig_dims))

        elif isinstance(item, ObjectDetectionData):
            new_objects = []
            if item.has_annotation():
                for ann_lobj in item.annotation:
                    ratio = region_lobj.overlap_ratio(ann_lobj)
                    if ((ratio > 0) and include_partial) or (ratio >= 1):
                        ann_lobj_new = fit_located_object(region_index, region_lobj, ann_lobj, logger=logger, context=item.image_name + "/" + str(ann_lobj))
                        if (ratio < 1) and (min_partial_iou > 0):
                            ann_lobj_part = fit_located_object(region_index, region_lobj, ann_lobj, relocate=False, logger=logger, context=item.image_name + "/" + str(ann_lobj))
                            poly_full = locatedobject_polygon_to_shapely(ann_lobj)
                            poly_part = locatedobject_polygon_to_shapely(ann_lobj_part)
                            iou = intersect_over_union(poly_part, poly_full)
                            if iou >= min_partial_iou:
                                new_objects.append(ann_lobj_new)
                        else:
                            new_objects.append(ann_lobj_new)
            if not suppress_empty or (len(new_objects) > 0):
                item_new = ObjectDetectionData(image_name=image_name_new, data=sub_bytes.getvalue(),
                                               annotation=LocatedObjects(new_objects), metadata=item.get_metadata())
                result.append((region_lobj, item_new, orig_dims))

        elif isinstance(item, ImageSegmentationData):
            new_annotations = ImageSegmentationAnnotations(list(), dict())
            if item.has_annotation():
                new_annotations = fit_layers(region_lobj, item.annotation, suppress_empty)
            if not suppress_empty or (len(new_annotations.layers) > 0):
                for label in new_annotations.layers:
                    new_annotations.layers[label] = pad_image(new_annotations.layers[label], pad_width=pad_width, pad_height=pad_height)
                if len(new_annotations.layers) == 0:
                    new_annotations = None
                item_new = ImageSegmentationData(image_name=image_name_new, data=sub_bytes.getvalue(),
                                                 annotation=new_annotations, metadata=item.get_metadata())
                result.append((region_lobj, item_new, orig_dims))

        elif isinstance(item, DepthData):
            new_annotations = fit_matrix(region_lobj, item.annotation.data, suppress_empty)
            if not suppress_empty or (new_annotations is not None):
                item_new = DepthData(image_name=image_name_new, data=sub_bytes.getvalue(),
                                     annotation=DepthInformation(data=new_annotations), metadata=item.get_metadata())
                result.append((region_lobj, item_new, orig_dims))

        else:
            logger.warning("Unhandled data (%s), skipping!" % str(type(item)))
            return None

    return result


def new_from_template(item, rebuild_image: bool = False):
    """
    Creates an empty image container using the provided template.

    :param item: the template container
    :param rebuild_image: whether to rebuild the image from the sub-images (ie start with empty) or use input one
    :type rebuild_image: bool
    :return: the new container
    """
    if rebuild_image:
        _, img_bytes = empty_image(item.image.mode, item.image_width, item.image_height, item.image_format)
        data = img_bytes.getvalue()
    else:
        data = item.image_bytes

    if isinstance(item, ImageClassificationData):
        result = ImageClassificationData(image_name=item.image_name, data=data,
                                         metadata=item.get_metadata(), annotation=None)
    elif isinstance(item, ObjectDetectionData):
        result = ObjectDetectionData(image_name=item.image_name, data=data,
                                     metadata=item.get_metadata(), annotation=LocatedObjects())
    elif isinstance(item, ImageSegmentationData):
        labels = list()
        layers = dict()
        if item.has_annotation():
            labels = item.annotation.labels[:]
            for label in item.annotation.labels:
                layers[label] = np.zeros((item.image_height, item.image_width), dtype=np.uint8)
        annotation = ImageSegmentationAnnotations(labels, layers)
        result = ImageSegmentationData(image_name=item.image_name, data=data,
                                       metadata=item.get_metadata(), annotation=annotation)
    elif isinstance(item, DepthData):
        result = DepthData(image_name=item.image_name, data=data,
                           metadata=item.get_metadata(), annotation=DepthInformation())
    else:
        raise Exception("Unhandled type of data: %s" % str(type(item)))

    return result


def transfer_region(full_image, sub_image, region: LocatedObject, rebuild_image: bool = False,
                    crop_width: int = None, crop_height: int = None, logger: logging.Logger = None,
                    context: str = None):
    """
    Transfers the sub image into the full image according to the region.
    Annotations get transferred as well.

    :param full_image: the image to transfer the sub image into
    :param sub_image: the sub image to transfer
    :param region: the region in the full image to update
    :type region: LocatedObject
    :param rebuild_image: whether to rebuild the image from the sub-images (ie start with empty) or use input one
    :type rebuild_image: bool
    :param crop_width: the width to crop to, ignored if None
    :type crop_width: int
    :param crop_height: the height to crop to, ignored if None
    :type crop_height: int
    :param logger: the optional logger instance to use
    :type logger: logging.Logger
    :param context: the optional context to use in the logging output
    :type context: str
    """
    # transfer image
    if rebuild_image:
        cropped = crop_image(sub_image.image, crop_width=crop_width, crop_height=crop_height)
        full_image.image.paste(cropped, (region.x, region.y))

    is_debug = (logger is not None) and logger.isEnabledFor(logging.DEBUG)

    # transfer annotations
    if sub_image.annotation is not None:
        # image classification (comma-separated list of labels)
        if isinstance(full_image, ImageClassificationData):
            if full_image.annotation is None:
                full_image.annotation = sub_image.annotation
            else:
                labels = full_image.annotation.split(",")
                if sub_image.annotation not in labels:
                    labels.append(sub_image.annotation)
                    full_image.annotation = ",".join(labels)

        # object detection (relocate located objects)
        elif isinstance(full_image, ObjectDetectionData):
            if is_debug:
                logger.debug("Transfer to region: %s" % str(region))
            img_width = full_image.image_width
            img_height = full_image.image_height
            for lobj in sub_image.annotation:
                new_lobj = LocatedObject(lobj.x + region.x, lobj.y + region.y, lobj.width, lobj.height, **lobj.metadata)
                if lobj.has_polygon:
                    xs = [x+region.x for x in lobj.get_polygon_x()]
                    ys = [y+region.y for y in lobj.get_polygon_y()]
                    new_lobj.set_polygon(WaiPolygon(*(WaiPoint(x, y) for x, y in zip(xs, ys))))
                # skip objects to the right of the image
                if new_lobj.x >= img_width:
                    if is_debug:
                        logger.debug("Skipping object beyond the right side of image: %s" % str(new_lobj))
                    continue
                # skip objects below the bottom of the image
                if new_lobj.y >= img_height:
                    if is_debug:
                        logger.debug("Skipping object below bottom of image: %s" % str(new_lobj))
                    continue
                # fit object if necessary
                fit = False
                if (new_lobj.x < img_width) and (new_lobj.x + new_lobj.width >= img_width):
                    fit = True
                if (new_lobj.y < img_height) and (new_lobj.y + new_lobj.height >= img_height):
                    fit = True
                if fit:
                    if is_debug:
                        logger.debug("Fitting object: %s" % str(new_lobj))
                    image_bounds = LocatedObject(0, 0, img_width, img_height)
                    new_lobj = fit_located_object(-1, image_bounds, new_lobj, logger=logger, context=context)
                # add object
                full_image.annotation.append(new_lobj)

        # image segmentation
        elif isinstance(full_image, ImageSegmentationData):
            if is_debug:
                logger.debug("Transfer to region: %s" % str(region))
            for label in sub_image.annotation.layers:
                x = region.x
                y = region.y
                w = region.width
                h = region.height
                if (not full_image.has_annotation()) or (not full_image.has_layer(label)):
                    full_image.new_layer(label, labels=sub_image.annotation.labels)
                layer = sub_image.annotation.layers[label]
                layer = crop_image(layer, crop_width=crop_width, crop_height=crop_height)
                full_image.annotation.layers[label][y:y + h, x:x + w] += layer
                full_image.annotation.layers[label] = np.where(full_image.annotation.layers[label] > 0, 255, full_image.annotation.layers[label])

        # depth
        elif isinstance(full_image, DepthInformation):
            if is_debug:
                logger.debug("Transfer to region: %s" % str(region))
            x = region.x
            y = region.y
            w = region.width
            h = region.height
            sub = crop_image(sub_image.annotation.data, crop_width=crop_width, crop_height=crop_height)
            full_image.annotation.data[y:y + h, x:x + w] += sub
            full_image.annotation.data = np.where(full_image.annotation.data > 0, 255, full_image.annotation.data)

        # unknown
        else:
            raise Exception("Unhandled type of data: %s" % str(type(full_image)))


def prune_annotations(image):
    """
    Prunes the annotations.

    :param image: the image container to process
    """
    if isinstance(image, ImageClassificationData):
        # nothing to do
        pass

    elif isinstance(image, ObjectDetectionData):
        # no object? -> remove annotations
        if len(image.annotation) == 0:
            image.annotation = None

    elif isinstance(image, ImageSegmentationData):
        # check which layers are empty
        empty = []
        for label in image.annotation.layers:
            unique = np.unique(image.annotation.layers[label])
            if (len(unique) == 1) and (unique[0] == 0):
                empty.append(label)

        # remove empty layers
        for label in empty:
            del image.annotation.layers[label]

        # no layers left? -> remove annotations
        if len(image.annotation.layers) == 0:
            image.annotation = None

    elif isinstance(image, DepthData):
        unique = np.unique(image.annotation.data)
        if (len(unique) == 1) and (unique[0] == 0):
            image.annotation = None

    else:
        raise Exception("Unhandled type of data: %s" % str(type(image)))


@dataclass
class Region:
    x: int
    y: int
    w: int
    h: int


def generate_regions(width: int, height: int,
                     num_rows: int = None, num_cols: int = None, fixed_size: bool = False,
                     row_height: int = None, col_width: int = None, margin: int = 0,
                     overlap_right: int = 0, overlap_bottom: int = 0,
                     margin_left: int = 0, margin_top: int = 0, margin_right: int = 0, margin_bottom: int = 0,
                     partial: bool = False, logger=None) -> List[Region]:
    """
    Generates the regions and returns them. Either specify num_rows/num_cols or row_height/col_width.

    :param width: the width of the image
    :type width: int
    :param height: the height of the image
    :type height: int
    :param num_rows: the number of rows
    :type num_rows: int
    :param num_cols: the number of columns
    :type num_cols: int
    :param fixed_size: whether to use fixed width/heights for sub-images and omit left-over bits bottom/right when using num_rows/num_cols
    :type fixed_size: bool
    :param row_height: the height of rows
    :type row_height: int
    :param col_width: the width of columns
    :type col_width: int
    :param overlap_right: the overlap in pixels between two images (on the right)
    :type overlap_right: int
    :param overlap_bottom: the overlap in pixels between two images (on the bottom)
    :type overlap_bottom: int
    :param margin: the margin around the actual section to generate the regions from
    :type margin: int
    :param margin_left: the left margin of the actual section to generate the regions from
    :type margin_left: int
    :param margin_top: the top margin of the actual section to generate the regions from
    :type margin_top: int
    :param margin_right: the right margin of the actual section to generate the regions from
    :type margin_right: int
    :param margin_bottom: the bottom margin of the actual section to generate the regions from
    :type margin_bottom: int
    :param partial: whether to return partial regions right/bottom when using fixed row height/col width
    :type partial: bool
    :param logger: the optional logger to use for logging output
    :return: the list of generated Region objects
    :rtype: list
    """
    result = []

    # image dimensions
    if width < 1:
        raise ValueError("Image width must be at least 1!")
    if logger is not None:
        logger.info("width: %d" % width)
    if height < 1:
        raise ValueError("Image height must be at least 1!")
    if logger is not None:
        logger.info("height: %d" % height)

    # margins
    if margin > 0:
        if margin_left == 0:
            margin_left = margin
        if margin_top == 0:
            margin_top = margin
        if margin_right == 0:
            margin_right = margin
        if margin_bottom == 0:
            margin_bottom = margin
    if margin_left > 0:
        if logger is not None:
            logger.info("margin (left): %d" % margin_left)
    if margin_top > 0:
        if logger is not None:
            logger.info("margin (top): %d" % margin_top)
    if margin_right > 0:
        if logger is not None:
            logger.info("margin (right): %d" % margin_right)
    if margin_bottom > 0:
        if logger is not None:
            logger.info("margin (bottom): %d" % margin_bottom)

    # overlaps
    if overlap_right > 0:
        if logger is not None:
            logger.info("overlap (right): %d" % overlap_right)
    if overlap_bottom > 0:
        if logger is not None:
            logger.info("overlap (bottom): %d" % overlap_bottom)

    # section
    section_width = width - margin_left - margin_right
    if section_width < width:
        if logger is not None:
            logger.info("section width: %d" % section_width)
    section_height = height - margin_top - margin_bottom
    if section_height < height:
        if logger is not None:
            logger.info("section height: %d" % section_height)

    mode = None
    if (num_rows is not None) and (num_cols is not None):
        mode = "grid"
    if (row_height is not None) and (col_width is not None):
        mode = "size"
    if mode is None:
        raise ValueError("Either num_rows/num_cols or row_height/col_width must be specified!")

    # fixed grid
    if mode == "grid":
        if logger is not None:
            logger.info("#rows: %d" % num_rows)
            logger.info("#cols: %d" % num_cols)
            logger.info("fixed width/height: %s" % str(fixed_size))
        for row in range(num_rows):
            y = row * section_height // num_rows + margin_top
            if (row == num_rows - 1) and not fixed_size:
                h = section_height - (y - margin_top)
            else:
                h = section_height // num_rows
            if row < num_rows - 1:
                h += overlap_bottom
            for col in range(num_cols):
                x = col * section_width // num_cols + margin_left
                if (col == num_cols - 1) and not fixed_size:
                    w = section_width - (x - margin_left)
                else:
                    w = section_width // num_cols
                if col < num_cols - 1:
                    w += overlap_right
                result.append(Region(x=x, y=y, w=w, h=h))

    # fixed row/col size
    elif mode == "size":
        if logger is not None:
            logger.info("#row-height: %d" % row_height)
            logger.info("#col-width: %d" % col_width)
            logger.info("partial: %s" % str(partial))
        y = margin_top
        while True:
            x = margin_left
            h = row_height + overlap_bottom
            if y + h - 1 > height:
                if partial:
                    h = height - y
                else:
                    h = 0

            while True:
                w = col_width + overlap_right
                if x + w - 1 > width:
                    if partial:
                        w = width - x
                    else:
                        w = 0
                if (w > 0) and (h > 0):
                    result.append(Region(x=x, y=y, w=w, h=h))

                x += col_width
                if x > width - 1:
                    break

            y += row_height
            if y > height - 1:
                break

    else:
        raise Exception("Unhandled split mode: %s" % mode)

    return result


def regions_to_string(regions: List[Region], one_based: bool = False, logger=None) -> str:
    """
    Turns the regions into a single string.

    :param regions: the regions to convert
    :type regions: list
    :param one_based: whether to use 1-based coordinates
    :type one_based: bool
    :param logger: optional logger for logging output
    :return: the generated string
    :rtype: str
    """
    if logger is not None:
        logger.info("#regions: %d" % len(regions))
        logger.info("1-based coordinates: %s" % str(one_based))
    regions_str = ""
    for region in regions:
        if one_based:
            regions_str += " %d,%d,%d,%d" % ((region.x+1), (region.y+1), region.w, region.h)
        else:
            regions_str += " %d,%d,%d,%d" % (region.x, region.y, region.w, region.h)
    return regions_str.strip()


def locatedobject_to_region(lobj: LocatedObject) -> Region:
    """
    Converts a LocatedObject instance into a Region object.

    :param lobj: the LocatedObject instance to convert
    :type lobj: LocatedObject
    :return: the Region object
    :rtype: Region
    """
    return Region(x=lobj.x, y=lobj.y, w=lobj.width, h=lobj.height)


def locatedobject_to_xyxy(lobj: LocatedObject) -> Tuple[int, int, int, int]:
    """
    Converts a LocatedObject instance into a tuple of x0,y0,x1,y1.

    :param lobj: the LocatedObject instance to convert
    :type lobj: LocatedObject
    :return: the tuple of the coordinates
    :rtype: tuple
    """
    return lobj.x, lobj.y, lobj.x + lobj.width - 1, lobj.y + lobj.height - 1


@dataclass
class Detection:
    bbox: np.ndarray
    score: float
    category_id: int
    polygon: Optional[BaseGeometry] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def geometry(self):
        if self.polygon is not None:
            return self.polygon
        return box(*self.bbox)


def locatedobject_to_detection(lobj: LocatedObject, categories: Dict[str, int]) -> Detection:
    """
    Turns a LocatedObject into a Detection one.

    :param lobj: the object to convert
    :type lobj: LocatedObject
    :param categories: the category/label lookup to use/expand
    :type categories: dict
    :return: the Detection object
    :rtype: Detection
    """
    try:
        score = float(lobj.metadata["score"]) if ("score" in lobj.metadata) else 1.0
    except:
        score = 1.0
    label = get_object_label(lobj)
    if label not in categories:
        categories[label] = len(categories)
    polygon = None
    if lobj.has_polygon():
        polygon = locatedobject_polygon_to_shapely(lobj)
    return Detection(bbox=np.asarray(locatedobject_to_xyxy(lobj)),
                     score=score, category_id=categories[label],
                     polygon=polygon, metadata=lobj.metadata.copy())


def reverse_categories_lookup(categories: Dict[str, int]) -> Dict[int, str]:
    """
    Generates a reverse lookup.

    :param categories: the categories lookup to invert.
    :type categories: dict
    :return: the new lookup
    :rtype: dict
    """
    result = dict()
    for k, v in categories.items():
        result[v] = k
    return result


def detection_to_locatedobject(detection: Detection, categories: Dict[int, str]) -> LocatedObject:
    """
    Turns the detection object back into a located one.

    :param detection: the object to convert
    :type detection: Detection
    :param categories: the lookup for the categories
    :type categories: dict
    :return: the converted object
    :rtype: LocatedObject
    """
    x0, y0, x1, y1 = detection.bbox
    if detection.metadata is None:
        meta = {"type": categories[detection.category_id]}
    else:
        meta = detection.metadata.copy()
        if "type" not in meta:
            meta["type"] = categories[detection.category_id]
    result = LocatedObject(x0, y0, x1-x0+1, y1-y0+1, **meta)
    if detection.polygon is not None:
        poly = shapely_to_polygon(detection.polygon)
        result.set_polygon(poly)
    return result


def bbox_ios(box_a, box_b) -> float:
    """
    Intersection over Smaller (IoS).

    :param box_a: first bounding box
    :param box_b: second bounding box
    :return: IoS
    :rtype: float
    """
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    ix1 = max(xa1, xb1)
    iy1 = max(ya1, yb1)
    ix2 = min(xa2, xb2)
    iy2 = min(ya2, yb2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)

    smaller = min(area_a, area_b)

    if smaller <= 0:
        return 0.0

    return inter / smaller


def polygon_ios(geom_a: BaseGeometry, geom_b: BaseGeometry) -> float:
    """
    Polygon Intersection over Smaller (IoS).

    :param geom_a: first polygon/bbox
    :param geom_b: second polygon/bbox
    :return: the IoS value
    :rtype: float
    """
    if geom_a.is_empty or geom_b.is_empty:
        return 0.0

    inter_area = geom_a.intersection(geom_b).area
    smaller_area = min(geom_a.area, geom_b.area)

    if smaller_area <= 0:
        return 0.0

    return inter_area / smaller_area


class UnionFind:

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_clusters(detections: List[Detection], ios_threshold: float = 0.5):
    """
    Build connected components using STRtree and IoS.

    :param detections: the list of detections
    :type detections: list
    :param ios_threshold: the IoS threshold to use
    :type ios_threshold: float
    """
    geometries = [det.geometry for det in detections]
    tree = STRtree(geometries)
    uf = UnionFind(len(detections))

    for idx, det in enumerate(detections):
        candidate_indices = tree.query(det.geometry)
        for j in candidate_indices:
            j = int(j)
            if j <= idx:
                continue

            other = detections[j]
            if det.category_id != other.category_id:
                continue

            if (det.polygon is not None) and (other.polygon is not None):
                overlap = polygon_ios(det.polygon, other.polygon)
            else:
                overlap = bbox_ios(det.bbox, other.bbox)
            if overlap >= ios_threshold:
                uf.union(idx, j)

    clusters = {}
    for idx in range(len(detections)):
        root = uf.find(idx)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(idx)

    return list(clusters.values())


def merge_cluster(cluster_indices: List[int], detections: List[Detection]) -> Detection:
    """
    Merge detections in one connected component.

    :param cluster_indices: the indices of the detection to merge
    :type cluster_indices: list
    :param detections: the full list of detections
    :type: list
    :return: the merged detection
    :rtype: Detection
    """
    cluster = [detections[i] for i in cluster_indices]
    best = max(cluster, key=lambda d: d.score)
    polygons = [
        det.polygon
        for det in cluster
        if det.polygon is not None
    ]

    merged_geometry = None
    if len(polygons) > 0:
        merged_geometry = unary_union(polygons)
        minx, miny, maxx, maxy = merged_geometry.bounds
        merged_bbox = np.array([minx, miny, maxx, maxy], dtype=np.float32)
    else:
        boxes = np.stack([det.bbox for det in cluster])
        merged_bbox = np.array(
            [
                boxes[:, 0].min(),
                boxes[:, 1].min(),
                boxes[:, 2].max(),
                boxes[:, 3].max(),
            ],
            dtype=np.float32,
        )

    return Detection(
        bbox=merged_bbox,
        score=best.score,
        category_id=best.category_id,
        polygon=merged_geometry,
    )


def merge_detections(detections: List[Detection], ios_threshold: float = 0.5):
    """
    Merge all overlapping detections.

    :param detections: the detections to merge
    :type detections: list
    :param ios_threshold: the IoS threshold to use
    :type ios_threshold: float
    :return: the merged detections
    :rtype: list
    """
    if len(detections) == 0:
        return []

    clusters = build_clusters(detections, ios_threshold=ios_threshold)
    merged = [merge_cluster(cluster, detections) for cluster in clusters]

    return merged


def polygon_nms(detections: List[Detection], ios_threshold: float = 0.7) -> List[Detection]:
    """
    Removes overlapping polygons.

    :param detections: the detections to process
    :type detections: list
    :param ios_threshold: the IoS threshold to use
    :type ios_threshold: float
    :return: the updated list of detections
    :rtype: list
    """
    detections = sorted(
        detections,
        key=lambda d: d.score,
        reverse=True,
    )

    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        remaining = []
        for det in detections:
            if (det.polygon is not None) and (best.polygon is not None):
                overlap = polygon_ios(best.polygon, det.polygon)
            else:
                overlap = bbox_ios(best.bbox, det.bbox)
            if overlap < ios_threshold:
                remaining.append(det)

        detections = remaining

    return keep
