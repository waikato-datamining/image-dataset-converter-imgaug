import logging
import io
import os
from typing import List, Tuple, Optional

import numpy as np
from shapely import Polygon, GeometryCollection, MultiPolygon
from wai.common.adams.imaging.locateobjects import LocatedObject, LocatedObjects
from wai.common.geometry import Point as WaiPoint, Polygon as WaiPolygon

from idc.api import ImageSegmentationAnnotations, ImageClassificationData, ImageSegmentationData, ObjectDetectionData, ImageData

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


def bbox_to_shapely(lobj: LocatedObject) -> Polygon:
    """
    Converts the located object rectangle into a shapely Polygon.

    :param lobj: the bbox to convert
    :return: the Polygon
    """
    coords = [
        (lobj.x, lobj.y),
        (lobj.x + lobj.width - 1, lobj.y),
        (lobj.x + lobj.width - 1, lobj.y + lobj.height - 1),
        (lobj.x, lobj.y + lobj.height - 1),
        (lobj.x, lobj.y),
    ]
    return Polygon(coords)


def polygon_to_shapely(lobj: LocatedObject) -> Polygon:
    """
    Converts the located object polygon into a shapely Polygon.

    :param lobj: the polygon to convert
    :return: the Polygon
    """
    if not lobj.has_polygon():
        return bbox_to_shapely(lobj)
    x_list = lobj.get_polygon_x()
    y_list = lobj.get_polygon_y()
    coords = []
    for x, y in zip(x_list, y_list):
        coords.append((x, y))
    coords.append((x_list[0], y_list[0]))
    return Polygon(coords)


def fit_located_object(index: int, region: LocatedObject, annotation: LocatedObject, logger: logging.Logger) -> LocatedObject:
    """
    Fits the annotation into the specified region, adjusts size if necessary.

    :param index: the index of the current region
    :type index: int
    :param region: the region object to fit the annotation in
    :type region: LocatedObject
    :param annotation: the annotation to fit
    :type annotation: LocatedObject
    :param logger: the logger to use
    :type logger: logging.Logger
    :return: the adjusted annotation
    :rtype: LocatedObject
    """
    sregion = bbox_to_shapely(region)
    sbbox = bbox_to_shapely(annotation)
    sintersect = sbbox.intersection(sregion)
    minx, miny, maxx, maxy = [int(x) for x in sintersect.bounds]
    result = LocatedObject(x=minx-region.x, y=miny-region.y, width=maxx-minx+1, height=maxy-miny+1, **annotation.metadata)
    result.metadata["region_index"] = index
    result.metadata["region_xywh"] = "%d,%d,%d,%d" % (region.x, region.y, region.width, region.height)

    if annotation.has_polygon():
        spolygon = polygon_to_shapely(annotation)
    else:
        spolygon = bbox_to_shapely(annotation)

    try:
        sintersect = spolygon.intersection(sregion)
    except:
        logger.warning("Failed to compute intersection!")
        sintersect = None

    if isinstance(sintersect, GeometryCollection):
        for x in sintersect.geoms:
            if isinstance(x, Polygon):
                sintersect = x
                break
    elif isinstance(sintersect, MultiPolygon):
        for x in sintersect.geoms:
            if isinstance(x, Polygon):
                sintersect = x
                break

    if isinstance(sintersect, Polygon):
        x_list, y_list = sintersect.exterior.coords.xy
        points = []
        for i in range(len(x_list)):
            points.append(WaiPoint(x=x_list[i]-region.x, y=y_list[i]-region.y))
        result.set_polygon(WaiPolygon(*points))
    else:
        logger.warning("Unhandled geometry type returned from intersection, skipping: %s" % str(type(sintersect)))

    return result


def fit_layers(region: LocatedObject, annotations: ImageSegmentationAnnotations, suppress_empty: bool) -> ImageSegmentationAnnotations:
    """
    Crops the layers to the region.

    :param region: the region to crop the layers to
    :type region: LocatedObject
    :param annotations: the annotations to crop
    :type annotations: ImageSegmentationAnnotations
    :param suppress_empty: whether to suppress empty annotations
    :type suppress_empty: bool
    :return: the updated annotations
    :rtype: ImageSegmentationAnnotations
    """
    layers = dict()
    for label in annotations.layers:
        layer = annotations.layers[label][region.y:region.y+region.height-1, region.x:region.x+region.width-1]
        add = True
        if suppress_empty:
            unique = np.unique(layer)
            # only background? -> skip
            if (len(unique) == 1) and (unique[0] == 0):
                add = False
        if add:
            layers[label] = layer
    return ImageSegmentationAnnotations(annotations.labels[:], layers)


def process_image(item: ImageData, regions_lobj: List[LocatedObject], regions_xyxy: List[Tuple], suffix: str,
                  suppress_empty: bool, include_partial: bool, logger: logging.Logger) -> Optional[List[Tuple[LocatedObject, ImageData]]]:
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
    :param suppress_empty: whether to suppress sub-images with no annotatoins (object detection and image segmentation only)
    :type suppress_empty: bool
    :param include_partial: whether to include partial annotations (ones that get cut off by the region)
    :type include_partial: bool
    :param logger: for logging purposes
    :type logger: logging.Logger
    :return: the list of tuples (located object of region, new image)
    :rtype: list
    """
    result = []

    pil = item.image
    for region_index, region_xyxy in enumerate(regions_xyxy):
        logger.info("Applying region %d :%s" % (region_index, str(region_xyxy)))

        # crop image
        x0, y0, x1, y1 = region_xyxy
        if x1 > item.image_width:
            x1 = item.image_width
        if y1 > item.image_height:
            y1 = item.image_height
        sub_image = pil.crop((x0, y0, x1, y1))
        sub_bytes = io.BytesIO()
        sub_image.save(sub_bytes, format=item.image_format)
        image_name_new = region_filename(item.image_name, regions_lobj, regions_xyxy, region_index, suffix)

        # crop annotations and forward
        region_lobj = regions_lobj[region_index]
        if isinstance(item, ImageClassificationData):
            item_new = ImageClassificationData(image_name=image_name_new, data=sub_bytes.getvalue(),
                                               annotation=item.annotation, metadata=item.get_metadata())
            result.append((region_lobj, item_new))
        elif isinstance(item, ObjectDetectionData):
            new_objects = []
            for ann_lobj in item.annotation:
                ratio = region_lobj.overlap_ratio(ann_lobj)
                if ((ratio > 0) and include_partial) or (ratio >= 1):
                    new_objects.append(fit_located_object(region_index, region_lobj, ann_lobj, logger))
            if not suppress_empty or (len(new_objects) > 0):
                item_new = ObjectDetectionData(image_name=image_name_new, data=sub_bytes.getvalue(),
                                               annotation=LocatedObjects(new_objects), metadata=item.get_metadata())
                result.append((region_lobj, item_new))
        elif isinstance(item, ImageSegmentationData):
            new_annotations = fit_layers(region_lobj, item.annotation, suppress_empty)
            if not suppress_empty or (len(new_annotations.layers) > 0):
                item_new = ImageSegmentationData(image_name=image_name_new, data=sub_bytes.getvalue(),
                                                 annotation=new_annotations, metadata=item.get_metadata())
                result.append((region_lobj, item_new))
        else:
            logger.warning("Unhandled data (%s), skipping!" % str(type(item)))
            return None

    return result
