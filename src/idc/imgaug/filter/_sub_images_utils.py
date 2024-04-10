import logging
import os
from typing import List, Tuple

import numpy as np
from shapely import Polygon, GeometryCollection, MultiPolygon
from wai.common.adams.imaging.locateobjects import LocatedObject
from wai.common.geometry import Point as WaiPoint, Polygon as WaiPolygon

from idc.api import ImageSegmentationAnnotations

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
    :return: the tuple of xyxy and LocatedObject lists
    :rtype: tuple
    """
    regions_xyxy = []
    region_lobjs = []
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
    return regions_xyxy, region_lobjs


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
