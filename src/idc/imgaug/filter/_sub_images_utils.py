import io
import logging
import math
import os
import statistics
from typing import List, Tuple, Optional, Union

import numpy as np
import shapely
from PIL import Image
from shapely import Polygon, GeometryCollection, MultiPolygon, LineString, distance
from wai.common.adams.imaging.locateobjects import LocatedObject, LocatedObjects
from wai.common.geometry import Point as WaiPoint, Polygon as WaiPolygon

from idc.api import ImageSegmentationAnnotations, ImageClassificationData, ImageSegmentationData, ObjectDetectionData, \
    ImageData, get_object_label, locatedobject_polygon_to_shapely, locatedobject_bbox_to_shapely, shapely_to_locatedobject

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


def fit_located_object(index: int, region: LocatedObject, annotation: LocatedObject, logger: Optional[logging.Logger]) -> LocatedObject:
    """
    Fits the annotation into the specified region, adjusts size if necessary.

    :param index: the index of the current region, gets added to meta-data if >=0
    :type index: int
    :param region: the region object to fit the annotation in
    :type region: LocatedObject
    :param annotation: the annotation to fit
    :type annotation: LocatedObject
    :param logger: the logger to use, can be None
    :type logger: logging.Logger
    :return: the adjusted annotation
    :rtype: LocatedObject
    """
    sregion = locatedobject_bbox_to_shapely(region)
    sbbox = locatedobject_bbox_to_shapely(annotation)
    sintersect = sbbox.intersection(sregion)
    minx, miny, maxx, maxy = [int(x) for x in sintersect.bounds]
    result = LocatedObject(x=minx-region.x, y=miny-region.y, width=maxx-minx+1, height=maxy-miny+1, **annotation.metadata)
    if index > -1:
        result.metadata["region_index"] = index
        result.metadata["region_xywh"] = "%d,%d,%d,%d" % (region.x, region.y, region.width, region.height)

    if annotation.has_polygon():
        spolygon = locatedobject_polygon_to_shapely(annotation)
    else:
        spolygon = locatedobject_bbox_to_shapely(annotation)

    try:
        sintersect = spolygon.intersection(sregion)
    except:
        msg = "Failed to compute intersection!"
        if logger is None:
            print(msg)
        else:
            logger.warning(msg)
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
        msg = "Unhandled geometry type returned from intersection, skipping: %s" % str(type(sintersect))
        if logger is None:
            print(msg)
        else:
            logger.warning(msg)

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
        layer = annotations.layers[label][region.y:region.y+region.height, region.x:region.x+region.width]
        add = True
        if suppress_empty:
            unique = np.unique(layer)
            # only background? -> skip
            if (len(unique) == 1) and (unique[0] == 0):
                add = False
        if add:
            layers[label] = layer
    return ImageSegmentationAnnotations(annotations.labels[:], layers)


def pad_image(img: Union[Image.Image, np.ndarray], pad_width: Optional[int] = None, pad_height: Optional[int] = None) -> Image:
    """
    Pads the image/layer if necessary (on the right/bottom).

    :param img: the image to pad
    :type img: Image.Image/np.ndarray
    :param pad_width: the width to pad to, return as is if None
    :type pad_width: int
    :param pad_height: the height to pad to, return as is if None
    :type pad_height: int
    :return: the (potentially) padded image
    :rtype: Image.Image/np.ndarray
    """
    result = img
    if isinstance(img, Image.Image):
        width, height = img.size
    else:
        height = img.shape[0]
        width = img.shape[1]
    pad = False

    if (pad_width is not None) and (pad_height is not None):
        pad = (width != pad_width) or (height != pad_height)
    elif pad_width is not None:
        pad = width != pad_width
        pad_height = height
    elif pad_height is not None:
        pad = height != pad_height
        pad_width = width

    if pad:
        if isinstance(img, Image.Image):
            result = Image.new(img.mode, (pad_width, pad_height))
            result.paste(img)
        else:
            result = np.zeros((pad_height, pad_width), dtype=img.dtype)
            result[0:height, 0:width] = img

    return result


def crop_image(img: Union[Image.Image, np.ndarray], crop_width: Optional[int] = None, crop_height: Optional[int] = None) -> Image:
    """
    Crops the image/layer if necessary (removes on the right/bottom).

    :param img: the image to pad
    :type img: Image.Image/np.ndarray
    :param crop_width: the width to crop to, return as is if None
    :type crop_width: int
    :param crop_height: the height to crop to, return as is if None
    :type crop_height: int
    :return: the (potentially) cropped image
    :rtype: Image.Image/np.ndarray
    """
    result = img
    if isinstance(img, Image.Image):
        width, height = img.size
    else:
        height = img.shape[0]
        width = img.shape[1]
    crop = False

    if (crop_width is not None) and (crop_height is not None):
        crop = (width != crop_width) or (height != crop_height)
    elif crop_width is not None:
        crop = width != crop_width
        crop_height = height
    elif crop_height is not None:
        crop = height != crop_height
        crop_width = width

    if crop:
        if isinstance(img, Image.Image):
            result = img.crop((0, 0, crop_width, crop_height))
        else:
            result = img[0:crop_height, 0:crop_width]

    return result


def process_image(item: ImageData, regions_lobj: List[LocatedObject], regions_xyxy: List[Tuple], suffix: str,
                  suppress_empty: bool, include_partial: bool, logger: logging.Logger,
                  pad_width: Optional[int] = None, pad_height: Optional[int] = None) -> Optional[List[Tuple[LocatedObject, ImageData, LocatedObject]]]:
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
    :param include_partial: whether to include partial annotations (ones that get cut off by the region)
    :type include_partial: bool
    :param logger: for logging purposes
    :type logger: logging.Logger
    :param pad_width: the width to pad to, return as is if None
    :type pad_width: int
    :param pad_height: the height to pad to, return as is if None
    :type pad_height: int
    :return: the list of tuples (located object of region, new image, located object of original dims)
    :rtype: list
    """
    result = []

    pil = item.image
    for region_index, region_xyxy in enumerate(regions_xyxy):
        logger.info("Applying region %d :%s" % (region_index, str(region_xyxy)))

        # crop image
        x0, y0, x1, y1 = region_xyxy
        if x1 >= item.image_width:
            x1 = item.image_width - 1
        if y1 >= item.image_height:
            y1 = item.image_height - 1
        sub_image = pil.crop((x0, y0, x1+1, y1+1))
        orig_dims = LocatedObject(0, 0, sub_image.size[0], sub_image.size[1])
        sub_image = pad_image(sub_image, pad_width=pad_width, pad_height=pad_height)
        sub_bytes = io.BytesIO()
        sub_image.save(sub_bytes, format=item.image_format)
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
                        new_objects.append(fit_located_object(region_index, region_lobj, ann_lobj, logger))
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
        img = Image.new(item.image.mode, item.image_size)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format=item.image_format)
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
    else:
        raise Exception("Unhandled type of data: %s" % str(type(item)))

    return result


def transfer_region(full_image, sub_image, region: LocatedObject, rebuild_image: bool = False,
                    crop_width: int = None, crop_height: int = None):
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
    """
    # transfer image
    if rebuild_image:
        cropped = crop_image(sub_image.image, crop_width=crop_width, crop_height=crop_height)
        full_image.image.paste(cropped, (region.x, region.y))

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
                    continue
                # skip objects below the bottom of the image
                if new_lobj.y >= img_height:
                    continue
                # fit object if necessary
                fit = False
                if (new_lobj.x < img_width) and (new_lobj.x + new_lobj.width >= img_width):
                    fit = True
                if (new_lobj.y < img_height) and (new_lobj.y + new_lobj.height >= img_height):
                    fit = True
                if fit:
                    region = LocatedObject(0, 0, img_width, img_height)
                    new_lobj = fit_located_object(-1, region, new_lobj, None)
                # add object
                full_image.annotation.append(new_lobj)

        # image segmentation
        elif isinstance(full_image, ImageSegmentationData):
            for label in sub_image.annotation.layers:
                x = region.x
                y = region.y
                w = region.width
                h = region.height
                if (not full_image.has_annotation()) or (not full_image.has_layer(label)):
                    full_image.new_layer(label)
                layer = sub_image.annotation.layers[label]
                layer = crop_image(layer, crop_width=crop_width, crop_height=crop_height)
                full_image.annotation.layers[label][y:y + h, x:x + w] = layer

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

    else:
        raise Exception("Unhandled type of data: %s" % str(type(image)))


def overlapping_lines(line1: Tuple[int, int], line2: Tuple[int, int]) -> bool:
    """
    Checks whether the lines are overlapping.

    :param line1: the first line (start,end)
    :type line1: tuple
    :param line2: the second line (start,eng)
    :type line2: tuple
    :return: True if overlap
    :rtype: bool
    """
    l1_s, l1_e = line1
    l2_s, l2_e = line2
    # line1 is completely to the left of line2
    if (l1_s < l2_s) and (l1_e < l2_s):
        return False
    # line1 is completely to the right of line2
    if (l1_s > l2_e) and (l1_e > l2_e):
        return False
    # line2 is completely to the left of line1
    if (l2_s < l1_s) and (l2_e < l1_s):
        return False
    # line2 is completely to the right of line1
    if (l2_s > l1_e) and (l2_e > l1_e):
        return False
    # some overlap
    return True


def merge_polygons(combined: ObjectDetectionData, max_slope_diff: float = 1e-6, max_dist: float = 1.0) -> ObjectDetectionData:
    """
    Merges adjacent polygons. Discards metadata apart from score, which it averages across merged objects,
    and the label, which has to be the same across objects.

    :param combined: the input data
    :type combined: ObjectDetectionData
    :param max_slope_diff: the maximum difference between slopes while still being considered parallel
    :type max_slope_diff: float
    :param max_dist: the maximum distance between parallel vertices
    :type max_dist: float
    :return: the (potentially) updated annotations
    :rtype: ObjectDetectionData
    """
    # for each polygon
    #   for each vertex in polygon
    #      compute slope
    # determine parallel vertices between objects
    # compute distance between parallel vertices
    # determine sets of objects to merge

    # determine vertices/slopes/intercepts
    vertices = dict()
    slopes = dict()
    normalized = combined.is_normalized()
    absolute = combined.get_absolute()
    for i in range(len(absolute)):
        vertices[i] = []
        slopes[i] = []
        if absolute[i].has_polygon():
            xs = absolute[i].get_polygon_x()
            ys = absolute[i].get_polygon_y()
        else:
            xs = [absolute[i].x, absolute[i].x + absolute[i].width - 1, absolute[i].x + absolute[i].width - 1, absolute[i].x]
            ys = [absolute[i].y, absolute[i].y, absolute[i].y + absolute[i].height - 1, absolute[i].y + absolute[i].height - 1]
        for n in range(len(xs)):
            # vertex: (x0,y0,x1,y1)
            vertices[i].append(LineString([(xs[n - 1], ys[n - 1]), (xs[n], ys[n])]))
            # slope: m = (y1-y0) / (x1-x0)
            if xs[n] - xs[n - 1] == 0:
                slope = math.inf
            else:
                slope = (ys[n] - ys[n - 1]) / (xs[n] - xs[n - 1])
            slopes[i].append(slope)

    # determine parallel vertices of objects with same label
    parallel = dict()
    for i in range(len(slopes)):
        label_i = get_object_label(absolute[i])
        for n in range(i + 1, len(slopes), 1):
            # only consider objects with the same label
            label_n = get_object_label(absolute[n])
            if label_i != label_n:
                continue
            for i_i in range(len(slopes[i])):
                for n_n in range(len(slopes[n])):
                    is_parallel = False

                    # horizontal lines
                    if (slopes[i][i_i] == 0) and (slopes[n][n_n] == 0):
                        is_parallel = True

                    # vertical lines
                    elif math.isinf(slopes[i][i_i]) and math.isinf(slopes[n][n_n]):
                        is_parallel = True

                    # compare slope
                    else:
                        slope_diff = abs(slopes[i][i_i] - slopes[n][n_n])
                        if slope_diff <= max_slope_diff:
                            is_parallel = True

                    # check distance of parallel vertices
                    if is_parallel:
                        d = distance(vertices[i][i_i], vertices[n][n_n])
                        if d <= max_dist:
                            if i not in parallel:
                                parallel[i] = set()
                            parallel[i].add(n)

    # create sets of objects to merge
    merge_sets = []
    to_merge = set()
    for i, ns in parallel.items():
        all_ = [i, *ns]
        found = None
        for a in all_:
            to_merge.add(a)
            for n, merge_set in enumerate(merge_sets):
                if a in merge_set:
                    found = n
                    break
            if found is not None:
                break
        if found is None:
            merge_sets.append(set(all_))
        else:
            for a in all_:
                merge_sets[found].add(a)

    if len(merge_sets) > 0:
        # transfer all objects that won't get merged
        annotation_new = LocatedObjects()
        for i, obj in enumerate(absolute):
            if i not in to_merge:
                annotation_new.append(obj)

        # merge sets
        for merge_set in merge_sets:
            label = None
            merged = None
            scores = []
            for i in merge_set:
                if label is None:
                    label = get_object_label(absolute[i])
                if "score" in absolute[i].metadata:
                    scores.append(float(absolute[i].metadata["score"]))
                if merged is None:
                    merged = locatedobject_polygon_to_shapely(absolute[i])
                else:
                    merged = shapely.union(merged, locatedobject_polygon_to_shapely(absolute[i]))
            obj = shapely_to_locatedobject(merged, label=label)
            # set average score
            if len(scores) > 0:
                score = statistics.mean(scores)
                obj.metadata["score"] = score
            annotation_new.append(obj)

        # update container
        combined.annotation = annotation_new
        if normalized:
            combined.to_normalized()

    return combined
