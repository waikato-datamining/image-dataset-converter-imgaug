import argparse
import io
import numpy as np
import os
from typing import List

from shapely.geometry import Polygon, GeometryCollection, MultiPolygon
from seppl.io import Filter
from wai.logging import LOGGING_WARNING
from wai.common.geometry import Polygon as WaiPolygon
from wai.common.geometry import Point as WaiPoint
from wai.common.adams.imaging.locateobjects import LocatedObjects, LocatedObject
from idc.api import ImageClassificationData, ObjectDetectionData, ImageSegmentationData, ImageSegmentationAnnotations, flatten_list, make_list


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


class SubImages(Filter):
    """
    Extracts sub-images (incl their annotations) from the images coming through, using the defined regions.
    """

    def __init__(self, regions: List[str] = None, region_sorting: str = REGION_SORTING_NONE,
                 include_partial: bool = False, suppress_empty: bool = False, suffix: str = DEFAULT_SUFFIX,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param regions: the regions (X,Y,WIDTH,HEIGHT) to crop and forward with their annotations
        :type regions: list
        :param region_sorting: how to sort the supplied region definitions
        :type region_sorting: str
        :param include_partial: whether to include only annotations that fit fully into a region or also partial ones
        :type include_partial: bool
        :param suppress_empty: suppresses sub-images that have no annotations (object detection)
        :type suppress_empty: bool
        :param suffix: the suffix pattern to use for the generated sub-images (with placeholders)
        :type suffix: str
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.regions = regions
        self.region_sorting = region_sorting
        self.include_partial = include_partial
        self.suppress_empty = suppress_empty
        self.suffix = suffix
        self._regions_xyxy = None
        self._region_lobjs = None

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "sub-images"

    def description(self) -> str:
        """
        Returns a description of the filter.

        :return: the description
        :rtype: str
        """
        return "Extracts sub-images (incl their annotations) from the images coming through, using the defined regions."

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
        parser.add_argument("-r", "--regions", type=str, default=None, help="The regions (X,Y,WIDTH,HEIGHT) to crop and forward with their annotations (0-based coordinates)", required=True, nargs="+")
        parser.add_argument("-s", "--region_sorting", choices=REGION_SORTING, default=REGION_SORTING_NONE, help="How to sort the supplied region definitions", required=False)
        parser.add_argument("-p", "--include_partial", action="store_true", help="Whether to include only annotations that fit fully into a region or also partial ones", required=False)
        parser.add_argument("-e", "--suppress_empty", action="store_true", help="Suppresses sub-images that have no annotations (object detection and image segmentation)", required=False)
        parser.add_argument("-S", "--suffix", type=str, default=DEFAULT_SUFFIX, help="The suffix pattern to use for the generated sub-images, available placeholders: " + "|".join(PLACEHOLDERS), required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.regions = ns.regions
        self.region_sorting = ns.region_sorting
        self.include_partial = ns.include_partial
        self.suppress_empty = ns.suppress_empty
        self.suffix = ns.suffix

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()

        if (self.regions is None) or (len(self.regions) == 0):
            raise Exception("No region definitions supplied!")
        if self.region_sorting is None:
            self.region_sorting = REGION_SORTING_NONE
        if self.include_partial is None:
            self.include_partial = False
        if self.suppress_empty is None:
            self.suppress_empty = False
        if self.suffix is None:
            self.suffix = DEFAULT_SUFFIX

        self._regions_xyxy = []
        self._region_lobjs = []
        for region in self.regions:
            coords = [int(x) for x in region.split(",")]
            if len(coords) == 4:
                x, y, w, h = coords
                self._region_lobjs.append(LocatedObject(x=x, y=y, width=w, height=h))

        self.logger().info("unsorted regions: %s" % str([str(x) for x in self._region_lobjs]))

        if self.region_sorting is not REGION_SORTING_NONE:
            if self.region_sorting == REGION_SORTING_XY:
                def sorting(obj: LocatedObject):
                    return "%06d %06d" % (obj.x, obj.y)
            elif self.region_sorting == REGION_SORTING_YX:
                def sorting(obj: LocatedObject):
                    return "%06d %06d" % (obj.y, obj.x)
            else:
                raise Exception("Unhandled region sorting: %s" % self.region_sorting)
            self._region_lobjs.sort(key=sorting)
            self.logger().info("sorted regions: %s" % str([str(x) for x in self._region_lobjs]))

        for lobj in self._region_lobjs:
            self._regions_xyxy.append((lobj.x, lobj.y, lobj.x + lobj.width - 1, lobj.y + lobj.height - 1))
        self.logger().info("sorted xyxy: %s" % str(self._regions_xyxy))

    def _new_filename(self, path: str, index: int) -> str:
        """
        Generates a new filename based on the original and the index of the region.

        :param path: the base filename
        :type path: str
        :param index: the region index
        :type index: int
        :return: the generated filename
        :rtype: str
        """
        parts = os.path.splitext(path)
        index_pattern = "%0" + str(len(str(len(self._region_lobjs)))) + "d"
        index_str = index_pattern % index
        suffix = self.suffix
        suffix = suffix.replace(PH_INDEX, index_str)
        suffix = suffix.replace(PH_X0, str(self._regions_xyxy[index][0]))
        suffix = suffix.replace(PH_Y0, str(self._regions_xyxy[index][1]))
        suffix = suffix.replace(PH_X1, str(self._regions_xyxy[index][2]))
        suffix = suffix.replace(PH_Y1, str(self._regions_xyxy[index][3]))
        suffix = suffix.replace(PH_X, str(self._region_lobjs[index].x))
        suffix = suffix.replace(PH_Y, str(self._region_lobjs[index].y))
        suffix = suffix.replace(PH_W, str(self._region_lobjs[index].width))
        suffix = suffix.replace(PH_H, str(self._region_lobjs[index].height))
        return parts[0] + suffix + parts[1]

    def _bbox_to_shapely(self, lobj: LocatedObject) -> Polygon:
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

    def _polygon_to_shapely(self, lobj: LocatedObject) -> Polygon:
        """
        Converts the located object polygon into a shapely Polygon.

        :param lobj: the polygon to convert
        :return: the Polygon
        """
        if not lobj.has_polygon():
            return self._bbox_to_shapely(lobj)
        x_list = lobj.get_polygon_x()
        y_list = lobj.get_polygon_y()
        coords = []
        for x, y in zip(x_list, y_list):
            coords.append((x, y))
        coords.append((x_list[0], y_list[0]))
        return Polygon(coords)

    def _fit_located_object(self, index: int, region: LocatedObject, annotation: LocatedObject) -> LocatedObject:
        """
        Fits the annotation into the specified region, adjusts size if necessary.

        :param index: the index of the current region
        :param region: the region object to fit the annotation in
        :param annotation: the annotation to fit
        :return: the adjust annotation
        """
        sregion = self._bbox_to_shapely(region)
        sbbox = self._bbox_to_shapely(annotation)
        sintersect = sbbox.intersection(sregion)
        minx, miny, maxx, maxy = [int(x) for x in sintersect.bounds]
        result = LocatedObject(x=minx-region.x, y=miny-region.y, width=maxx-minx+1, height=maxy-miny+1, **annotation.metadata)
        result.metadata["region_index"] = index
        result.metadata["region_xywh"] = "%d,%d,%d,%d" % (region.x, region.y, region.width, region.height)

        if annotation.has_polygon():
            spolygon = self._polygon_to_shapely(annotation)
        else:
            spolygon = self._bbox_to_shapely(annotation)

        try:
            sintersect = spolygon.intersection(sregion)
        except:
            self.logger().warning("Failed to compute intersection!")
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
            self.logger().warning("Unhandled geometry type returned from intersection, skipping: %s" % str(type(sintersect)))

        return result

    def _fit_layers(self, region: LocatedObject, annotations: ImageSegmentationAnnotations) -> ImageSegmentationAnnotations:
        """
        Crops the layers to the region.

        :param region: the region to crop the layers to
        :type region: LocatedObject
        :param annotations: the annotations to crop
        :type annotations: ImageSegmentationAnnotations
        :return: the updated annotations
        :rtype: ImageSegmentationAnnotations
        """
        layers = dict()
        for label in annotations.layers:
            layer = annotations.layers[label][region.y:region.y+region.height-1, region.x:region.x+region.width-1]
            add = True
            if self.suppress_empty:
                unique = np.unique(layer)
                # only background? -> skip
                if (len(unique) == 1) and (unique[0] == 0):
                    add = False
            if add:
                layers[label] = layer
        return ImageSegmentationAnnotations(annotations.labels[:], layers)

    def _do_process(self, data):
        """
        Processes the data record(s).

        :param data: the record(s) to process
        :return: the potentially updated record(s)
        """
        result = []

        for item in make_list(data):
            pil = item.image
            for region_index, region_xyxy in enumerate(self._regions_xyxy):
                self.logger().info("Applying region %d :%s" % (region_index, str(region_xyxy)))

                # crop image
                x0, y0, x1, y1 = region_xyxy
                if x1 > item.image_width:
                    x1 = item.image_width
                if y1 > item.image_height:
                    y1 = item.image_height
                sub_image = pil.crop((x0, y0, x1, y1))
                sub_bytes = io.BytesIO()
                sub_image.save(sub_bytes, format=item.image_format)
                image_name_new = self._new_filename(item.image_name, region_index)

                # crop annotations and forward
                region_lobj = self._region_lobjs[region_index]
                if isinstance(item, ImageClassificationData):
                    item_new = ImageClassificationData(image_name=image_name_new, data=sub_bytes.getvalue(),
                                                       annotation=item.annotation, metadata=item.get_metadata())
                    result.append(item_new)
                elif isinstance(item, ObjectDetectionData):
                    new_objects = []
                    for ann_lobj in item.annotation:
                        ratio = region_lobj.overlap_ratio(ann_lobj)
                        if ((ratio > 0) and self.include_partial) or (ratio >= 1):
                            new_objects.append(self._fit_located_object(region_index, region_lobj, ann_lobj))
                    if not self.suppress_empty or (len(new_objects) > 0):
                        item_new = ObjectDetectionData(image_name=image_name_new, data=sub_bytes.getvalue(),
                                                       annotation=LocatedObjects(new_objects), metadata=item.get_metadata())
                        result.append(item_new)
                elif isinstance(item, ImageSegmentationData):
                    new_annotations = self._fit_layers(region_lobj, item.annotation)
                    if not self.suppress_empty or (len(new_annotations.layers) > 0):
                        item_new = ImageSegmentationData(image_name=image_name_new, data=sub_bytes.getvalue(),
                                                         annotation=new_annotations, metadata=item.get_metadata())
                        result.append(item_new)
                else:
                    self.logger().warning("Unhandled data (%s), skipping!" % str(type(item)))
                    result.append(item)

        return flatten_list(result)
