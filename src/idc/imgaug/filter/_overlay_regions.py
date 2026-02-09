import argparse
import copy
from typing import List, Union

from PIL import Image, ImageDraw
from simple_palette_utils import COLOR_LISTS, COLOR_LIST_X11, ColorProvider, parse_rgb, color_lists
from wai.logging import LOGGING_WARNING

from idc.api import ImageData, image_to_bytesio
from idc.imgaug.filter._sub_images_utils import REGION_SORTING_NONE, REGION_SORTING, parse_regions, generate_regions, \
    regions_to_string
from kasperl.api import make_list, flatten_list
from seppl.io import BatchFilter


class OverlayRegions(BatchFilter):
    """
    Overlays the regions on the image.
    """

    def __init__(self, regions: List[str] = None, region_sorting: str = REGION_SORTING_NONE,
                 num_rows: int = None, num_cols: int = None, row_height: int = None, col_width: int = None,
                 overlap_right: int = None, overlap_bottom: int = None,
                 colors: Union[str, List[str]] = None,
                 outline_thickness: int = None, outline_alpha: int = None,
                 fill: bool = False, fill_alpha: int = None, vary_colors: bool = False,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param regions: the regions (X,Y,WIDTH,HEIGHT) to crop and forward with their annotations
        :type regions: list
        :param region_sorting: how to sort the supplied region definitions
        :type region_sorting: str
        :param num_rows: the number of rows to use, if no regions defined
        :type num_rows: int
        :param num_cols: the number of columns to use, if no regions defined
        :type num_cols: int
        :param row_height: the height of rows
        :type row_height: int
        :param col_width: the width of columns
        :type col_width: int
        :param overlap_right: the overlap in pixels on the right, if no regions defined
        :type overlap_right: int
        :param overlap_bottom: the overlap in pixels on the bottom, if no regions defined
        :type overlap_bottom: int
        :param colors: the color list name or list of RGB triplets (R,G,B) of custom colors to use, uses default colors if not supplied
        :type colors: list
        :param outline_thickness: the line thickness to use for the outline, <1 to turn off
        :type outline_thickness: int
        :param outline_alpha: the alpha value to use for the outline (0: transparent, 255: opaque)
        :type outline_alpha: int
        :param fill: whether to fill the bounding boxes/polygons
        :type fill: bool
        :param fill_alpha: the alpha value to use for the filling (0: transparent, 255: opaque)
        :type fill_alpha: int
        :param vary_colors: whether to vary the colors of the outline/filling regardless of label
        :type vary_colors: bool
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.regions = regions
        self.region_sorting = region_sorting
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.row_height = row_height
        self.col_width = col_width
        self.overlap_right = overlap_right
        self.overlap_bottom = overlap_bottom
        if isinstance(colors, str):
            colors = [colors]
        self.colors = colors
        self.outline_thickness = outline_thickness
        self.outline_alpha = outline_alpha
        self.fill = fill
        self.fill_alpha = fill_alpha
        self.vary_colors = vary_colors
        self._regions_lobj = None
        self._color_provider = None

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "overlay-regions"

    def description(self) -> str:
        """
        Returns a description of the filter.

        :return: the description
        :rtype: str
        """
        return "Overlays the regions on the images coming through, using either the explicitly defined regions or ones derived from #rows/cols."

    def accepts(self) -> List:
        """
        Returns the list of classes that are accepted.

        :return: the list of classes
        :rtype: list
        """
        return [ImageData]

    def generates(self) -> List:
        """
        Returns the list of classes that get produced.

        :return: the list of classes
        :rtype: list
        """
        return [ImageData]

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        parser.add_argument("-r", "--regions", type=str, default=None, help="The regions (X,Y,WIDTH,HEIGHT) to crop and forward with their annotations (0-based coordinates)", required=False, nargs="*")
        parser.add_argument("-s", "--region_sorting", choices=REGION_SORTING, default=REGION_SORTING_NONE, help="How to sort the supplied region definitions", required=False)
        parser.add_argument("--num_rows", type=int, help="The number of rows, if no regions defined.", default=None, required=False)
        parser.add_argument("--num_cols", type=int, help="The number of columns, if no regions defined.", default=None, required=False)
        parser.add_argument("--row_height", type=int, help="The height of rows.", default=None, required=False)
        parser.add_argument("--col_width", type=int, help="The width of columns.", default=None, required=False)
        parser.add_argument("--overlap_right", type=int, help="The overlap between two images (on the right of the left-most image), if no regions defined.", default=0, required=False)
        parser.add_argument("--overlap_bottom", type=int, help="The overlap between two images (on the bottom of the top-most image), if no regions defined.", default=0, required=False)
        parser.add_argument("-c", "--colors", type=str, metavar="R,G,B", help="The color list name (available: " + ",".join(color_lists()) + ") or list of RGB triplets (R,G,B) of custom colors to use, uses default colors if not supplied (X11 colors, without dark/light colors)", required=False, nargs="*")
        parser.add_argument("--outline_thickness", type=int, metavar="INT", help="The line thickness to use for the outline, <1 to turn off.", required=False, default=3)
        parser.add_argument("--outline_alpha", type=int, metavar="INT", help="The alpha value to use for the outline (0: transparent, 255: opaque).", required=False, default=255)
        parser.add_argument("--fill", action="store_true", help="Whether to fill the bounding boxes/polygons.", required=False)
        parser.add_argument("--fill_alpha", type=int, metavar="INT", help="The alpha value to use for the filling (0: transparent, 255: opaque).", required=False, default=128)
        parser.add_argument("--vary_colors", action="store_true", help="Whether to vary the colors of the outline/filling regardless of label.", required=False)
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
        self.num_rows = ns.num_rows
        self.num_cols = ns.num_cols
        self.row_height = ns.row_height
        self.col_width = ns.col_width
        self.overlap_right = ns.overlap_right
        self.overlap_bottom = ns.overlap_bottom
        self.colors = ns.colors
        self.outline_thickness = ns.outline_thickness
        self.outline_alpha = ns.outline_alpha
        self.fill = ns.fill
        self.fill_alpha = ns.fill_alpha
        self.vary_colors = ns.vary_colors

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()

        if (self.regions is not None) and (len(self.regions) == 0):
            self.regions = None
        if (self.num_rows is None) or (self.num_cols is None):
            self.num_rows = None
            self.num_cols = None
        if (self.row_height is None) or (self.col_width is None):
            self.row_height = None
            self.col_width = None
        if (self.regions is None) and (self.num_rows is None) and (self.row_height is None):
            raise Exception("Neither regions nor #rows/cols nor row height/col width specified!")
        if self.region_sorting is None:
            self.region_sorting = REGION_SORTING_NONE
        if self.overlap_right is None:
            self.overlap_right = 0
        if self.overlap_bottom is None:
            self.overlap_bottom = 0
        if self.outline_thickness is None:
            self.outline_thickness = 3
        if self.outline_alpha is None:
            self.outline_alpha = 255
        if self.fill is None:
            self.fill = False
        if self.vary_colors is None:
            self.vary_colors = False
        color_list = COLOR_LIST_X11
        custom_colors = None
        if self.colors is not None:
            # color list name?
            if (len(self.colors) == 1) and ("," not in self.colors[0]):
                color_list = self.colors[0]
                if self.colors[0] not in COLOR_LISTS:
                    raise Exception("Unknown color list '%s'! Available lists: %s" % (color_list, ",".join(sorted(COLOR_LISTS.keys()))))
            else:
                custom_colors = parse_rgb(self.colors)
        self._color_provider = ColorProvider(color_list=color_list, custom_colors=custom_colors)

        self._regions_lobj = None
        if self.regions is not None:
            self._regions_lobj, _ = parse_regions(self.regions, self.region_sorting, self.logger())

    def _do_process(self, data):
        """
        Processes the data record(s).

        :param data: the record(s) to process
        :return: the potentially updated record(s)
        """
        result = []

        for item in make_list(data):
            if self._regions_lobj is not None:
                regions_lobj = self._regions_lobj
            elif (self.num_rows is not None) and (self.num_cols is not None):
                regions = generate_regions(item.image_width, item.image_height,
                                           num_rows=self.num_rows, num_cols=self.num_cols,
                                           overlap_right=self.overlap_right, overlap_bottom=self.overlap_bottom,
                                           logger=self.logger())
                regions_str = regions_to_string(regions, logger=self.logger())
                regions_lobj, _ = parse_regions(regions_str.split(" "), self.region_sorting, self.logger())
            else:
                regions = generate_regions(item.image_width, item.image_height,
                                           row_height=self.row_height, col_width=self.col_width,
                                           overlap_right=self.overlap_right, overlap_bottom=self.overlap_bottom,
                                           logger=self.logger())
                regions_str = regions_to_string(regions, logger=self.logger())
                regions_lobj, _ = parse_regions(regions_str.split(" "), self.region_sorting, self.logger())

            img_pil = item.image.copy()

            overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            for i, lobj in enumerate(regions_lobj):
                # determine label/color
                label = "object"
                if self.vary_colors:
                    color_label = "object-%d" % i
                else:
                    color_label = label

                # assemble polygon
                points = []
                rect = lobj.get_rectangle()
                points.append((rect.left(), rect.top()))
                points.append((rect.right(), rect.top()))
                points.append((rect.right(), rect.bottom()))
                points.append((rect.left(), rect.bottom()))
                self.logger().info("Drawing polygon: %s" % str(lobj))
                if self.fill:
                    draw.polygon(tuple(points), outline=self._color_provider.get_color(color_label, alpha=self.outline_alpha),
                                 fill=self._color_provider.get_color(color_label, alpha=self.fill_alpha), width=self.outline_thickness)
                else:
                    draw.polygon(tuple(points), outline=self._color_provider.get_color(color_label, alpha=self.outline_alpha),
                                 width=self.outline_thickness)

            self.logger().info("Adding overlay")
            img_pil.paste(overlay, (0, 0), mask=overlay)

            # update image
            img_bytes = image_to_bytesio(img_pil, item.image_format)
            item_new = type(item)(image_name=item.image_name, data=img_bytes.getvalue(),
                                   annotation=copy.deepcopy(item.annotation), metadata=item.get_metadata())
            result.append(item_new)

        return flatten_list(result)
