import argparse
import logging
import sys
import traceback
from dataclasses import dataclass
from typing import List

from wai.logging import init_logging, set_logging_level, add_logging_level
from idc.core import ENV_IDC_LOGLEVEL

GENERATE_REGIONS = "idc-generate-regions"

_logger = logging.getLogger(GENERATE_REGIONS)


@dataclass
class Region:
    x: int
    y: int
    w: int
    h: int


def generate(width: int, height: int,
             num_rows: int = None, num_cols: int = None, fixed_size: bool = False,
             row_height: int = None, col_width: int = None, margin: int = 0,
             overlap_right: int = 0, overlap_bottom: int = 0,
             margin_left: int = 0, margin_top: int = 0, margin_right: int = 0, margin_bottom: int = 0,
             partial: bool = False) -> List[Region]:
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
    """
    result = []

    # image dimensions
    if width < 1:
        raise ValueError("Image width must be at least 1!")
    _logger.info("width: %d" % width)
    if height < 1:
        raise ValueError("Image height must be at least 1!")
    _logger.info("height: %d" % height)

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
        _logger.info("margin (left): %d" % margin_left)
    if margin_top > 0:
        _logger.info("margin (top): %d" % margin_top)
    if margin_right > 0:
        _logger.info("margin (right): %d" % margin_right)
    if margin_bottom > 0:
        _logger.info("margin (bottom): %d" % margin_bottom)

    # overlaps
    if overlap_right > 0:
        _logger.info("overlap (right): %d" % overlap_right)
    if overlap_bottom > 0:
        _logger.info("overlap (bottom): %d" % overlap_bottom)

    # section
    section_width = width - margin_left - margin_right
    if section_width < width:
        _logger.info("section width: %d" % section_width)
    section_height = height - margin_top - margin_bottom
    if section_height < height:
        _logger.info("section height: %d" % section_height)

    mode = None
    if (num_rows is not None) and (num_cols is not None):
        mode = "grid"
    if (row_height is not None) and (col_width is not None):
        mode = "size"
    if mode is None:
        raise ValueError("Either num_rows/num_cols or row_height/col_width must be specified!")

    # fixed grid
    if mode == "grid":
        _logger.info("#rows: %d" % num_rows)
        _logger.info("#cols: %d" % num_cols)
        _logger.info("fixed width/height: %s" % str(fixed_size))
        for row in range(num_rows):
            y = row * section_height // num_rows + margin_top
            if (row == num_rows - 1) and not fixed_size:
                h = section_height - y
            else:
                h = section_height // num_rows
            if row < num_rows - 1:
                h += overlap_bottom
            for col in range(num_cols):
                x = col * section_width // num_cols + margin_left
                if (col == num_cols - 1) and not fixed_size:
                    w = section_width - x
                else:
                    w = section_width // num_cols
                if col < num_cols - 1:
                    w += overlap_right
                result.append(Region(x=x, y=y, w=w, h=h))

    # fixed row/col size
    elif mode == "size":
        _logger.info("#row-height: %d" % row_height)
        _logger.info("#col-width: %d" % col_width)
        _logger.info("partial: %s" % str(partial))
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


def regions_to_string(regions: List[Region], one_based: bool = False) -> str:
    """
    Turns the regions into a single string.

    :param regions: the regions to convert
    :type regions: list
    :param one_based: whether to use 1-based coordinates
    :type one_based: bool
    :return: the generated string
    :rtype: str
    """
    _logger.info("#regions: %d" % len(regions))
    _logger.info("1-based coordinates: %s" % str(one_based))
    regions_str = ""
    for region in regions:
        if one_based:
            regions_str += " %d,%d,%d,%d" % ((region.x+1), (region.y+1), region.w, region.h)
        else:
            regions_str += " %d,%d,%d,%d" % (region.x, region.y, region.w, region.h)
    return regions_str.strip()


def main(args=None):
    """
    The main method for parsing command-line arguments.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """
    init_logging(env_var=ENV_IDC_LOGLEVEL)
    parser = argparse.ArgumentParser(
        description="Tool turns an image size into regions to be used, e.g., with the 'sub-images' filter. Either specify the number of rows/cols or the height/width of rows/cols.",
        prog=GENERATE_REGIONS,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-W", "--width", type=int, help="The width of the image.", default=None, required=True)
    parser.add_argument("-H", "--height", type=int, help="The height of the image.", default=None, required=True)
    parser.add_argument("-r", "--num_rows", type=int, help="The number of rows.", default=None, required=False)
    parser.add_argument("-c", "--num_cols", type=int, help="The number of columns.", default=None, required=False)
    parser.add_argument("-R", "--row_height", type=int, help="The height of rows.", default=None, required=False)
    parser.add_argument("-C", "--col_width", type=int, help="The width of columns.", default=None, required=False)
    parser.add_argument("--overlap_right", type=int, help="The overlap between two images (on the right of the left-most image).", default=0, required=False)
    parser.add_argument("--overlap_bottom", type=int, help="The overlap between two images (on the bottom of the top-most image).", default=0, required=False)
    parser.add_argument("-m", "--margin", type=int, help="The margin around the actual section to generate the regions from.", default=0, required=False)
    parser.add_argument("--margin_left", type=int, help="The left margin for the actual section to generate the regions from.", default=0, required=False)
    parser.add_argument("--margin_top", type=int, help="The top margin for the actual section to generate the regions from.", default=0, required=False)
    parser.add_argument("--margin_right", type=int, help="The right margin for the actual section to generate the regions from.", default=0, required=False)
    parser.add_argument("--margin_bottom", type=int, help="The bottom margin for the actual section to generate the regions from.", default=0, required=False)
    parser.add_argument("-f", "--fixed_size", action="store_true", help="Whether to use fixed row height/col width, omitting any left-over bits at right/bottom, when using num_rows/num_cols", required=False)
    parser.add_argument("-p", "--partial", action="store_true", help="Whether to output partial regions, the left-over bits at right/bottom, when using row_height/col_width", required=False)
    parser.add_argument("-1", "--one_based", action="store_true", help="Whether to use 1-based coordinates", required=False)
    add_logging_level(parser)
    parsed = parser.parse_args(args=args)
    set_logging_level(_logger, parsed.logging_level)
    regions = generate(parsed.width, parsed.height,
                       num_rows=parsed.num_rows, num_cols=parsed.num_cols, fixed_size=parsed.fixed_size,
                       row_height=parsed.row_height, col_width=parsed.col_width, margin=parsed.margin,
                       overlap_right=parsed.overlap_right, overlap_bottom=parsed.overlap_bottom,
                       margin_left=parsed.margin_left, margin_top=parsed.margin_top,
                       margin_right=parsed.margin_right, margin_bottom=parsed.margin_bottom,
                       partial=parsed.partial)
    print(regions_to_string(regions, one_based=parsed.one_based))


def sys_main() -> int:
    """
    Runs the main function using the system cli arguments, and
    returns a system error code.

    :return: 0 for success, 1 for failure.
    """
    try:
        main()
        return 0
    except Exception:
        traceback.print_exc()
        print("options: %s" % str(sys.argv[1:]), file=sys.stderr)
        return 1


if __name__ == '__main__':
    main()
