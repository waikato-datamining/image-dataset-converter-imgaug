import argparse
import logging
import sys
import traceback

from wai.logging import init_logging, set_logging_level, add_logging_level

from idc.core import ENV_IDC_LOGLEVEL
from idc.imgaug.filter import generate_regions, regions_to_string

GENERATE_REGIONS = "idc-generate-regions"

_logger = logging.getLogger(GENERATE_REGIONS)


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
    regions = generate_regions(parsed.width, parsed.height,
                               num_rows=parsed.num_rows, num_cols=parsed.num_cols, fixed_size=parsed.fixed_size,
                               row_height=parsed.row_height, col_width=parsed.col_width, margin=parsed.margin,
                               overlap_right=parsed.overlap_right, overlap_bottom=parsed.overlap_bottom,
                               margin_left=parsed.margin_left, margin_top=parsed.margin_top,
                               margin_right=parsed.margin_right, margin_bottom=parsed.margin_bottom,
                               partial=parsed.partial, logger=_logger)
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
