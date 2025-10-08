import argparse
import logging
import sys
import traceback

from wai.logging import init_logging, set_logging_level, add_logging_level

from idc.core import ENV_IDC_LOGLEVEL
from idc.imgaug.filter import ARUCO_TYPES, DEFAULT_ARUCO_TYPE, generate_aruco

GENERATE_ARUCO = "idc-generate-aruco"

_logger = logging.getLogger(GENERATE_ARUCO)


def main(args=None):
    """
    The main method for parsing command-line arguments.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """
    init_logging(env_var=ENV_IDC_LOGLEVEL)
    parser = argparse.ArgumentParser(
        description="Tool for generating ArUco code marker images.",
        prog=GENERATE_ARUCO,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--size", type=int, help="The size of the image (width and height).", default=200, required=False)
    parser.add_argument("-i", "--aruco_id", type=int, default=None, help="The ID to encode.", required=True)
    parser.add_argument("-t", "--aruco_type", choices=ARUCO_TYPES.keys(), default=DEFAULT_ARUCO_TYPE, help="The type of markers to detect.", required=False)
    parser.add_argument("-o", "--output", metavar="FILE", type=str, help="The file to save the image in.", default="./marker.png", required=False)
    add_logging_level(parser)
    parsed = parser.parse_args(args=args)
    set_logging_level(_logger, parsed.logging_level)
    if generate_aruco(parsed.size, parsed.aruco_id, parsed.aruco_type, parsed.output, logger=_logger):
        _logger.info("Successfully generated!")
    else:
        _logger.error("Failed to generate!")


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
