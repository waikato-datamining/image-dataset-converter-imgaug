import argparse
import logging
import os.path
import re
import sys
import traceback
from typing import List, Dict, Tuple

from seppl import Initializable, init_initializable, Session
from seppl.io import locate_files, StreamWriter, BatchWriter, Writer
from seppl.placeholders import placeholder_list
from wai.common.adams.imaging.locateobjects import LocatedObject
from wai.logging import init_logging, set_logging_level, add_logging_level

from idc.api import ImageData, parse_reader, parse_writer, Reader, ObjectDetectionData, merge_polygons, empty_image
from idc.core import ENV_IDC_LOGLEVEL
from idc.imgaug.filter import new_from_template, transfer_region, prune_annotations

COMBINE_SUB_IMAGES = "idc-combine-sub-images"

_logger = logging.getLogger(COMBINE_SUB_IMAGES)


def group_files(input_files: List[str], regexp: str) -> Dict[str, List[str]]:
    """
    Groups the files using the specified regexp.

    :param input_files: the files to group
    :type input_files: list
    :param regexp: the regexp to use for grouping (1st group is image group ID)
    :type regexp: str
    :return: the grouped images, key is group ID
    :rtype: dict
    """
    result = dict()
    for input_file in input_files:
        input_name = os.path.basename(input_file)
        m = re.search(regexp, input_name)
        if m is None:
            raise Exception("Failed to extract 1st group from '%s' using '%s'" % (input_name, regexp))
        group_id = m.group(1)
        if group_id not in result:
            result[group_id] = []
        result[group_id].append(input_file)
    return result


def extract_coordinate(input_file: str, regexp: str) -> int:
    """
    Extracts the coordinate using the specified regexp (1st group is coordinate).

    :param input_file: the file to get the coordinate for
    :type input_file: str
    :param regexp: the regexp to use for the coordinate (uses 1st group)
    :type regexp: str
    :return: the coordinate
    :rtype: int
    """
    input_name = os.path.basename(input_file)
    m = re.search(regexp, input_name)
    if m is None:
        raise Exception("Failed to extract 1st group from '%s' using '%s'" % (input_name, regexp))
    coordinate = m.group(1)
    return int(coordinate)


def extract_coordinates(input_files: List[str], x: str, y: str, one_based: bool) -> List[Tuple[int, int]]:
    """
    Extracts the x/y coordinates from the files and returns a corresponding list with x/y tuples.

    :param input_files: the files to process
    :type input_files: list
    :param x: the regexp for the x coordinate (1st group)
    :type x: str
    :param y: the regexp for the y coordinate (1st group)
    :type y: str
    :param one_based: whether the coordinates are 1-based or 0-based
    :type one_based: bool
    :return: the list of x/y tuples
    :rtype: list
    """
    result = []
    for input_file in input_files:
        coord_x = extract_coordinate(input_file, x)
        coord_y = extract_coordinate(input_file, y)
        if one_based:
            coord_x -= 1
            coord_y -= 1
        result.append((coord_x, coord_y))
    return result


def read_images(input_files: List[str], reader: Reader) -> List[ImageData]:
    """
    Reads the images using the specified reader command-line.

    :param input_files: the files to read
    :type input_files: list
    :param reader: the reader to use
    :type reader: Reader
    :return: the list of image containers
    :rtype: list
    """
    reader.source = input_files
    init_initializable(reader, "reader", raise_again=True)
    result = []
    while not reader.has_finished():
        for item in reader.read():
            result.append(item)
    reader.finalize()
    return result


def merge_images(input_images: List[ImageData], coordinates: [List[Tuple[int, int]]], width: int, height: int, image_name: str) -> ImageData:
    """
    Merges the images into a new one.

    :param input_images: the images to merge
    :type input_images: list
    :param coordinates: the coordinates for the images (list of x/y tuples)
    :type coordinates: list
    :param width: the width of the combined image
    :type width: int
    :param height: the height of the combined image
    :type height: int
    :param image_name: the name for the image
    :type image_name: str
    :return: the combined image/annotations
    :rtype: ImageData
    """
    _, img_bytes = empty_image("RGB", width, height, input_images[0].image_format)
    data = img_bytes.getvalue()

    cls = type(input_images[0])
    obj = cls(image_name=image_name, data=data)
    result = new_from_template(obj, rebuild_image=False)

    for coords, sub_image in zip(coordinates, input_images):
        region = LocatedObject(coords[0], coords[1], sub_image.image_width, sub_image.image_height)
        transfer_region(result, sub_image, region, rebuild_image=True)

    return result


def write_image(combined: ImageData, writer: Writer):
    """
    Writes the combined image using the specified writer.

    :param combined: the combined image
    :type combined: ImageData
    :param writer: the writer to use
    :type writer: Writer
    """
    if isinstance(writer, Initializable):
        init_initializable(writer, "writer", raise_again=True)

    if isinstance(writer, StreamWriter):
        writer.write_stream(combined)
    elif isinstance(writer, BatchWriter):
        writer.write_batch([combined])
    else:
        raise Exception("Unhandled type of writer: %s" % str(type(writer)))

    if isinstance(writer, Initializable):
        writer.finalize()


def combine(input_files: List[str], group: str, x: str, y: str, width: int, height: int, one_based: bool,
            reader: str, writer: str, merge_adjacent_polygons: bool = False):
    """
    Generates the regions and returns them. Either specify num_rows/num_cols or row_height/col_width.

    :param input_files: the list of files to combine
    :type input_files: list
    :param group: the regexp to identify the sub-images that belong together (1st group)
    :type group: str
    :param x: the regexp to identify the x coordinate (1st group)
    :type x: str
    :param y: the regexp to identify the y coordinate (1st group)
    :type y: str
    :param width: the width of the image
    :type width: int
    :param height: the height of the image
    :type height: int
    :param one_based: whether the coordinates are 1-based or 0-based
    :type one_based: bool
    :param reader: the reader command-line to use for reading the sub-images
    :type reader: str
    :param writer: the writer command-line to use for writing the combined images
    :type writer: str
    :param merge_adjacent_polygons: whether to merge adjacent polygons (object detection only)
    :type merge_adjacent_polygons: bool
    """
    _logger.info("Instantiating reader: %s" % reader)
    reader = parse_reader(reader)
    reader.session = Session()
    if not hasattr(reader, "source"):
        raise Exception("Reader does not have 'source' attribute: %s" % str(type(reader)))

    _logger.info("Instantiating writer: %s" % writer)
    writer = parse_writer(writer)
    writer.session = Session()

    _logger.info("Locating files...")
    input_files = locate_files(input_files, None, fail_if_empty=True)
    _logger.info("Found %d files" % len(input_files))

    _logger.info("Grouping files...")
    grouped = group_files(input_files, group)
    _logger.info("%d groups determined" % len(grouped))

    for group_id in grouped:
        _logger.info("Processing group: %s" % group_id)
        gfiles = grouped[group_id]
        gimages = read_images(gfiles, reader)
        gcoords = extract_coordinates(gfiles, x, y, one_based)
        image_name = group_id + "." + gimages[0].image_format.lower().replace("jpeg", "jpg")
        combined = merge_images(gimages, gcoords, width, height, image_name)
        prune_annotations(combined)
        if merge_adjacent_polygons and isinstance(combined, ObjectDetectionData):
            combined = merge_polygons(combined)
        write_image(combined, writer)


def main(args=None):
    """
    The main method for parsing command-line arguments.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """
    init_logging(env_var=ENV_IDC_LOGLEVEL)
    parser = argparse.ArgumentParser(
        description="Tool for combining image regions and annotations generated by the 'sub-images' filter. NB: it does not combine adjacent annotations.",
        prog=COMBINE_SUB_IMAGES,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", type=str, help="Path to the report file(s) to read; glob syntax is supported; " + placeholder_list(input_based=False), required=True, nargs="+")
    parser.add_argument("-g", "--group", metavar="REGEXP", type=str, help="Regular expression for grouping the sub-images into ones that belong to a single image (first group is used a group ID)", required=True)
    parser.add_argument("-x", metavar="REGEXP", type=str, help="Regular expression for extracting the x coordinate (first group)", required=True)
    parser.add_argument("-y", metavar="REGEXP", type=str, help="Regular expression for extracting the y coordinate (first group)", required=True)
    parser.add_argument("-W", "--width", type=int, help="The width of the image.", default=None, required=True)
    parser.add_argument("-H", "--height", type=int, help="The height of the image.", default=None, required=True)
    parser.add_argument("-1", "--one_based", action="store_true", help="Whether the coordinates are 1-based", required=False)
    parser.add_argument("-r", "--reader", metavar="CMDLINE", type=str, help="The reader command-line to use for reading the sub-images.", required=True, default=None)
    parser.add_argument("-m", "--merge_adjacent_polygons", action="store_true", help="Whether to merge adjacent polygons (object detection only).", required=False)
    parser.add_argument("-w", "--writer", metavar="CMDLINE", type=str, help="The writer command-line to use for writing the combined images, must contain parameters for storing the output.", required=True, default=None)
    add_logging_level(parser)
    parsed = parser.parse_args(args=args)
    set_logging_level(_logger, parsed.logging_level)
    combine(parsed.input, parsed.group, parsed.x, parsed.y, parsed.width, parsed.height, parsed.one_based,
            parsed.reader, parsed.writer, merge_adjacent_polygons=parsed.merge_adjacent_polygons)


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
