# meta-sub-images

* accepts: idc.api.ImageClassificationData, idc.api.ObjectDetectionData, idc.api.ImageSegmentationData
* generates: idc.api.ImageClassificationData, idc.api.ObjectDetectionData, idc.api.ImageSegmentationData

Extracts sub-images (incl their annotations) from the images coming through, using the defined regions or #rows/cols, and passes them through the base filter before reassembling them again.

```
usage: meta-sub-images [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                       [-N LOGGER_NAME] [-r [REGIONS ...]]
                       [--num_rows NUM_ROWS] [--num_cols NUM_COLS]
                       [--row_height ROW_HEIGHT] [--col_width COL_WIDTH]
                       [--overlap_right OVERLAP_RIGHT]
                       [--overlap_bottom OVERLAP_BOTTOM]
                       [-s {none,x-then-y,y-then-x}] [-p] [-e] [-S SUFFIX]
                       [-b BASE_FILTER] [-R] [-m] [--pad_width PAD_WIDTH]
                       [--pad_height PAD_HEIGHT]

Extracts sub-images (incl their annotations) from the images coming through,
using the defined regions or #rows/cols, and passes them through the base
filter before reassembling them again.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  -r [REGIONS ...], --regions [REGIONS ...]
                        The regions (X,Y,WIDTH,HEIGHT) to crop and forward
                        with their annotations (0-based coordinates) (default:
                        None)
  --num_rows NUM_ROWS   The number of rows, if no regions defined. (default:
                        None)
  --num_cols NUM_COLS   The number of columns, if no regions defined.
                        (default: None)
  --row_height ROW_HEIGHT
                        The height of rows. (default: None)
  --col_width COL_WIDTH
                        The width of columns. (default: None)
  --overlap_right OVERLAP_RIGHT
                        The overlap between two images (on the right of the
                        left-most image), if no regions defined. (default: 0)
  --overlap_bottom OVERLAP_BOTTOM
                        The overlap between two images (on the bottom of the
                        top-most image), if no regions defined. (default: 0)
  -s {none,x-then-y,y-then-x}, --region_sorting {none,x-then-y,y-then-x}
                        How to sort the supplied region definitions (default:
                        none)
  -p, --include_partial
                        Whether to include only annotations that fit fully
                        into a region or also partial ones (default: False)
  -e, --suppress_empty  Suppresses sub-images that have no annotations
                        (default: False)
  -S SUFFIX, --suffix SUFFIX
                        The suffix pattern to use for the generated sub-
                        images, available placeholders:
                        {X}|{Y}|{W}|{H}|{X0}|{Y0}|{X1}|{Y1}|{INDEX} (default:
                        -{INDEX})
  -b BASE_FILTER, --base_filter BASE_FILTER
                        The base filter to pass the sub-images through
                        (default: passthrough)
  -R, --rebuild_image   Rebuilds the image from the filtered sub-images rather
                        than using the input image. (default: False)
  -m, --merge_adjacent_polygons
                        Whether to merge adjacent polygons (object detection
                        only). (default: False)
  --pad_width PAD_WIDTH
                        The width to pad the sub-images to (on the right).
                        (default: None)
  --pad_height PAD_HEIGHT
                        The height to pad the sub-images to (at the bottom).
                        (default: None)
```
