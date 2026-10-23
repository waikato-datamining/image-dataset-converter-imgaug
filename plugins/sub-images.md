# sub-images

* accepts: idc.api.ImageClassificationData, idc.api.ObjectDetectionData, idc.api.ImageSegmentationData, idc.api.DepthData
* generates: idc.api.ImageClassificationData, idc.api.ObjectDetectionData, idc.api.ImageSegmentationData, idc.api.DepthData

Extracts sub-images (incl their annotations) from the images coming through, using the defined regions or #rows/cols. When using x/y in the suffix, these images can be reassembled using the 'idc-combine-sub-images' tool.

```
usage: sub-images [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                  [-N LOGGER_NAME] [--skip] [-r [REGION ...]]
                  [--num_rows ROWS] [--num_cols COLS] [--row_height HEIGHT]
                  [--col_width WIDTH] [--partial_sub_images]
                  [--overlap_right RIGHT] [--overlap_bottom BOTTOM]
                  [-s {none,x-then-y,y-then-x}] [-p] [--min_partial_iou IOU]
                  [-e] [-S SUFFIX] [--pad_width WIDTH] [--pad_height HEIGHT]

Extracts sub-images (incl their annotations) from the images coming through,
using the defined regions or #rows/cols. When using x/y in the suffix, these
images can be reassembled using the 'idc-combine-sub-images' tool.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  --skip                Disables the plugin, removing it from the pipeline.
                        (default: False)
  -r [REGION ...], --regions [REGION ...]
                        The regions (X,Y,WIDTH,HEIGHT) to crop and forward
                        with their annotations (0-based coordinates) (default:
                        None)
  --num_rows ROWS       The number of rows, if no regions defined. (default:
                        None)
  --num_cols COLS       The number of columns, if no regions defined.
                        (default: None)
  --row_height HEIGHT   The height of rows. (default: None)
  --col_width WIDTH     The width of columns. (default: None)
  --partial_sub_images  Whether to use sub-images that don't have the required
                        --col_width/--row_height (default: False)
  --overlap_right RIGHT
                        The overlap between two images (on the right of the
                        left-most image), if no regions defined, gets added to
                        the sub-image width. (default: 0)
  --overlap_bottom BOTTOM
                        The overlap between two images (on the bottom of the
                        top-most image), if no regions defined, gets added to
                        the sub-image width. (default: 0)
  -s {none,x-then-y,y-then-x}, --region_sorting {none,x-then-y,y-then-x}
                        How to sort the supplied region definitions (default:
                        none)
  -p, --include_partial
                        Whether to include only objdet annotations that fit
                        fully into a region or also partial ones (default:
                        False)
  --min_partial_iou IOU
                        When including partial object detection annotations,
                        the partial ones must have at least this IoU to be
                        included; uses polygon as basis if available,
                        otherwise the bounding box. (default: 0.0)
  -e, --suppress_empty  Suppresses sub-images that have no annotations
                        (default: False)
  -S SUFFIX, --suffix SUFFIX
                        The suffix pattern to use for the generated sub-
                        images, available placeholders:
                        {X}|{Y}|{W}|{H}|{X0}|{Y0}|{X1}|{Y1}|{INDEX} (default:
                        -{INDEX})
  --pad_width WIDTH     The width to pad the sub-images to (on the right).
                        (default: None)
  --pad_height HEIGHT   The height to pad the sub-images to (at the bottom).
                        (default: None)
```
