# find-contours

* accepts: idc.api.ImageData
* generates: idc.api.ObjectDetectionData

Detects blobs images using scikit-image's find_contours method and turns them into object detection polygons. In case of image segmentation data, the annotations are analyzed, otherwise the base image.

```
usage: find-contours [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                     [-N LOGGER_NAME] [--skip] [-t MASK_THRESHOLD]
                     [-n MASK_NTH] [-v VIEW_MARGIN] [-f {low,high}]
                     [--label LABEL] [-m MIN_SIZE] [-M MAX_SIZE]

Detects blobs images using scikit-image's find_contours method and turns them
into object detection polygons. In case of image segmentation data, the
annotations are analyzed, otherwise the base image.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  --skip                Disables the plugin, removing it from the pipeline.
                        (default: False)
  -t MASK_THRESHOLD, --mask_threshold MASK_THRESHOLD
                        The (lower) probability threshold for mask values in
                        order to be considered part of the object (0-1).
                        (default: 0.1)
  -n MASK_NTH, --mask_nth MASK_NTH
                        The contour tracing can be slow for large masks, by
                        using only every nth row/col, this can be sped up
                        dramatically. (default: 1)
  -v VIEW_MARGIN, --view_margin VIEW_MARGIN
                        The margin in pixels to enlarge the view with in each
                        direction. (default: 5)
  -f {low,high}, --fully_connected {low,high}
                        Whether regions of high or low values should be fully-
                        connected at isthmuses. (default: low)
  --label LABEL         The label to use when processing images other than
                        image segmentation ones. (default: object)
  -m MIN_SIZE, --min_size MIN_SIZE
                        The minimum width or height that detected contours
                        must have. (default: None)
  -M MAX_SIZE, --max_size MAX_SIZE
                        The maximum width or height that detected contours can
                        have. (default: None)
```
