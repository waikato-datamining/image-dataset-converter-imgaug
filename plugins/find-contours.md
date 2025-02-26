# find-contours

* accepts: idc.api.ImageSegmentationData
* generates: idc.api.ObjectDetectionData

Detects blobs in the annotations of the image segmentation data and turns them into object detection polygons.

```
usage: find-contours [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                     [-N LOGGER_NAME] [-t MASK_THRESHOLD] [-n MASK_NTH]
                     [-m VIEW_MARGIN] [-f {low,high}]

Detects blobs in the annotations of the image segmentation data and turns them
into object detection polygons.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  -t MASK_THRESHOLD, --mask_threshold MASK_THRESHOLD
                        The (lower) probability threshold for mask values in
                        order to be considered part of the object (0-1).
                        (default: 0.1)
  -n MASK_NTH, --mask_nth MASK_NTH
                        The contour tracing can be slow for large masks, by
                        using only every nth row/col, this can be sped up
                        dramatically. (default: 1)
  -m VIEW_MARGIN, --view_margin VIEW_MARGIN
                        The margin in pixels to enlarge the view with in each
                        direction. (default: 5)
  -f {low,high}, --fully_connected {low,high}
                        Whether regions of high or low values should be fully-
                        connected at isthmuses. (default: low)
```
