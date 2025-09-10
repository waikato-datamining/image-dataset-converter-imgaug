# simple-blob-detector

* accepts: idc.api.ImageData
* generates: idc.api.ObjectDetectionData

Finds blobs in grayscale images/annotations and stores them as rectangle annotations.

```
usage: simple-blob-detector [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                            [-N LOGGER_NAME] [--skip] [-I {skip,fail}]
                            [-a {both,image,annotations}] [--label LABEL]
                            [--min_threshold MIN_THRESHOLD]
                            [--max_threshold MAX_THRESHOLD]
                            [--threshold_step THRESHOLD_STEP]
                            [--filter_by_color] [--blob_color BLOB_COLOR]
                            [--filter_by_area] [--min_area MIN_AREA]
                            [--max_area MAX_AREA] [--filter_by_circularity]
                            [--min_circularity MIN_CIRCULARITY]
                            [--max_circularity MAX_CIRCULARITY]
                            [--filter_by_convexity]
                            [--min_convexity MIN_CONVEXITY]
                            [--max_convexity MAX_CONVEXITY]
                            [--filter_by_inertia]
                            [--min_inertia_ratio MIN_INERTIA_RATIO]
                            [--max_inertia_ratio MAX_INERTIA_RATIO]
                            [--min_dist_between_blobs MIN_DIST_BETWEEN_BLOBS]

Finds blobs in grayscale images/annotations and stores them as rectangle
annotations.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  --skip                Disables the plugin, removing it from the pipeline.
                        (default: False)
  -I {skip,fail}, --incorrect_format_action {skip,fail}
                        The action to undertake if an invalid input format is
                        encountered. (default: skip)
  -a {both,image,annotations}, --apply_to {both,image,annotations}
                        Where to apply the filter to. (default: image)
  --label LABEL         The label to use for the blob annotations. (default:
                        object)
  --min_threshold MIN_THRESHOLD
                        The minimum threshold (inclusive) for converting to
                        binary. (default: None)
  --max_threshold MAX_THRESHOLD
                        The maximum threshold (exclusive) for converting to
                        binary. (default: None)
  --threshold_step THRESHOLD_STEP
                        The distance thresholdStep between neighboring
                        thresholds. (default: None)
  --filter_by_color     This filter compares the intensity of a binary image
                        at the center of a blob to blobColor. If they differ,
                        the blob is filtered out. Use blobColor = 0 to extract
                        dark blobs and blobColor = 255 to extract light blobs.
                        (default: False)
  --blob_color BLOB_COLOR
                        The blob color to use. (default: None)
  --filter_by_area      Extracted blobs have an area between minArea
                        (inclusive) and maxArea (exclusive). (default: False)
  --min_area MIN_AREA   The minimum area to use. (default: None)
  --max_area MAX_AREA   The maximum area. (default: None)
  --filter_by_circularity
                        Extracted blobs have circularity
                        ((4∗π∗Area)/(perimeter∗perimeter)) between
                        minCircularity (inclusive) and maxCircularity
                        (exclusive). (default: False)
  --min_circularity MIN_CIRCULARITY
                        The minimum circularity. (default: None)
  --max_circularity MAX_CIRCULARITY
                        The maximum circularity. (default: None)
  --filter_by_convexity
                        Extracted blobs have convexity (area / area of blob
                        convex hull) between minConvexity (inclusive) and
                        maxConvexity (exclusive). (default: False)
  --min_convexity MIN_CONVEXITY
                        The minimum convexity. (default: None)
  --max_convexity MAX_CONVEXITY
                        The maximum convexity. (default: None)
  --filter_by_inertia   Extracted blobs have this ratio between
                        minInertiaRatio (inclusive) and maxInertiaRatio
                        (exclusive). (default: False)
  --min_inertia_ratio MIN_INERTIA_RATIO
                        The minimum inertia ratio. (default: None)
  --max_inertia_ratio MAX_INERTIA_RATIO
                        The maximum inertia ratio. (default: None)
  --min_dist_between_blobs MIN_DIST_BETWEEN_BLOBS
                        The minimum distance between detected blobs. (default:
                        None)
```
