# simple-crop

* accepts: idc.api.ImageClassificationData, idc.api.ImageSegmentationData, idc.api.ObjectDetectionData, idc.api.DepthData
* generates: idc.api.ImageClassificationData, idc.api.ImageSegmentationData, idc.api.ObjectDetectionData, idc.api.DepthData

Crops the image to the specified width/height.

```
usage: simple-crop [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                   [-N LOGGER_NAME] [--skip] [-W WIDTH] [-H HEIGHT] [-p]

Crops the image to the specified width/height.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  --skip                Disables the plugin, removing it from the pipeline.
                        (default: False)
  -W WIDTH, --width WIDTH
                        The width to crop to, ignored if not specified.
                        (default: None)
  -H HEIGHT, --height HEIGHT
                        The height to crop to, ignored if not specified.
                        (default: None)
  -p, --include_partial
                        Whether to include only annotations that fit fully
                        into a region or also partial ones (default: False)
```
