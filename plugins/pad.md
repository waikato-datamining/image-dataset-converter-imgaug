# pad

* accepts: idc.api.ImageClassificationData, idc.api.ImageSegmentationData, idc.api.ObjectDetectionData, idc.api.DepthData
* generates: idc.api.ImageClassificationData, idc.api.ImageSegmentationData, idc.api.ObjectDetectionData, idc.api.DepthData

Pads the images to have at least the specified width/height.

```
usage: pad [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [-N LOGGER_NAME]
           [--skip] [-W WIDTH] [-H HEIGHT] [-b R,G,B]

Pads the images to have at least the specified width/height.

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
                        The minimum width to pad to, ignored if not specified.
                        (default: None)
  -H HEIGHT, --height HEIGHT
                        The minimum height to pad to, ignored if not
                        specified. (default: None)
  -b R,G,B, --background R,G,B
                        The RGB triplet (R,G,B) to use for the background
                        color (default: 0,0,0)
```
