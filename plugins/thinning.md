# thinning

* accepts: idc.api.ImageClassificationData, idc.api.ObjectDetectionData, idc.api.ImageSegmentationData
* generates: idc.api.ImageClassificationData, idc.api.ObjectDetectionData, idc.api.ImageSegmentationData

Thinning algorithm developed by Lingdong Huang: https://github.com/LingDong-/skeleton-tracing/blob/master/py/trace_skeleton.py

```
usage: thinning [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [-N LOGGER_NAME]
                [--skip] [-f FACTOR]

Thinning algorithm developed by Lingdong Huang:
https://github.com/LingDong-/skeleton-tracing/blob/master/py/trace_skeleton.py

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  --skip                Disables the plugin, removing it from the pipeline.
                        (default: False)
  -f FACTOR, --factor FACTOR
                        The factor with which to scale the image before
                        applying the thinning algorithm. (default: 1.0)
```
