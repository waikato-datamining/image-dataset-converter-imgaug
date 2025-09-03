# thinning

* accepts: idc.api.ImageClassificationData, idc.api.ObjectDetectionData, idc.api.ImageSegmentationData
* generates: idc.api.ImageClassificationData, idc.api.ObjectDetectionData, idc.api.ImageSegmentationData

Thinning algorithm developed by Lingdong Huang: https://github.com/LingDong-/skeleton-tracing/blob/master/py/trace_skeleton.py

```
usage: thinning [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [-N LOGGER_NAME]
                [--skip] [-a {both,image,annotations}]
                [-o {as-is,binary,grayscale,rgb}] [-I {skip,fail}]

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
  -a {both,image,annotations}, --apply_to {both,image,annotations}
                        Where to apply the filter to. (default: image)
  -o {as-is,binary,grayscale,rgb}, --output_format {as-is,binary,grayscale,rgb}
                        The image format to generate as output. (default: as-
                        is)
  -I {skip,fail}, --incorrect_format_action {skip,fail}
                        The action to undertake if an invalid input format is
                        encountered. (default: skip)
```
