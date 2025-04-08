# change-grayscale

* accepts: idc.api.ImageClassificationData, idc.api.ObjectDetectionData, idc.api.ImageSegmentationData
* generates: idc.api.ImageClassificationData, idc.api.ObjectDetectionData, idc.api.ImageSegmentationData

Changes the pixel values of grayscale images either by a factor or by a fixed value.

```
usage: change-grayscale [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                        [-N LOGGER_NAME] [--skip] [--factor FACTOR]
                        [--increment INCREMENT]

Changes the pixel values of grayscale images either by a factor or by a fixed
value.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  --skip                Disables the plugin, removing it from the pipeline.
                        (default: False)
  --factor FACTOR       The factor with which to scale the pixel values.
                        (default: None)
  --increment INCREMENT
                        The value to change the pixel values by. (default:
                        None)
```
