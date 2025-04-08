# crop-to-label

* accepts: idc.api.ObjectDetectionData
* generates: idc.api.ObjectDetectionData

Crops an image to the bbox with the specified label.

```
usage: crop-to-label [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                     [-N LOGGER_NAME] [--skip] -r REGION_LABEL [-k]

Crops an image to the bbox with the specified label.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  --skip                Disables the plugin, removing it from the pipeline.
                        (default: False)
  -r REGION_LABEL, --region_label REGION_LABEL
                        The label of the bbox to crop the image to. (default:
                        None)
  -k, --keep_missing    For keeping images that don't have the label instead
                        of suppressing them. (default: False)
```
