# clip-grayscale

* accepts: idc.api.ImageClassificationData, idc.api.ObjectDetectionData, idc.api.ImageSegmentationData
* generates: idc.api.ImageClassificationData, idc.api.ObjectDetectionData, idc.api.ImageSegmentationData

Changes the grayscale values that fall below the minimum or go above the maximum to the specified replacement values.

```
usage: clip-grayscale [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                      [-N LOGGER_NAME] [-m MIN_VALUE] [-r MIN_REPLACEMENT]
                      [-M MAX_VALUE] [-R MAX_REPLACEMENT]

Changes the grayscale values that fall below the minimum or go above the
maximum to the specified replacement values.

optional arguments:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  -m MIN_VALUE, --min_value MIN_VALUE
                        The smallest allowed grayscale pixel value. (default:
                        0)
  -r MIN_REPLACEMENT, --min_replacement MIN_REPLACEMENT
                        The replacement grayscale pixel value for values that
                        fall below the minimum. (default: 0)
  -M MAX_VALUE, --max_value MAX_VALUE
                        The largest allowed grayscale pixel value. (default:
                        255)
  -R MAX_REPLACEMENT, --max_replacement MAX_REPLACEMENT
                        The replacement grayscale pixel value for values that
                        go above the minimum. (default: 255)
```
