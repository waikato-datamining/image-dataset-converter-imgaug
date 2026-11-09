# merge-overlapping-polygons

* accepts: idc.api.ObjectDetectionData
* generates: idc.api.ObjectDetectionData

Merges overlapping polygons with the same label. NB: loses any meta-data apart from label due to merge operation.

```
usage: merge-overlapping-polygons [-h]
                                  [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                                  [-N LOGGER_NAME] [--skip] [-t THRESH]

Merges overlapping polygons with the same label. NB: loses any meta-data apart
from label due to merge operation.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  --skip                Disables the plugin, removing it from the pipeline.
                        (default: False)
  -t THRESH, --ios_threshold THRESH
                        The IoS (intersect over small) threshold for merging
                        polygons. (default: 0.5)
```
