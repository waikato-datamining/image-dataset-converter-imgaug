# aruco-detector

* accepts: idc.api.ImageData
* generates: idc.api.ImageData

Detects ArUco markers and adds them to the meta-data.

```
usage: aruco-detector [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                      [-N LOGGER_NAME] [--skip] [-I {skip,fail}] [-p PREFIX]
                      [-t {DICT_4X4_50,DICT_4X4_100,DICT_4X4_250,DICT_4X4_1000,DICT_5X5_50,DICT_5X5_100,DICT_5X5_250,DICT_5X5_1000,DICT_6X6_50,DICT_6X6_100,DICT_6X6_250,DICT_6X6_1000,DICT_7X7_50,DICT_7X7_100,DICT_7X7_250,DICT_7X7_1000}]

Detects ArUco markers and adds them to the meta-data.

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
  -p PREFIX, --prefix PREFIX
                        The prefix to use for the detected markers in the
                        meta-data. (default: aruco-)
  -t {DICT_4X4_50,DICT_4X4_100,DICT_4X4_250,DICT_4X4_1000,DICT_5X5_50,DICT_5X5_100,DICT_5X5_250,DICT_5X5_1000,DICT_6X6_50,DICT_6X6_100,DICT_6X6_250,DICT_6X6_1000,DICT_7X7_50,DICT_7X7_100,DICT_7X7_250,DICT_7X7_1000}, --aruco_type {DICT_4X4_50,DICT_4X4_100,DICT_4X4_250,DICT_4X4_1000,DICT_5X5_50,DICT_5X5_100,DICT_5X5_250,DICT_5X5_1000,DICT_6X6_50,DICT_6X6_100,DICT_6X6_250,DICT_6X6_1000,DICT_7X7_50,DICT_7X7_100,DICT_7X7_250,DICT_7X7_1000}
                        The type of markers to detect. (default: DICT_6X6_250)
```
