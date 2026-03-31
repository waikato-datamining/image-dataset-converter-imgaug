# aruco-autorotate

* accepts: idc.api.ImageData
* generates: idc.api.ImageData

Automatically rotates the image according to the orientation of the ArUco marker(s) in 90degree increments.

```
usage: aruco-autorotate [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                        [-N LOGGER_NAME] [--skip]
                        [-t {DICT_4X4_50,DICT_4X4_100,DICT_4X4_250,DICT_4X4_1000,DICT_5X5_50,DICT_5X5_100,DICT_5X5_250,DICT_5X5_1000,DICT_6X6_50,DICT_6X6_100,DICT_6X6_250,DICT_6X6_1000,DICT_7X7_50,DICT_7X7_100,DICT_7X7_250,DICT_7X7_1000}]

Automatically rotates the image according to the orientation of the ArUco
marker(s) in 90degree increments.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  --skip                Disables the plugin, removing it from the pipeline.
                        (default: False)
  -t {DICT_4X4_50,DICT_4X4_100,DICT_4X4_250,DICT_4X4_1000,DICT_5X5_50,DICT_5X5_100,DICT_5X5_250,DICT_5X5_1000,DICT_6X6_50,DICT_6X6_100,DICT_6X6_250,DICT_6X6_1000,DICT_7X7_50,DICT_7X7_100,DICT_7X7_250,DICT_7X7_1000}, --aruco_type {DICT_4X4_50,DICT_4X4_100,DICT_4X4_250,DICT_4X4_1000,DICT_5X5_50,DICT_5X5_100,DICT_5X5_250,DICT_5X5_1000,DICT_6X6_50,DICT_6X6_100,DICT_6X6_250,DICT_6X6_1000,DICT_7X7_50,DICT_7X7_100,DICT_7X7_250,DICT_7X7_1000}
                        The type of markers to detect. (default: DICT_6X6_250)
```
