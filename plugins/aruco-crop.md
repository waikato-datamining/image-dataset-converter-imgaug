# aruco-crop

* accepts: idc.api.ImageData
* generates: idc.api.ImageData

Crops the image according to the ArUco markers.

```
usage: aruco-crop [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                  [-N LOGGER_NAME] [--skip] [-I {skip,fail}]
                  [-t {DICT_4X4_50,DICT_4X4_100,DICT_4X4_250,DICT_4X4_1000,DICT_5X5_50,DICT_5X5_100,DICT_5X5_250,DICT_5X5_1000,DICT_6X6_50,DICT_6X6_100,DICT_6X6_250,DICT_6X6_1000,DICT_7X7_50,DICT_7X7_100,DICT_7X7_250,DICT_7X7_1000}]
                  [-m MIN_NUM_MARKERS] [-c {outside,inside}]
                  [--crop_success_key CROP_SUCCESS_KEY]

Crops the image according to the ArUco markers.

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
  -t {DICT_4X4_50,DICT_4X4_100,DICT_4X4_250,DICT_4X4_1000,DICT_5X5_50,DICT_5X5_100,DICT_5X5_250,DICT_5X5_1000,DICT_6X6_50,DICT_6X6_100,DICT_6X6_250,DICT_6X6_1000,DICT_7X7_50,DICT_7X7_100,DICT_7X7_250,DICT_7X7_1000}, --aruco_type {DICT_4X4_50,DICT_4X4_100,DICT_4X4_250,DICT_4X4_1000,DICT_5X5_50,DICT_5X5_100,DICT_5X5_250,DICT_5X5_1000,DICT_6X6_50,DICT_6X6_100,DICT_6X6_250,DICT_6X6_1000,DICT_7X7_50,DICT_7X7_100,DICT_7X7_250,DICT_7X7_1000}
                        The type of markers to detect. (default: DICT_6X6_250)
  -m MIN_NUM_MARKERS, --min_num_markers MIN_NUM_MARKERS
                        The minimum number of markers that need to be detected
                        in order to proceed with cropping. (default: 3)
  -c {outside,inside}, --crop_type {outside,inside}
                        How to crop in relation to the markers. (default:
                        outside)
  --crop_success_key CROP_SUCCESS_KEY
                        The meta-data key to store the crop success under
                        (true/false). (default: None)
```
