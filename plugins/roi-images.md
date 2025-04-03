# roi-images

* accepts: idc.api.ObjectDetectionData
* generates: idc.api.ObjectDetectionData

Extracts sub-images using the bbox of all the object detection annotations that have matching labels. If no labels are specified, all annotations are extracted.

```
usage: roi-images [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                  [-N LOGGER_NAME] [--labels [LABELS ...]] [-S SUFFIX]

Extracts sub-images using the bbox of all the object detection annotations
that have matching labels. If no labels are specified, all annotations are
extracted.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  --labels [LABELS ...]
                        The label(s) of the annotations to forward as sub-
                        images, uses all annotations if not specified.
                        (default: None)
  -S SUFFIX, --suffix SUFFIX
                        The suffix pattern to use for the generated sub-
                        images, available placeholders:
                        {X}|{Y}|{W}|{H}|{X0}|{Y0}|{X1}|{Y1}|{INDEX} (default:
                        -{INDEX})
```
