# roi-images

* accepts: idc.api.ObjectDetectionData
* generates: idc.api.ObjectDetectionData

Extracts sub-images using the bbox of all the object detection annotations that have matching labels. If no labels are specified, all annotations are extracted. For polygons, the background outside the polygon can be blanked.

```
usage: roi-images [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                  [-N LOGGER_NAME] [--skip] [--labels [LABELS ...]]
                  [-S SUFFIX] [-f] [-1] [-b]

Extracts sub-images using the bbox of all the object detection annotations
that have matching labels. If no labels are specified, all annotations are
extracted. For polygons, the background outside the polygon can be blanked.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  --skip                Disables the plugin, removing it from the pipeline.
                        (default: False)
  --labels [LABELS ...]
                        The label(s) of the annotations to forward as sub-
                        images, uses all annotations if not specified.
                        (default: None)
  -S SUFFIX, --suffix SUFFIX
                        The suffix pattern to use for the generated sub-
                        images, available placeholders:
                        {X}|{Y}|{W}|{H}|{X0}|{Y0}|{X1}|{Y1}|{INDEX} (default:
                        -{INDEX})
  -f, --skip_failed     Removes images with no annotations or when failing to
                        extract objects. (default: False)
  -1, --separate_objects
                        Generates output with only a single annotation in each
                        item. (default: False)
  -b, --blank_background
                        In case of polygon annotations, this flag enables
                        removing the background outside of the polygon.
                        Requires --separate_objects (default: False)
```
