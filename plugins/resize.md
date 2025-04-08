# resize

* accepts: idc.api.ImageClassificationData, idc.api.ObjectDetectionData, idc.api.ImageSegmentationData
* generates: idc.api.ImageClassificationData, idc.api.ObjectDetectionData, idc.api.ImageSegmentationData

Resizes all images according to the specified width/height. When only resizing one dimension, use 'keep-aspect-ratio' for the other one to keep the aspect ratio intact.

```
usage: resize [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [-N LOGGER_NAME]
              [--skip] [-W WIDTH] [-H HEIGHT]

Resizes all images according to the specified width/height. When only resizing
one dimension, use 'keep-aspect-ratio' for the other one to keep the aspect
ratio intact.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  --skip                Disables the plugin, removing it from the pipeline.
                        (default: False)
  -W WIDTH, --width WIDTH
                        The new width for the image; use 'keep-aspect-ratio'
                        when only supplying height and you want to keep the
                        aspect ratio intact. (default: keep-aspect-ratio)
  -H HEIGHT, --height HEIGHT
                        The new height for the image; use 'keep-aspect-ratio'
                        when only supplying width and you want to keep the
                        aspect ratio intact. (default: keep-aspect-ratio)
```
