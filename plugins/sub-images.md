# sub-images

* accepts: idc.api.ImageClassificationData, idc.api.ObjectDetectionData
* generates: idc.api.ImageClassificationData, idc.api.ObjectDetectionData

Extracts sub-images (incl their annotations) from the images coming through, using the defined regions.

```
usage: sub-images [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                  [-N LOGGER_NAME] -r REGIONS [REGIONS ...]
                  [-s {none,x-then-y,y-then-x}] [-p] [-e]

Extracts sub-images (incl their annotations) from the images coming through,
using the defined regions.

optional arguments:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  -r REGIONS [REGIONS ...], --regions REGIONS [REGIONS ...]
                        The regions (X,Y,WIDTH,HEIGHT) to crop and forward
                        with their annotations (0-based coordinates) (default:
                        None)
  -s {none,x-then-y,y-then-x}, --region_sorting {none,x-then-y,y-then-x}
                        How to sort the supplied region definitions (default:
                        none)
  -p, --include_partial
                        Whether to include only annotations that fit fully
                        into a region or also partial ones (default: False)
  -e, --suppress_empty  Suppresses sub-images that have no annotations (object
                        detection) (default: False)
```
