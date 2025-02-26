# linear-contrast

* accepts: idc.api.ImageClassificationData, idc.api.ObjectDetectionData, idc.api.ImageSegmentationData
* generates: idc.api.ImageClassificationData, idc.api.ObjectDetectionData, idc.api.ImageSegmentationData

Applies linear contrast to images.

```
usage: linear-contrast [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                       [-N LOGGER_NAME] [-m {replace,add}] [--suffix SUFFIX]
                       [-s SEED] [-a] [-T THRESHOLD] [-f FROM_ALPHA]
                       [-t TO_ALPHA]

Applies linear contrast to images.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  -m {replace,add}, --mode {replace,add}
                        The image augmentation mode to use. (default: replace)
  --suffix SUFFIX       The suffix to use for the file names in case of
                        augmentation mode add. (default: None)
  -s SEED, --seed SEED  The seed value to use for the random number generator;
                        randomly seeded if not provided (default: None)
  -a, --seed_augmentation
                        Whether to seed the augmentation; if specified, uses
                        the seeded random generator to produce a seed value
                        from 0 to 1000 for the augmentation. (default: False)
  -T THRESHOLD, --threshold THRESHOLD
                        the threshold to use for Random.rand(): if equal or
                        above, augmentation gets applied; range: 0-1; default:
                        0 (= always) (default: 0.0)
  -f FROM_ALPHA, --from_alpha FROM_ALPHA
                        The minimum alpha to apply. (default: None)
  -t TO_ALPHA, --to_alpha TO_ALPHA
                        The minimum alpha to apply. (default: None)
```
