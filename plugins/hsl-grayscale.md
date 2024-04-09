# hsl-grayscale

* accepts: idc.api.ImageClassificationData, idc.api.ObjectDetectionData, idc.api.ImageSegmentationData
* generates: idc.api.ImageClassificationData, idc.api.ObjectDetectionData, idc.api.ImageSegmentationData

Turns RGB images into fake grayscale ones by converting them to HSL and then using the L channel for all channels. The brightness can be influenced and varied even.

```
usage: hsl-grayscale [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                     [-N LOGGER_NAME] [-m {replace,add}] [--suffix SUFFIX]
                     [-s SEED] [-a] [-T THRESHOLD] [-f FROM_FACTOR]
                     [-t TO_FACTOR]

Turns RGB images into fake grayscale ones by converting them to HSL and then
using the L channel for all channels. The brightness can be influenced and
varied even.

optional arguments:
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
  -f FROM_FACTOR, --from_factor FROM_FACTOR
                        The start of the factor range to apply to the L
                        channel to darken or lighten the image (<1: darker,
                        >1: lighter). (default: None)
  -t TO_FACTOR, --to_factor TO_FACTOR
                        The end of the factor range to apply to the L channel
                        to darken or lighten the image (<1: darker, >1:
                        lighter). (default: None)
```
