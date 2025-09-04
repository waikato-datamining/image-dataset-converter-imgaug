# trace-skeleton

* accepts: idc.api.ImageData
* generates: idc.api.ObjectDetectionData

Thinning and tracing algorithm developed by Lingdong Huang: https://github.com/LingDong-/skeleton-tracing/blob/master/py/trace_skeleton.py

```
usage: trace-skeleton [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                      [-N LOGGER_NAME] [--skip] [-I {skip,fail}]
                      [-a {both,image,annotations}] [-c SIZE] [-m ITER]
                      [-o {as-is,binary,grayscale,rgb}] [--label LABEL]

Thinning and tracing algorithm developed by Lingdong Huang:
https://github.com/LingDong-/skeleton-tracing/blob/master/py/trace_skeleton.py

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
  -a {both,image,annotations}, --apply_to {both,image,annotations}
                        Where to apply the filter to. (default: image)
  -c SIZE, --chunk_size SIZE
                        The chunk size to use. (default: 10)
  -m ITER, --max_iter ITER
                        The maximum number of iterations to perform. (default:
                        999)
  -o {as-is,binary,grayscale,rgb}, --output_format {as-is,binary,grayscale,rgb}
                        The image format to generate as output. (default: as-
                        is)
  --label LABEL         The label to use when processing images other than
                        image segmentation ones. (default: object)
```
