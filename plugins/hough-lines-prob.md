# hough-lines-prob

* accepts: idc.api.ImageData
* generates: idc.api.ObjectDetectionData

Finds line segments in a binary image using the probabilistic Hough transform. 

```
usage: hough-lines-prob [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                        [-N LOGGER_NAME] [--skip] [--label LABEL] [--rho RHO]
                        [--theta THETA] [--threshold THRESHOLD]
                        [--min_line_length MIN_LINE_LENGTH]
                        [--max_line_gap MAX_LINE_GAP]

Finds line segments in a binary image using the probabilistic Hough transform.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  --skip                Disables the plugin, removing it from the pipeline.
                        (default: False)
  --label LABEL         The label to use when processing images other than
                        image segmentation ones. (default: object)
  --rho RHO             Distance resolution of the accumulator in pixels.
                        (default: 1.0)
  --theta THETA         Angle resolution of the accumulator in radians.
                        (default: 0.017453292519943295)
  --threshold THRESHOLD
                        Accumulator threshold parameter. Only those lines are
                        returned that get enough votes (>threshold). (default:
                        50)
  --min_line_length MIN_LINE_LENGTH
                        Minimum line length. Line segments shorter than that
                        are rejected. (default: 0)
  --max_line_gap MAX_LINE_GAP
                        Maximum allowed gap between points on the same line to
                        link them. (default: 0)
```
