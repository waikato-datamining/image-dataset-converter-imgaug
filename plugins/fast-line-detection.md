# fast-line-detection

* accepts: idc.api.ImageData
* generates: idc.api.ObjectDetectionData

Detects lines in the image and stores them as polygons.

```
usage: fast-line-detection [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                           [-N LOGGER_NAME] [--skip] [--label LABEL]
                           [--length_threshold LENGTH_THRESHOLD]
                           [--distance_threshold DISTANCE_THRESHOLD]
                           [--canny_th1 CANNY_TH1] [--canny_th2 CANNY_TH2]
                           [--canny_aperture_size CANNY_APERTURE_SIZE]
                           [--do_merge]

Detects lines in the image and stores them as polygons.

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
  --length_threshold LENGTH_THRESHOLD
                        Segment shorter than this will be discarded. (default:
                        10)
  --distance_threshold DISTANCE_THRESHOLD
                        A point placed from a hypothesis line segment farther
                        than this will be regarded as an outlier. (default:
                        1.414213562)
  --canny_th1 CANNY_TH1
                        First threshold for hysteresis procedure in Canny().
                        (default: 50.0)
  --canny_th2 CANNY_TH2
                        Second threshold for hysteresis procedure in Canny().
                        (default: 50.0)
  --canny_aperture_size CANNY_APERTURE_SIZE
                        Aperture size for the sobel operator in Canny(). If
                        zero, Canny() is not applied and the input image is
                        taken as an edge image. (default: 3)
  --do_merge            If true, incremental merging of segments will be
                        performed. (default: False)
```
