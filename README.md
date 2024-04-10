# image-dataset-converter-imgaug
Image augmentation extension for the image-dataset-converter library.


## Installation

Via PyPI:

```bash
pip install image-dataset-converter-imgaug
```

The latest code straight from the repository:

```bash
pip install git+https://github.com/waikato-datamining/image-dataset-converter-imgaug.git
```

## Tools

### Generate regions

```
usage: idc-generate-regions [-h] -W WIDTH -H HEIGHT [-r NUM_ROWS]
                            [-c NUM_COLS] [-R ROW_HEIGHT] [-C COL_WIDTH] [-f]
                            [-p] [-1] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

Tool turns an image size into regions to be used, e.g., with the 'sub-images'
filter. Either specify the number of rows/cols or the height/width of
rows/cols.

optional arguments:
  -h, --help            show this help message and exit
  -W WIDTH, --width WIDTH
                        The width of the image. (default: None)
  -H HEIGHT, --height HEIGHT
                        The height of the image. (default: None)
  -r NUM_ROWS, --num_rows NUM_ROWS
                        The number of rows. (default: None)
  -c NUM_COLS, --num_cols NUM_COLS
                        The number of columns. (default: None)
  -R ROW_HEIGHT, --row_height ROW_HEIGHT
                        The height of rows. (default: None)
  -C COL_WIDTH, --col_width COL_WIDTH
                        The width of columns. (default: None)
  -f, --fixed_size      Whether to use fixed row height/col width, omitting
                        any left-over bits at right/bottom, when using
                        num_rows/num_cols (default: False)
  -p, --partial         Whether to output partial regions, the left-over bits
                        at right/bottom, when using row_height/col_width
                        (default: False)
  -1, --one_based       Whether to use 1-based coordinates (default: False)
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
```


## Plugins

See [here](plugins/README.md) for an overview of all plugins.

