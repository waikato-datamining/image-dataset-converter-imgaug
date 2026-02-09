# overlay-regions

* accepts: idc.api.ImageData
* generates: idc.api.ImageData

Overlays the regions on the images coming through, using either the explicitly defined regions or ones derived from #rows/cols.

```
usage: overlay-regions [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                       [-N LOGGER_NAME] [--skip] [-r [REGIONS ...]]
                       [-s {none,x-then-y,y-then-x}] [--num_rows NUM_ROWS]
                       [--num_cols NUM_COLS] [--row_height ROW_HEIGHT]
                       [--col_width COL_WIDTH] [--overlap_right OVERLAP_RIGHT]
                       [--overlap_bottom OVERLAP_BOTTOM] [-c [R,G,B ...]]
                       [--outline_thickness INT] [--outline_alpha INT]
                       [--fill] [--fill_alpha INT] [--vary_colors]

Overlays the regions on the images coming through, using either the explicitly
defined regions or ones derived from #rows/cols.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  --skip                Disables the plugin, removing it from the pipeline.
                        (default: False)
  -r [REGIONS ...], --regions [REGIONS ...]
                        The regions (X,Y,WIDTH,HEIGHT) to crop and forward
                        with their annotations (0-based coordinates) (default:
                        None)
  -s {none,x-then-y,y-then-x}, --region_sorting {none,x-then-y,y-then-x}
                        How to sort the supplied region definitions (default:
                        none)
  --num_rows NUM_ROWS   The number of rows, if no regions defined. (default:
                        None)
  --num_cols NUM_COLS   The number of columns, if no regions defined.
                        (default: None)
  --row_height ROW_HEIGHT
                        The height of rows. (default: None)
  --col_width COL_WIDTH
                        The width of columns. (default: None)
  --overlap_right OVERLAP_RIGHT
                        The overlap between two images (on the right of the
                        left-most image), if no regions defined. (default: 0)
  --overlap_bottom OVERLAP_BOTTOM
                        The overlap between two images (on the bottom of the
                        top-most image), if no regions defined. (default: 0)
  -c [R,G,B ...], --colors [R,G,B ...]
                        The color list name (available: colorblind12,colorblin
                        d15,colorblind24,colorblind8,dark,light,x11) or list
                        of RGB triplets (R,G,B) of custom colors to use, uses
                        default colors if not supplied (X11 colors, without
                        dark/light colors) (default: None)
  --outline_thickness INT
                        The line thickness to use for the outline, <1 to turn
                        off. (default: 3)
  --outline_alpha INT   The alpha value to use for the outline (0:
                        transparent, 255: opaque). (default: 255)
  --fill                Whether to fill the bounding boxes/polygons. (default:
                        False)
  --fill_alpha INT      The alpha value to use for the filling (0:
                        transparent, 255: opaque). (default: 128)
  --vary_colors         Whether to vary the colors of the outline/filling
                        regardless of label. (default: False)
```
