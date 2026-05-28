# merge-segmentation-layers

* accepts: idc.api.ImageSegmentationData
* generates: idc.api.ImageSegmentationData

Merges the specified source layer from an image segmentation item in internal storage with the target layer of the item passing through the filter.

```
usage: merge-segmentation-layers [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                                 [-N LOGGER_NAME] [--skip] -n SOURCE_NAME -s
                                 SOURCE_LAYER -t TARGET_LAYER
                                 [-o {add,subtract}]

Merges the specified source layer from an image segmentation item in internal
storage with the target layer of the item passing through the filter.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  --skip                Disables the plugin, removing it from the pipeline.
                        (default: False)
  -n SOURCE_NAME, --source_name SOURCE_NAME
                        The name of the item in internal storage. (default:
                        None)
  -s SOURCE_LAYER, --source_layer SOURCE_LAYER
                        The source layer/label to use from the item in
                        internal storage. (default: None)
  -t TARGET_LAYER, --target_layer TARGET_LAYER
                        The target layer/label to use from the item passing
                        through. (default: None)
  -o {add,subtract}, --merge_operation {add,subtract}
                        How to merge the two layers. (default: add)
```
