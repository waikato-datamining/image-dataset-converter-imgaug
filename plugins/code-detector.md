# code-detector

* accepts: idc.api.ImageData
* generates: idc.api.ImageData

Detects 1-dimensional barcodes and QR codes. Adds them to the meta-data.

```
usage: code-detector [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                     [-N LOGGER_NAME] [--skip] [-p PREFIX]
                     [-t {ean2,ean5,ean8,ean13,upca,upce,isbn10,composite,i25,databar,databar-exp,codabar,code39,code93,code128,pdf417,qrcode,sqcode}]

Detects 1-dimensional barcodes and QR codes. Adds them to the meta-data.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  --skip                Disables the plugin, removing it from the pipeline.
                        (default: False)
  -p PREFIX, --prefix PREFIX
                        The prefix to use for the detected markers in the
                        meta-data. (default: code-)
  -t {ean2,ean5,ean8,ean13,upca,upce,isbn10,composite,i25,databar,databar-exp,codabar,code39,code93,code128,pdf417,qrcode,sqcode}, --code_type {ean2,ean5,ean8,ean13,upca,upce,isbn10,composite,i25,databar,databar-exp,codabar,code39,code93,code128,pdf417,qrcode,sqcode}
                        The specific type of code to detect, auto-detect any
                        supported code if not specified. (default: None)
```
