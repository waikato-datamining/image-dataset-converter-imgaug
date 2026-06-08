import argparse
from typing import List

from pyzbar.pyzbar import ZBarSymbol, decode
from wai.common.adams.imaging.locateobjects import LocatedObject, LocatedObjects
from wai.logging import LOGGING_WARNING

from idc.api import ImageData, ObjectDetectionData, LABEL_KEY
from kasperl.api import make_list, flatten_list
from seppl.io import BatchFilter

DEFAULT_PREFIX = "qrcode-"

CODE_EAN2 = "ean2"
CODE_EAN5 = "ean5"
CODE_EAN8 = "ean8"
CODE_EAN13 = "ean13"
CODE_UPCA = "upca"
CODE_UPCE = "upce"
CODE_ISBN10 = "isbn10"
CODE_COMPOSITE = "composite"
CODE_I25 = "i25"
CODE_DATABAR = "databar"
CODE_DATABAR_EXP = "databar-exp"
CODE_CODABAR = "codabar"
CODE_CODE39 = "code39"
CODE_CODE93 = "code93"
CODE_CODE128 = "code128"
CODE_PDF417 = "pdf417"
CODE_QRCODE = "qrcode"
CODE_SQCODE = "sqcode"

CODE_TYPES = [
    CODE_EAN2,
    CODE_EAN5,
    CODE_EAN8,
    CODE_EAN13,
    CODE_UPCA,
    CODE_UPCE,
    CODE_ISBN10,
    CODE_COMPOSITE,
    CODE_I25,
    CODE_DATABAR,
    CODE_DATABAR_EXP,
    CODE_CODABAR,
    CODE_CODE39,
    CODE_CODE93,
    CODE_CODE128,
    CODE_PDF417,
    CODE_QRCODE,
    CODE_SQCODE,
]

CODE_SYMBOLS = {
    CODE_EAN2: ZBarSymbol.EAN2,
    CODE_EAN5: ZBarSymbol.EAN5,
    CODE_EAN8: ZBarSymbol.EAN8,
    CODE_EAN13: ZBarSymbol.EAN13,
    CODE_UPCA: ZBarSymbol.UPCA,
    CODE_UPCE: ZBarSymbol.UPCE,
    CODE_ISBN10: ZBarSymbol.ISBN10,
    CODE_COMPOSITE: ZBarSymbol.COMPOSITE,
    CODE_I25: ZBarSymbol.I25,
    CODE_DATABAR: ZBarSymbol.DATABAR,
    CODE_DATABAR_EXP: ZBarSymbol.DATABAR_EXP,
    CODE_CODABAR: ZBarSymbol.CODABAR,
    CODE_CODE39: ZBarSymbol.CODE39,
    CODE_CODE93: ZBarSymbol.CODE93,
    CODE_CODE128: ZBarSymbol.CODE128,
    CODE_PDF417: ZBarSymbol.PDF417,
    CODE_QRCODE: ZBarSymbol.QRCODE,
    CODE_SQCODE: ZBarSymbol.SQCODE,
}


class CodeDetector(BatchFilter):
    """
    Detects QR codes and adds the content to the meta-data.
    """

    def __init__(self, incorrect_format_action: str = None,
                 prefix: str = None, code_type: str = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param incorrect_format_action: how to react to incorrect input format
        :type incorrect_format_action: str
        :param prefix: the prefix to use in the meta-data
        :type prefix: str
        :param code_type: the type of code to identify, None for auto
        :type code_type: str
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.prefix = prefix
        self.code_type = code_type

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "code-detector"

    def description(self) -> str:
        """
        Returns a description of the handler.

        :return: the description
        :rtype: str
        """
        return "Detects 1-dimensional barcodes and QR codes. Adds them to the meta-data."

    def accepts(self) -> List:
        """
        Returns the list of classes that are accepted.

        :return: the list of classes
        :rtype: list
        """
        return [ImageData]

    def generates(self) -> List:
        """
        Returns the list of classes that get produced.

        :return: the list of classes
        :rtype: list
        """
        return [ImageData]

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        parser.add_argument("-p", "--prefix", type=str, default=DEFAULT_PREFIX, help="The prefix to use for the detected markers in the meta-data.", required=False)
        parser.add_argument("-t", "--code_type", choices=CODE_TYPES, default=None, help="The specific type of code to detect, auto-detect any supported code if not specified.", required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.prefix = ns.prefix
        self.code_type = ns.code_type

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if self.prefix is None:
            self.prefix = DEFAULT_PREFIX
        if (self.code_type is not None) and (len(self.code_type) == 0):
            self.code_type = None
        if (self.code_type is not None) and (self.code_type not in CODE_TYPES):
            raise Exception("Unsupported code type: %s" % self.code_type)

    def _do_process(self, data):
        """
        Processes the data record(s).

        :param data: the record(s) to process
        :return: the potentially updated record(s)
        """
        result = []

        for item in make_list(data):
            symbols = None
            if self.code_type is not None:
                symbols = [CODE_SYMBOLS[self.code_type]]
            codes = decode(item.image, symbols=symbols)
            self.logger().info("# codes detected: %d" % len(codes))

            # store results
            meta = dict()
            meta[self.prefix + "code_count"] = len(codes)
            objs = []
            if len(codes) > 0:
                for i, code in enumerate(codes):
                    self.logger().info("code: %s" % str(code.data))
                    meta[self.prefix + str(i) + "-type"] = str(code.type)
                    meta[self.prefix + str(i) + "-data"] = code.data.decode("utf-8")
                    meta[self.prefix + str(i) + "-orientation"] = str(code.orientation)
                    meta[self.prefix + str(i) + "-quality"] = str(code.quality)
                    meta[self.prefix + str(i) + "-topleft.x"] = int(code.polygon[0][0])
                    meta[self.prefix + str(i) + "-topleft.y"] = int(code.polygon[0][1])
                    meta[self.prefix + str(i) + "-topright.x"] = int(code.polygon[1][0])
                    meta[self.prefix + str(i) + "-topright.y"] = int(code.polygon[1][1])
                    meta[self.prefix + str(i) + "-bottomright.x"] = int(code.polygon[2][0])
                    meta[self.prefix + str(i) + "-bottomright.y"] = int(code.polygon[2][1])
                    meta[self.prefix + str(i) + "-bottomleft.x"] = int(code.polygon[3][0])
                    meta[self.prefix + str(i) + "-bottomleft.y"] = int(code.polygon[3][1])

                    if isinstance(item, ObjectDetectionData):
                        xs = [int(code.polygon[0][0]), int(code.polygon[1][0]), int(code.polygon[2][0]), int(code.polygon[3][0])]
                        ys = [int(code.polygon[0][1]), int(code.polygon[1][1]), int(code.polygon[2][1]), int(code.polygon[3][1])]
                        left = min(xs)
                        right = max(xs)
                        top = min(ys)
                        bottom = max(ys)
                        obj = LocatedObject(left, top, right - left + 1, bottom - top + 1)
                        obj.metadata[LABEL_KEY] = str(code.type)
                        obj.metadata["data"] = code.data.decode("utf-8")
                        obj.metadata["quality"] = str(code.quality)
                        obj.metadata["orientation"] = str(code.orientation)
                        objs.append(obj)

            item_new = item.duplicate()
            if not item_new.has_metadata():
                item_new.set_metadata(meta)
            else:
                item_new.get_metadata().extend(meta)
            if len(objs) > 0:
                if not item_new.has_annotation():
                    item_new.annotation = LocatedObjects()
                item_new.annotation.extend(objs)

            result.append(item_new)

        return flatten_list(result)
