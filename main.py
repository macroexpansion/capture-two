"""Main entry point for running capture_two module."""

import os

os.environ["QT_QPA_PLATFORM"] = "xcb"


from capture_two import ImageProcessor, Portra400

if __name__ == "__main__":
    processor = ImageProcessor(filepath="/mnt/second/100LEICA/L1000401.DNG")
    print(processor.camera)
    processor.compare(processor.apply(Portra400()))
