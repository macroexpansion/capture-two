"""Main entry point for running capture_two module."""

import os

os.environ["QT_QPA_PLATFORM"] = "xcb"


from capture_two import ImageProcessor, Original, Portra400

if __name__ == "__main__":
    processor = ImageProcessor(filepath="/mnt/second/leica/L1007949.DNG")

    original = processor.apply(Original(exposure=0.6))
    edited = processor.apply(Portra400(exposure=0.6))

    processor.compare(original, edited)
