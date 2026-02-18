"""Main entry point for running capture_two module."""

import os

os.environ["QT_QPA_PLATFORM"] = "xcb"


from capture_two import ImageProcessor, Portra400

if __name__ == "__main__":
    processor = ImageProcessor(filepath="/mnt/second/leica/L1007949.DNG")

    # original = processor.adjust_exposure(processor.image, 0.5)
    # edited = processor.adjust_exposure(processor.apply(Portra400()), 0.5)

    original = processor.image
    edited = processor.apply(Portra400())

    processor.compare(original, edited)
