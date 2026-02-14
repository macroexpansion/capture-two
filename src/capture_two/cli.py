"""Command line interface for Capture Two."""

import argparse
import sys

from capture_two.core import ImageProcessor


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Capture Two - Image editing CLI")
    parser.add_argument("--input", "-i", help="Input image file")
    parser.add_argument("--output", "-o", help="Output image file")
    parser.add_argument(
        "--exposure", "-e", type=float, default=0.0, help="Exposure adjustment in stops"
    )

    args = parser.parse_args()

    processor = ImageProcessor()

    if args.input:
        if args.input.lower().endswith((".raw", ".cr2", ".nef", ".arw")):
            processor.load_raw(args.input)
        print(f"Loaded: {args.input}")
    else:
        print("Hello from capture-two!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
