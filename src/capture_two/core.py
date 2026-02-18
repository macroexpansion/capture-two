"""Core image processing functionality."""

from typing import Optional

import colour
import cv2
import imageio
import numpy as np
import rawpy

from .enum import Camera
from .preset import FilmSimulation


class ImageProcessor:
    """Main image processor class for RAW and standard image editing."""

    def __init__(self, filepath: str, camera=Camera.LEICA_M11):
        self.camera: Camera = camera
        self.image: np.ndarray
        self.metadata = {}
        self.filepath = filepath

        self._load_dng(self.filepath)

    def apply(self, preset: FilmSimulation) -> np.ndarray:
        return preset(self.image)

    def _convert_to_linear_float(self, rgb16):
        return rgb16.astype(np.float32) / 65535.0

    def _load_dng(self, filepath: str) -> None:
        """Load a RAW image file."""
        with rawpy.imread(filepath) as raw:
            match self.camera:
                case Camera.LEICA_M11:
                    rgb16 = raw.postprocess(
                        use_camera_wb=True,
                        no_auto_bright=True,
                        bright=3.0,
                        output_bps=16,
                        output_color=rawpy.ColorSpace.sRGB,  # type: ignore
                    )
                    self.image = self._convert_to_linear_float(rgb16)
                case _:
                    raise ValueError(f"Camera not supported: {self.camera.value}")

            self.metadata = {
                "camera": raw.camera_whitebalance,
                "daylight": raw.daylight_whitebalance,
            }

    def adjust_exposure(self, img: np.ndarray, stops: float) -> np.ndarray:
        """Adjust image exposure by given stops."""
        factor = 2**stops
        adjusted = img.astype(np.float32) * factor
        # return np.clip(adjusted, 0, 255).astype(np.uint8)
        return adjusted

    def save_image(self, path: str):
        img = (self.image * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img)

    def display(self, img: Optional[np.ndarray] = None) -> None:
        """Display the linear float image using OpenCV."""
        if img is None:
            img = self.image

        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imshow("DNG Image", img)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        cv2.destroyAllWindows()

    def apply_lut(self, lut_path: str):
        # Load LUT
        lut = colour.io.read_LUT(lut_path)
        # Apply LUT
        rgb_lut = lut.apply(self.image)
        # Clip to safe range
        rgb_lut = np.clip(rgb_lut, 0, 1)

        self.image = rgb_lut

        # Convert back to 16-bit
        # rgb_16 = (rgb_lut * 65535).astype(np.uint16)
        # imageio.imwrite("output_porta400.tiff", rgb_16)

    def compare(self, img1: np.ndarray, img2: np.ndarray) -> None:
        """Display and compare two images side by side using OpenCV."""
        img1 = np.clip((img1 * 255).astype(np.uint8), 0, 255)
        img2 = np.clip((img2 * 255).astype(np.uint8), 0, 255)

        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

        comparison = np.hstack((img1, img2))

        cv2.imshow("Comparison (Left | Right)", comparison)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        cv2.destroyAllWindows()
