"""Core image processing functionality."""

import numpy as np
import rawpy

from .enum import Camera, FilmSimulation


class ImageProcessor:
    """Main image processor class for RAW and standard image editing."""

    def __init__(self, camera=Camera.LEICA_M11):
        self.camera: Camera = camera
        self.image = None
        self.metadata = {}

    def apply(self, preset=FilmSimulation.PORTRA_400):
        pass

    def _convert_to_linear_float(rgb16):
        return rgb16.astype(np.float32) / 65535.0

    def load_dng(self, filepath: str) -> None:
        """Load a RAW image file."""
        with rawpy.imread(filepath) as raw:
            match self.camera:
                case Camera.LEICA_M11:
                    rgb16 = raw.postprocess(
                        use_camera_wb=True,
                        no_auto_bright=True,
                        output_bps=16,
                        highlight_mode=rawpy.HighlightMode.Blend,
                    )
                    self.image = _convert_to_linear_float(rgb16)
                case _:
                    raise ValueError(f"Camera not supported: {self.camera.value}")

            self.metadata = {
                "camera": raw.camera_whitebalance,
                "daylight": raw.daylight_whitebalance,
            }

    def adjust_exposure(self, stops: float) -> np.ndarray:
        """Adjust image exposure by given stops."""
        if self.image is None:
            raise ValueError("No image loaded")

        factor = 2**stops
        adjusted = self.image.astype(np.float32) * factor
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def save_image(self, path: str):
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img)
