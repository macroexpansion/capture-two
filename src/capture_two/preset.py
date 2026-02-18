from abc import ABC, abstractmethod

import cv2
import numpy as np
import rawpy
from scipy.ndimage import gaussian_filter

from .effect import (
    Clip,
    ColorBalance,
    ContrastToneCurve,
    Effect,
    Exposure,
    Grain,
    HableFilmicToneCurve,
    Halation,
    HighlightCompression,
    LinearToSRGB,
    MircroContrast,
)


class FilmSimulation(ABC):
    """Supported film simulations."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __call__(self, raw: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def _apply_effects(img: np.ndarray, effects: list[Effect]) -> np.ndarray:
        for effect in effects:
            img = effect(img)
        return img


class Original(FilmSimulation):
    def __init__(self, exposure: float = 0.0):
        self.name = "Original"
        self.description = "No processing"
        self.effects = [
            Exposure(stops=exposure),
            Clip(),
            LinearToSRGB(),
        ]

    def __repr__(self):
        return f"FilmSimulation.{self.name}"

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        return self._apply_effects(raw, self.effects)


class Portra400(FilmSimulation):
    def __init__(self, exposure: float = 0.0):
        self.name = "Portra 400"
        self.description = "Kodak Portra 400"
        self.exposure = exposure

    def __repr__(self):
        return f"FilmSimulation.{self.name}"

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        img = self._apply_effects(
            raw,
            [
                HableFilmicToneCurve(exposure=-0.5),
                # ContrastToneCurve(shadow=-0.2, highlight=-0.2, p=2),
                ColorBalance(
                    r=1.01, g=1.02, b=0.97, clip=False
                ),  # Red slightly up, green slightly up, blue slightly down
                ColorBalance(
                    g=0.97, clip=False
                ),  # Soften greens (Portra pastel greens)
                ColorBalance(b=0.97, clip=False),  # Reduce blue saturation (Mute Blues)
                Clip(),
                MircroContrast(sigma=1.0, amount=0.20),
                Halation(sigma=6, amount=0.03),
                # Grain(strength=0.003),
                Exposure(stops=self.exposure),
                Clip(),
                LinearToSRGB(),
            ],
        )
        return img

    def _filmic_curve(self, img: np.ndarray):
        """Apply a filmic tone curve."""
        # soft S curve
        img = np.clip((img - 0.5) * 0.85 + 0.5, 0, 1)

        # Lift Shadows (matte feel)
        # img = img * 0.95 + 0.03

        return img
