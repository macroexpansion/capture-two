from abc import ABC, abstractmethod

import cv2
import numpy as np
import rawpy
from scipy.ndimage import gaussian_filter

from .effect import (
    Clip,
    ColorBalance,
    Effect,
    Grain,
    Halation,
    HighlightCompression,
    MircroContrast,
    ContrastToneCurve,
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


class Portra400(FilmSimulation):
    def __init__(self):
        self.name = "Portra 400"
        self.description = "Kodak Portra 400"

    def __repr__(self):
        return f"FilmSimulation.{self.name}"

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        img = raw.copy()
        img = self._apply_effects(
            img,
            [
                ContrastToneCurve(shadow=-0.2, highlight=-0.2, p=2),
                # Red slightly up, green slightly up, blue slightly down
                ColorBalance(r=1.01, g=1.02, b=0.97, clip=False),
                # # Soften greens (Portra pastel greens)
                ColorBalance(g=0.97, clip=False),
                # # Reduce blue saturation (Mute Blues)
                ColorBalance(b=0.97, clip=False),
                Clip(),
                MircroContrast(sigma=1.0, amount=0.20),
                Halation(sigma=6, amount=0.03),
                Grain(strength=0.005),
                Clip(),
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

    def _add_grain(self, img, strength=0.008):
        """Add fine film grain."""
        noise = np.random.normal(0, strength, img.shape)
        return np.clip(img + noise, 0, 1)
