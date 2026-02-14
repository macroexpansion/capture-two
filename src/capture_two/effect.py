from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter


class Effect(ABC):
    """Base class for all effects."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, img: np.ndarray) -> np.ndarray:
        pass


class HighlightCompression(Effect):
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img / (img + self.threshold)


class SCurve(Effect):
    def __init__(self, a: float = 0.85, b: float = 0.5):
        self.a = a
        self.b = b

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return np.clip((img - self.b) * self.a + self.b, 0, 1)


class ColorBalance(Effect):
    def __init__(self, r: float = 1.0, g: float = 1.0, b: float = 1.0):
        self.r = r
        self.g = g
        self.b = b
        self.matrix = np.array([[self.r, 0, 0], [0, self.g, 0], [0, 0, self.b]])

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return np.clip(np.dot(img, self.matrix), 0, 1)


class MircroContrast(Effect):
    def __init__(self, sigma: float = 1.0, amount: float = 0.15):
        self.sigma = sigma
        self.amount = amount

    def __call__(self, img: np.ndarray) -> np.ndarray:
        blurred = gaussian_filter(img, sigma=self.sigma)
        return img * (1 - self.amount) + blurred * self.amount


class Grain(Effect):
    def __init__(self, strength: float = 0.008):
        self.strength = strength

    def __call__(self, img: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.strength, img.shape)
        return np.clip(img + noise, 0, 1)


class Halation(Effect):
    def __init__(
        self, sigma: float = 6, amount: float = 0.03, glow: Optional[np.ndarray] = None
    ):
        self.sigma = sigma
        self.amount = amount
        self.glow = None

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.glow is None:
            self.glow = gaussian_filter(img, sigma=self.sigma)
        return np.clip(img + self.glow * self.amount, 0, 1)


class Clip(Effect):
    def __init__(self):
        pass

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return np.clip(img, 0, 1)
