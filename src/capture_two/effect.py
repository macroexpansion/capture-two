from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter


class ImageFloat:
    """
    Represents an image in linear float format.
    Values are in the range [0, 1].
    """

    def __init__(self, img: np.ndarray):
        self.img = img


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
    def __init__(
        self, r: float = 1.0, g: float = 1.0, b: float = 1.0, clip: bool = True
    ):
        self.r = r
        self.g = g
        self.b = b
        self.matrix = np.array([[self.r, 0, 0], [0, self.g, 0], [0, 0, self.b]])
        self.clip = clip

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.clip:
            return np.clip(np.dot(img, self.matrix), 0, 1)
        return np.dot(img, self.matrix)


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


class ContrastToneCurve(Effect):
    def __init__(
        self, shadow: float = 0.0, highlight: float = 0.0, p: int = 2, clip: bool = True
    ):
        """
        shadow:  [-1, 1] positive lifts shadows
        highlight: [-1, 1] positive lifts highlights
        p: localization strength (2–4 recommended), lower = more localized
        """
        self.shadow = shadow
        self.highlight = highlight
        self.p = p
        self.clip = clip

    def __call__(self, img: np.ndarray) -> np.ndarray:
        # Compute masks
        w_shadow = (1 - img) ** self.p
        w_high = img**self.p

        # Nonlinear adjustments
        shadow_adjust = np.power(img, 1 - self.shadow)
        highlight_adjust = np.power(img, 1 - self.highlight)

        # Blend
        result = (
            img + w_shadow * (shadow_adjust - img) + w_high * (highlight_adjust - img)
        )

        if self.clip:
            return np.clip(result, 0, 1)

        return result


class HableFilmicToneCurve(Effect):
    def __init__(self, exposure=0.0, clip: bool = True):
        """
        exposure: stops adjustment (-2 to +2 typical)
        """
        self.exposure = exposure
        self.clip = clip

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img_f = img

        # Apply exposure
        img_f *= 2**self.exposure

        # Apply curve
        result = self._filmic_curve(img_f)

        # Normalize white point
        white_scale = 1.0 / self._filmic_curve(1.0)
        result *= white_scale

        if self.clip:
            return np.clip(result, 0, 1)

        return result

    def _filmic_curve(self, x):
        # Hable filmic constants (cinematic look)
        A = 0.22  # stronger highlight compression
        B = 0.30
        C = 0.10
        D = 0.20  # softer shadows
        E = 0.01  # lifted blacks
        F = 0.30

        return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F


class Exposure(Effect):
    def __init__(self, stops: float = 0.0):
        self.stops = stops

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img * 2**self.stops


class LinearToSRGB(Effect):
    def __init__(self):
        pass

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self._linear_to_srgb(img)

    def _linear_to_srgb(self, x: np.ndarray) -> np.ndarray:
        return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1 / 2.4)) - 0.055)
