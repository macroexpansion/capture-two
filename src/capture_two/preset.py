# def portra_400(image: n):
#     img = image.copy()
#
#     # --- 1. Soft Contrast Curve ---
#     img = np.clip((img - 0.5) * 0.9 + 0.5, 0, 1)
#
#     # --- 2. Lift Shadows (matte feel) ---
#     img = img * 0.95 + 0.03
#
#     # --- 3. Warm Highlights ---
#     img[:, :, 0] *= 1.04  # Red slightly up
#     img[:, :, 1] *= 1.02  # Green slightly up
#     img[:, :, 2] *= 0.96  # Blue slightly down
#
#     # --- 4. Mute Blues ---
#     img[:, :, 2] *= 0.9
#
#     # --- 5. Soften Greens ---
#     img[:, :, 1] *= 0.97
#
#     return np.clip(img, 0, 1)

import rawpy
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


# ------------------------
# Filmic Tone Curve
# ------------------------
def filmic_curve(img: np.ndarray):
    # highlight compression
    img = img / (img + 0.6)

    # soft S curve
    img = np.clip((img - 0.5) * 0.85 + 0.5, 0, 1)

    return img


# ------------------------
# Portra 400 Color Science
# ------------------------
def portra_400(img: np.ndarray):
    # Slight global warmth
    img[:, :, 0] *= 1.05  # red
    img[:, :, 1] *= 1.02  # green
    img[:, :, 2] *= 0.92  # blue

    # Reduce blue saturation
    img[:, :, 2] *= 0.9

    # Soften greens (Portra pastel greens)
    img[:, :, 1] *= 0.97

    return np.clip(img, 0, 1)


# ------------------------
# Reduce Microcontrast
# ------------------------
def soften_microcontrast(img: np.ndarray):
    blurred = gaussian_filter(img, sigma=1)
    return img * 0.85 + blurred * 0.15


# ------------------------
# Fine Film Grain
# ------------------------
def add_grain(img, strength=0.008):
    noise = np.random.normal(0, strength, img.shape)
    return np.clip(img + noise, 0, 1)


# ------------------------
# Subtle Halation
# ------------------------
def halation(img):
    glow = gaussian_filter(img, sigma=6)
    return np.clip(img + glow * 0.03, 0, 1)


# ------------------------
# Save
# ------------------------
def save_image(img, path):
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


# ------------------------
# Main
# ------------------------
input_path = "input.DNG"
output_path = "M11_Portra400.jpg"

img = load_dng(input_path)
img = filmic_curve(img)
img = portra_color(img)
img = soften_microcontrast(img)
img = halation(img)
img = add_grain(img)

save_image(img, output_path)
