from enum import Enum


class Camera(Enum):
    """Supported camera models."""

    LEICA_M11 = "Leica M11"
    LUMIX_S9 = "Lumix S9"


class FilmSimulation(Enum):
    """Supported film simulations."""

    PORTRA_400 = "Porta 400"
