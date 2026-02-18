"""Tests for core image processing functionality."""

import numpy as np
import pytest

from capture_two.core import ImageProcessor


class TestImageProcessor:
    """Test cases for ImageProcessor class."""

    def test_init(self):
        """Test processor initialization."""
        processor = ImageProcessor()
        assert processor.linear_rgb is None
        assert processor.metadata == {}

    def test_adjust_exposure_no_image(self):
        """Test exposure adjustment without loaded image."""
        processor = ImageProcessor()
        with pytest.raises(ValueError, match="No image loaded"):
            processor.adjust_exposure(1.0)
