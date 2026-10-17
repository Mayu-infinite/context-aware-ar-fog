from __future__ import annotations

import math

from cafog.perception.detector import Detection


def horizontal_angle_from_center(
    detection: Detection,
    image_width: int,
    horizontal_fov_deg: float,
) -> float:
    """
    Estimate horizontal angle of an object relative to the camera center.

    Positive = right
    Negative = left
    """

    image_center_x = image_width / 2.0
    offset_pixels = detection.center_x - image_center_x

    degrees_per_pixel = horizontal_fov_deg / image_width

    return offset_pixels * degrees_per_pixel


def approximate_distance_from_object_width(
    pixel_width: float,
    real_width_m: float,
    focal_length_pixels: float,
) -> float | None:
    """
    Estimate distance using the pinhole camera approximation.

    distance = real_width * focal_length / pixel_width

    This is only an approximation and requires a known object width
    and calibrated camera.
    """

    if pixel_width <= 0:
        return None

    distance = (
        real_width_m
        * focal_length_pixels
        / pixel_width
    )

    return float(distance)


def angle_difference_deg(
    first_angle: float,
    second_angle: float,
) -> float:
    """Return the smallest absolute difference between two angles."""

    difference = abs(first_angle - second_angle)

    while difference > 180:
        difference -= 360

    return float(abs(difference))


def radians_to_degrees(value: float) -> float:
    return math.degrees(value)