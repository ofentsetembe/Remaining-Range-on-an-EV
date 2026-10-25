"""Editable example speed profiles for offline simulation."""

import numpy as np

from .route_builder import RouteData


def make_mixed_speed_profile(
    total_time_s: float = 1800.0,
    dt_s: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an urban, highway and slower-road speed trace."""

    time_s = np.arange(0.0, total_time_s + dt_s, dt_s)
    key_time_s = np.array(
        [
            0, 30, 150, 170, 210, 300, 330, 900, 930, 1050,
            1080, 1200, 1230, 1350, 1380, 1500, 1530, 1650,
            1680, 1800,
        ]
    )
    key_speed_mps = np.array(
        [
            0, 13.9, 13.9, 0, 0, 13.9, 27.8, 27.8, 16.7, 16.7,
            0, 0, 11.1, 11.1, 0, 0, 16.7, 16.7, 0, 0,
        ]
    )
    return time_s, np.interp(time_s, key_time_s, key_speed_mps)


def make_route_speed_profile(
    route_data: RouteData,
    dt_s: float = 0.1,
    maximum_acceleration_mps2: float = 1.5,
    maximum_deceleration_mps2: float = 2.0,
    driver_speed_factor: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a frame profile from route speed limits and traffic sections.

    Legal limits define the base target. Traffic effective speeds can only
    reduce it. A forward/backward kinematic pass prevents instantaneous speed
    changes, and a final downward scaling matches a longer TomTom travel time.
    """

    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    if maximum_acceleration_mps2 <= 0.0:
        raise ValueError("maximum acceleration must be positive")
    if maximum_deceleration_mps2 <= 0.0:
        raise ValueError("maximum deceleration must be positive")
    if not 0.0 < driver_speed_factor <= 1.0:
        raise ValueError("driver_speed_factor must be in (0, 1]")

    distance_m = route_data.route.distance_m
    if len(distance_m) < 3:
        raise ValueError("route speed-profile generation needs at least 3 points")
    if (
        route_data.expected_travel_time_s is None
        or route_data.expected_travel_time_s <= 0.0
    ):
        raise ValueError("a positive expected route travel time is required")

    fallback_speed_mps = (
        route_data.route.length_m / route_data.expected_travel_time_s
    )
    target_speed_mps = np.full(len(distance_m), fallback_speed_mps)

    # Apply legal limits first. Points without speed-limit coverage retain the
    # traffic-aware whole-route average as a conservative fallback.
    for section in route_data.speed_sections:
        if section.source != "speed_limit":
            continue
        section_slice = slice(
            section.start_point_index,
            section.end_point_index + 1,
        )
        target_speed_mps[section_slice] = (
            section.speed_mps * driver_speed_factor
        )

    # A traffic incident is a local upper bound; it must never raise speed.
    for section in route_data.speed_sections:
        if section.source != "traffic":
            continue
        section_slice = slice(
            section.start_point_index,
            section.end_point_index + 1,
        )
        target_speed_mps[section_slice] = np.minimum(
            target_speed_mps[section_slice],
            section.speed_mps,
        )

    target_speed_mps = np.maximum(target_speed_mps, 0.1)
    target_speed_mps[0] = 0.0
    target_speed_mps[-1] = 0.0
    target_speed_mps = _apply_kinematic_speed_limits(
        distance_m,
        target_speed_mps,
        maximum_acceleration_mps2,
        maximum_deceleration_mps2,
    )

    route_time_s = _route_point_times(distance_m, target_speed_mps)
    generated_travel_time_s = route_time_s[-1]

    # TomTom's route time includes traffic effects not always represented by an
    # explicit traffic section. Slow the whole profile if necessary, but never
    # speed it above the legal/kinematic profile merely to shorten the trip.
    if generated_travel_time_s < route_data.expected_travel_time_s:
        time_scale = generated_travel_time_s / route_data.expected_travel_time_s
        target_speed_mps *= time_scale
        target_speed_mps[0] = 0.0
        target_speed_mps[-1] = 0.0
        route_time_s = _route_point_times(distance_m, target_speed_mps)

    frame_time_s = np.arange(0.0, route_time_s[-1], dt_s)
    frame_time_s = np.append(frame_time_s, route_time_s[-1])
    frame_speed_mps = np.interp(
        frame_time_s,
        route_time_s,
        target_speed_mps,
    )
    return frame_time_s, frame_speed_mps


def _apply_kinematic_speed_limits(
    distance_m: np.ndarray,
    target_speed_mps: np.ndarray,
    maximum_acceleration_mps2: float,
    maximum_deceleration_mps2: float,
) -> np.ndarray:
    speed_mps = target_speed_mps.copy()

    # Forward pass: maximum speed reachable from the previous point.
    for index in range(1, len(speed_mps)):
        distance_increment = distance_m[index] - distance_m[index - 1]
        reachable_speed = np.sqrt(
            speed_mps[index - 1] ** 2
            + 2.0 * maximum_acceleration_mps2 * distance_increment
        )
        speed_mps[index] = min(speed_mps[index], reachable_speed)

    # Backward pass: maximum prior speed that can brake to the next target.
    for index in range(len(speed_mps) - 2, -1, -1):
        distance_increment = distance_m[index + 1] - distance_m[index]
        braking_speed = np.sqrt(
            speed_mps[index + 1] ** 2
            + 2.0 * maximum_deceleration_mps2 * distance_increment
        )
        speed_mps[index] = min(speed_mps[index], braking_speed)
    return speed_mps


def _route_point_times(
    distance_m: np.ndarray,
    speed_mps: np.ndarray,
) -> np.ndarray:
    segment_distance_m = np.diff(distance_m)
    average_segment_speed_mps = 0.5 * (speed_mps[:-1] + speed_mps[1:])
    if np.any(average_segment_speed_mps <= 0.0):
        raise ValueError("route contains a segment with zero traversal speed")
    segment_time_s = segment_distance_m / average_segment_speed_mps
    return np.concatenate(([0.0], np.cumsum(segment_time_s)))
