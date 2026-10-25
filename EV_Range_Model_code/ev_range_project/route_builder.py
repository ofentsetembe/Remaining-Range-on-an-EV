"""Build simulation routes from manual data or external API results."""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, atan2, cos, degrees, radians, sin, sqrt
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class GeoPoint:
    """A WGS84 latitude/longitude coordinate."""

    latitude: float
    longitude: float


@dataclass(frozen=True)
class RouteSpeedSection:
    """A speed target applying between two route-polyline point indices."""

    start_point_index: int
    end_point_index: int
    speed_mps: float
    source: str


@dataclass
class Route:
    """Elevation sampled at monotonically increasing route distances."""

    distance_m: np.ndarray
    elevation_m: np.ndarray

    def __post_init__(self) -> None:
        self.distance_m = np.asarray(self.distance_m, dtype=float)
        self.elevation_m = np.asarray(self.elevation_m, dtype=float)

        if self.distance_m.ndim != 1 or self.elevation_m.ndim != 1:
            raise ValueError("route distance and elevation must be 1-D arrays")
        if len(self.distance_m) < 2 or len(self.distance_m) != len(self.elevation_m):
            raise ValueError("route arrays must have the same length and at least 2 points")
        if not np.all(np.diff(self.distance_m) > 0.0):
            raise ValueError("route distances must be strictly increasing")

        self._segment_slope = np.diff(self.elevation_m) / np.diff(self.distance_m)
        self._segment_angle_rad = np.arctan(self._segment_slope)

    @property
    def length_m(self) -> float:
        return float(self.distance_m[-1])

    def elevation_at(self, position_m: float) -> float:
        return float(np.interp(position_m, self.distance_m, self.elevation_m))

    def angle_at(self, position_m: float) -> float:
        position_m = float(
            np.clip(position_m, self.distance_m[0], self.distance_m[-1])
        )
        segment_index = (
            np.searchsorted(self.distance_m, position_m, side="right") - 1
        )
        segment_index = int(
            np.clip(segment_index, 0, len(self._segment_angle_rad) - 1)
        )
        return float(self._segment_angle_rad[segment_index])

    def grade_percent_at(self, position_m: float) -> float:
        return 100.0 * np.tan(self.angle_at(position_m))


@dataclass(frozen=True)
class RouteData:
    """A route plus its geographic geometry and routing metadata."""

    route: Route
    points: tuple[GeoPoint, ...]
    expected_travel_time_s: float | None = None
    speed_sections: tuple[RouteSpeedSection, ...] = ()


class RoutingClient(Protocol):
    def calculate_route(
        self,
        origin: GeoPoint,
        destination: GeoPoint,
        *,
        traffic: bool = True,
    ) -> object: ...


class ElevationClient(Protocol):
    def get_elevations(self, points: Sequence[GeoPoint]) -> list[float]: ...


class RouteBuilder:
    """Convert route geometry into the distance/elevation model."""

    @staticmethod
    def cumulative_distances_m(points: Sequence[GeoPoint]) -> np.ndarray:
        if len(points) < 2:
            raise ValueError("at least two geographic points are required")

        distances = [0.0]
        for start, end in zip(points[:-1], points[1:]):
            segment_length = RouteBuilder._haversine_distance_m(start, end)
            if segment_length <= 0.0:
                raise ValueError("consecutive route points must be different")
            distances.append(distances[-1] + segment_length)
        return np.asarray(distances, dtype=float)

    @staticmethod
    def route_bearing_deg(points: Sequence[GeoPoint]) -> float:
        """Approximate route heading using its first and last coordinates."""

        if len(points) < 2:
            raise ValueError("at least two geographic points are required")
        start = points[0]
        end = points[-1]
        latitude_1 = radians(start.latitude)
        latitude_2 = radians(end.latitude)
        delta_longitude = radians(end.longitude - start.longitude)
        x = sin(delta_longitude) * cos(latitude_2)
        y = (
            cos(latitude_1) * sin(latitude_2)
            - sin(latitude_1) * cos(latitude_2) * cos(delta_longitude)
        )
        return (degrees(atan2(x, y)) + 360.0) % 360.0

    @staticmethod
    def headwind_component_mps(
        route_bearing_deg: float,
        wind_from_direction_deg: float,
        wind_speed_mps: float,
    ) -> float:
        """Project meteorological wind direction onto the route heading."""

        direction_difference = radians(
            wind_from_direction_deg - route_bearing_deg
        )
        return wind_speed_mps * cos(direction_difference)

    @staticmethod
    def from_points_and_elevations(
        points: Sequence[GeoPoint],
        elevation_m: Sequence[float],
        *,
        expected_travel_time_s: float | None = None,
        speed_sections: Sequence[RouteSpeedSection] = (),
    ) -> RouteData:
        if len(points) != len(elevation_m):
            raise ValueError("each geographic point must have one elevation")

        points_tuple = tuple(points)
        route = Route(
            distance_m=RouteBuilder.cumulative_distances_m(points_tuple),
            elevation_m=np.asarray(elevation_m, dtype=float),
        )
        return RouteData(
            route=route,
            points=points_tuple,
            expected_travel_time_s=expected_travel_time_s,
            speed_sections=tuple(speed_sections),
        )

    @staticmethod
    def from_api_clients(
        origin: GeoPoint,
        destination: GeoPoint,
        routing_client: RoutingClient,
        elevation_client: ElevationClient,
        *,
        traffic: bool = True,
    ) -> RouteData:
        """Fetch geometry first, then attach an elevation to every point."""

        routing_result = routing_client.calculate_route(
            origin,
            destination,
            traffic=traffic,
        )
        points = routing_result.points
        elevations = elevation_client.get_elevations(points)
        return RouteBuilder.from_points_and_elevations(
            points,
            elevations,
            expected_travel_time_s=routing_result.travel_time_s,
            speed_sections=routing_result.speed_sections,
        )

    @staticmethod
    def make_demo_route() -> RouteData:
        """Long mixed route: urban hills, flat highway and hilly roads."""

        distance_m = np.array(
            [
                0, 500, 1_000, 1_500, 2_000, 3_000, 4_000,
                7_000, 10_000, 13_000, 16_000, 19_000, 22_000,
                23_000, 24_000, 25_000, 26_000, 27_000, 28_000,
                29_000, 30_000,
            ],
            dtype=float,
        )
        elevation_m = np.array(
            [
                1_650, 1_660, 1_648, 1_670, 1_660, 1_662, 1_660,
                1_661, 1_660, 1_661, 1_660, 1_661, 1_660,
                1_680, 1_650, 1_685, 1_645, 1_675, 1_655,
                1_680, 1_660,
            ],
            dtype=float,
        )
        return RouteData(route=Route(distance_m, elevation_m), points=())

    @staticmethod
    def _haversine_distance_m(start: GeoPoint, end: GeoPoint) -> float:
        earth_radius_m = 6_371_000.0
        latitude_1 = radians(start.latitude)
        latitude_2 = radians(end.latitude)
        delta_latitude = radians(end.latitude - start.latitude)
        delta_longitude = radians(end.longitude - start.longitude)
        haversine = (
            sin(delta_latitude / 2.0) ** 2
            + cos(latitude_1)
            * cos(latitude_2)
            * sin(delta_longitude / 2.0) ** 2
        )
        return 2.0 * earth_radius_m * asin(sqrt(haversine))
