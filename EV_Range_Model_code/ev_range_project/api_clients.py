"""Small standard-library clients for the external project APIs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .route_builder import GeoPoint, RouteSpeedSection


class ApiClientError(RuntimeError):
    """Raised when an external service cannot return usable data."""


@dataclass(frozen=True)
class RoutingResult:
    points: tuple[GeoPoint, ...]
    distance_m: float
    travel_time_s: float
    speed_sections: tuple[RouteSpeedSection, ...]


@dataclass(frozen=True)
class WeatherResult:
    temperature_c: float
    surface_pressure_pa: float
    wind_speed_mps: float
    wind_direction_deg: float


def _read_json(request: Request, timeout_s: float) -> dict[str, Any]:
    try:
        with urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise ApiClientError(
            f"API returned HTTP {error.code}: {details[:300]}"
        ) from error
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        raise ApiClientError(f"API request failed: {error}") from error


class TomTomRoutingClient:
    """Retrieve route geometry and traffic-aware travel time from TomTom."""

    base_url = "https://api.tomtom.com/routing/1/calculateRoute"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout_s: float = 20.0,
    ) -> None:
        self.api_key = api_key or os.getenv("TOMTOM_API_KEY", "")
        self.timeout_s = timeout_s
        if not self.api_key:
            raise ValueError(
                "TomTom API key missing. Set the TOMTOM_API_KEY environment variable."
            )

    def calculate_route(
        self,
        origin: GeoPoint,
        destination: GeoPoint,
        *,
        traffic: bool = True,
        depart_at: datetime | None = None,
    ) -> RoutingResult:
        locations = (
            f"{origin.latitude},{origin.longitude}:"
            f"{destination.latitude},{destination.longitude}"
        )
        # A list of pairs permits repeated TomTom query parameters.
        parameters = [
            ("key", self.api_key),
            ("traffic", str(traffic).lower()),
            ("travelMode", "car"),
            ("routeRepresentation", "polyline"),
            ("sectionType", "speedLimit"),
            ("sectionType", "traffic"),
            ("extendedRouteRepresentation", "distance"),
            ("extendedRouteRepresentation", "travelTime"),
            ("computeTravelTimeFor", "all"),
        ]
        if depart_at is not None:
            parameters.append(("departAt", depart_at.isoformat()))

        url = f"{self.base_url}/{locations}/json?{urlencode(parameters)}"
        payload = _read_json(
            Request(url, headers={"Accept": "application/json"}),
            self.timeout_s,
        )

        try:
            route = payload["routes"][0]
            raw_points = tuple(
                GeoPoint(point["latitude"], point["longitude"])
                for leg in route["legs"]
                for point in leg["points"]
            )
            # Adjacent legs can repeat a shared waypoint. Record an index map so
            # TomTom section indices still refer to the correct deduplicated point.
            unique_points: list[GeoPoint] = []
            raw_to_unique_index: list[int] = []
            for point in raw_points:
                if not unique_points or point != unique_points[-1]:
                    unique_points.append(point)
                raw_to_unique_index.append(len(unique_points) - 1)
            points = tuple(unique_points)

            speed_sections = []
            for section in route.get("sections", []):
                section_type = section.get("sectionType")
                speed_kmh = None
                source = ""
                if section_type == "SPEED_LIMIT":
                    speed_kmh = section.get("maxSpeedLimitInKmh")
                    source = "speed_limit"
                elif section_type == "TRAFFIC":
                    speed_kmh = section.get("effectiveSpeedInKmh")
                    source = "traffic"

                if speed_kmh is None:
                    continue

                start_index = int(section["startPointIndex"])
                end_index = int(section["endPointIndex"])
                if not 0 <= start_index <= end_index < len(raw_points):
                    raise ApiClientError(
                        "TomTom returned an invalid speed-section point index"
                    )
                speed_sections.append(
                    RouteSpeedSection(
                        start_point_index=raw_to_unique_index[start_index],
                        end_point_index=raw_to_unique_index[end_index],
                        speed_mps=max(float(speed_kmh) / 3.6, 0.0),
                        source=source,
                    )
                )

            summary = route["summary"]

            TomTomRoutingClient.last_route_points = points
            
            return RoutingResult(
                points=points,
                distance_m=float(summary["lengthInMeters"]),
                travel_time_s=float(summary["travelTimeInSeconds"]),
                speed_sections=tuple(speed_sections),
            )
        except (KeyError, IndexError, TypeError) as error:
            raise ApiClientError("TomTom response had an unexpected format") from error


class OpenElevationClient:
    """Retrieve elevations for route coordinates from Open-Meteo using smart sampling."""

    endpoint = "https://api.open-meteo.com/v1/elevation"

    def __init__(self, *, timeout_s: float = 30.0) -> None:
        self.timeout_s = timeout_s

    def get_elevations(self, points: Sequence[GeoPoint]) -> list[float]:
        if not points:
            raise ValueError("at least one point is required")

        # 1. We will sample exactly 100 points along the route to stay under limits
        max_samples = 100
        
        if len(points) <= max_samples:
            indices = list(range(len(points)))
        else:
            # Pick evenly spaced indices from start to finish
            indices = [int(i * (len(points) - 1) / (max_samples - 1)) for i in range(max_samples)]
            
        sampled_points = [points[i] for i in indices]

        # 2. Make exactly ONE request to the API
        lats = ",".join(str(p.latitude) for p in sampled_points)
        lons = ",".join(str(p.longitude) for p in sampled_points)
        
        url = f"{self.endpoint}?latitude={lats}&longitude={lons}"
        request = Request(url, headers={"Accept": "application/json"})
        payload = _read_json(request, self.timeout_s)
        
        try:
            sampled_elevations = [float(elev) for elev in payload["elevation"]]
        except (KeyError, TypeError, ValueError) as error:
            raise ApiClientError("Open-Meteo response had an unexpected format") from error

        # 3. Interpolate the gaps so we return the exact number of points expected
        full_elevations = []
        sample_idx = 0
        
        for i in range(len(points)):
            # Move to the next interval if we passed it
            if sample_idx < len(indices) - 2 and i >= indices[sample_idx + 1]:
                sample_idx += 1
                
            idx_start = indices[sample_idx]
            idx_end = indices[sample_idx + 1] if sample_idx + 1 < len(indices) else idx_start
            
            elev_start = sampled_elevations[sample_idx]
            elev_end = sampled_elevations[sample_idx + 1] if sample_idx + 1 < len(sampled_elevations) else elev_start
            
            if idx_start == idx_end:
                full_elevations.append(elev_start)
            else:
                # Calculate the exact slope between our two sampled points
                fraction = (i - idx_start) / (idx_end - idx_start)
                full_elevations.append(elev_start + fraction * (elev_end - elev_start))
                
        if len(full_elevations) != len(points):
            raise ApiClientError("Internal sampling error returned wrong number of points")
            
        return full_elevations


class OpenMeteoClient:
    """Retrieve current temperature, pressure and wind conditions."""

    endpoint = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, *, timeout_s: float = 20.0) -> None:
        self.timeout_s = timeout_s

    def get_current_weather(self, point: GeoPoint) -> WeatherResult:
        parameters = {
            "latitude": point.latitude,
            "longitude": point.longitude,
            "current": (
                "temperature_2m,surface_pressure,wind_speed_10m,"
                "wind_direction_10m"
            ),
            "wind_speed_unit": "ms",
        }
        url = f"{self.endpoint}?{urlencode(parameters)}"
        payload = _read_json(
            Request(url, headers={"Accept": "application/json"}),
            self.timeout_s,
        )
        try:
            current = payload["current"]
            return WeatherResult(
                temperature_c=float(current["temperature_2m"]),
                surface_pressure_pa=float(current["surface_pressure"]) * 100.0,
                wind_speed_mps=float(current["wind_speed_10m"]),
                wind_direction_deg=float(current["wind_direction_10m"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ApiClientError("Open-Meteo response had an unexpected format") from error
