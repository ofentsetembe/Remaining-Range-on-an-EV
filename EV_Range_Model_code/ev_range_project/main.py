"""Configuration and execution layer for the EV range simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from .api_clients import (
    ApiClientError,
    OpenElevationClient,
    OpenMeteoClient,
    TomTomRoutingClient,
)
from .driving_profiles import make_mixed_speed_profile, make_route_speed_profile
from .physical_model import (
    BatteryModel,
    Environment,
    Powertrain,
    VehicleDynamicsSimulator,
    VehicleParameters,
)
from .plotting import plot_results
from .route_builder import GeoPoint, RouteBuilder


StatusCallback = Callable[[str], None]


@dataclass
class SimulationConfig:
    """All user-facing settings needed to run one simulation."""

    frame_time_s: float = 0.1
    offline_total_time_s: float = 1_800.0
    use_live_route: bool = False
    use_live_weather: bool = False
    use_traffic: bool = True

    origin_latitude: float = -26.2041
    origin_longitude: float = 28.0473
    destination_latitude: float = -26.1076
    destination_longitude: float = 28.0567

    maximum_acceleration_mps2: float = 1.5
    maximum_deceleration_mps2: float = 2.0
    driver_speed_factor: float = 0.95

    default_temperature_c: float = 20.0
    default_headwind_mps: float = 2.0
    default_pressure_pa: float = 101_325.0

    minimum_range_estimation_distance_m: float = 500.0
    initial_consumption_estimate_wh_per_km: float = 160.0
    consumption_prior_distance_km: float = 10.0
    minimum_consumption_estimate_wh_per_km: float = 40.0

    vehicle: VehicleParameters = field(default_factory=VehicleParameters)
    powertrain: Powertrain = field(default_factory=Powertrain)
    battery: BatteryModel = field(
        default_factory=lambda: BatteryModel(
            soc_percent=80.0,
            soh_percent=95.0,
            capacity_kwh=42.0,
        )
    )

    def validate(self) -> None:
        if self.frame_time_s <= 0.0:
            raise ValueError("Frame time must be positive")
        if self.offline_total_time_s <= 0.0:
            raise ValueError("Offline profile duration must be positive")
        if self.use_live_weather and not self.use_live_route:
            raise ValueError("Live weather requires a live route location")
        if not -90.0 <= self.origin_latitude <= 90.0:
            raise ValueError("Origin latitude must be between -90 and 90")
        if not -90.0 <= self.destination_latitude <= 90.0:
            raise ValueError("Destination latitude must be between -90 and 90")
        if not -180.0 <= self.origin_longitude <= 180.0:
            raise ValueError("Origin longitude must be between -180 and 180")
        if not -180.0 <= self.destination_longitude <= 180.0:
            raise ValueError("Destination longitude must be between -180 and 180")
        if self.maximum_acceleration_mps2 <= 0.0:
            raise ValueError("Maximum acceleration must be positive")
        if self.maximum_deceleration_mps2 <= 0.0:
            raise ValueError("Maximum deceleration must be positive")
        if not 0.0 < self.driver_speed_factor <= 1.0:
            raise ValueError("Driver speed factor must be in (0, 1]")
        if self.default_pressure_pa <= 0.0:
            raise ValueError("Atmospheric pressure must be positive")
        if self.default_temperature_c <= -273.15:
            raise ValueError("Temperature must be above absolute zero")
        if self.minimum_range_estimation_distance_m <= 0.0:
            raise ValueError("Range-estimation distance must be positive")
        if self.initial_consumption_estimate_wh_per_km <= 0.0:
            raise ValueError("Initial consumption assumption must be positive")
        if self.consumption_prior_distance_km <= 0.0:
            raise ValueError("Consumption prior distance must be positive")
        if self.minimum_consumption_estimate_wh_per_km <= 0.0:
            raise ValueError("Minimum consumption must be positive")
        if (
            self.minimum_consumption_estimate_wh_per_km
            > self.initial_consumption_estimate_wh_per_km
        ):
            raise ValueError(
                "Minimum consumption cannot exceed the initial assumption"
            )
        self.vehicle.validate()
        self.powertrain.validate()
        self.battery.validate()


@dataclass(frozen=True)
class SimulationOutcome:
    simulator: VehicleDynamicsSimulator
    initial_soc_percent: float
    messages: tuple[str, ...]


def run_simulation(
    config: SimulationConfig,
    status_callback: StatusCallback | None = None,
) -> SimulationOutcome:
    """Build the selected inputs and run one complete speed profile."""

    config.validate()
    messages: list[str] = []

    def report(message: str) -> None:
        messages.append(message)
        if status_callback is not None:
            status_callback(message)

    # The battery changes during simulation, so each run needs a fresh instance.
    vehicle = VehicleParameters(**asdict(config.vehicle))
    powertrain = Powertrain(**asdict(config.powertrain))
    battery = BatteryModel(**asdict(config.battery))
    initial_soc_percent = battery.soc_percent

    if config.use_live_route:
        report("Requesting route and elevation data...")
        origin = GeoPoint(config.origin_latitude, config.origin_longitude)
        destination = GeoPoint(
            config.destination_latitude,
            config.destination_longitude,
        )
        route_data = RouteBuilder.from_api_clients(
            origin,
            destination,
            TomTomRoutingClient(),
            OpenElevationClient(),
            traffic=config.use_traffic,
        )

        environment = _build_environment(config, route_data, report)
        report("Generating the route speed profile...")
        time_s, speed_mps = make_route_speed_profile(
            route_data,
            dt_s=config.frame_time_s,
            maximum_acceleration_mps2=config.maximum_acceleration_mps2,
            maximum_deceleration_mps2=config.maximum_deceleration_mps2,
            driver_speed_factor=config.driver_speed_factor,
        )
        report("Using live route, elevation and route-speed data.")
    else:
        route_data = RouteBuilder.make_demo_route()
        environment = Environment.from_temperature(
            temperature_c=config.default_temperature_c,
            wind_speed_mps=config.default_headwind_mps,
            pressure_pa=config.default_pressure_pa,
        )
        time_s, speed_mps = make_mixed_speed_profile(
            total_time_s=config.offline_total_time_s,
            dt_s=config.frame_time_s,
        )
        report("Using the offline demonstration route and speed profile.")

    report("Running the frame-by-frame simulation...")
    simulator = VehicleDynamicsSimulator(
        vehicle,
        powertrain,
        battery,
        route_data.route,
        environment,
        minimum_range_estimation_distance_m=(
            config.minimum_range_estimation_distance_m
        ),
        initial_consumption_estimate_wh_per_km=(
            config.initial_consumption_estimate_wh_per_km
        ),
        consumption_prior_distance_km=config.consumption_prior_distance_km,
        minimum_consumption_estimate_wh_per_km=(
            config.minimum_consumption_estimate_wh_per_km
        ),
    )
    simulator.run_speed_profile(time_s, speed_mps)
    report("Simulation complete.")
    return SimulationOutcome(
        simulator=simulator,
        initial_soc_percent=initial_soc_percent,
        messages=tuple(messages),
    )


def _build_environment(config, route_data, report) -> Environment:
    if not config.use_live_weather:
        return Environment.from_temperature(
            temperature_c=config.default_temperature_c,
            wind_speed_mps=config.default_headwind_mps,
            pressure_pa=config.default_pressure_pa,
        )

    report("Requesting current weather data...")
    midpoint = route_data.points[len(route_data.points) // 2]
    try:
        weather = OpenMeteoClient().get_current_weather(midpoint)
    except ApiClientError as error:
        report(
            "Open-Meteo unavailable; using the entered default weather "
            f"conditions. ({error})"
        )
        return Environment.from_temperature(
            temperature_c=config.default_temperature_c,
            wind_speed_mps=config.default_headwind_mps,
            pressure_pa=config.default_pressure_pa,
        )

    route_bearing = RouteBuilder.route_bearing_deg(route_data.points)
    headwind = RouteBuilder.headwind_component_mps(
        route_bearing,
        weather.wind_direction_deg,
        weather.wind_speed_mps,
    )
    report("Using current Open-Meteo weather data.")
    return Environment.from_temperature(
        temperature_c=weather.temperature_c,
        wind_speed_mps=headwind,
        pressure_pa=weather.surface_pressure_pa,
    )


def format_summary(outcome: SimulationOutcome) -> str:
    simulator = outcome.simulator
    state = simulator.state
    battery = simulator.battery
    final_frame = simulator.history[-1]
    lines = [
        "Simulation summary",
        f"Time simulated:              {state.time_s:9.1f} s",
        f"Distance travelled:          {state.position_m / 1000:9.3f} km",
        f"Initial SOC:                 {outcome.initial_soc_percent:9.3f} %",
        f"Final SOC:                   {battery.soc_percent:9.3f} %",
        (
            "Net battery energy used:     "
            f"{state.net_battery_energy_j / 3.6e6:9.4f} kWh"
        ),
    ]
    if np.isfinite(final_frame.average_battery_consumption_wh_per_km):
        lines.append(
            "Observed consumption:        "
            f"{final_frame.average_battery_consumption_wh_per_km:9.1f} Wh/km"
        )
    lines.extend(
        [
            (
                "Range-estimator consumption: "
                f"{final_frame.range_estimate_consumption_wh_per_km:8.1f} Wh/km"
            ),
            (
                "Estimated remaining range:   "
                f"{final_frame.estimated_remaining_range_km:8.1f} km"
            ),
        ]
    )
    warnings = _simulation_warnings(simulator)
    if warnings:
        lines.extend(["", "Warnings:", *warnings])
    return "\n".join(lines)


def _simulation_warnings(simulator: VehicleDynamicsSimulator) -> list[str]:
    warnings: list[str] = []
    maximum_unmet_power_w = max(
        frame.unmet_wheel_power_w for frame in simulator.history
    )
    if maximum_unmet_power_w > 1.0:
        warnings.append(
            "- Requested speed exceeded the motor limit by up to "
            f"{maximum_unmet_power_w / 1000.0:.2f} kW at the wheels."
        )
    if simulator.state.battery_energy_limit_difference_j > 1.0:
        warnings.append(
            "- Battery reached an SOC boundary; some requested energy flow "
            "could not be accepted."
        )
    return warnings


def main() -> None:
    """Run the default offline configuration without opening the GUI."""

    outcome = run_simulation(SimulationConfig(), print)
    print("\n" + format_summary(outcome))
    plot_results(outcome.simulator)


if __name__ == "__main__":
    main()
