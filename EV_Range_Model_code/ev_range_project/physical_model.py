"""Frame-by-frame longitudinal vehicle, powertrain and battery model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np

from .route_builder import Route

G = 9.81


@dataclass
class VehicleParameters:
    mass_kg: float = 1_300.0
    rolling_resistance_coefficient: float = 0.012
    drag_coefficient: float = 0.29
    frontal_area_m2: float = 2.2
    wheel_radius_m: float = 0.31
    gear_ratio: float = 9.0
    rotor_inertia_kg_m2: float = 0.03

    def validate(self) -> None:
        if self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be positive")
        if self.rolling_resistance_coefficient < 0.0:
            raise ValueError("rolling resistance cannot be negative")
        if self.drag_coefficient < 0.0 or self.frontal_area_m2 < 0.0:
            raise ValueError("drag coefficient and frontal area cannot be negative")
        if self.wheel_radius_m <= 0.0:
            raise ValueError("wheel_radius_m must be positive")
        if self.gear_ratio < 0.0 or self.rotor_inertia_kg_m2 < 0.0:
            raise ValueError("gear ratio and rotor inertia cannot be negative")

    @property
    def equivalent_mass_kg(self) -> float:
        reflected_rotating_mass = (
            self.rotor_inertia_kg_m2
            * self.gear_ratio**2
            / self.wheel_radius_m**2
        )
        return self.mass_kg + reflected_rotating_mass


@dataclass
class PowertrainFrame:
    traction_battery_power_w: float
    regenerative_battery_power_w: float
    auxiliary_power_w: float
    battery_power_w: float
    friction_brake_power_w: float
    unmet_wheel_power_w: float


@dataclass
class Powertrain:
    drivetrain_efficiency: float = 0.90
    regen_efficiency: float = 0.70
    max_motor_power_w: float = 100_000
    max_regen_power_w: float = 30_000
    auxiliary_power_w: float = 500

    def validate(self) -> None:
        if not 0.0 < self.drivetrain_efficiency <= 1.0:
            raise ValueError("drivetrain_efficiency must be in (0, 1]")
        if not 0.0 <= self.regen_efficiency <= 1.0:
            raise ValueError("regen_efficiency must be in [0, 1]")
        if self.max_motor_power_w < 0.0 or self.max_regen_power_w < 0.0:
            raise ValueError("power limits cannot be negative")
        if self.auxiliary_power_w < 0.0:
            raise ValueError("auxiliary_power_w cannot be negative")

    def convert_wheel_power(self, wheel_power_w: float) -> PowertrainFrame:
        """Convert signed wheel demand into battery and braking powers."""

        traction_battery_power_w = 0.0
        regenerative_battery_power_w = 0.0
        friction_brake_power_w = 0.0
        unmet_wheel_power_w = 0.0

        if wheel_power_w >= 0.0:
            requested_traction_power_w = (
                wheel_power_w / self.drivetrain_efficiency
            )
            traction_battery_power_w = min(
                requested_traction_power_w,
                self.max_motor_power_w,
            )
            available_wheel_power_w = (
                traction_battery_power_w * self.drivetrain_efficiency
            )
            unmet_wheel_power_w = max(
                wheel_power_w - available_wheel_power_w,
                0.0,
            )
        else:
            required_braking_power_w = -wheel_power_w
            potential_regen_power_w = (
                required_braking_power_w * self.regen_efficiency
            )
            regenerative_battery_power_w = min(
                potential_regen_power_w,
                self.max_regen_power_w,
            )
            if self.regen_efficiency > 0.0:
                regenerative_wheel_power_w = (
                    regenerative_battery_power_w / self.regen_efficiency
                )
            else:
                regenerative_wheel_power_w = 0.0
            friction_brake_power_w = max(
                required_braking_power_w - regenerative_wheel_power_w,
                0.0,
            )

        battery_power_w = (
            traction_battery_power_w
            + self.auxiliary_power_w
            - regenerative_battery_power_w
        )
        return PowertrainFrame(
            traction_battery_power_w=traction_battery_power_w,
            regenerative_battery_power_w=regenerative_battery_power_w,
            auxiliary_power_w=self.auxiliary_power_w,
            battery_power_w=battery_power_w,
            friction_brake_power_w=friction_brake_power_w,
            unmet_wheel_power_w=unmet_wheel_power_w,
        )


@dataclass
class BatteryModel:
    """Simple energy-accounting battery model; no electrochemistry."""

    soc_percent: float
    soh_percent: float
    capacity_kwh: float

    def validate(self) -> None:
        if not 0.0 <= self.soc_percent <= 100.0:
            raise ValueError("soc_percent must be in [0, 100]")
        if not 0.0 < self.soh_percent <= 100.0:
            raise ValueError("soh_percent must be in (0, 100]")
        if self.capacity_kwh <= 0.0:
            raise ValueError("capacity_kwh must be positive")

    @property
    def usable_capacity_kwh(self) -> float:
        return self.capacity_kwh * self.soh_percent / 100.0

    @property
    def remaining_energy_kwh(self) -> float:
        return self.usable_capacity_kwh * self.soc_percent / 100.0

    def apply_energy(self, requested_battery_energy_j: float) -> float:
        """Update SOC and return energy actually removed from the battery."""

        energy_before_kwh = self.remaining_energy_kwh
        requested_energy_kwh = requested_battery_energy_j / 3.6e6
        energy_after_kwh = float(
            np.clip(
                energy_before_kwh - requested_energy_kwh,
                0.0,
                self.usable_capacity_kwh,
            )
        )
        self.soc_percent = 100.0 * energy_after_kwh / self.usable_capacity_kwh
        return (energy_before_kwh - energy_after_kwh) * 3.6e6


@dataclass
class Environment:
    """Positive wind speed is a headwind; negative is a tailwind."""

    air_density_kg_m3: float = 1.225
    wind_speed_mps: float = 0.0

    @classmethod
    def from_temperature(
        cls,
        temperature_c: float,
        wind_speed_mps: float = 0.0,
        pressure_pa: float = 101_325.0,
    ) -> Environment:
        absolute_temperature_k = temperature_c + 273.15
        if absolute_temperature_k <= 0.0:
            raise ValueError("temperature must be above absolute zero")
        density = pressure_pa / (287.05 * absolute_temperature_k)
        return cls(air_density_kg_m3=density, wind_speed_mps=wind_speed_mps)


@dataclass
class SimulationState:
    time_s: float = 0.0
    position_m: float = 0.0
    speed_mps: float = 0.0
    positive_wheel_energy_j: float = 0.0
    braking_energy_available_j: float = 0.0
    signed_wheel_energy_j: float = 0.0
    traction_energy_from_battery_j: float = 0.0
    regenerative_energy_to_battery_j: float = 0.0
    auxiliary_energy_j: float = 0.0
    requested_net_battery_energy_j: float = 0.0
    net_battery_energy_j: float = 0.0
    battery_energy_limit_difference_j: float = 0.0


@dataclass
class FrameResult:
    time_s: float
    dt_s: float
    position_m: float
    elevation_m: float
    speed_mps: float
    acceleration_mps2: float
    road_angle_rad: float
    grade_percent: float
    rolling_force_n: float
    aerodynamic_force_n: float
    gradient_force_n: float
    inertial_force_n: float
    required_wheel_force_n: float
    wheel_power_w: float
    frame_wheel_energy_j: float
    traction_battery_power_w: float
    regenerative_battery_power_w: float
    auxiliary_power_w: float
    battery_power_w: float
    friction_brake_power_w: float
    unmet_wheel_power_w: float
    requested_frame_battery_energy_j: float
    actual_frame_battery_energy_j: float
    actual_battery_power_w: float
    battery_soc_percent: float
    battery_remaining_energy_kwh: float
    cumulative_positive_wheel_energy_wh: float
    cumulative_braking_energy_available_wh: float
    cumulative_signed_wheel_energy_wh: float
    cumulative_traction_energy_from_battery_wh: float
    cumulative_regenerative_energy_to_battery_wh: float
    cumulative_auxiliary_energy_wh: float
    cumulative_requested_net_battery_energy_wh: float
    cumulative_net_battery_energy_wh: float
    cumulative_battery_energy_limit_difference_wh: float
    average_battery_consumption_wh_per_km: float
    range_estimate_consumption_wh_per_km: float
    estimated_remaining_range_km: float
    estimated_total_range_km: float


class VehicleDynamicsSimulator:
    """Discrete-time vehicle model updated once per simulation frame."""

    def __init__(
        self,
        vehicle: VehicleParameters,
        powertrain: Powertrain,
        battery: BatteryModel,
        route: Route,
        environment: Environment,
        initial_speed_mps: float = 0.0,
        minimum_range_estimation_distance_m: float = 500.0,
        initial_consumption_estimate_wh_per_km: float = 160.0,
        consumption_prior_distance_km: float = 10.0,
        minimum_consumption_estimate_wh_per_km: float = 40.0,
    ) -> None:
        vehicle.validate()
        powertrain.validate()
        battery.validate()
        if initial_speed_mps < 0.0:
            raise ValueError("this model supports forward speed only")
        if minimum_range_estimation_distance_m <= 0.0:
            raise ValueError("minimum range-estimation distance must be positive")
        if initial_consumption_estimate_wh_per_km <= 0.0:
            raise ValueError("initial consumption estimate must be positive")
        if consumption_prior_distance_km <= 0.0:
            raise ValueError("consumption prior distance must be positive")
        if minimum_consumption_estimate_wh_per_km <= 0.0:
            raise ValueError("minimum consumption estimate must be positive")
        if (
            minimum_consumption_estimate_wh_per_km
            > initial_consumption_estimate_wh_per_km
        ):
            raise ValueError(
                "minimum consumption estimate cannot exceed the initial estimate"
            )

        self.vehicle = vehicle
        self.powertrain = powertrain
        self.battery = battery
        self.route = route
        self.environment = environment
        self.minimum_range_estimation_distance_m = (
            minimum_range_estimation_distance_m
        )
        self.initial_consumption_estimate_wh_per_km = (
            initial_consumption_estimate_wh_per_km
        )
        self.consumption_prior_distance_km = consumption_prior_distance_km
        self.minimum_consumption_estimate_wh_per_km = (
            minimum_consumption_estimate_wh_per_km
        )
        self.state = SimulationState(speed_mps=initial_speed_mps)
        self.history: List[FrameResult] = []

    def step(self, next_speed_mps: float, dt_s: float) -> FrameResult:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if next_speed_mps < 0.0:
            raise ValueError("this model supports forward speed only")

        previous_speed = self.state.speed_mps
        average_speed = 0.5 * (previous_speed + next_speed_mps)
        acceleration = (next_speed_mps - previous_speed) / dt_s
        distance_increment = average_speed * dt_s
        midpoint_position = min(
            self.state.position_m + 0.5 * distance_increment,
            self.route.length_m,
        )
        theta = self.route.angle_at(midpoint_position)
        parameters = self.vehicle

        rolling_force = 0.0
        if average_speed > 0.0:
            rolling_force = (
                parameters.rolling_resistance_coefficient
                * parameters.mass_kg
                * G
                * np.cos(theta)
            )

        relative_air_speed = average_speed + self.environment.wind_speed_mps
        aerodynamic_force = (
            0.5
            * self.environment.air_density_kg_m3
            * parameters.drag_coefficient
            * parameters.frontal_area_m2
            * relative_air_speed
            * abs(relative_air_speed)
        )
        gradient_force = parameters.mass_kg * G * np.sin(theta)
        inertial_force = parameters.equivalent_mass_kg * acceleration
        required_wheel_force = (
            rolling_force
            + aerodynamic_force
            + gradient_force
            + inertial_force
        )
        wheel_power = required_wheel_force * average_speed
        frame_wheel_energy_j = wheel_power * dt_s
        powertrain_frame = self.powertrain.convert_wheel_power(wheel_power)
        requested_battery_energy_j = powertrain_frame.battery_power_w * dt_s
        actual_battery_energy_j = self.battery.apply_energy(
            requested_battery_energy_j
        )

        if frame_wheel_energy_j >= 0.0:
            self.state.positive_wheel_energy_j += frame_wheel_energy_j
        else:
            self.state.braking_energy_available_j += -frame_wheel_energy_j
        self.state.signed_wheel_energy_j += frame_wheel_energy_j
        self.state.traction_energy_from_battery_j += (
            powertrain_frame.traction_battery_power_w * dt_s
        )
        self.state.regenerative_energy_to_battery_j += (
            powertrain_frame.regenerative_battery_power_w * dt_s
        )
        self.state.auxiliary_energy_j += powertrain_frame.auxiliary_power_w * dt_s
        self.state.requested_net_battery_energy_j += requested_battery_energy_j
        self.state.net_battery_energy_j += actual_battery_energy_j
        self.state.battery_energy_limit_difference_j += abs(
            requested_battery_energy_j - actual_battery_energy_j
        )

        self.state.time_s += dt_s
        self.state.position_m = min(
            self.state.position_m + distance_increment,
            self.route.length_m,
        )
        self.state.speed_mps = next_speed_mps

        (
            observed_consumption,
            estimated_consumption,
            remaining_range,
            total_range,
        ) = self._range_estimate()
        result = FrameResult(
            time_s=self.state.time_s,
            dt_s=dt_s,
            position_m=self.state.position_m,
            elevation_m=self.route.elevation_at(self.state.position_m),
            speed_mps=next_speed_mps,
            acceleration_mps2=acceleration,
            road_angle_rad=theta,
            grade_percent=100.0 * np.tan(theta),
            rolling_force_n=float(rolling_force),
            aerodynamic_force_n=float(aerodynamic_force),
            gradient_force_n=float(gradient_force),
            inertial_force_n=float(inertial_force),
            required_wheel_force_n=float(required_wheel_force),
            wheel_power_w=float(wheel_power),
            frame_wheel_energy_j=float(frame_wheel_energy_j),
            traction_battery_power_w=powertrain_frame.traction_battery_power_w,
            regenerative_battery_power_w=(
                powertrain_frame.regenerative_battery_power_w
            ),
            auxiliary_power_w=powertrain_frame.auxiliary_power_w,
            battery_power_w=powertrain_frame.battery_power_w,
            friction_brake_power_w=powertrain_frame.friction_brake_power_w,
            unmet_wheel_power_w=powertrain_frame.unmet_wheel_power_w,
            requested_frame_battery_energy_j=requested_battery_energy_j,
            actual_frame_battery_energy_j=actual_battery_energy_j,
            actual_battery_power_w=actual_battery_energy_j / dt_s,
            battery_soc_percent=self.battery.soc_percent,
            battery_remaining_energy_kwh=self.battery.remaining_energy_kwh,
            cumulative_positive_wheel_energy_wh=(
                self.state.positive_wheel_energy_j / 3600.0
            ),
            cumulative_braking_energy_available_wh=(
                self.state.braking_energy_available_j / 3600.0
            ),
            cumulative_signed_wheel_energy_wh=(
                self.state.signed_wheel_energy_j / 3600.0
            ),
            cumulative_traction_energy_from_battery_wh=(
                self.state.traction_energy_from_battery_j / 3600.0
            ),
            cumulative_regenerative_energy_to_battery_wh=(
                self.state.regenerative_energy_to_battery_j / 3600.0
            ),
            cumulative_auxiliary_energy_wh=(
                self.state.auxiliary_energy_j / 3600.0
            ),
            cumulative_requested_net_battery_energy_wh=(
                self.state.requested_net_battery_energy_j / 3600.0
            ),
            cumulative_net_battery_energy_wh=(
                self.state.net_battery_energy_j / 3600.0
            ),
            cumulative_battery_energy_limit_difference_wh=(
                self.state.battery_energy_limit_difference_j / 3600.0
            ),
            average_battery_consumption_wh_per_km=float(observed_consumption),
            range_estimate_consumption_wh_per_km=float(estimated_consumption),
            estimated_remaining_range_km=float(remaining_range),
            estimated_total_range_km=float(total_range),
        )
        self.history.append(result)
        return result

    def _range_estimate(self) -> tuple[float, float, float, float]:
        """Return observed consumption and a stable prior-weighted estimate.

        The prior acts as if the vehicle had already travelled a configurable
        distance at the initial consumption estimate. Real trip energy and
        distance are then added to that prior. This prevents a few early frames
        with nearly zero net energy from implying an enormous driving range.
        """

        distance_km = self.state.position_m / 1000.0
        net_battery_energy_wh = self.state.net_battery_energy_j / 3600.0

        observed_consumption = np.nan
        if (
            self.state.position_m >= self.minimum_range_estimation_distance_m
            and distance_km > 0.0
        ):
            observed_consumption = net_battery_energy_wh / distance_km

        prior_energy_wh = (
            self.initial_consumption_estimate_wh_per_km
            * self.consumption_prior_distance_km
        )
        estimated_consumption = (
            prior_energy_wh + net_battery_energy_wh
        ) / (self.consumption_prior_distance_km + distance_km)
        estimated_consumption = max(
            estimated_consumption,
            self.minimum_consumption_estimate_wh_per_km,
        )

        remaining_range = (
            self.battery.remaining_energy_kwh * 1000.0
            / estimated_consumption
        )
        total_range = distance_km + remaining_range
        return (
            observed_consumption,
            estimated_consumption,
            remaining_range,
            total_range,
        )

    def run_speed_profile(
        self,
        time_s: np.ndarray,
        speed_mps: np.ndarray,
    ) -> List[FrameResult]:
        time_s = np.asarray(time_s, dtype=float)
        speed_mps = np.asarray(speed_mps, dtype=float)
        if time_s.ndim != 1 or speed_mps.ndim != 1:
            raise ValueError("time_s and speed_mps must be 1-D arrays")
        if len(time_s) != len(speed_mps):
            raise ValueError("time_s and speed_mps must have equal length")
        if len(time_s) < 2 or not np.all(np.diff(time_s) > 0.0):
            raise ValueError("timestamps must be strictly increasing")

        self.state.speed_mps = float(speed_mps[0])
        self.state.time_s = float(time_s[0])
        for index in range(1, len(time_s)):
            if self.state.position_m >= self.route.length_m:
                break
            self.step(
                next_speed_mps=float(speed_mps[index]),
                dt_s=float(time_s[index] - time_s[index - 1]),
            )
        return self.history

    def history_as_arrays(self) -> Dict[str, np.ndarray]:
        if not self.history:
            raise RuntimeError("no frames have been simulated")
        keys = asdict(self.history[0]).keys()
        return {
            key: np.array([getattr(frame, key) for frame in self.history])
            for key in keys
        }
