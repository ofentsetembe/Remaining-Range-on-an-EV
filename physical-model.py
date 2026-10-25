# This model takes in a user-inputted route profile (speed and elevation)
# and calculates the vehicle and powertrain dynamics to determine the energy requirement at the wheels and battery.

from dataclasses import asdict, dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

G = 9.81

@dataclass
class VehicleParameters:

    mass_kg: float = 1_300.0
    rolling_resistance_coefficient: float = 0.012
    drag_coefficient: float = 0.29
    frontal_area_m2: float = 2.2

    # Optional rotating-inertia model. Set rotor_inertia_kg_m2 to zero if the
    # value is unknown. The rotor inertia is reflected to the wheels by G^2.
    wheel_radius_m: float = 0.31
    gear_ratio: float = 9.0
    rotor_inertia_kg_m2: float = 0.03

    def validate(self) -> None:
        if self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be positive")
        if self.rolling_resistance_coefficient < 0.0:
            raise ValueError("rolling_resistance_coefficient cannot be negative")
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
class Powertrain:
# constant-efficiency EV powertrain (update for temperature effects?).

# The power limits are relative to the battery/electrical side:
#      - max_motor_power_w limits propulsion power drawn by the motor/inverter.
#      - max_regen_power_w limits charging power returned during regeneration.

    drivetrain_efficiency: float = 0.90
    regen_efficiency: float = 0.70

    max_motor_power_w: float = 100_000
    max_regen_power_w: float = 30_000

    auxiliary_power_w: float = 500

    def validate(self) -> None:
        if not 0.0 < self.drivetrain_efficiency <= 1.0:
            raise ValueError("drivetrain_efficiency must be in the interval (0, 1]")
        if not 0.0 <= self.regen_efficiency <= 1.0:
            raise ValueError("regen_efficiency must be in the interval [0, 1]")
        if self.max_motor_power_w < 0.0 or self.max_regen_power_w < 0.0:
            raise ValueError("power limits cannot be negative")
        if self.auxiliary_power_w < 0.0:
            raise ValueError("auxiliary_power_w cannot be negative")

    def convert_wheel_power(self, wheel_power_w: float) -> "PowertrainFrame":

        """Convert signed wheel power into battery and braking powers.
        This model is retroactive, so a drive demand above the motor
        limit cannot change the requested speed. Instead, `unmet_wheel_power_w`
        records how much wheel power the powertrain cannot supply. A nonzero
        value means that the requested speed is physically infeasible for
        the specified power limit."""

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
class PowertrainFrame:
    # Instantaneous output of the powertrain calculation.

    traction_battery_power_w: float
    regenerative_battery_power_w: float
    auxiliary_power_w: float
    battery_power_w: float
    friction_brake_power_w: float
    unmet_wheel_power_w: float

@dataclass
class BatteryModel:
# Simple energy-accounting battery model.
    soc_percent: float
    soh_percent: float
    capacity_kwh: float

    def validate(self) -> None:
        if not 0.0 <= self.soc_percent <= 100.0:
            raise ValueError("soc_percent must be in the interval [0, 100]")
        if not 0.0 < self.soh_percent <= 100.0:
            raise ValueError("soh_percent must be in the interval (0, 100]")
        if self.capacity_kwh <= 0.0:
            raise ValueError("capacity_kwh must be positive")

    @property
    def usable_capacity_kwh(self) -> float:
        return self.capacity_kwh * self.soh_percent / 100.0

    @property
    def remaining_energy_kwh(self) -> float:
        return self.usable_capacity_kwh * self.soc_percent / 100.0

    def apply_energy(self, requested_battery_energy_j: float) -> float:
    # Update SOC and return the energy actually removed from the battery.

        energy_before_kwh = self.remaining_energy_kwh
        requested_energy_kwh = requested_battery_energy_j / 3.6e6
        requested_energy_after_kwh = energy_before_kwh - requested_energy_kwh
        energy_after_kwh = float(
            np.clip(
                requested_energy_after_kwh,
                0.0,
                self.usable_capacity_kwh,
            )
        )

        self.soc_percent = 100.0 * energy_after_kwh / self.usable_capacity_kwh
        actual_battery_energy_kwh = energy_before_kwh - energy_after_kwh
        return actual_battery_energy_kwh * 3.6e6


@dataclass
class Environment:
    """ Positive wind_speed_mps is a headwind. A negative value is a tailwind.
    Air density may be entered directly or estimated from temperature and
    pressure with Environment.from_temperature(). """

    air_density_kg_m3: float = 1.225
    wind_speed_mps: float = 0.0

    @classmethod
    def from_temperature(
        cls,
        temperature_c: float,
        wind_speed_mps: float = 0.0,
        pressure_pa: float = 101_325.0,
    ) -> "Environment":
        absolute_temperature_k = temperature_c + 273.15
        if absolute_temperature_k <= 0.0:
            raise ValueError("temperature must be above absolute zero")
        specific_gas_constant_air = 287.05  # J/(kg K)
        density = pressure_pa / (specific_gas_constant_air * absolute_temperature_k)
        return cls(air_density_kg_m3=density, wind_speed_mps=wind_speed_mps)


@dataclass
class Route:
# Route elevation sampled at monotonically increasing distances.

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

        # Calculate the slope of every route segment.
        # For each segment: slope = change in elevation / change in route distance
        self._segment_slope = (np.diff(self.elevation_m) / np.diff(self.distance_m))

        # Since grade = tan(theta), convert slope to road angle.
        self._segment_angle_rad = np.arctan(self._segment_slope)

    @property
    def length_m(self) -> float:
        return float(self.distance_m[-1])

    def elevation_at(self, position_m: float) -> float:
        return float(np.interp(position_m, self.distance_m, self.elevation_m))

    def angle_at(self, position_m: float) -> float:
        # Keep the requested position within the route boundaries.
        position_m = float(np.clip(position_m, self.distance_m[0], self.distance_m[-1],))

        # Find which route segment contains the vehicle.
        segment_index = (np.searchsorted(self.distance_m, position_m, side="right",) - 1 )

        # At the final route point, use the final available segment.
        segment_index = int(np.clip(segment_index, 0, len(self._segment_angle_rad) - 1,))

        return float(self._segment_angle_rad[segment_index])

    def grade_percent_at(self, position_m: float) -> float:
        return 100.0 * np.tan(self.angle_at(position_m))


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
    # Calculated values for one frame.

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


class VehicleDynamicsSimulator:
    # Discrete-time vehicle model updated once per simulation frame.

    def __init__(
        self,
        vehicle: VehicleParameters,
        powertrain: Powertrain,
        battery: BatteryModel,
        route: Route,
        environment: Environment,
        initial_speed_mps: float = 0.0,
    ) -> None:
        vehicle.validate()
        powertrain.validate()
        battery.validate()
        if initial_speed_mps < 0.0:
            raise ValueError("this model supports forward speed only")

        self.vehicle = vehicle
        self.powertrain = powertrain
        self.battery = battery
        self.route = route
        self.environment = environment
        self.state = SimulationState(speed_mps=initial_speed_mps)
        self.history: List[FrameResult] = []

    def step(self, next_speed_mps: float, dt_s: float) -> FrameResult:
        # Advance the model by one frame.

        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if next_speed_mps < 0.0:
            raise ValueError("this starter model supports forward speed only")

        previous_speed = self.state.speed_mps
        average_speed = 0.5 * (previous_speed + next_speed_mps)
        acceleration = (next_speed_mps - previous_speed) / dt_s
        distance_increment = average_speed * dt_s

        # Evaluate route conditions near the middle of the frame.
        midpoint_position = min(
            self.state.position_m + 0.5 * distance_increment,
            self.route.length_m,
        )
        theta = self.route.angle_at(midpoint_position)

        p = self.vehicle
        env = self.environment

        # Kinetic rolling resistance acts only while the vehicle is moving.
        rolling_force = 0.0
        if average_speed > 0.0:
            rolling_force = (
                p.rolling_resistance_coefficient
                * p.mass_kg
                * G
                * np.cos(theta)
            )

        # Positive relative air speed produces drag that opposes forward motion.
        # Tailwind gives a forward aerodynamic force.
        relative_air_speed = average_speed + env.wind_speed_mps
        aerodynamic_force = (
            0.5
            * env.air_density_kg_m3
            * p.drag_coefficient
            * p.frontal_area_m2
            * relative_air_speed
            * abs(relative_air_speed)
        )

        gradient_force = p.mass_kg * G * np.sin(theta)
        inertial_force = p.equivalent_mass_kg * acceleration

        required_wheel_force = (
            rolling_force
            + aerodynamic_force
            + gradient_force
            + inertial_force
        )
        wheel_power = required_wheel_force * average_speed
        frame_energy = wheel_power * dt_s
        powertrain_frame = self.powertrain.convert_wheel_power(wheel_power)
        requested_frame_battery_energy_j = (
            powertrain_frame.battery_power_w * dt_s
        )
        actual_frame_battery_energy_j = self.battery.apply_energy(
            requested_frame_battery_energy_j
        )

        if frame_energy >= 0.0:
            self.state.positive_wheel_energy_j += frame_energy
        else:
            self.state.braking_energy_available_j += -frame_energy
        self.state.signed_wheel_energy_j += frame_energy
        self.state.traction_energy_from_battery_j += (
            powertrain_frame.traction_battery_power_w * dt_s
        )
        self.state.regenerative_energy_to_battery_j += (
            powertrain_frame.regenerative_battery_power_w * dt_s
        )
        self.state.auxiliary_energy_j += (
            powertrain_frame.auxiliary_power_w * dt_s
        )
        self.state.requested_net_battery_energy_j += (
            requested_frame_battery_energy_j
        )
        self.state.net_battery_energy_j += actual_frame_battery_energy_j
        self.state.battery_energy_limit_difference_j += abs(
            requested_frame_battery_energy_j - actual_frame_battery_energy_j
        )

        self.state.time_s += dt_s
        self.state.position_m = min(
            self.state.position_m + distance_increment,
            self.route.length_m,
        )
        self.state.speed_mps = next_speed_mps

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
            frame_wheel_energy_j=float(frame_energy),
            traction_battery_power_w=powertrain_frame.traction_battery_power_w,
            regenerative_battery_power_w=powertrain_frame.regenerative_battery_power_w,
            auxiliary_power_w=powertrain_frame.auxiliary_power_w,
            battery_power_w=powertrain_frame.battery_power_w,
            friction_brake_power_w=powertrain_frame.friction_brake_power_w,
            unmet_wheel_power_w=powertrain_frame.unmet_wheel_power_w,
            requested_frame_battery_energy_j=requested_frame_battery_energy_j,
            actual_frame_battery_energy_j=actual_frame_battery_energy_j,
            actual_battery_power_w=actual_frame_battery_energy_j / dt_s,
            battery_soc_percent=self.battery.soc_percent,
            battery_remaining_energy_kwh=self.battery.remaining_energy_kwh,
            cumulative_positive_wheel_energy_wh=self.state.positive_wheel_energy_j / 3600.0,
            cumulative_braking_energy_available_wh=self.state.braking_energy_available_j / 3600.0,
            cumulative_signed_wheel_energy_wh=self.state.signed_wheel_energy_j / 3600.0,
            cumulative_traction_energy_from_battery_wh=(
                self.state.traction_energy_from_battery_j / 3600.0
            ),
            cumulative_regenerative_energy_to_battery_wh=(
                self.state.regenerative_energy_to_battery_j / 3600.0
            ),
            cumulative_auxiliary_energy_wh=self.state.auxiliary_energy_j / 3600.0,
            cumulative_requested_net_battery_energy_wh=(
                self.state.requested_net_battery_energy_j / 3600.0
            ),
            cumulative_net_battery_energy_wh=self.state.net_battery_energy_j / 3600.0,
            cumulative_battery_energy_limit_difference_wh=(
                self.state.battery_energy_limit_difference_j / 3600.0
            ),
        )
        self.history.append(result)
        return result

    def run_speed_profile(
        self,
        time_s: np.ndarray,
        speed_mps: np.ndarray,
    ) -> List[FrameResult]:
        # Run a complete speed profile, timestamps may have unequal spacing.

        time_s = np.asarray(time_s, dtype=float)
        speed_mps = np.asarray(speed_mps, dtype=float)
        if time_s.ndim != 1 or speed_mps.ndim != 1 or len(time_s) != len(speed_mps):
            raise ValueError("time_s and speed_mps must be equal-length 1-D arrays")
        if len(time_s) < 2 or not np.all(np.diff(time_s) > 0.0):
            raise ValueError("timestamps must be strictly increasing")

        # The first speed value defines the initial state; each later value is
        # the end-of-frame speed passed to step().
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
        """Convert the recorded dataclass results to NumPy arrays for plotting."""

        if not self.history:
            raise RuntimeError("no frames have been simulated")
        keys = asdict(self.history[0]).keys()
        return {
            key: np.array([getattr(frame, key) for frame in self.history])
            for key in keys
        }


def make_example_speed_profile(total_time_s: float, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Create an editable stop-and-go speed trace for the demonstration."""

    time_s = np.arange(0.0, total_time_s + dt_s, dt_s)

    # Key points are easier to edit than hundreds of frame values. np.interp
    # linearly fills the speed between them. Speeds are in m/s (3.6 km/h per m/s).
    key_time_s = np.array([0, 12, 55, 65, 80, 125, 135, 155, 205, 220, 260, 280])
    key_speed_mps = np.array([0, 12, 12, 0, 0, 16, 16, 0, 0, 10, 10, 0])
    speed_mps = np.interp(time_s, key_time_s, key_speed_mps)
    return time_s, speed_mps


def plot_results(simulator: VehicleDynamicsSimulator) -> None:
    data = simulator.history_as_arrays()
    time_s = data["time_s"]

    # Window 1: driving profile and mechanical wheel energy.
    trip_figure, trip_axes = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        sharex=True,
        num="Trip overview",
    )

    trip_axes[0].plot(
        time_s,
        data["speed_mps"] * 3.6,
        color="tab:blue",
    )
    trip_axes[0].set_ylabel("Speed [km/h]")
    trip_axes[0].set_title("Prescribed speed profile")

    trip_axes[1].plot(
        time_s,
        data["cumulative_positive_wheel_energy_wh"],
        label="Propulsion wheel energy",
        color="tab:red",
    )
    trip_axes[1].plot(
        time_s,
        data["cumulative_braking_energy_available_wh"],
        label="Mechanical braking energy available",
        color="tab:green",
    )
    trip_axes[1].set_ylabel("Wheel energy [Wh]")
    trip_axes[1].set_xlabel("Time [s]")
    trip_axes[1].set_title("Cumulative mechanical energy at the wheels")
    trip_axes[1].legend()

    for axis in trip_axes:
        axis.grid(True, alpha=0.3)
    trip_figure.suptitle("Trip overview")
    trip_figure.tight_layout()

    # Window 2: battery state and cumulative battery-side energy flows.
    battery_figure, battery_axes = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        sharex=True,
        num="Battery results",
    )

    battery_axes[0].plot(
        time_s,
        data["battery_remaining_energy_kwh"],
        label="Remaining battery energy",
        color="tab:blue",
        linewidth=2.0,
    )
    battery_axes[0].set_ylabel("Remaining energy [kWh]")
    battery_axes[0].set_title("Remaining battery energy and SOC")

    soc_axis = battery_axes[0].twinx()
    soc_axis.plot(
        time_s,
        data["battery_soc_percent"],
        label="SOC",
        color="tab:orange",
        linestyle="--",
    )
    soc_axis.set_ylabel("SOC [%]", color="tab:orange")
    soc_axis.tick_params(axis="y", labelcolor="tab:orange")
    energy_lines, energy_labels = battery_axes[0].get_legend_handles_labels()
    soc_lines, soc_labels = soc_axis.get_legend_handles_labels()
    battery_axes[0].legend(
        energy_lines + soc_lines,
        energy_labels + soc_labels,
        loc="best",
    )

    battery_axes[1].plot(
        time_s,
        data["cumulative_traction_energy_from_battery_wh"],
        label="Traction energy drawn",
        color="tab:red",
    )
    battery_axes[1].plot(
        time_s,
        data["cumulative_regenerative_energy_to_battery_wh"],
        label="Regenerative energy returned",
        color="tab:green",
    )
    battery_axes[1].plot(
        time_s,
        data["cumulative_auxiliary_energy_wh"],
        label="Auxiliary energy",
        color="tab:orange",
    )
    battery_axes[1].plot(
        time_s,
        data["cumulative_net_battery_energy_wh"],
        label="Net battery energy used",
        color="black",
        linewidth=2.0,
    )
    battery_axes[1].set_ylabel("Energy [Wh]")
    battery_axes[1].set_xlabel("Time [s]")
    battery_axes[1].set_title("Cumulative battery energy used")
    battery_axes[1].legend(ncol=2)

    for axis in battery_axes:
        axis.grid(True, alpha=0.3)
    battery_figure.suptitle("Battery results")
    battery_figure.tight_layout()

    # Window 3: detailed vehicle and powertrain quantities.
    dynamics_figure, dynamics_axes = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        sharex=True,
        num="Vehicle dynamics",
    )

    dynamics_axes[0].plot(time_s, data["rolling_force_n"], label="Rolling")
    dynamics_axes[0].plot(
        time_s,
        data["aerodynamic_force_n"],
        label="Aerodynamic",
    )
    dynamics_axes[0].plot(time_s, data["gradient_force_n"], label="Gradient")
    dynamics_axes[0].plot(
        time_s,
        data["inertial_force_n"],
        label="Inertial",
        alpha=0.8,
    )
    dynamics_axes[0].set_ylabel("Force [N]")
    dynamics_axes[0].set_title("Longitudinal force components")
    dynamics_axes[0].legend(ncol=2)

    dynamics_axes[1].plot(
        time_s,
        data["wheel_power_w"] / 1000.0,
        label="Wheel power",
        color="tab:purple",
    )
    dynamics_axes[1].plot(
        time_s,
        data["actual_battery_power_w"] / 1000.0,
        label="Actual battery power",
        color="black",
        alpha=0.8,
    )
    dynamics_axes[1].axhline(0.0, color="grey", linewidth=0.8)
    dynamics_axes[1].set_ylabel("Power [kW]")
    dynamics_axes[1].set_xlabel("Time [s]")
    dynamics_axes[1].set_title("Wheel and battery power")
    dynamics_axes[1].legend()

    for axis in dynamics_axes:
        axis.grid(True, alpha=0.3)
    dynamics_figure.suptitle("Vehicle dynamics and powertrain")
    dynamics_figure.tight_layout()

    # Window 4: elevation and road grade shown as separate plots.
    route_figure, route_axes = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        sharex=True,
        num="Route conditions",
    )

    route_axes[0].plot(
        time_s,
        data["elevation_m"],
        label="Elevation",
        color="tab:brown",
    )
    route_axes[0].set_ylabel("Elevation [m]")
    route_axes[0].set_title("Route elevation")
    route_axes[0].legend()

    route_axes[1].plot(
        time_s,
        data["grade_percent"],
        label="Road grade",
        color="tab:green",
    )
    route_axes[1].axhline(0.0, color="grey", linewidth=0.8)
    route_axes[1].set_ylabel("Road grade [%]")
    route_axes[1].set_xlabel("Time [s]")
    route_axes[1].set_title("Road grade")
    route_axes[1].legend()

    for axis in route_axes:
        axis.grid(True, alpha=0.3)
    route_figure.suptitle("Route conditions")
    route_figure.tight_layout()

    plt.show()


def main() -> None:
    # ------------------------- EDITABLE INPUTS -------------------------
    frame_time_s = 0.1  # 10 updates per second

    vehicle = VehicleParameters(
        mass_kg=1_300.0,
        rolling_resistance_coefficient=0.012,
        drag_coefficient=0.29,
        frontal_area_m2=2.2,
        wheel_radius_m=0.31,
        gear_ratio=9.0,
        rotor_inertia_kg_m2=0.03,
    )

    powertrain = Powertrain(
        drivetrain_efficiency=0.90,
        regen_efficiency=0.70,
        max_motor_power_w=100_000,
        max_regen_power_w=30_000,
        auxiliary_power_w=500,
    )

    battery = BatteryModel(
        soc_percent=80.0,
        soh_percent=95.0,
        capacity_kwh=42.0,
    )
    initial_soc_percent = battery.soc_percent
    initial_battery_energy_kwh = battery.remaining_energy_kwh

    # Positive wind is a headwind. This method demonstrates how ambient
    # temperature affects air density in the aerodynamic model.
    environment = Environment.from_temperature(
        temperature_c=20.0,
        wind_speed_mps=2.0,
    )

    route = Route(
        distance_m=np.array([0, 500, 1_000, 1_600, 2_200, 3_000, 4_000]),
        elevation_m=np.array([1_650, 1_655, 1_680, 1_700, 1_685, 1_710, 1_700]),
    )
    # ------------------------------------------------------------------

    time_s, speed_mps = make_example_speed_profile(
        total_time_s=280.0,
        dt_s=frame_time_s,
    )

    simulator = VehicleDynamicsSimulator(
        vehicle,
        powertrain,
        battery,
        route,
        environment,
    )
    simulator.run_speed_profile(time_s, speed_mps)

    state = simulator.state
    print("\nSimulation summary")
    print(f"Time simulated:                    {state.time_s:8.1f} s")
    print(f"Distance travelled:                {state.position_m / 1000:8.3f} km")
    print(f"SOH-adjusted battery capacity:     {battery.usable_capacity_kwh:8.3f} kWh")
    print(f"Initial battery energy:            {initial_battery_energy_kwh:8.3f} kWh")
    print(f"Remaining battery energy:          {battery.remaining_energy_kwh:8.3f} kWh")
    print(f"Initial SOC:                       {initial_soc_percent:8.3f} %")
    print(f"Final SOC:                         {battery.soc_percent:8.3f} %")
    print(f"Positive wheel energy required:    {state.positive_wheel_energy_j / 3.6e6:8.4f} kWh")
    print(f"Braking energy mechanically avail.:{state.braking_energy_available_j / 3.6e6:8.4f} kWh")
    print(f"Signed net wheel work:             {state.signed_wheel_energy_j / 3.6e6:8.4f} kWh")
    print(f"Traction energy from battery:      {state.traction_energy_from_battery_j / 3.6e6:8.4f} kWh")
    print(f"Regenerative energy returned:      {state.regenerative_energy_to_battery_j / 3.6e6:8.4f} kWh")
    print(f"Auxiliary energy used:              {state.auxiliary_energy_j / 3.6e6:8.4f} kWh")
    print(f"Net battery energy used:            {state.net_battery_energy_j / 3.6e6:8.4f} kWh")

    if state.position_m > 0.0:
        mechanical_consumption_wh_per_km = (
            state.positive_wheel_energy_j / 3600.0
        ) / (state.position_m / 1000.0)
        battery_consumption_wh_per_km = (
            state.net_battery_energy_j / 3600.0
        ) / (state.position_m / 1000.0)
        print(f"Mechanical propulsion consumption: {mechanical_consumption_wh_per_km:8.1f} Wh/km")
        print(f"Net battery consumption:            {battery_consumption_wh_per_km:8.1f} Wh/km")

    maximum_unmet_power_w = max(
        frame.unmet_wheel_power_w for frame in simulator.history
    )
    if maximum_unmet_power_w > 1.0:
        print(
            "WARNING: The requested speed profile exceeded the motor limit by "
            f"up to {maximum_unmet_power_w / 1000.0:.2f} kW at the wheels."
        )
    else:
        print("The requested speed profile remained within the motor power limit.")

    if state.battery_energy_limit_difference_j > 1.0:
        print(
            "WARNING: The battery reached an SOC boundary; some requested "
            "discharge or regeneration could not be accepted."
        )
    plot_results(simulator)


if __name__ == "__main__":
    main()