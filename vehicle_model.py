"""Frame-by-frame longitudinal vehicle dynamics starter model.

This is a *backward-facing* vehicle model: a speed profile tells the simulator
how fast the vehicle is travelling each frame, and the model calculates the
wheel force, wheel power and mechanical energy needed to follow that profile.

Scope of this version
---------------------
Included:
    - acceleration/inertial force
    - rolling resistance
    - aerodynamic drag, including headwind or tailwind
    - road-gradient force from a route elevation profile
    - rotating inertia reflected through a gear ratio
    - frame-by-frame integration of distance and wheel energy

Deliberately not included yet:
    - motor/inverter/drivetrain efficiency maps
    - regenerative-braking efficiency and limits
    - battery power, SOC or battery thermal behaviour
    - auxiliary electrical loads

Sign convention
---------------
Forward is positive. A positive road angle is uphill. Positive wind speed is a
headwind; negative wind speed is a tailwind. Positive wheel force/power means
the wheels must propel the vehicle. Negative wheel force/power means braking
is required and mechanical energy is available to a future regenerative-
braking model.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


G = 9.81  # gravitational acceleration [m/s^2]


@dataclass
class VehicleParameters:
    """Parameters that remain constant for one vehicle."""

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
        """Translational mass plus motor/rotor inertia reflected to the road.

        From rotational kinetic energy:
            0.5*J*omega_motor^2 = 0.5*m_equivalent*v^2
            omega_motor = gear_ratio*v/wheel_radius

        Therefore m_equivalent = J*gear_ratio^2/wheel_radius^2.
        Gearbox efficiency is intentionally not included here; losses belong
        in the later powertrain model, not in the physical inertia itself.
        """

        reflected_rotating_mass = (
            self.rotor_inertia_kg_m2
            * self.gear_ratio**2
            / self.wheel_radius_m**2
        )
        return self.mass_kg + reflected_rotating_mass


@dataclass
class Environment:
    """Environmental values used in a frame.

    Positive wind_speed_mps is a headwind. A negative value is a tailwind.
    Air density may be entered directly or estimated from temperature and
    pressure with Environment.from_temperature().
    """

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
    """Route elevation sampled at monotonically increasing distances.

    `distance_m` is route chainage (distance from the route start), not vehicle
    position in a global coordinate system. Between points, elevation and road
    slope are linearly interpolated.
    """

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

        # np.gradient gives a smoother slope at internal sampled points than a
        # one-sided segment calculation. For ordinary road gradients, treating
        # elevation change / route distance as tan(theta) is a good approximation.
        self._slope = np.gradient(self.elevation_m, self.distance_m)
        self._angle_rad = np.arctan(self._slope)

    @property
    def length_m(self) -> float:
        return float(self.distance_m[-1])

    def elevation_at(self, position_m: float) -> float:
        return float(np.interp(position_m, self.distance_m, self.elevation_m))

    def angle_at(self, position_m: float) -> float:
        return float(np.interp(position_m, self.distance_m, self._angle_rad))

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


@dataclass
class FrameResult:
    """Calculated values for one frame."""

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
    cumulative_positive_wheel_energy_wh: float
    cumulative_braking_energy_available_wh: float
    cumulative_signed_wheel_energy_wh: float


class VehicleDynamicsSimulator:
    """Discrete-time vehicle model updated once per simulation frame."""

    def __init__(
        self,
        vehicle: VehicleParameters,
        route: Route,
        environment: Environment,
        initial_speed_mps: float = 0.0,
    ) -> None:
        vehicle.validate()
        if initial_speed_mps < 0.0:
            raise ValueError("this starter model supports forward speed only")

        self.vehicle = vehicle
        self.route = route
        self.environment = environment
        self.state = SimulationState(speed_mps=initial_speed_mps)
        self.history: List[FrameResult] = []

    def step(self, next_speed_mps: float, dt_s: float) -> FrameResult:
        """Advance the model by one frame.

        This backward-facing step receives the measured or commanded speed at
        the *end* of the frame. Acceleration and distance use finite differences
        and the trapezoidal rule, respectively.

        C++ analogy: this method mutates the object's `state` and returns a
        struct-like FrameResult containing the outputs for this update.
        """

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

        rolling_force = p.rolling_resistance_coefficient * p.mass_kg * G * np.cos(theta)

        # Positive relative-air speed produces drag opposing forward motion.
        # If a tailwind is faster than the vehicle, this signed expression
        # correctly gives a forward aerodynamic force.
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

        if frame_energy >= 0.0:
            self.state.positive_wheel_energy_j += frame_energy
        else:
            self.state.braking_energy_available_j += -frame_energy
        self.state.signed_wheel_energy_j += frame_energy

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
            cumulative_positive_wheel_energy_wh=self.state.positive_wheel_energy_j / 3600.0,
            cumulative_braking_energy_available_wh=self.state.braking_energy_available_j / 3600.0,
            cumulative_signed_wheel_energy_wh=self.state.signed_wheel_energy_j / 3600.0,
        )
        self.history.append(result)
        return result

    def run_speed_profile(
        self,
        time_s: np.ndarray,
        speed_mps: np.ndarray,
    ) -> List[FrameResult]:
        """Run a complete speed trace; timestamps may have unequal spacing."""

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

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    axes[0].plot(time_s, data["speed_mps"] * 3.6, color="tab:blue")
    axes[0].set_ylabel("Speed [km/h]")

    axes[1].plot(time_s, data["rolling_force_n"], label="Rolling")
    axes[1].plot(time_s, data["aerodynamic_force_n"], label="Aerodynamic")
    axes[1].plot(time_s, data["gradient_force_n"], label="Gradient")
    axes[1].plot(time_s, data["inertial_force_n"], label="Inertial", alpha=0.8)
    axes[1].set_ylabel("Force [N]")
    axes[1].legend(ncol=2)

    axes[2].plot(time_s, data["wheel_power_w"] / 1000.0, color="tab:purple")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_ylabel("Wheel power [kW]")

    axes[3].plot(
        time_s,
        data["cumulative_positive_wheel_energy_wh"],
        label="Propulsion energy required",
        color="tab:red",
    )
    axes[3].plot(
        time_s,
        data["cumulative_braking_energy_available_wh"],
        label="Mechanical braking energy available",
        color="tab:green",
    )
    axes[3].set_ylabel("Energy [Wh]")
    axes[3].set_xlabel("Time [s]")
    axes[3].legend()

    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.suptitle("Frame-by-frame longitudinal vehicle dynamics")
    fig.tight_layout()
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

    # Positive wind is a headwind. This method also demonstrates how ambient
    # temperature can affect air density without modelling the battery yet.
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

    simulator = VehicleDynamicsSimulator(vehicle, route, environment)
    simulator.run_speed_profile(time_s, speed_mps)

    state = simulator.state
    print("\nSimulation summary")
    print(f"Time simulated:                    {state.time_s:8.1f} s")
    print(f"Distance travelled:                {state.position_m / 1000:8.3f} km")
    print(f"Positive wheel energy required:    {state.positive_wheel_energy_j / 3.6e6:8.4f} kWh")
    print(f"Braking energy mechanically avail.:{state.braking_energy_available_j / 3.6e6:8.4f} kWh")
    print(f"Signed net wheel work:             {state.signed_wheel_energy_j / 3.6e6:8.4f} kWh")

    if state.position_m > 0.0:
        consumption_wh_per_km = (
            state.positive_wheel_energy_j / 3600.0
        ) / (state.position_m / 1000.0)
        print(f"Mechanical propulsion consumption: {consumption_wh_per_km:8.1f} Wh/km")

    print("\nImportant: these are wheel-level mechanical values, not battery energy.")
    plot_results(simulator)


if __name__ == "__main__":
    main()
