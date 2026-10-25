"""Matplotlib views for simulation results."""

import matplotlib.pyplot as plt

from .physical_model import VehicleDynamicsSimulator


def plot_results(
    simulator: VehicleDynamicsSimulator,
    *,
    block: bool = True,
) -> None:
    data = simulator.history_as_arrays()
    time_s = data["time_s"]
    position_km = data["position_m"] / 1000.0

    trip_figure, trip_axes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, num="Trip overview"
    )
    trip_axes[0].plot(time_s, data["speed_mps"] * 3.6, color="tab:blue")
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
    _finish_figure(trip_figure, trip_axes, "Trip overview")

    battery_figure, battery_axes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, num="Battery results"
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
    energy_lines, energy_labels = battery_axes[0].get_legend_handles_labels()
    soc_lines, soc_labels = soc_axis.get_legend_handles_labels()
    battery_axes[0].legend(
        energy_lines + soc_lines,
        energy_labels + soc_labels,
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
    _finish_figure(battery_figure, battery_axes, "Battery results")

    dynamics_figure, dynamics_axes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, num="Vehicle dynamics"
    )
    dynamics_axes[0].plot(time_s, data["rolling_force_n"], label="Rolling")
    dynamics_axes[0].plot(
        time_s, data["aerodynamic_force_n"], label="Aerodynamic"
    )
    dynamics_axes[0].plot(time_s, data["gradient_force_n"], label="Gradient")
    dynamics_axes[0].plot(time_s, data["inertial_force_n"], label="Inertial")
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
    )
    dynamics_axes[1].axhline(0.0, color="grey", linewidth=0.8)
    dynamics_axes[1].set_ylabel("Power [kW]")
    dynamics_axes[1].set_xlabel("Time [s]")
    dynamics_axes[1].set_title("Wheel and battery power")
    dynamics_axes[1].legend()
    _finish_figure(
        dynamics_figure,
        dynamics_axes,
        "Vehicle dynamics and powertrain",
    )

    route_figure, route_axes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, num="Route conditions"
    )
    route_axes[0].plot(
        position_km, data["elevation_m"], label="Elevation", color="tab:brown"
    )
    route_axes[0].set_ylabel("Elevation [m]")
    route_axes[0].set_title("Route elevation")
    route_axes[0].legend()
    route_axes[1].plot(
        position_km, data["grade_percent"], label="Road grade", color="tab:green"
    )
    route_axes[1].axhline(0.0, color="grey", linewidth=0.8)
    route_axes[1].set_ylabel("Road grade [%]")
    route_axes[1].set_xlabel("Distance [km]")
    route_axes[1].set_title("Road grade")
    route_axes[1].legend()
    _finish_figure(route_figure, route_axes, "Route conditions")

    range_figure, range_axes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, num="Range estimate"
    )
    range_axes[0].plot(
        position_km,
        data["average_battery_consumption_wh_per_km"],
        color="tab:grey",
        alpha=0.7,
        label="Raw observed consumption",
    )
    range_axes[0].plot(
        position_km,
        data["range_estimate_consumption_wh_per_km"],
        color="tab:purple",
        linewidth=2.0,
        label="Prior-weighted consumption estimate",
    )
    range_axes[0].set_ylabel("Consumption [Wh/km]")
    range_axes[0].set_title("Observed and stabilized energy consumption")
    range_axes[0].legend()
    range_axes[1].plot(
        position_km,
        data["estimated_remaining_range_km"],
        color="tab:blue",
        label="Estimated remaining range",
    )
    range_axes[1].plot(
        position_km,
        data["estimated_total_range_km"],
        color="tab:orange",
        linestyle="--",
        label="Estimated total range from trip start",
    )
    range_axes[1].set_ylabel("Range [km]")
    range_axes[1].set_xlabel("Distance travelled [km]")
    range_axes[1].set_title("Consumption-based range estimate")
    range_axes[1].legend()
    _finish_figure(range_figure, range_axes, "Range estimation")
    plt.show(block=block)


def _finish_figure(figure, axes, title: str) -> None:
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.suptitle(title)
    figure.tight_layout()
