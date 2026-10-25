"""Tkinter configuration window for the EV range simulation."""

from __future__ import annotations

import os
os.environ["TOMTOM_API_KEY"] = "WcphEXuxcztIXfHgUQ3X1gVLEqOjxKlJ"
import traceback
import tkinter as tk
from tkinter import messagebox, ttk
import urllib.parse  
import urllib.request 
import json      
import tkintermapview  
from .api_clients import TomTomRoutingClient

import matplotlib.pyplot as plt

from .main import SimulationConfig, format_summary, run_simulation
from .physical_model import BatteryModel, Powertrain, VehicleParameters
from .plotting import plot_results


class SimulationApp:
    """Collect configuration values and launch the existing simulation."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("EV Range Estimation Simulator")
        self.root.geometry("860x760")
        self.root.minsize(760, 680)

        self.entries: dict[str, ttk.Entry] = {}
        self.live_route_var = tk.BooleanVar()
        self.live_weather_var = tk.BooleanVar()
        self.traffic_var = tk.BooleanVar()
        self.status_var = tk.StringVar(value="Ready")
        self.api_key_status_var = tk.StringVar()

        self._build_window()
        self._load_defaults()
        self._update_api_controls()

    def _build_window(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        heading = ttk.Frame(self.root, padding=(14, 12, 14, 4))
        heading.grid(row=0, column=0, sticky="ew")
        heading.columnconfigure(0, weight=1)
        ttk.Label(
            heading,
            text="EV Range Estimation Simulator",
            font=("TkDefaultFont", 17, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            heading,
            text="Configure the vehicle, route and estimator, then run the model.",
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=1, column=0, padx=14, pady=8, sticky="nsew")

        simulation_tab = self._new_tab(self.notebook, "Simulation")

        # Create the Route Map tab FIRST
        self.map_tab = self._new_tab(self.notebook, "Route Map")

        # Initialise the map widget
        self.map_widget = tkintermapview.TkinterMapView(
            self.map_tab,
            width=600,
            height=400,
            corner_radius=0
        )

        # Configure TomTom map tiles
        api_key = os.getenv("TOMTOM_API_KEY")

        if api_key:
            tile_server = (
                "https://api.tomtom.com/map/1/tile/basic/main/"
                "{z}/{x}/{y}.png?key=" + api_key
            )

            self.map_widget.set_tile_server(
                tile_server,
                max_zoom=22
            )

        # Display the map
        self.map_widget.grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="nsew"
        )

        self.map_tab.columnconfigure(0, weight=1)
        self.map_tab.rowconfigure(0, weight=1)

        # Centre the map on Johannesburg
        self.map_widget.set_position(-26.2041, 28.0473)
        self.map_widget.set_zoom(10)

        # Create the remaining tabs
        route_tab = self._new_tab(self.notebook, "Route & APIs")
        vehicle_tab = self._new_tab(self.notebook, "Vehicle")
        
        ttk.Label(vehicle_tab, text="Compare Petrol Car (L/100km):").grid(row=15, column=0, padx=10, pady=5, sticky="w")
        self.petrol_entry = ttk.Entry(vehicle_tab, width=10)
        self.petrol_entry.grid(row=15, column=1, padx=10, pady=5, sticky="w")
        self.petrol_entry.insert(0, "8.0")

        ttk.Label(vehicle_tab, text="Petrol Price (R/L):").grid(row=16, column=0, padx=10, pady=5, sticky="w")
        self.petrol_price_entry = ttk.Entry(vehicle_tab, width=10)
        self.petrol_price_entry.grid(row=16, column=1, padx=10, pady=5, sticky="w")
        self.petrol_price_entry.insert(0, "30.25") # Defaulting to Gauteng Oct 2026 prices!

        ttk.Label(vehicle_tab, text="EV Charging Price (R/kWh):").grid(row=17, column=0, padx=10, pady=5, sticky="w")
        self.ev_price_entry = ttk.Entry(vehicle_tab, width=10)
        self.ev_price_entry.grid(row=17, column=1, padx=10, pady=5, sticky="w")
        self.ev_price_entry.insert(0, "2.50")

        powertrain_tab = self._new_tab(self.notebook, "Powertrain")
        battery_tab = self._new_tab(self.notebook, "Battery & environment")
        range_tab = self._new_tab(self.notebook, "Range estimate")

        self._add_entry(simulation_tab, 0, "frame_time_s", "Frame time [s]")
        self._add_entry(
            simulation_tab,
            1,
            "offline_total_time_s",
            "Offline speed-profile duration [s]",
        )
        self._add_entry(
            simulation_tab,
            2,
            "maximum_acceleration_mps2",
            "Maximum acceleration [m/s²]",
        )
        self._add_entry(
            simulation_tab,
            3,
            "maximum_deceleration_mps2",
            "Maximum deceleration [m/s²]",
        )
        self._add_entry(
            simulation_tab,
            4,
            "driver_speed_factor",
            "Driver speed factor [0–1]",
        )
        ttk.Label(
            simulation_tab,
            text=(
                "Acceleration, deceleration and driver factor apply to the "
                "live route-speed profile."
            ),
            wraplength=620,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(16, 0))

        self.live_route_check = ttk.Checkbutton(
            route_tab,
            text="Use TomTom and Open-Elevation for a live route",
            variable=self.live_route_var,
            command=self._update_api_controls,
        )
        self.live_route_check.grid(row=0, column=0, columnspan=2, sticky="w")
        self.live_weather_check = ttk.Checkbutton(
            route_tab,
            text="Use current Open-Meteo weather (defaults are the fallback)",
            variable=self.live_weather_var,
        )
        self.live_weather_check.grid(row=1, column=0, columnspan=2, sticky="w")
        self.traffic_check = ttk.Checkbutton(
            route_tab,
            text="Use TomTom traffic information",
            variable=self.traffic_var,
        )
        self.traffic_check.grid(row=2, column=0, columnspan=2, sticky="w")
        self._add_entry(route_tab, 3, "origin_address", "Origin address")
        self._add_entry(route_tab, 4, "destination_address", "Destination address")
        ttk.Separator(route_tab).grid(
            row=7,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=12,
        )
        ttk.Label(route_tab, textvariable=self.api_key_status_var).grid(
            row=8,
            column=0,
            columnspan=2,
            sticky="w",
        )

        vehicle_fields = (
            ("mass_kg", "Vehicle mass [kg]"),
            ("rolling_resistance", "Rolling-resistance coefficient"),
            ("drag_coefficient", "Aerodynamic drag coefficient"),
            ("frontal_area_m2", "Frontal area [m²]"),
            ("wheel_radius_m", "Wheel radius [m]"),
            ("gear_ratio", "Gear ratio"),
            ("rotor_inertia", "Motor rotor inertia [kg·m²]"),
        )
        for row, (key, label) in enumerate(vehicle_fields):
            self._add_entry(vehicle_tab, row, key, label)

        powertrain_fields = (
            ("drivetrain_efficiency", "Drivetrain efficiency [0–1]"),
            ("regen_efficiency", "Regenerative-braking efficiency [0–1]"),
            ("max_motor_power_kw", "Maximum motor input power [kW]"),
            ("max_regen_power_kw", "Maximum battery regen power [kW]"),
            ("auxiliary_power_w", "Auxiliary power [W]"),
        )
        for row, (key, label) in enumerate(powertrain_fields):
            self._add_entry(powertrain_tab, row, key, label)

        battery_environment_fields = (
            ("soc_percent", "Initial SOC [%]"),
            ("soh_percent", "Battery SOH [%]"),
            ("capacity_kwh", "Nominal battery capacity [kWh]"),
            ("temperature_c", "Default temperature [°C]"),
            ("headwind_mps", "Default headwind [m/s]"),
            ("pressure_kpa", "Default atmospheric pressure [kPa]"),
        )
        for row, (key, label) in enumerate(battery_environment_fields):
            self._add_entry(battery_tab, row, key, label)
        ttk.Label(
            battery_tab,
            text=(
                "Default weather is used for offline runs and whenever the "
                "live weather service is unavailable. Negative wind is a tailwind."
            ),
            wraplength=620,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(16, 0))

        range_fields = (
            (
                "minimum_range_distance_m",
                "Distance before showing raw consumption [m]",
            ),
            ("initial_consumption", "Initial consumption assumption [Wh/km]"),
            ("prior_distance_km", "Consumption prior distance [km]"),
            ("minimum_consumption", "Minimum consumption guard [Wh/km]"),
        )
        for row, (key, label) in enumerate(range_fields):
            self._add_entry(range_tab, row, key, label)
        ttk.Label(
            range_tab,
            text=(
                "The prior distance controls how quickly measured trip "
                "consumption replaces the initial assumption."
            ),
            wraplength=620,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(16, 0))

        output_frame = ttk.LabelFrame(
            self.root,
            text="Simulation output",
            padding=8,
        )
        output_frame.grid(row=2, column=0, padx=14, pady=(0, 8), sticky="ew")
        output_frame.columnconfigure(0, weight=1)
        self.output_text = tk.Text(
            output_frame,
            height=10,
            wrap="word",
            font=("TkFixedFont", 10),
        )
        self.output_text.grid(row=0, column=0, sticky="ew")
        output_scrollbar = ttk.Scrollbar(
            output_frame,
            orient="vertical",
            command=self.output_text.yview,
        )
        output_scrollbar.grid(row=0, column=1, sticky="ns")
        self.output_text.configure(yscrollcommand=output_scrollbar.set)

        controls = ttk.Frame(self.root, padding=(14, 0, 14, 12))
        controls.grid(row=3, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)
        ttk.Label(controls, textvariable=self.status_var).grid(
            row=0,
            column=0,
            sticky="w",
        )
        self.run_button = ttk.Button(
            controls,
            text="Run simulation",
            command=self._run,
        )
        self.run_button.grid(row=0, column=1, padx=(8, 0))
        ttk.Button(
            controls,
            text="Reset defaults",
            command=self._load_defaults,
        ).grid(row=0, column=2, padx=(8, 0))
        ttk.Button(
            controls,
            text="Close",
            command=self.root.destroy,
        ).grid(row=0, column=3, padx=(8, 0))

    @staticmethod
    def _new_tab(notebook: ttk.Notebook, title: str) -> ttk.Frame:
        tab = ttk.Frame(notebook, padding=18)
        tab.columnconfigure(1, weight=1)
        notebook.add(tab, text=title)
        return tab

    def _add_entry(
        self,
        parent: ttk.Frame,
        row: int,
        key: str,
        label: str,
    ) -> None:
        ttk.Label(parent, text=label).grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 16),
            pady=5,
        )
        entry = ttk.Entry(parent, width=24)
        entry.grid(row=row, column=1, sticky="ew", pady=5)
        self.entries[key] = entry

    def _load_defaults(self) -> None:
        config = SimulationConfig()
        values = {
            "frame_time_s": config.frame_time_s,
            "offline_total_time_s": config.offline_total_time_s,
            "maximum_acceleration_mps2": config.maximum_acceleration_mps2,
            "maximum_deceleration_mps2": config.maximum_deceleration_mps2,
            "driver_speed_factor": config.driver_speed_factor,
            "origin_address": "Johannesburg, South Africa",
            "destination_address": "Pretoria, South Africa",
            "mass_kg": config.vehicle.mass_kg,
            "rolling_resistance": config.vehicle.rolling_resistance_coefficient,
            "drag_coefficient": config.vehicle.drag_coefficient,
            "frontal_area_m2": config.vehicle.frontal_area_m2,
            "wheel_radius_m": config.vehicle.wheel_radius_m,
            "gear_ratio": config.vehicle.gear_ratio,
            "rotor_inertia": config.vehicle.rotor_inertia_kg_m2,
            "drivetrain_efficiency": config.powertrain.drivetrain_efficiency,
            "regen_efficiency": config.powertrain.regen_efficiency,
            "max_motor_power_kw": config.powertrain.max_motor_power_w / 1000.0,
            "max_regen_power_kw": config.powertrain.max_regen_power_w / 1000.0,
            "auxiliary_power_w": config.powertrain.auxiliary_power_w,
            "soc_percent": config.battery.soc_percent,
            "soh_percent": config.battery.soh_percent,
            "capacity_kwh": config.battery.capacity_kwh,
            "temperature_c": config.default_temperature_c,
            "headwind_mps": config.default_headwind_mps,
            "pressure_kpa": config.default_pressure_pa / 1000.0,
            "minimum_range_distance_m": config.minimum_range_estimation_distance_m,
            "initial_consumption": config.initial_consumption_estimate_wh_per_km,
            "prior_distance_km": config.consumption_prior_distance_km,
            "minimum_consumption": config.minimum_consumption_estimate_wh_per_km,
        }
        for key, value in values.items():
            entry = self.entries[key]
            entry.configure(state="normal")
            entry.delete(0, tk.END)
            entry.insert(0, str(value))
        self.live_route_var.set(config.use_live_route)
        self.live_weather_var.set(config.use_live_weather)
        self.traffic_var.set(config.use_traffic)
        self.output_text.delete("1.0", tk.END)
        self.status_var.set("Ready")
        self._update_api_key_status()
        self._update_api_controls()

    def _update_api_key_status(self) -> None:
        if os.getenv("TOMTOM_API_KEY"):
            self.api_key_status_var.set("TomTom API key: detected")
        else:
            self.api_key_status_var.set(
                "TomTom API key: not detected (set TOMTOM_API_KEY before launch)"
            )

    def _update_api_controls(self) -> None:
        live_route = self.live_route_var.get()
        state = "normal" if live_route else "disabled"
        for key in (
                "origin_address",
                "destination_address",
            ):
            self.entries[key].configure(state=state)
        self.traffic_check.configure(state=state)
        self.live_weather_check.configure(state=state)
        if not live_route:
            self.live_weather_var.set(False)

    def _float(self, key: str, label: str) -> float:
        value = self.entries[key].get().strip()
        try:
            return float(value)
        except ValueError as error:
            raise ValueError(f"{label} must be a number") from error


    def _geocode_address(self, address: str) -> tuple[float, float]:
        api_key = os.getenv("TOMTOM_API_KEY")
        if not api_key:
            raise ValueError("TomTom API key not found. Please set TOMTOM_API_KEY.")
            
        encoded_address = urllib.parse.quote(address.strip())
        url = f"https://api.tomtom.com/search/2/geocode/{encoded_address}.json?key={api_key}&limit=1"
        
        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
                if not data.get("results"):
                    raise ValueError(f"Could not find coordinates for: {address}")
                position = data["results"][0]["position"]
                return float(position["lat"]), float(position["lon"])
        except Exception as e:
            raise ValueError(f"Geocoding failed for '{address}': {e}")

    def _build_config(self) -> SimulationConfig:
        vehicle = VehicleParameters(
            mass_kg=self._float("mass_kg", "Vehicle mass"),
            rolling_resistance_coefficient=self._float(
                "rolling_resistance",
                "Rolling-resistance coefficient",
            ),
            drag_coefficient=self._float("drag_coefficient", "Drag coefficient"),
            frontal_area_m2=self._float("frontal_area_m2", "Frontal area"),
            wheel_radius_m=self._float("wheel_radius_m", "Wheel radius"),
            gear_ratio=self._float("gear_ratio", "Gear ratio"),
            rotor_inertia_kg_m2=self._float("rotor_inertia", "Rotor inertia"),
        )
        powertrain = Powertrain(
            drivetrain_efficiency=self._float(
                "drivetrain_efficiency",
                "Drivetrain efficiency",
            ),
            regen_efficiency=self._float(
                "regen_efficiency",
                "Regenerative-braking efficiency",
            ),
            max_motor_power_w=(
                self._float("max_motor_power_kw", "Maximum motor power") * 1000.0
            ),
            max_regen_power_w=(
                self._float("max_regen_power_kw", "Maximum regen power") * 1000.0
            ),
            auxiliary_power_w=self._float(
                "auxiliary_power_w",
                "Auxiliary power",
            ),
        )
        battery = BatteryModel(
            soc_percent=self._float("soc_percent", "SOC"),
            soh_percent=self._float("soh_percent", "SOH"),
            capacity_kwh=self._float("capacity_kwh", "Battery capacity"),
        )

        if self.live_route_var.get():
            origin_lat, origin_lon = self._geocode_address(self.entries["origin_address"].get())
            dest_lat, dest_lon = self._geocode_address(self.entries["destination_address"].get())
        else:
            origin_lat, origin_lon = 0.0, 0.0 
            dest_lat, dest_lon = 0.0, 0.0

            
        return SimulationConfig(
            frame_time_s=self._float("frame_time_s", "Frame time"),
            offline_total_time_s=self._float(
                "offline_total_time_s",
                "Offline profile duration",
            ),
            use_live_route=self.live_route_var.get(),
            use_live_weather=self.live_weather_var.get(),
            use_traffic=self.traffic_var.get(),
            origin_latitude=origin_lat,
            origin_longitude=origin_lon,
            destination_latitude=dest_lat,
            destination_longitude=dest_lon,
            maximum_acceleration_mps2=self._float(
                "maximum_acceleration_mps2",
                "Maximum acceleration",
            ),
            maximum_deceleration_mps2=self._float(
                "maximum_deceleration_mps2",
                "Maximum deceleration",
            ),
            driver_speed_factor=self._float(
                "driver_speed_factor",
                "Driver speed factor",
            ),
            default_temperature_c=self._float("temperature_c", "Temperature"),
            default_headwind_mps=self._float("headwind_mps", "Headwind"),
            default_pressure_pa=(
                self._float("pressure_kpa", "Atmospheric pressure") * 1000.0
            ),
            minimum_range_estimation_distance_m=self._float(
                "minimum_range_distance_m",
                "Minimum range-estimation distance",
            ),
            initial_consumption_estimate_wh_per_km=self._float(
                "initial_consumption",
                "Initial consumption assumption",
            ),
            consumption_prior_distance_km=self._float(
                "prior_distance_km",
                "Consumption prior distance",
            ),
            minimum_consumption_estimate_wh_per_km=self._float(
                "minimum_consumption",
                "Minimum consumption",
            ),
            vehicle=vehicle,
            powertrain=powertrain,
            battery=battery,
        )

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)
        self.root.update_idletasks()

    def update_map_with_route(self, route_points):
        """Draws the route on the map tab when simulation finishes."""
        self.map_widget.delete_all_path()
        self.map_widget.delete_all_marker()

        # Extract coordinates from the points
        path_coordinates = [(point.latitude, point.longitude) for point in route_points]

        # Draw line
        self.map_widget.set_path(path_coordinates, color="blue", width=3)

        # Add Start/End pins
        start_coord = path_coordinates[0]
        end_coord = path_coordinates[-1]
        self.map_widget.set_marker(start_coord[0], start_coord[1], text="Origin")
        self.map_widget.set_marker(end_coord[0], end_coord[1], text="Destination")

        # Auto-zoom to fit the route
        lats = [p[0] for p in path_coordinates]
        lons = [p[1] for p in path_coordinates]
        self.map_widget.fit_bounding_box(
            (max(lats), min(lons)), 
            (min(lats), max(lons))
        )

    def _run(self) -> None:
        self.run_button.configure(state="disabled")
        self.output_text.delete("1.0", tk.END)
        self._set_status("Validating inputs...")
        try:
            config = self._build_config()
            outcome = run_simulation(config, self._set_status)
            if hasattr(TomTomRoutingClient, 'last_route_points'):
                self.update_map_with_route(TomTomRoutingClient.last_route_points)
                self.notebook.select(self.map_tab)
            output = "\n".join(outcome.messages)
            output += "\n\n" + format_summary(outcome)

            try:
                dist_m = getattr(outcome.simulator.route, 'length_m', 0)
                if not dist_m:
                    dist_m = getattr(outcome.simulator.route, 'distance_m', 0)
                
                if dist_m > 0:
                    distance_km = dist_m / 1000.0
                    
                    # 1. Read all three user inputs safely
                    try:
                        user_l100km = float(self.petrol_entry.get())
                    except ValueError:
                        user_l100km = 8.0 
                        
                    try:
                        petrol_price = float(self.petrol_price_entry.get())
                    except ValueError:
                        petrol_price = 30.25
                        
                    try:
                        ev_price = float(self.ev_price_entry.get())
                    except ValueError:
                        ev_price = 2.50
                    
                    # 2. Extract the EXACT simulated energy from your summary text!
                    import re
                    match = re.search(r"Net battery energy used:\s*([0-9.]+)\s*kWh", output)
                    if match:
                        actual_ev_kwh = float(match.group(1))
                    else:
                        actual_ev_kwh = (distance_km / 100) * 15.0 # Fallback
                    
                    # 3. Calculate costs based on exact usage and user prices
                    petrol_liters_used = (distance_km / 100) * user_l100km
                    petrol_cost = petrol_liters_used * petrol_price
                    
                    ev_cost = actual_ev_kwh * ev_price
                    
                    savings = petrol_cost - ev_cost
                    
                    # 4. Attach to the bottom of the summary
                    output += f"REAL WORLD SAVINGS"
                    output += f"Trip Distance             : {distance_km:.1f} km\n"
                    output += f"Petrol Car ({user_l100km:.1f} L/100km) : R {petrol_cost:.2f} (at R{petrol_price:.2f}/L)\n"
                    output += f"Your EV ({actual_ev_kwh:.1f} kWh used)   : R {ev_cost:.2f} (at R{ev_price:.2f}/kWh)\n"
                    output += f"Total Saved on this trip  : R {savings:.2f}\n"
                    output += f"--------------------------------------\n"
            except Exception as e:
                print("Error calculating savings:", e)

            self.output_text.insert("1.0", output)
            self.output_text.see("1.0")

            plt.close("all")
            plot_results(outcome.simulator, block=False)
            self._set_status("Simulation complete; plot windows opened.")
        except Exception as error:
            traceback.print_exc()
            self._set_status("Simulation failed.")
            messagebox.showerror("Simulation error", str(error), parent=self.root)
        finally:
            self.run_button.configure(state="normal")


def main() -> None:
    root = tk.Tk()
    SimulationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
