# EV Remaining Range Estimation Model

This project estimates electric vehicle battery energy use and remaining range from a known battery state of charge and a planned route. It combines:

- A longitudinal vehicle dynamics model.
- Battery, regenerative braking and remaining range calculations.
- TomTom routing, Open Elevation and Open Meteo data.
- Random Forest Regression (RFR) residual correction.
- A Tkinter interface with an interactive map.

The pre trip RFR uses only route, weather and user inputs that are available before departure. A real time model is also included for comparison using onboard vehicle measurements.

## Project structure

```text
EV_Range_Model/
├── Launcher.py
├── clean_ev_data.py
├── build_physics_dataset.py
├── train_rfr.py
├── master_ev_trips_dataset.csv       # Raw input, if cleaning the data
├── cleaned_ev_trips.csv              # Input to build_physics_dataset.py
├── physics_enriched_ev_trips.csv     # Input to train_rfr.py
├── ev_range_project/
│   ├── api_clients.py
│   ├── driving_profiles.py
│   ├── main.py
│   ├── physical_model.py
│   ├── plotting.py
│   ├── route_builder.py
│   ├── user_interface.py
│   └── requirements.txt
└── rfr_results_planner/
    ├── trained_rfr.joblib
    ├── rfr_metrics.txt
    ├── rfr_trip_metrics.csv
    ├── rfr_feature_importance.csv
    └── rfr_test_predictions.csv
```

Place the required CSV files in the main `EV_Range_Model` folder beside the three data-processing scripts:

- `master_ev_trips_dataset.csv` is read by `clean_ev_data.py`.
- `cleaned_ev_trips.csv` is read by `build_physics_dataset.py`.
- `physics_enriched_ev_trips.csv` is read by `train_rfr.py`.
- The trained planner model and its metrics remain inside `rfr_results_planner/`.

Run all commands from the main `EV_Range_Model` folder. Large datasets and generated results may be supplied separately.

## Setup

Python 3.11 or newer is recommended. From the `EV_Range_Model` folder, run:

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ev_range_project/requirements.txt
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r ev_range_project/requirements.txt
```

Test Tkinter with:

```bash
python -m tkinter
```

## Run the interface

```bash
python Launcher.py
```

For the first run, disable the live route and live weather options. The program will use its offline demonstration route and default environmental values.

The interface allows the user to select a route and vehicle parameters, run the physics model, apply the pre trip RFR correction and view numerical and graphical results.

## Optional TomTom route data

Save the TomTom key as an environment variable. Do not place the key in the code or shared ZIP.

macOS or Linux:

```bash
export TOMTOM_API_KEY="your_key_here"
python Launcher.py
```

Windows PowerShell:

```powershell
$env:TOMTOM_API_KEY="your_key_here"
python Launcher.py
```

Open Meteo and the public Open Elevation service do not require API keys.

## Reproduce the machine learning results

Run these commands from the folder containing the three Python scripts.

### 1. Clean the trip data

```bash
python clean_ev_data.py master_ev_trips_dataset.csv --sample-period-s 0.1
```

Output: `cleaned_ev_trips.csv`

### 2. Add the physics calculations

```bash
python build_physics_dataset.py cleaned_ev_trips.csv
```

Output: `physics_enriched_ev_trips.csv`

### 3. Train the pre trip RFR

```bash
python train_rfr.py physics_enriched_ev_trips.csv \
    --feature-set planner \
    --output-directory rfr_results_planner
```

The model learns the residual:

```text
measured battery power - physics battery power
```

Important outputs include:

```text
rfr_results_planner/
├── trained_rfr.joblib
├── rfr_metrics.txt
├── rfr_trip_metrics.csv
├── rfr_feature_importance.csv
└── rfr_test_predictions.csv
```

The validated dataset contains 54 training trips and 13 unseen test trips.

## Dataset

The large datasets are available separately:

[Google Drive dataset folder](https://drive.google.com/drive/folders/11fvGVq2hUVXennN19YmN2PKLMGRITqrb?usp=share_link)

Use one of the following starting points:

- `master_ev_trips_dataset.csv` to reproduce the complete pipeline.
- `cleaned_ev_trips.csv` to begin with physics enrichment.
- `physics_enriched_ev_trips.csv` to train the RFR immediately.
- `trained_rfr.joblib` to use an existing trained model.

## Files to exclude when sharing

Do not include API keys, virtual environments, caches or temporary validation files:

```text
.venv/
__pycache__/
*.pyc
.DS_Store
problem_trips.zip
validation.zip
validation_check/
```

Do not include duplicate compressed and uncompressed copies of the same dataset.
