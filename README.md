# EV Remaining Range Estimation Model

This project estimates electric-vehicle battery energy use and remaining range from a known state of charge, a speed profile and route information. It contains:

- A frame-by-frame longitudinal vehicle physics model.
- Battery, regenerative-braking and range calculations.
- Optional TomTom, Open-Elevation and Open-Meteo inputs.
- A Tkinter desktop interface.
- A Random Forest Regression pipeline that learns the difference between measured BMW i3 battery power and the physics-model estimate.

The Random Forest model is currently a separate training and evaluation pipeline. It has not yet been connected to the Tkinter range simulator.

## Recommended project structure

```text
EV_Range_Model/
├── README.md
├── Launcher.py
├── clean_ev_data.py
├── build_physics_dataset.py
├── train_rfr.py
└── ev_range_project/
    ├── __init__.py
    ├── api_clients.py
    ├── driving_profiles.py
    ├── main.py
    ├── physical_model.py
    ├── plotting.py
    ├── route_builder.py
    ├── user_interface.py
    ├── requirements.txt
    └── .env.example
```

Data files and generated results are not required inside the source-code ZIP. See [Data files](#data-files) for the options.

## File guide

| File | Purpose |
|---|---|
| `Launcher.py` | Recommended way to open the Tkinter desktop interface. |
| `clean_ev_data.py` | Cleans and combines the recorded BMW i3 trip CSV data. |
| `build_physics_dataset.py` | Applies the physics model to the cleaned trips and assigns complete trips to training or testing. |
| `train_rfr.py` | Trains and evaluates the Random Forest residual-correction model. |
| `ev_range_project/main.py` | Validates the configuration and coordinates one simulation. |
| `ev_range_project/physical_model.py` | Calculates vehicle forces, wheel power, battery power, energy, SOC and range. |
| `ev_range_project/user_interface.py` | Defines the tabbed Tkinter input window. |
| `ev_range_project/route_builder.py` | Converts route geometry and elevation into distance and road grade. |
| `ev_range_project/api_clients.py` | Connects to TomTom, Open-Elevation and Open-Meteo. |
| `ev_range_project/driving_profiles.py` | Creates the offline and route-based speed profiles. |
| `ev_range_project/plotting.py` | Produces the Matplotlib result windows. |

## Requirements

- Python 3.11 or newer is recommended.
- Internet access is only needed for the optional live route and weather features.
- A TomTom API key is only needed when the live-route option is enabled.

The required Python packages are listed in:

```text
ev_range_project/requirements.txt
```

## First-time setup on macOS

Open Terminal, move into the unzipped project folder and create a virtual environment:

```bash
cd /path/to/EV_Range_Model
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ev_range_project/requirements.txt
```

### Tkinter on a Homebrew Python installation

Test whether Tkinter is available:

```bash
python -m tkinter
```

If a small test window opens, Tkinter is installed. If Python reports that Tkinter is missing and Python 3.14 was installed through Homebrew, run:

```bash
brew install python-tk@3.14
```

If the existing virtual environment still cannot import Tkinter, recreate only the virtual environment with the same Homebrew Python:

```bash
deactivate
rm -rf .venv
/opt/homebrew/bin/python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r ev_range_project/requirements.txt
python -m tkinter
```

Do not delete the project folder. Only `.venv` is recreated.

## First-time setup on Windows

Install Python from python.org with the Tcl/Tk option enabled. Open PowerShell in the unzipped project folder and run:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r ev_range_project/requirements.txt
python -m tkinter
```

## Run the desktop simulator

Activate the virtual environment and run this command from the folder containing `Launcher.py`:

```bash
python Launcher.py
```

The application opens with six groups of inputs:

- Simulation settings.
- Route and API settings.
- Vehicle parameters.
- Powertrain parameters.
- Battery and environmental conditions.
- Range-estimator settings.

For the first run, leave the live-route and live-weather options switched off. The program will use the offline demonstration route and does not require an API key.

Select **Run simulation**. The application validates the inputs, displays a summary and opens the Matplotlib result windows.

## Run the simulator without the interface

The physics model can be tested independently of Tkinter:

```bash
python -m ev_range_project.main
```

This runs the default offline configuration.

## Optional live route and weather

### TomTom route key

Create a TomTom developer key and save it in the `TOMTOM_API_KEY` environment variable. Never put the real key in the Python files, README, ZIP or version control.

For the current macOS or Linux terminal session:

```bash
export TOMTOM_API_KEY="your_real_key_here"
python Launcher.py
```

For the current Windows PowerShell session:

```powershell
$env:TOMTOM_API_KEY="your_real_key_here"
python Launcher.py
```

The UI reports whether the key was detected. Open-Meteo and the public Open-Elevation endpoint do not require keys in the current implementation. If live weather fails, the simulator falls back to the weather values entered in the UI.

## Machine-learning workflow

Run all commands from the folder containing `clean_ev_data.py`, `build_physics_dataset.py` and `train_rfr.py`.

### Step 1 Clean the trip data

If starting from the complete merged trip file:

```bash
python clean_ev_data.py master_ev_trips_dataset.csv --sample-period-s 0.1
```

If starting from separate trip files:

```bash
python clean_ev_data.py TripA01.csv TripA02.csv TripA03.csv --sample-period-s 0.1
```

Using `--sample-period-s 0.1` retains the original 10 Hz resolution used for the current Random Forest baseline.

Outputs:

```text
cleaned_ev_trips.csv
cleaning_report.txt
```

If `cleaned_ev_trips.csv` has already been supplied, skip this step.

### Step 2 Add the physics calculations and split the trips

```bash
python build_physics_dataset.py cleaned_ev_trips.csv
```

This applies the force and power equations to every row and makes a reproducible 80/20 split by complete trip. Adjacent 10 Hz rows from one journey are therefore never divided between training and testing.

Default conditions for the enrichment run are:

- Temperature: 20 degrees C.
- Wind speed: 0 m/s.
- Pressure: 101325 Pa.
- Random seed: 42.

Output:

```text
physics_enriched_ev_trips.csv
```

To specify the test trips manually:

```bash
python build_physics_dataset.py cleaned_ev_trips.csv --test-trip-ids 3 8 14 21
```

The current validated baseline excludes trips 36 and 41 because they contain plugged-in charging periods, and trip 70 because its velocity data are missing. These exclusions are included in the packaged version of `build_physics_dataset.py`.

If `physics_enriched_ev_trips.csv` has already been supplied, skip Steps 1 and 2.

### Step 3 Train and evaluate the Random Forest

```bash
python train_rfr.py physics_enriched_ev_trips.csv
```

The model uses these features:

- Vehicle velocity.
- Longitudinal acceleration.
- Road gradient.
- Estimated wheel power.
- Throttle percentage.
- Regenerative-braking signal.

The learning target is:

```text
battery_power_residual_w = measured battery power - physics battery power
```

The corrected prediction is:

```text
hybrid battery power = physics battery power + predicted residual
```

The baseline model uses 60 trees, a maximum depth of 18, a minimum of 10 samples per leaf and 25 percent of the training rows for each tree.

Outputs are written to:

```text
rfr_results/
├── trained_rfr.joblib
├── rfr_test_predictions.csv
├── rfr_trip_metrics.csv
├── rfr_feature_importance.csv
└── rfr_metrics.txt
```

The main files to inspect are `rfr_metrics.txt` and `rfr_trip_metrics.csv`. The current validated baseline used 54 training trips and 13 testing trips.

## Data files

The source-code ZIP does not include the datasets because of their large file sizes. Download the required datasets from the Google Drive folder below before running the machine-learning pipeline.

Google Drive — Dataset Download: [https://drive.google.com/drive/folders/11fvGVq2hUVXennN19YmN2PKLMGRITqrb?usp=share_link]

The folder contains the datasets required to run the machine-learning pipeline, including the raw driving data, cleaned data, physics-enriched data, and trained model results. Download the files required for your chosen workflow, as described in the Machine-learning workflow section.

### Option A Reproduce the complete ML pipeline

Send `master_ev_trips_dataset.csv` separately. The recipient runs all three ML steps.

### Option B Start from cleaned data

Send `cleaned_ev_trips.csv` separately. The recipient runs the physics-enrichment and training steps.

### Option C Train immediately

Send `physics_enriched_ev_trips.csv.zip` separately. The recipient extracts it beside `train_rfr.py` and runs only the training step.

### Option D Use the already trained model

Send `rfr_results/trained_rfr.joblib` and the metrics files separately. This is useful for inspection, but the trained model is not yet called by the Tkinter simulator.

Do not include both the uncompressed and compressed copy of the same large CSV.

## Files that should not be included in the code ZIP

Exclude the following:

```text
__pycache__/
.venv/
.DS_Store
*.pyc
problem_trips.zip
validation.zip
validation_check/
cleaning_report.txt
make_validation_bundle.py
```

The problem-trip and validation files were temporary investigation artifacts. `make_validation_bundle.py` was only used to generate diagnostic files and is not required to run the simulator or train the final baseline.

The following are also unnecessary when the current files above are present:

- `Pasted code.py`, because `Launcher.py` replaces it.
- `vehicle_model.py`, if it is an older standalone copy of the physics model.
- Duplicate CSV and ZIP versions of the same dataset.
- Old screenshots, PDFs and unrelated project files.
- Any file containing a real API key.

## Current modelling limits

- Powertrain and regenerative efficiencies are constant rather than map-based.
- The physics-enrichment run uses default weather and a constant auxiliary load.
- The Random Forest model is not yet connected to the Tkinter range output.
- Throttle and regenerative-braking signals are onboard measurements and are not normally known for a future planned route.
- The model improves the overall baseline but does not improve every unseen trip.

## Troubleshooting

### Python cannot find a script

Check that Terminal is inside the project folder:

```bash
pwd
ls
```

The listing should include `Launcher.py`, `clean_ev_data.py`, `build_physics_dataset.py`, `train_rfr.py` and the `ev_range_project` folder.

### Relative import error

Do not run `ev_range_project/main.py` directly. Use:

```bash
python -m ev_range_project.main
```

### Input CSV not found

Place the CSV beside the top-level scripts or provide its full path. File names are case-sensitive on some systems.

### TomTom key missing

The offline simulator still works. Set `TOMTOM_API_KEY` only when using the live-route option.

### Open-Meteo timeout

Disable live weather or retry later. The application uses the entered default weather when the live request fails.
