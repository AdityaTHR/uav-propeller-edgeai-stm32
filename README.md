# UAV Propeller Edge-AI Fault Diagnosis on STM32F446RE

A vibration-based five-class UAV propeller fault diagnosis system that combines a Python ML reference pipeline, embedded C inference, STM32F446RE deployment, Renode firmware validation, and an interactive Streamlit dashboard.

## What this project does

The system classifies five propeller-health states from UAV vibration measurements:

- Healthy
- Damaged Bottom Right Blade
- Damaged Top Right Blade
- Unbalanced Bottom Right Blade
- Unbalanced Top Right Blade

The deployment pipeline is:

```text
UAV vibration
    ↓
500-sample X/Y window
    ↓
Time-domain DSP + mean removal
    ↓
Zero-pad to 512 + FFT
    ↓
16-feature vector
    ↓
50-tree AdaBoost classifier
    ↓
5-class diagnosis
    ↓
Equivalent embedded C pipeline on STM32F446RE
```

## Project architecture

```text
                         ┌─────────────────────────┐
                         │  Recorded UAV vibration │
                         │      X/Y acceleration   │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ DSP preprocessing       │
                         │ 500 samples / axis      │
                         │ statistics + FFT512     │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ 16-feature contract     │
                         └────────────┬────────────┘
                                      │
                ┌─────────────────────┴─────────────────────┐
                │                                           │
                ▼                                           ▼
     ┌────────────────────────┐                  ┌────────────────────────┐
     │ Python reference path  │                  │ Embedded deployment    │
     │ AdaBoost + Streamlit   │                  │ C + CMSIS-DSP + STM32 │
     └────────────┬───────────┘                  └────────────┬───────────┘
                  │                                           │
                  └────────────── parity checks ──────────────┘
                                      │
                                      ▼
                              Equivalent diagnosis
```

The dashboard and STM32 firmware are **two execution targets of the same frozen inference contract**. Streamlit does not pretend to execute MCU firmware inside the browser; the C implementation is validated separately and its deployment evidence is exposed in the dashboard.

## Dataset

Source dataset: UAV propeller/blade vibration dataset published through Figshare.

DOI: `10.6084/m9.figshare.28765640`

The original recordings contain time plus X/Y/Z acceleration measurements. This repository should not duplicate the full raw dataset. A small public demo subset can be generated locally for the Streamlit deployment using `scripts/make_demo_data.py`.

Observed sampling rate used by the project:

```text
1023.541 Hz
```

A deployment window contains:

```text
500 samples ≈ 0.4885 s
```

## Validation strategy

Neighboring vibration samples are highly related, so random row-level splitting was avoided.

The recordings were divided chronologically into domains. Development used grouped cross-validation, while the final temporal block was held out until model selection was completed.

This supports temporal/domain generalization **within the controlled experiment**. It does not prove generalization to arbitrary UAVs, sensors, RPM ranges, payloads, mounting positions, or outdoor operating conditions.

## Model development

Three principal classifiers were evaluated:

| Model | Development CV Macro-F1 | Temporal holdout Macro-F1 |
|---|---:|---:|
| Decision Tree | ~0.884 | ~0.813 |
| AdaBoost | ~0.928 | ~0.905 |
| XGBoost | ~0.961 | ~0.925 |

Further resource-aware experiments reduced the sensor axes and feature bank and aligned preprocessing with the embedded implementation.

### Final embedded-compatible comparison

| Model | Temporal holdout Macro-F1 | Minimum class recall | Trees |
|---|---:|---:|---:|
| AdaBoost | 0.9081 | 0.7170 | 50 |
| Compact XGBoost | 0.9099 | 0.6918 | 375 |

Compact XGBoost improved holdout Macro-F1 by only about 0.18 percentage points while using 7.5× more trees. AdaBoost was therefore selected for the embedded deployment.

## Final inference contract

### Input

```text
500 X-axis acceleration samples
500 Y-axis acceleration samples
```

### DSP

For each axis:

- mean
- standard deviation
- RMS
- peak-to-peak amplitude
- crest factor
- skewness
- kurtosis
- spectral centroid

For spectral processing:

```text
500 samples
→ subtract 500-sample mean
→ zero-pad to 512
→ 512-point real FFT
→ spectral centroid
```

Total:

```text
8 features × 2 axes = 16 features
```

### Classifier

```text
50-tree AdaBoost
```

### Output

One of five propeller-health classes.

## Python ↔ Embedded-C validation

The final implementation was checked at multiple levels:

| Validation | Result |
|---|---:|
| Generated C classifier parity | 798 / 798 |
| Raw signal → features → prediction parity | 15 / 15 |
| Maximum observed host feature difference | ~3.05e-05 |
| Renode firmware self-test | 15 / 15 |
| STM32 Release build | 0 errors / 0 warnings |

## STM32F446RE deployment footprint

Optimized Release build:

```text
text = 18,888 bytes
data =     88 bytes
bss  = 10,064 bytes
```

Approximate deployment usage:

```text
Flash       ≈ 18.5 KiB
Static RAM  ≈  9.9 KiB
```

Target:

```text
STM32F446RE
Arm Cortex-M4F
CMSIS-DSP FFT
```

The physical NUCLEO-F446RE board run is a future hardware-validation step; the current firmware has been compiled for STM32F446RE and executed in an STM32F4/Cortex-M4 Renode emulation environment.

## Repository structure

```text
uav-propeller-edgeai-stm32/
├── dashboard/
│   ├── app.py
│   └── assets/
│       ├── stm32_release_build.png
│       └── renode_validation.png
├── data/
│   ├── README.md
│   └── demo/
├── embedded/
│   └── STM32/
│       ├── Inc/
│       ├── Src/
│       ├── Startup/
│       └── linker/
├── ml/
│   ├── training/
│   ├── evaluation/
│   └── export/
├── models/
│   └── adaboost_xy_fft512_final.joblib
├── results/
├── scripts/
│   └── make_demo_data.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Dashboard

The Streamlit application provides:

- selectable recorded operating conditions
- upload of compatible new `.xlsx` recordings
- 3D UAV digital twin
- fault-specific rotor visualization
- recorded X/Y vibration plots
- 512-point FFT visualization
- 16-feature inspection
- Python reference-model diagnosis
- embedded-C parity summary
- STM32 Flash/RAM footprint
- Renode and STM32CubeIDE evidence
- architecture and validation diagrams

Uploaded files are used for **inference only**. The deployed model does not automatically retrain itself.

## Running locally

```bash
cd uav-propeller-edgeai-stm32
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Generating lightweight demo data

Keep the five original `.xlsx` recordings outside Git.

Place them temporarily under a local source directory and run:

```bash
python scripts/make_demo_data.py \
  --source data_full \
  --output data/demo \
  --windows 30
```

This creates a compact public demo subset for each class while preserving valid 500-sample windows.

## New-data behavior

A new compatible recording can be uploaded to the dashboard and classified by the **frozen** model.

```text
new recording
→ 500-sample windows
→ same preprocessing
→ same 16 features
→ frozen AdaBoost
→ diagnosis
```

The upload is **not automatically incorporated into training**. Incorporating a new UAV, sensor, RPM regime, or new fault distribution requires labelled data, retraining/revalidation, C re-export, parity testing, and embedded revalidation.

## Limitations

- Controlled experimental dataset
- Five known fault classes only
- No external cross-UAV validation yet
- No physical NUCLEO-F446RE sensor-stream test yet
- Sensor mounting, RPM, payload and operating environment may change the vibration distribution
- Probability-like AdaBoost class outputs should not be interpreted as calibrated real-world certainty without calibration testing

## Next hardware step

```text
real accelerometer
→ STM32F446RE acquisition
→ live 500-sample buffering
→ embedded DSP
→ AdaBoost inference
→ UART/BLE/telemetry
→ live dashboard
```

## License / third-party components

The project uses open-source Python packages and ARM CMSIS-DSP. Review and preserve the respective upstream licenses when redistributing third-party code.
