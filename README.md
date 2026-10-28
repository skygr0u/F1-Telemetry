# F1-Telemetry

How is a Formula 1 race won, and how much of the result is decided before the lights go out? This project uses lap-by-lap timing data from [FastF1](https://docs.fastf1.dev/) to analyse race pace, tyre strategy, tyre degradation, position changes, and the link between qualifying and the race result, for any Grand Prix.

Course project for *Coding for Data Science and Data Management* (Python module), University of Milan, A.Y. 2026/27. Proposal 5, "Box, Box".

> **Status:** work in progress. Results, figures and findings will be added as the analysis is completed.

## Research questions

1. **B1 Pace:** who had the real pace in the race?
2. **B2 Strategy:** what tyre strategies were used?
3. **B3 Degradation:** how fast did tyres lose performance?
4. **B4 Race evolution:** how did positions change during the race?
5. **B5 Qualifying vs race:** how much did the starting grid decide?

## Data

- Source: FastF1 3.8.3 (Formula 1 timing data), loaded through its local cache.
- Development race: 2022 Bahrain Grand Prix. Test race for wet conditions: 2022 Monaco Grand Prix.
- Small CSV snapshots of the data used will be stored in `data/raw/`, so the analysis can run without downloading.

## Setup (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Project structure

```
src/f1_telemetry/
    data_loader.py     load sessions, read and write CSV
    cleaning.py        validity flags, clean laps, stints
    analysis.py        pace, degradation, correlations
    visualization.py   plotting functions
notebooks/             exploration and the race report
tests/                 unit tests
data/raw/              small CSV snapshots
reports/               exported figures and tables
```

## How to run

To be completed: notebooks, tests, and the optional Streamlit app.

## Methodology

To be completed: definition of a clean race lap, statistics used.

## Main findings

To be completed after the analysis.

## Limitations

To be completed. Results are descriptive associations, not causal claims.
