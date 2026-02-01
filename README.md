# Conv2 (CDN Pyodide)

This build loads Pyodide + NumPy from a CDN and runs the DSP offline *after* it loads.

## Run
You must use a local static server (ES modules don't work reliably with file://):
- `python -m http.server 8000`
- open `http://localhost:8000`

## UI workflow
1. Click **Start Pyodide**
2. Select **Input WAV**
3. Select **IR Bank** (one or more WAV files)
4. Click **Render**

## Notes
- Internet is required to fetch Pyodide the first time you click Start.
- All WAV sample rates must match (input, control wav if used, IR bank). No resampling is implemented.

## Output processing
- Optional high-pass filter at 5 Hz or 20 Hz (post-convolution)
- Optional peak normalize to -0.01 dBFS


## UI theme
Brutalist monochrome ('a symphony in grey') layout with denser controls.

- Click **GREY EDITION** in the header to open a terminal quickstart popup.

## Resampling
If enabled (Linear or Windowed Sinc), control WAV + IR bank files are automatically resampled to the input WAV sample rate.

- The UI shows detected sample rates for input/control/bank and indicates whether resampling will occur.
