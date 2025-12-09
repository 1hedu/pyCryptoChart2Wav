# pyCryptoChart2Wav — Chart-to-Audio Tools

Convert cryptocurrency price charts into audio waveforms and synthesizer wavetables.

This project includes two main tools:
1. **chart2wav** - Command-line chart-to-WAV converter (works on desktop and Android)
2. **Chart FM Synth Pro** - Full-featured FM synthesizer with chart-to-wavetable import (desktop only)

---

# 📊 chart2wav — Chart-to-Waveform Converter

`chart2wav` is a lightweight Python tool that converts PNG price charts into **audio waveforms**, ready for samplers, synths, DAWs and wavetable engines.

It extracts the blue chart line from your screenshot, converts it into amplitude data, shapes it into a clean audio waveform, and exports a 16-bit mono WAV.

No external dependencies beyond Pillow.

Supports **three synthesis modes**:

---

## 🎛 Modes

### **1. cycle (default)**  
Turns the chart into a **single-cycle oscillator** (like Serum/PPG/DW style wavetable waves).  
Useful for synth engines and looped tones.

### **2. sweep**  
Stretches the **entire chart** across the output duration.  
A one-shot gesture of the full price motion.

### **3. multi**  
Carves the chart into **N sequential cycles**, scanning from left → right.  
Produces evolving multi-cycle patterns rather than a single repeated shape.

---

## ✨ Features

- PNG → waveform extraction based only on the DEXSCREENER chart’s blue color (#2962FF by default)[NOTE: to work on another chart with a different color line, you must update this hex color value]
- Baseline (dashed center line) auto-detected
- Missing data automatically interpolated
- Loop-safe output
  - endpoint ramp removal  
  - symmetric cosine fade  
- Three waveform modes (cycle / sweep / multi)
- User-defined number of cycles in multi-mode
- Fixed-duration output for sampler compatibility (Digitakt, MPC, Octatrack, etc.)
- Python **2.7 compatible**
- No dependencies except Pillow

---

## 📦 Requirements

**Python 3** (recommended) - Works on desktop and Android:
- Python 3.6+
- Pillow (`pip install pillow`)

**Python 2.7** (legacy) - Desktop only:
- Python 2.7
- Pillow (`pip install pillow`)

---

## 🚀 Usage

**Python 3** (recommended):

```sh
python3 chart2wav.py chart.png chart.wav
```

**Python 2.7** (legacy):

```sh
python chart_png_to_wave.py chart.png chart.wav
```

### Advanced Options

```sh
python3 chart2wav.py chart.png output.wav \
  --mode cycle \
  --points 256 \
  --samplerate 48000 \
  --seconds 1.0 \
  --fade 0.05 \
  --blue 2962FF \
  --thresh 35
```

**Options:**
- `--mode` : cycle, sweep, or multi
- `--points` : samples per cycle (default: 64)
- `--cycles` : number of cycles for multi mode (default: 8)
- `--samplerate` : output sample rate (default: 44100)
- `--seconds` : duration in seconds (default: 1.0)
- `--fade` : fade amount 0.0-0.5 (default: 0.05)
- `--blue` : hex color to detect (default: 2962FF)
- `--thresh` : color matching threshold (default: 35)

---

# 🎹 Chart FM Synth Pro — Full-Featured FM Synthesizer

A complete FM synthesizer application that imports cryptocurrency chart images as custom wavetables for sound design.

**Platform:** Desktop only (Linux, Mac, Windows) - Requires PyQt5 which is not available on Android/Pydroid3.

## ✨ Features

### Synthesis Engine
- **6 FM operators** with chart-derived wavetables
- **32 classic FM algorithms** (DX7-inspired) + custom matrix routing
- **ADSR envelopes** per operator + master amplitude envelope
- **3 LFOs** with 6 waveform shapes (sine, triangle, saw up/down, square, sample & hold)
- **Multi-mode filter** (lowpass, highpass, bandpass) with resonance
- **Polyphonic** voice management (up to 8 voices)
- **Real-time audio** with Numba JIT optimization (~50x faster DSP)

### Chart Integration
- **Import PNG charts** directly from the UI
- **3 conversion modes** (cycle, sweep, multi)
- **Automatic wavetable generation** from price action
- **Preview and adjust** conversion parameters
- **Preset configurations** for common use cases

### User Interface
- **49-key on-screen keyboard** with velocity sensitivity
- **Real-time oscilloscope** and spectrum analyzer
- **Hardware-inspired dark UI** with rotary knobs
- **Patch save/load system** (JSON format)
- **MIDI support** (note on/off, mod wheel, pitch bend, aftertouch)
- **Mod matrix** for flexible modulation routing

## 📦 Requirements

**Required:**
```sh
pip install sounddevice numpy scipy pyqt5 pillow
```

**Optional (for MIDI support):**
```sh
pip install mido python-rtmidi
```

**Optional (for performance boost):**
```sh
pip install numba
```

## 🚀 Usage

```sh
python3 chart_fm_synth_pro.py
```

### First Steps

1. **Import chart WAVs** or create them using chart2wav
2. **Load wavetables** via "Chart Import" button
3. **Select algorithm** (1-32) or use custom matrix
4. **Assign wavetables** to operators
5. **Play notes** on the on-screen keyboard or via MIDI
6. **Adjust envelopes, LFOs, and filter** to shape the sound
7. **Save patches** for later use

### Chart Import Workflow

1. Click **"Chart Import"** button
2. Browse and select a chart PNG
3. Choose conversion mode:
   - **Cycle** - Single-cycle wavetable (recommended for most synth work)
   - **Sweep** - Full chart as one-shot gesture
   - **Multi** - Multiple evolving cycles
4. Adjust parameters (points, cycles, duration)
5. Click **"Import"**
6. The wavetable becomes available in operator dropdowns

### Keyboard Shortcuts

- **Z-M** keys - Bottom octave (C-B)
- **Q-U** keys - Middle octave (C-B)
- **Arrow keys** - Change octave
- **Space** - Panic (all notes off)

### MIDI Support

If `mido` is installed, the synth automatically connects to the first available MIDI input port and responds to:
- Note on/off messages
- Control Change 1 (Mod Wheel)
- Pitch bend
- Channel aftertouch

## 🎛 FM Algorithms

The synth includes 32 DX7-style FM algorithms. Each algorithm defines how the 6 operators modulate each other:

- **Algorithm 1-5**: Classic stacks (6→5→4→3→2→1 to various splits)
- **Algorithm 6-11**: Parallel carriers with modulators
- **Algorithm 12**: All 6 operators in parallel (additive)
- **Algorithm 13-20**: Mixed configurations
- **Algorithm 21-32**: Complex routings for evolving timbres

You can also create **custom matrix** routing by enabling "Custom Matrix" mode and adjusting the modulation amounts directly.

## 🔊 Audio Backend

- Low-latency real-time audio via `sounddevice` (PortAudio)
- Callback-based processing (1024 sample blocks)
- Configurable latency settings for stability

## 📄 Patch Format

Patches are saved as JSON files containing:
- Operator parameters (level, ratio, detune, feedback, envelope)
- Algorithm or custom matrix configuration
- Master settings (level, FM depth, portamento, tuning)
- LFO settings (rate, depth, shape, delay)
- Filter settings (type, cutoff, resonance)
- Modulation matrix entries

Example patch structure:
```json
{
  "name": "My Patch",
  "algorithm": 1,
  "operators": [...],
  "master_level": 0.7,
  "fm_depth": 0.5,
  "lfos": [...],
  "filter": {...}
}
```

## 🎨 Sound Design Tips

### Creating Evolving Pads
- Use **multi-mode** chart import with 8-16 cycles
- Slow attack/release envelopes on operators
- LFO modulation on operator levels
- Low FM depth (0.3-0.5)

### Creating Aggressive Basses
- Use **cycle mode** with sharp chart movements
- High FM depth (0.7-1.0)
- Fast attack, medium release
- Filter with resonance

### Creating Bell Tones
- Use **Algorithm 12** (all parallel)
- Enable 2-3 operators only
- Different ratios (1.0, 2.0, 3.5)
- Fast attack, long release

### Creating Plucks
- Use **cycle mode** with volatile chart patterns
- Very fast attack (0.001s)
- Short decay/release
- Higher operator ratios (2.0, 4.0, 8.0)

---

## 🛠 Technical Details

### DSP Optimization
- **Numba JIT compilation** for critical DSP kernels (~50x speedup)
- **Block processing** (1024 samples) for efficiency
- **Pre-allocated buffers** to minimize memory allocation
- **Vectorized operations** using NumPy

### Chart-to-Wavetable Pipeline
1. Load PNG image
2. Color-based pixel detection (Euclidean RGB distance)
3. Baseline detection (row with most chart pixels)
4. Per-column median Y extraction
5. Linear interpolation for missing data
6. DC offset removal (linear ramp subtraction)
7. Cosine window fade at endpoints
8. Normalization to [-1, 1]
9. Resampling to target points/duration

### Voice Architecture
- Polyphonic voice allocator with voice stealing
- Per-voice operator states (phase, envelope, feedback)
- Per-voice filter and LFO instances
- Portamento with linear frequency interpolation

---

## 📜 License

See LICENSE file for details.

## 🙏 Credits

- FM algorithms inspired by Yamaha DX7
- Chart2wav conversion based on cryptocurrency chart analysis
- Built with PyQt5, NumPy, SciPy, and Numba

---

## 🐛 Troubleshooting

### "No audio output"
- Check sounddevice installation: `pip install sounddevice`
- Verify audio device: `python3 -c "import sounddevice; print(sounddevice.query_devices())"`
- Try increasing buffer size or latency in the code

### "Numba warnings"
- Numba is optional - synth works without it (just slower)
- Install for better performance: `pip install numba`

### "MIDI not working"
- Install mido: `pip install mido python-rtmidi`
- Check available MIDI devices
- On Linux, may need to install `libasound2-dev`

### "Chart import fails"
- Ensure Pillow is installed: `pip install pillow`
- Check that chart has visible blue line (#2962FF)
- Try adjusting threshold (25-50 range)
- Different chart colors need `--blue RRGGBB` parameter

### "Can I run the FM synth on Android?"
- No - PyQt5 is not available on Android/Pydroid3
- Use the command-line `chart2wav` tool instead to create wavetables on Android
- Then load those WAVs into desktop synthesizers or DAWs
