# chart2wav — Convert Dexscreener (or any) charts into audio waveforms

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

- PNG → waveform extraction based only on the DEXSCREENER chart’s blue color (#2962FF by default)
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

- Python 2.7  
- Pillow (`pip install pillow`)

---

## 🚀 Usage

Basic:

```sh
python chart_png_to_wave.py chart.png chart.wav
