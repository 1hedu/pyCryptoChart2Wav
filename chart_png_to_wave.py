#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Convert a Dexscreener-style PNG chart into an audio waveform WAV.

Modes:
  - cycle : single-cycle oscillator, tiled to fill the duration.
  - sweep : full chart stretched across the duration (one-shot gesture).
  - multi : multiple cycles in sequence (chart carved into N equal cycles),
            then that multi-cycle pattern can be tiled to fill the duration.

Pipeline:
  1) Load PNG screenshot.
  2) Detect chart line pixels close to a given blue (default #2962FF).
  3) Find the horizontal dashed baseline (row with most blue pixels).
  4) For each column, get median y of blue pixels (price line).
  5) Interpolate missing columns.
  6) Convert y to amplitude relative to baseline.
  7) Depending on mode:
       - cycle: resample to `points` → 1 cycle
       - sweep: resample to `total_frames`
       - multi: resample to `points * cycles` → N cycles in sequence
  8) Remove linear ramp so endpoints are at 0.
  9) Apply symmetric cosine fade to both edges.
 10) Normalize to [-1, 1].
 11) For cycle/multi: tile pattern to reach desired duration.
 12) Write mono 16-bit WAV.

Usage example:
    python2 chart_png_to_wave.py 11-09-2025_pcock.png pcock.wav \
        --points 64 --samplerate 44100 --seconds 1.0 --fade 0.05 --mode multi --cycles 8
"""

from __future__ import division
import sys
import math
import argparse
import wave
import struct

try:
    from PIL import Image
except ImportError:
    sys.stderr.write("You need Pillow: pip install pillow\n")
    sys.exit(1)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="PNG chart -> loopable WAV")
    parser.add_argument("png_in", help="Input PNG screenshot")
    parser.add_argument("wav_out", help="Output WAV file")
    parser.add_argument("--points", type=int, default=64,
                        help="Number of waveform points per cycle (default: 64)")
    parser.add_argument("--samplerate", type=int, default=44100,
                        help="Sample rate for WAV (default: 44100)")
    parser.add_argument("--blue", default="2962FF",
                        help="Hex color for chart line (default: 2962FF)")
    parser.add_argument("--thresh", type=int, default=35,
                        help="Color distance threshold (default: 35)")
    parser.add_argument("--fade", type=float, default=0.05,
                        help="Fraction of waveform on each side to fade (default: 0.05 = 5%%)")
    parser.add_argument("--seconds", type=float, default=1.0,
                        help="Target sample length in seconds (default: 1.0)")
    parser.add_argument("--mode", choices=["cycle", "sweep", "multi"], default="cycle",
                        help="waveform mode: 'cycle' (looped osc), "
                             "'sweep' (one-shot gesture), or "
                             "'multi' (multi-cycle pattern)")
    parser.add_argument("--cycles", type=int, default=8,
                        help="Number of cycles in 'multi' mode (default: 8)")
    return parser.parse_args(argv)


def hex_to_rgb(hex_str):
    hex_str = hex_str.strip().lstrip('#')
    if len(hex_str) != 6:
        raise ValueError("Hex color must be 6 chars, got: %r" % hex_str)
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return (r, g, b)


def color_distance(c1, c2):
    # Euclidean distance in RGB
    dr = c1[0] - c2[0]
    dg = c1[1] - c2[1]
    db = c1[2] - c2[2]
    return math.sqrt(dr * dr + dg * dg + db * db)


def build_blue_mask(img, target_rgb, thresh):
    """
    Return a 2D list of booleans mask[y][x] = True if pixel is "chart blue".
    """
    w, h = img.size
    pixels = img.load()
    mask = [[False for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            c = pixels[x, y]
            d = color_distance(c, target_rgb)
            if d <= thresh:
                mask[y][x] = True
    return mask, w, h


def find_baseline(mask, w, h):
    """
    Baseline = row with the highest count of blue pixels.
    """
    best_row = 0
    best_count = -1
    for y in range(h):
        count = 0
        row = mask[y]
        for x in range(w):
            if row[x]:
                count += 1
        if count > best_count:
            best_count = count
            best_row = y
    return best_row


def extract_line_y(mask, w, h):
    """
    For each column x, find all blue pixels and return median y.
    Returns:
        ys_line: list of length w with y or None for each x.
    """
    ys_line = [None] * w
    for x in range(w):
        ys = []
        for y in range(h):
            if mask[y][x]:
                ys.append(y)
        if ys:
            ys.sort()
            m = len(ys) // 2
            if len(ys) % 2 == 1:
                ys_line[x] = ys[m]
            else:
                ys_line[x] = (ys[m - 1] + ys[m]) / 2.0
        else:
            ys_line[x] = None
    return ys_line


def interpolate_missing(ys):
    """
    Linearly interpolate None values in ys based on nearest neighbors.
    Modifies the list in-place and returns it.
    """
    n = len(ys)
    # find first non-None from left
    first_val = None
    first_idx = None
    for i in range(n):
        if ys[i] is not None:
            first_val = ys[i]
            first_idx = i
            break
    if first_val is None:
        raise ValueError("No line pixels found at all.")

    # fill leading Nones
    for i in range(first_idx):
        ys[i] = first_val

    # fill in-between gaps
    last_idx = first_idx
    last_val = first_val
    for i in range(first_idx + 1, n):
        if ys[i] is not None:
            cur_idx = i
            cur_val = ys[i]
            gap = cur_idx - last_idx
            if gap > 1:
                step = (cur_val - last_val) / float(gap)
                for k in range(1, gap):
                    ys[last_idx + k] = last_val + step * k
            last_idx = cur_idx
            last_val = cur_val

    # fill trailing Nones, if any
    if last_idx < n - 1:
        for i in range(last_idx + 1, n):
            ys[i] = last_val

    return ys


def resample_to_points(values, n_points):
    """
    Resample a list of numeric values to n_points using linear interpolation.
    """
    n = len(values)
    if n_points <= 1:
        return [values[0]]
    out = []
    for i in range(n_points):
        t = i * (n - 1) / float(n_points - 1)
        j = int(math.floor(t))
        k = int(math.ceil(t))
        if j == k:
            out.append(values[j])
        else:
            frac = t - j
            v = values[j] + frac * (values[k] - values[j])
            out.append(v)
    return out


def normalize_to_minus1_plus1(amps):
    """
    Normalize list of amplitudes to range [-1, 1].
    """
    max_abs = 0.0
    for a in amps:
        aa = abs(a)
        if aa > max_abs:
            max_abs = aa
    if max_abs == 0:
        return [0.0] * len(amps)
    return [a / max_abs for a in amps]


def force_endpoints_to_zero(amps):
    """
    Remove the straight-line ramp between endpoints so that
    amps[0] and amps[-1] become exactly 0, while preserving
    the internal shape as much as possible.
    """
    n = len(amps)
    if n < 2:
        return amps[:]
    a0 = amps[0]
    a1 = amps[-1]
    out = []
    for i, a in enumerate(amps):
        t = i / float(n - 1)
        baseline = a0 + (a1 - a0) * t
        out.append(a - baseline)
    return out


def apply_symmetric_fade(amps, percent=0.05):
    """
    Apply a cosine fade-in and fade-out to smoothly bring the waveform
    to zero near both ends. Percent determines fade length (0.05 = 5%).
    """
    n = len(amps)
    fade_len = int(n * percent)
    if fade_len < 1:
        return amps[:]

    out = amps[:]

    # fade-in
    for i in range(fade_len):
        t = i / float(fade_len)
        fade = 0.5 - 0.5 * math.cos(math.pi * t)
        out[i] *= fade

    # fade-out
    for i in range(fade_len):
        t = i / float(fade_len)
        fade = 0.5 - 0.5 * math.cos(math.pi * t)
        out[n - i - 1] *= fade

    return out


def write_wav(wav_path, amps, samplerate):
    """
    Write amps (float list in [-1, 1]) as mono 16-bit WAV.
    """
    wf = wave.open(wav_path, 'w')
    wf.setnchannels(1)
    wf.setsampwidth(2)  # 16-bit
    wf.setframerate(samplerate)
    frames = []
    for a in amps:
        if a > 1.0:
            a = 1.0
        elif a < -1.0:
            a = -1.0
        val = int(a * 32767.0)
        frames.append(struct.pack('<h', val))
    wf.writeframes(''.join(frames))
    wf.close()


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    target_rgb = hex_to_rgb(args.blue)
    img = Image.open(args.png_in).convert('RGB')

    mask, w, h = build_blue_mask(img, target_rgb, args.thresh)
    baseline_y = find_baseline(mask, w, h)

    ys_line = extract_line_y(mask, w, h)
    ys_line = interpolate_missing(ys_line)

    # Convert y to amplitude around baseline (SVG-style: y down, so invert)
    amps_raw = []
    for y in ys_line:
        a = baseline_y - y
        amps_raw.append(a)

    total_frames = int(args.samplerate * args.seconds)
    if total_frames <= 0:
        total_frames = len(amps_raw)

    if args.mode == "cycle":
        # --- single-cycle oscillator, tiled ---

        # 1) resample horizontally to N points (cycle resolution)
        amps_resampled = resample_to_points(amps_raw, args.points)

        # 2) remove DC ramp so endpoints are exactly 0
        amps_resampled = force_endpoints_to_zero(amps_resampled)

        # 3) apply symmetric fade to pull both sides gracefully into 0
        amps_resampled = apply_symmetric_fade(amps_resampled, percent=args.fade)

        # 4) normalize single cycle to [-1, 1]
        pattern = normalize_to_minus1_plus1(amps_resampled)

        # 5) tile the cycle to reach the desired total length
        if total_frames < len(pattern):
            total_frames = len(pattern)

        tiled = []
        while len(tiled) < total_frames:
            tiled.extend(pattern)
        amps_out = tiled[:total_frames]

    elif args.mode == "multi":
        # --- multi-cycle pattern: carve chart into N equal cycles ---

        cycles = max(1, args.cycles)
        n_points = args.points * cycles
        if n_points < 2:
            n_points = 2

        # 1) resample entire chart into points * cycles
        amps_resampled = resample_to_points(amps_raw, n_points)

        # 2) remove DC ramp and fade only at global ends
        amps_resampled = force_endpoints_to_zero(amps_resampled)
        amps_resampled = apply_symmetric_fade(amps_resampled, percent=args.fade)

        # 3) normalize the whole multi-cycle pattern
        pattern = normalize_to_minus1_plus1(amps_resampled)

        # 4) tile pattern to reach desired duration
        if total_frames < len(pattern):
            total_frames = len(pattern)

        tiled = []
        while len(tiled) < total_frames:
            tiled.extend(pattern)
        amps_out = tiled[:total_frames]

    else:
        # --- sweep mode: full chart stretched to duration, no tiling ---

        # 1) resample entire chart directly to total_frames
        amps_resampled = resample_to_points(amps_raw, total_frames)

        # 2) remove DC ramp + fade edges for clickless playback
        amps_resampled = force_endpoints_to_zero(amps_resampled)
        amps_resampled = apply_symmetric_fade(amps_resampled, percent=args.fade)

        # 3) normalize to [-1, 1]
        amps_out = normalize_to_minus1_plus1(amps_resampled)

    write_wav(args.wav_out, amps_out, args.samplerate)

    sys.stdout.write(
        "Done. Wrote %d-sample waveform (%.3f s, mode=%s) to %s (Fs=%d Hz), baseline row=%d\n"
        % (len(amps_out),
           len(amps_out) / float(args.samplerate),
           args.mode,
           args.wav_out,
           args.samplerate,
           baseline_y)
    )


if __name__ == "__main__":
    main()
