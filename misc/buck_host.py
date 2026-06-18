#!/usr/bin/env python3
"""
Unified host-side tool for the ESP32-S3 buck identification firmware (V2).

Supports the three firmware capture modes and saves self-describing .npz files
with reconstructed per-sample duty and all metadata (mode, note, chirp params).

Interactive commands (type at the > prompt):
    duty <pct> / d<pct>     set manual duty
    note <text>             tag the next capture
    seq                     run the random-step sequence and save
    hold <seconds>          continuous fixed-duty capture and save
                            (turn Vin/load knob by hand DURING this to record
                             a within-run perturbation; set a note first)
    chirp <nom%> <amp%> <f0> <f1> <sec>
                            swept-sine small-signal capture and save
    read / status / help    passthrough to firmware
    quit

One-shot:
    python buck_host.py /dev/ttyUSB0 --hold 40 -o hold1.npz --note "Vin 17->25 at ~12s"
    python buck_host.py /dev/ttyUSB0 --chirp 50 5 100 8000 4 -o chirp1.npz
    python buck_host.py /dev/ttyUSB0 --seq -o seq1.npz
"""

import sys
import time
import argparse
import threading
import numpy as np
import serial

BAUD = 1500000
DUMP_BYTES_PER_S = 187500.0     # ~1.5 Mbaud effective payload bytes/s


def read_line(ser):
    buf = bytearray()
    while True:
        b = ser.read(1)
        if not b:
            raise TimeoutError("timed out waiting for a header line")
        if b == b"\n":
            return buf.decode(errors="replace").rstrip("\r")
        buf += b


def reconstruct_duty(meta, steps, n_samples, fs):
    """Build the per-sample duty array for whichever mode was captured."""
    mode = meta.get("mode", "sequence")
    duty = np.zeros(n_samples, dtype=np.float32)

    if mode == "sequence":
        idx = 0
        for duty_pct, nsmp in steps:
            end = min(idx + nsmp, n_samples)
            duty[idx:end] = duty_pct
            idx = end
            if idx >= n_samples:
                break

    elif mode == "hold":
        duty[:] = float(meta.get("hold_duty", 0.0))

    elif mode == "chirp":
        nominal = float(meta["chirp_nominal"])
        amp     = float(meta["chirp_amp"])
        f0      = float(meta["chirp_f0"])
        f1      = float(meta["chirp_f1"])
        T       = float(meta["chirp_dur_s"])
        # SAME closed-form phase as the firmware's chirp_duty_at(), in float64
        k  = np.arange(n_samples, dtype=np.float64)
        tk = k / float(fs)
        phase = 2.0 * np.pi * (f0 * tk + (f1 - f0) * tk * tk / (2.0 * T))
        duty = (nominal + amp * np.sin(phase)).astype(np.float32)

    return duty, mode


def run_capture(ser, out_file, expected_capture_s=2.0):
    """Trigger has already been sent by the caller. Read header + payload, save."""
    old_timeout = ser.timeout
    # header arrives only after the whole capture completes (buffer-then-dump),
    # so wait at least the capture duration plus margin
    ser.timeout = max(expected_capture_s + 5.0, 10.0)

    meta = {}
    steps = []
    while True:
        line = read_line(ser)
        if line == "#BEGIN":
            break
        if line.startswith("S,"):
            _, duty, nsmp = line.split(",")
            steps.append((float(duty), int(nsmp)))
        elif line.startswith("#") and "=" in line:
            k, v = line[1:].split("=", 1)
            meta[k] = v
        elif line.startswith("ERR"):
            print(line)
            ser.timeout = old_timeout
            return

    fs = int(meta.get("sample_rate_hz", "0"))
    total = int(meta.get("total_samples", "0"))
    mode = meta.get("mode", "sequence")
    note = meta.get("note", "")
    print(f"mode={mode} sample_rate={fs} total_samples={total} "
          f"aborted={meta.get('aborted')} note={note!r}")

    # widen timeout for the binary payload based on its size
    capture_s = total / fs if fs else 0
    dump_s = (total * 2) / DUMP_BYTES_PER_S
    ser.timeout = capture_s + dump_s + 5

    need = total * 2
    raw = bytearray()
    t0 = time.time()
    while len(raw) < need:
        chunk = ser.read(need - len(raw))
        if not chunk:
            raise TimeoutError("timed out reading binary payload")
        raw += chunk
    dt = time.time() - t0

    footer = read_line(ser)
    ser.timeout = old_timeout
    if footer != "#END":
        print(f"warning: expected #END, got {footer!r}")

    samples = np.frombuffer(bytes(raw), dtype="<u2").astype(np.int32)
    duty, mode = reconstruct_duty(meta, steps, len(samples), fs)
    t = np.arange(len(samples)) / fs

    # save everything, including all metadata so the file is self-describing
    save_kwargs = dict(t=t, raw=samples, duty=duty, fs=fs, mode=mode, note=note)
    for key in ("hold_duty", "chirp_nominal", "chirp_amp", "chirp_f0",
                "chirp_f1", "chirp_dur_s", "pwm_freq_hz", "aborted"):
        if key in meta:
            save_kwargs[key] = meta[key]
    np.savez(out_file, **save_kwargs)
    print(f"saved {out_file}  ({len(samples)} samples, {dt:.1f}s transfer)")


def reader_thread(ser, stop_event):
    while not stop_event.is_set():
        try:
            line = ser.readline()
        except serial.SerialException:
            break
        if line:
            print(line.decode(errors="replace").rstrip())


def _capture_with_reader_paused(ser, trigger_cmd, out_file, expected_s,
                                stop_event, thread_ref):
    """Pause the background reader, send a trigger, capture, restart reader."""
    stop_event.set()
    thread_ref[0].join()
    old_timeout = ser.timeout
    try:
        ser.reset_input_buffer()
        ser.write((trigger_cmd + "\n").encode())
        run_capture(ser, out_file, expected_capture_s=expected_s)
    except TimeoutError as e:
        print(f"capture failed: {e}")
    finally:
        ser.timeout = old_timeout
    new_stop = threading.Event()
    t = threading.Thread(target=reader_thread, args=(ser, new_stop), daemon=True)
    t.start()
    thread_ref[0] = t
    return new_stop


def interactive(ser, default_out):
    stop_event = threading.Event()
    thread_ref = [threading.Thread(target=reader_thread,
                                   args=(ser, stop_event), daemon=True)]
    thread_ref[0].start()

    print("Commands: duty <pct>, note <text>, seq, hold <s>, "
          "chirp <nom> <amp> <f0> <f1> <s>, read, status, help, quit")
    run_count = {"seq": 0, "hold": 0, "chirp": 0}

    def out_name(tag):
        run_count[tag] += 1
        c = run_count[tag]
        base = default_out.replace(".npz", "")
        return f"{base}_{tag}{c}.npz"

    try:
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                continue
            low = line.lower()
            if low in ("quit", "exit"):
                break

            if low == "seq" or low == "start":
                stop_event = _capture_with_reader_paused(
                    ser, "seq", out_name("seq"), 12.0, stop_event, thread_ref)
                continue

            if low.startswith("hold"):
                parts = line.split()
                secs = float(parts[1]) if len(parts) > 1 else 30
                stop_event = _capture_with_reader_paused(
                    ser, line, out_name("hold"), secs + 2, stop_event, thread_ref)
                continue

            if low.startswith("chirp"):
                parts = line.split()
                secs = float(parts[5]) if len(parts) > 5 else 4
                stop_event = _capture_with_reader_paused(
                    ser, line, out_name("chirp"), secs + 2, stop_event, thread_ref)
                continue

            # everything else (duty, note, read, status, help) is passthrough
            ser.write((line + "\n").encode())
    finally:
        stop_event.set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("-o", "--output", default="buck_capture.npz")
    ap.add_argument("--note", default=None, help="set a note before capture")
    ap.add_argument("--seq", action="store_true", help="one-shot sequence")
    ap.add_argument("--hold", type=float, metavar="SEC",
                    help="one-shot hold capture for SEC seconds")
    ap.add_argument("--chirp", nargs=5, type=float,
                    metavar=("NOM", "AMP", "F0", "F1", "SEC"),
                    help="one-shot chirp capture")
    args = ap.parse_args()

    ser = serial.Serial(args.port, BAUD, timeout=5)

    one_shot = args.seq or args.hold or args.chirp
    if one_shot:
        if args.note is not None:
            ser.write(("note " + args.note + "\n").encode())
            time.sleep(0.1)
            ser.reset_input_buffer()
        if args.seq:
            ser.reset_input_buffer(); ser.write(b"seq\n")
            run_capture(ser, args.output, expected_capture_s=12.0)
        elif args.hold:
            ser.reset_input_buffer(); ser.write(f"hold {int(args.hold)}\n".encode())
            run_capture(ser, args.output, expected_capture_s=args.hold + 2)
        elif args.chirp:
            nom, amp, f0, f1, sec = args.chirp
            ser.reset_input_buffer()
            ser.write(f"chirp {nom} {amp} {f0} {f1} {sec}\n".encode())
            run_capture(ser, args.output, expected_capture_s=sec + 2)
    else:
        ser.timeout = 0.5
        interactive(ser, args.output)

    ser.close()


if __name__ == "__main__":
    main()
