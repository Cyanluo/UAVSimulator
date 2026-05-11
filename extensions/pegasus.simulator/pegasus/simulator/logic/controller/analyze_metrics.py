#!/usr/bin/env python3
"""Compute stable-window metrics for BaseControlController result logs.

Examples:
    python3 analyze_metrics.py debug/base_control_hover.npz --start 6
    python3 analyze_metrics.py run_a.npz run_b.npz --labels baseline,proposed --csv metrics.csv
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np


AXES = ("x", "y", "z")
ATT = ("roll", "pitch", "yaw")


def rms(values, axis=0):
    return np.sqrt(np.mean(np.square(values), axis=axis))


def max_abs(values, axis=0):
    return np.max(np.abs(values), axis=axis)


def load_window(path, start_time, end_time):
    data = np.load(path)
    time = data["time"]
    mask = time >= start_time
    if end_time is not None:
        mask &= time <= end_time
    if not np.any(mask):
        raise ValueError(f"No samples in selected time window for {path}")
    return data, mask


def vector_norm_rms(values):
    return float(np.sqrt(np.mean(np.sum(np.square(values), axis=1))))


def compute_metrics(path, label, start_time, end_time):
    data, mask = load_window(path, start_time, end_time)

    time = data["time"][mask]
    ep = data["ep"][mask]
    ev = data["ev"][mask]
    er = data["er"][mask]
    ew = data["ew"][mask]
    control = data["control"][mask]
    actuator = data["actuator"][mask]
    attitude = data["attitude"][mask]
    desired_attitude = data["desired_attitude"][mask]
    desired_attitude_delta = desired_attitude
    attitude_tracking_error = attitude - desired_attitude

    metrics = {
        "label": label,
        "file": str(path),
        "t_start": float(time[0]),
        "t_end": float(time[-1]),
        "samples": int(time.size),
        "pos_rms_3d": vector_norm_rms(ep),
        "vel_rms_3d": vector_norm_rms(ev),
    }

    for idx, axis in enumerate(AXES):
        metrics[f"pos_rms_{axis}"] = float(rms(ep[:, idx]))
        metrics[f"pos_max_{axis}"] = float(max_abs(ep[:, idx]))
        metrics[f"vel_rms_{axis}"] = float(rms(ev[:, idx]))
        metrics[f"vel_max_{axis}"] = float(max_abs(ev[:, idx]))

    for idx, axis in enumerate(ATT):
        metrics[f"er_rms_{axis}"] = float(rms(er[:, idx]))
        metrics[f"ew_rms_{axis}"] = float(rms(ew[:, idx]))
        metrics[f"att_track_rms_{axis}"] = float(rms(attitude_tracking_error[:, idx]))
        metrics[f"att_track_bias_{axis}"] = float(np.mean(attitude_tracking_error[:, idx]))
        metrics[f"att_des_rms_{axis}"] = float(rms(desired_attitude_delta[:, idx]))
        metrics[f"att_des_max_{axis}"] = float(max_abs(desired_attitude_delta[:, idx]))

    for idx in range(control.shape[1]):
        metrics[f"control_{idx}_rms"] = float(rms(control[:, idx]))
        metrics[f"control_{idx}_max"] = float(max_abs(control[:, idx]))
        metrics[f"control_{idx}_mean"] = float(np.mean(control[:, idx]))

    if control.shape[1] >= 6:
        metrics["force_x_rms"] = metrics["control_4_rms"]
        metrics["force_y_rms"] = metrics["control_5_rms"]
        metrics["force_x_max"] = metrics["control_4_max"]
        metrics["force_y_max"] = metrics["control_5_max"]
    else:
        metrics["force_x_rms"] = 0.0
        metrics["force_y_rms"] = 0.0
        metrics["force_x_max"] = 0.0
        metrics["force_y_max"] = 0.0

    actuator_names = ("rotor_0", "rotor_1", "servo_0", "servo_1")
    for idx, name in enumerate(actuator_names[: actuator.shape[1]]):
        metrics[f"{name}_rms"] = float(rms(actuator[:, idx]))
        metrics[f"{name}_max"] = float(max_abs(actuator[:, idx]))
        metrics[f"{name}_mean"] = float(np.mean(actuator[:, idx]))

    if actuator.shape[1] >= 4:
        metrics["rotor_max"] = float(max(max_abs(actuator[:, 0]), max_abs(actuator[:, 1])))
        metrics["servo_max"] = float(max(max_abs(actuator[:, 2]), max_abs(actuator[:, 3])))

    return metrics


def parse_labels(labels_arg, paths):
    if labels_arg is None:
        return [Path(path).stem for path in paths]
    labels = [item.strip() for item in labels_arg.split(",") if item.strip()]
    if len(labels) != len(paths):
        raise ValueError("--labels count must match the number of input files")
    return labels


def ordered_fields(rows):
    preferred = [
        "label",
        "t_start",
        "t_end",
        "samples",
        "pos_rms_x",
        "pos_rms_y",
        "pos_rms_z",
        "pos_rms_3d",
        "vel_rms_x",
        "vel_rms_y",
        "vel_rms_z",
        "vel_rms_3d",
        "att_track_rms_roll",
        "att_track_rms_pitch",
        "att_track_rms_yaw",
        "att_track_bias_pitch",
        "att_des_rms_roll",
        "att_des_rms_pitch",
        "force_x_rms",
        "force_y_rms",
        "force_x_max",
        "force_y_max",
        "servo_max",
        "rotor_max",
        "file",
    ]
    all_keys = sorted({key for row in rows for key in row})
    return [key for key in preferred if key in all_keys] + [key for key in all_keys if key not in preferred]


def format_value(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_markdown(rows, fields):
    print("| " + " | ".join(fields) + " |")
    print("| " + " | ".join("---" for _ in fields) + " |")
    for row in rows:
        print("| " + " | ".join(format_value(row.get(field, "")) for field in fields) + " |")


def write_csv(rows, fields, csv_path):
    with open(csv_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="Result .npz files to compare")
    parser.add_argument("--labels", help="Comma-separated labels matching input files")
    parser.add_argument("--start", type=float, default=6.0, help="Stable-window start time")
    parser.add_argument("--end", type=float, default=None, help="Stable-window end time")
    parser.add_argument("--csv", help="Optional CSV output path")
    parser.add_argument(
        "--all-fields",
        action="store_true",
        help="Print every computed field instead of the compact paper table",
    )
    args = parser.parse_args()

    labels = parse_labels(args.labels, args.files)
    rows = [
        compute_metrics(path, label, args.start, args.end)
        for path, label in zip(args.files, labels)
    ]
    fields = ordered_fields(rows)
    if not args.all_fields:
        fields = fields[: fields.index("file") + 1]

    print_markdown(rows, fields)
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        write_csv(rows, fields, args.csv)
        print(f"\nSaved CSV: {args.csv}")


if __name__ == "__main__":
    main()
