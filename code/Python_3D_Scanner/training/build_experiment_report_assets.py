"""Build reproducible loss summaries and figures for the final Word report."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "report_assets"
TELEMETRY_PATH = PROJECT_ROOT / "outputs" / "formal_gpu_telemetry.csv"

RUNS = [
    {
        "label": "Pilot 200 (256)",
        "path": Path(r"D:\code\DiffSynth-Studio\models\train\building_pilot_200_lora\loss.csv"),
        "rolling": 10,
    },
    {
        "label": "Dual smoke (256)",
        "path": Path(r"D:\code\DiffSynth-Studio\models\train\building_dual_3060_smoke_lossavg_lora\loss.csv"),
        "rolling": 1,
    },
    {
        "label": "Dual smoke (384)",
        "path": Path(r"D:\code\DiffSynth-Studio\models\train\building_dual_3060_smoke_384_lora\loss.csv"),
        "rolling": 1,
    },
    {
        "label": "Balanced 500 (384)",
        "path": Path(r"D:\code\DiffSynth-Studio\models\train\building_train_500_balanced_384_cooled_lora\loss.csv"),
        "rolling": 10,
    },
    {
        "label": "Formal 9560 clips (256)",
        "path": Path(r"D:\code\DiffSynth-Studio\models\train\building_train_1912_all_256_cooled_lora\loss.csv"),
        "rolling": 50,
    },
]


def read_loss(path: Path) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    values: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("key") != "loss":
                continue
            value = float(row["value"])
            if math.isfinite(value):
                steps.append(int(row["step"]))
                values.append(value)
    return steps, values


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values[:]
    result: list[float] = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        result.append(running / min(index + 1, window))
    return result


def load_font(size: int, bold: bool = False):
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def build_loss_figure(plot_data, output_path: Path) -> None:
    width = 1800
    panel_height = 390
    top = 95
    image = Image.new("RGB", (width, top + panel_height * len(plot_data)), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(34, bold=True)
    panel_font = load_font(23, bold=True)
    label_font = load_font(18)
    draw.text((70, 28), "Training loss by experiment", fill="#172B4D", font=title_font)

    for panel_index, (run, steps, values) in enumerate(plot_data):
        panel_top = top + panel_index * panel_height
        left, right = 135, width - 70
        chart_top, chart_bottom = panel_top + 70, panel_top + panel_height - 65
        smoothed = rolling_mean(values, int(run["rolling"]))
        ymin = min(min(values), min(smoothed))
        ymax = max(max(values), max(smoothed))
        padding = max((ymax - ymin) * 0.08, 1e-6)
        ymin -= padding
        ymax += padding
        xmin, xmax = min(steps), max(steps)
        if xmax == xmin:
            xmax = xmin + 1

        draw.text((70, panel_top + 14), run["label"], fill="#172B4D", font=panel_font)
        for tick in range(5):
            y = chart_top + (chart_bottom - chart_top) * tick / 4
            value = ymax - (ymax - ymin) * tick / 4
            draw.line((left, y, right, y), fill="#E4E8ED", width=2)
            draw.text((15, y - 10), f"{value:.3f}", fill="#5E6C84", font=label_font)
        draw.line((left, chart_bottom, right, chart_bottom), fill="#7A869A", width=2)
        draw.line((left, chart_top, left, chart_bottom), fill="#7A869A", width=2)

        def point(step: int, value: float) -> tuple[float, float]:
            x = left + (step - xmin) / (xmax - xmin) * (right - left)
            y = chart_bottom - (value - ymin) / (ymax - ymin) * (chart_bottom - chart_top)
            return x, y

        raw_points = [point(step, value) for step, value in zip(steps, values)]
        smooth_points = [point(step, value) for step, value in zip(steps, smoothed)]
        if len(raw_points) >= 2:
            draw.line(raw_points, fill="#A9BCD0", width=2)
            draw.line(smooth_points, fill="#2E74B5", width=5)
        else:
            x, y = raw_points[0]
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#2E74B5")
        draw.text((left, chart_bottom + 15), str(xmin), fill="#5E6C84", font=label_font)
        end_label = str(max(steps))
        end_width = draw.textbbox((0, 0), end_label, font=label_font)[2]
        draw.text((right - end_width, chart_bottom + 15), end_label, fill="#5E6C84", font=label_font)
        legend = f"raw + rolling mean ({run['rolling']})"
        legend_width = draw.textbbox((0, 0), legend, font=label_font)[2]
        draw.text((right - legend_width, panel_top + 22), legend, fill="#2E74B5", font=label_font)

    image.save(output_path, format="PNG", optimize=True)


def build_telemetry_summary() -> Path | None:
    if not TELEMETRY_PATH.is_file():
        return None
    groups: dict[tuple[str, int], list[dict]] = {}
    with TELEMETRY_PATH.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["stage"], int(row["gpu"]))
            groups.setdefault(key, []).append(row)
    summaries = []
    for (stage, gpu), rows in sorted(groups.items()):
        temperatures = [float(row["temperature_c"]) for row in rows]
        utilization = [float(row["utilization_pct"]) for row in rows]
        memory = [float(row["memory_mib"]) for row in rows]
        power = [float(row["power_w"]) for row in rows]
        summaries.append(
            {
                "stage": stage,
                "gpu": gpu,
                "samples": len(rows),
                "first_timestamp": rows[0]["timestamp"],
                "last_timestamp": rows[-1]["timestamp"],
                "temperature_avg_c": sum(temperatures) / len(temperatures),
                "temperature_max_c": max(temperatures),
                "utilization_avg_pct": sum(utilization) / len(utilization),
                "utilization_max_pct": max(utilization),
                "memory_max_mib": max(memory),
                "power_avg_w": sum(power) / len(power),
                "power_max_w": max(power),
            }
        )
    output = OUTPUT_DIR / "gpu_telemetry_summary.json"
    output.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().astimezone().isoformat(),
                "source": str(TELEMETRY_PATH),
                "groups": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    plot_data = []
    for run in RUNS:
        path = run["path"]
        if not path.is_file():
            continue
        steps, values = read_loss(path)
        if not values:
            continue
        summaries.append(
            {
                "label": run["label"],
                "path": str(path),
                "rows": len(values),
                "first": values[0],
                "last": values[-1],
                "minimum": min(values),
                "maximum": max(values),
                "average": sum(values) / len(values),
                "rolling_window": run["rolling"],
            }
        )
        plot_data.append((run, steps, values))

    summary_path = OUTPUT_DIR / "loss_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().astimezone().isoformat(),
                "runs": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    build_loss_figure(plot_data, OUTPUT_DIR / "loss_curves.png")
    telemetry_summary = build_telemetry_summary()
    print(summary_path)
    print(OUTPUT_DIR / "loss_curves.png")
    if telemetry_summary:
        print(telemetry_summary)


if __name__ == "__main__":
    main()
