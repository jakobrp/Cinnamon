import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from openpyxl import Workbook
except ImportError:  # pragma: no cover - optional dependency
    Workbook = None


DEFAULT_DATASET = "heart_failure"

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = BASE_DIR / "experiment_results" / DEFAULT_DATASET

# Ordner, die immer vorhanden sind und fest zugeordnet werden.
DEFAULT_SUBFOLDERS = [
    "anon/tech_eval",
    "anon/risk_eval",
    "anon_synth/tech_eval",
    "anon_synth/risk_eval",
    "real_risk",
]

# Ordnernamen, die NICHT als zusaetzliches Label gelten. Sind als Baseline schon vorhanden
FIXED_FOLDERS = {"anon", "anon_synth", "real_risk"}

# Baseline-Varianten in fester Reihenfolge, bestimmen die ersten Spalten in der Excel.
BASE_LABELS = ["real", "anon", "anon_synth"]

TECH_EVAL_FOLDERS = {
    "anon": "anon/tech_eval",
    "anon_synth": "anon_synth/tech_eval",
}

TECH_EVAL_METRICS = [
    "overall_resemblance",
    "overall_utility",
]

COMBINED_UTILITY_METRIC = "combined_utility"

RISK_FOLDERS = {
    "anon": "anon/risk_eval",
    "anon_synth": "anon_synth/risk_eval",
    "real": "real_risk",
}

RISK_METRICS = [
    "total_risk_score",
]

PRIVACY_METRIC = "privacy_score"


#Methode von Claude generiert
def discover_extras(root: Path) -> List[str]:
    """Findet automatisch alle Label-Ordner ausser den festen, die einen tech_eval-Unterordner haben."""
    if not root.exists():
        return []
    extras: List[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in FIXED_FOLDERS:
            continue
        # Nur Ordner mit tech_eval zaehlen als Ergebnis eines Laufs (z.B. dpgan_eps1/tech_eval).
        if (child / "tech_eval").is_dir():
            extras.append(child.name)
    return extras


def load_json_files(folder: Path) -> List[Tuple[Path, Any]]:
    if not folder.exists():
        print(f"Warning: folder not found, skipping: {folder}")
        return []
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")

    results: List[Tuple[Path, Any]] = []
    for json_path in sorted(folder.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as handle:
            try:
                data = json.load(handle)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {json_path}") from exc
        results.append((json_path, data))
    return results


def load_all(root: Path, folders: List[str]) -> Dict[str, List[Tuple[Path, Any]]]:
    loaded: Dict[str, List[Tuple[Path, Any]]] = {}
    for folder in folders:
        folder_path = root / folder
        loaded[folder] = load_json_files(folder_path)
    return loaded


def extract_overview_metrics(payload: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(payload, dict):
        return {}
    overview = payload.get("Overview") or payload.get("overview")
    if not isinstance(overview, dict):
        return {}
    aggregated = overview.get("aggregated_metrics", [])
    if isinstance(aggregated, dict):
        aggregated = [aggregated]  # Dict in einelementige Liste verpacken
    if not isinstance(aggregated, list):
        return {}

    metrics: Dict[str, Dict[str, Any]] = {}
    for item in aggregated:
        if not isinstance(item, dict):
            continue
        for metric in TECH_EVAL_METRICS:
            if metric in metrics:
                continue
            metric_payload = item.get(metric)
            if not isinstance(metric_payload, dict):
                continue
            values = metric_payload.get("values")
            if not isinstance(values, dict):
                continue
            metrics[metric] = {
                "real": values.get("real"),
                "synthetic": values.get("synthetic"),
            }
    return metrics


def extract_risk_metrics(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    metrics: Dict[str, Any] = {}
    for metric in RISK_METRICS:
        if metric in payload:
            metrics[metric] = payload.get(metric)
    return metrics

#Zu eigener Funktion gemacht
def combined_value(resemblance: Any, utility: Any) -> Any:
    # Fehlt einer der beiden Werte, bleibt die Zelle leer statt falsch gemittelt.
    if resemblance is None or utility is None:
        return None
    return (resemblance + utility) / 2


def build_rows(
    anon_items: List[Tuple[Path, Any]],
    anon_synth_items: List[Tuple[Path, Any]],
    anon_risk_items: List[Tuple[Path, Any]],
    anon_synth_risk_items: List[Tuple[Path, Any]],
    real_risk_items: List[Tuple[Path, Any]],
    # Neu: {label: (tech_eval-Dateien, risk_eval-Dateien)} fuer alle DP-Varianten
    extras: Dict[str, Tuple[List[Tuple[Path, Any]], List[Tuple[Path, Any]]]],
) -> List[Dict[str, Any]]:
    lengths = [
        len(anon_items),
        len(anon_synth_items),
        len(anon_risk_items),
        len(anon_synth_risk_items),
        len(real_risk_items),
    ]
    # Auch die Anzahl Runs der DP-Varianten beruecksichtigen (z.B. 25 Runs statt 5 bei der Baseline).
    for tech_items, risk_items in extras.values():
        lengths.append(len(tech_items))
        lengths.append(len(risk_items))
    max_runs = max(lengths) if lengths else 0

    rows: List[Dict[str, Any]] = []
    for run_index in range(max_runs):
        row: Dict[str, Any] = {"run": run_index + 1}

        anon_metrics = (
            extract_overview_metrics(anon_items[run_index][1])
            if run_index < len(anon_items)
            else {}
        )
        anon_synth_metrics = (
            extract_overview_metrics(anon_synth_items[run_index][1])
            if run_index < len(anon_synth_items)
            else {}
        )
        anon_risk = (
            extract_risk_metrics(anon_risk_items[run_index][1])
            if run_index < len(anon_risk_items)
            else {}
        )
        anon_synth_risk = (
            extract_risk_metrics(anon_synth_risk_items[run_index][1])
            if run_index < len(anon_synth_risk_items)
            else {}
        )
        real_risk = (
            extract_risk_metrics(real_risk_items[run_index][1])
            if run_index < len(real_risk_items)
            else {}
        )

        for metric in TECH_EVAL_METRICS:
            real_value = None
            if metric in anon_metrics and anon_metrics[metric].get("real") is not None:
                real_value = anon_metrics[metric].get("real")
            elif metric in anon_synth_metrics:
                real_value = anon_synth_metrics[metric].get("real")

            row[f"{metric}_real"] = real_value
            row[f"{metric}_anon"] = anon_metrics.get(metric, {}).get("synthetic")
            row[f"{metric}_anon_synth"] = anon_synth_metrics.get(metric, {}).get("synthetic")

        # Combined Utility = (Resemblance + Utility) / 2 fuer die Baseline
        row[f"{COMBINED_UTILITY_METRIC}_real"] = combined_value(
            row.get("overall_resemblance_real"), row.get("overall_utility_real")
        )
        row[f"{COMBINED_UTILITY_METRIC}_anon"] = combined_value(
            row.get("overall_resemblance_anon"), row.get("overall_utility_anon")
        )
        row[f"{COMBINED_UTILITY_METRIC}_anon_synth"] = combined_value(
            row.get("overall_resemblance_anon_synth"), row.get("overall_utility_anon_synth")
        )

        for metric in RISK_METRICS:
            real_value = real_risk.get(metric)
            anon_value = anon_risk.get(metric)
            anon_synth_value = anon_synth_risk.get(metric)
            row[f"{PRIVACY_METRIC}_real"] = (
                1.0 - real_value if real_value is not None else None
            )
            row[f"{PRIVACY_METRIC}_anon"] = (
                1.0 - anon_value if anon_value is not None else None
            )
            row[f"{PRIVACY_METRIC}_anon_synth"] = (
                1.0 - anon_synth_value if anon_synth_value is not None else None
            )

        # Zusaetzliche Label (dpgan_eps5, pategan_eps1, ...) generisch behandeln.
        for label, (tech_items, risk_items) in extras.items():
            tech_metrics = (
                extract_overview_metrics(tech_items[run_index][1])
                if run_index < len(tech_items)
                else {}
            )
            risk_metrics = (
                extract_risk_metrics(risk_items[run_index][1])
                if run_index < len(risk_items)
                else {}
            )

            # Nur "synthetic" uebernehmen, der "real"-Wert steht schon in der real-Spalte.
            for metric in TECH_EVAL_METRICS:
                row[f"{metric}_{label}"] = tech_metrics.get(metric, {}).get("synthetic")

            row[f"{COMBINED_UTILITY_METRIC}_{label}"] = combined_value(
                row.get(f"overall_resemblance_{label}"),
                row.get(f"overall_utility_{label}"),
            )

            # Privacy Score = 1 - Risiko, wie bei der Baseline
            for metric in RISK_METRICS:
                risk_value = risk_metrics.get(metric)
                row[f"{PRIVACY_METRIC}_{label}"] = (
                    1.0 - risk_value if risk_value is not None else None
                )

        rows.append(row)
    return rows


def write_excel(rows: List[Dict[str, Any]], output_path: Path, labels: List[str]) -> None:
    """labels wird uebergeben, damit auch die DP-Varianten eigene Spalten bekommen."""
    if Workbook is None:
        raise RuntimeError("openpyxl is required to write Excel files. Install it with: pip install openpyxl")
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "evaluation"

    # Spalten nach Metrik gruppiert: overall_resemblance_real, ..._anon, ..._dpgan_eps1, dann naechste Metrik
    headers = ["run"]
    for metric in TECH_EVAL_METRICS + [COMBINED_UTILITY_METRIC, PRIVACY_METRIC]:
        headers.extend(f"{metric}_{label}" for label in labels)

    sheet.append(headers)
    for row in rows:
        # Fehlende Werte (z.B. Baseline hat weniger Runs) werden zu leeren Zellen.
        sheet.append([row.get(header) for header in headers])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load JSON result files for aggregation.")
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Base folder containing result subfolders (e.g., simulation/experiment_results/heart_failure).",
    )
    # Ersetzt das alte --folders: optional bestimmte Varianten angeben, sonst automatisch erkennen.
    parser.add_argument(
        "--extras",
        nargs="*",
        help="Explicit extra label folders (e.g. dpgan_eps5 pategan_eps1). Default: auto-detect.",
    )
    parser.add_argument(
        "--output",
        help="Output Excel file path for the evaluation summary.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output) if args.output else root / "evaluation_summary.xlsx"

    # Ohne --extras werden alle Label-Ordner im Datensatzordner gesucht.
    extra_names = args.extras if args.extras is not None else discover_extras(root)
    if extra_names:
        print(f"Extra labels: {', '.join(extra_names)}")
    else:
        print("Extra labels: none")

    # Zu ladende Ordner: feste Baseline-Ordner plus tech_eval/risk_eval je DP-Variante.
    folders = list(DEFAULT_SUBFOLDERS)
    for name in extra_names:
        folders.append(f"{name}/tech_eval")
        folders.append(f"{name}/risk_eval")

    loaded = load_all(root, folders)
    for folder, items in loaded.items():
        print(f"{folder}: {len(items)} JSON files")

    anon_items = loaded.get(TECH_EVAL_FOLDERS["anon"], [])
    anon_synth_items = loaded.get(TECH_EVAL_FOLDERS["anon_synth"], [])
    anon_risk_items = loaded.get(RISK_FOLDERS["anon"], [])
    anon_synth_risk_items = loaded.get(RISK_FOLDERS["anon_synth"], [])
    real_risk_items = loaded.get(RISK_FOLDERS["real"], [])

    # Pro DP-Variante die geladenen tech_eval- und risk_eval-Dateien zusammenfassen.
    extras = {
        name: (
            loaded.get(f"{name}/tech_eval", []),
            loaded.get(f"{name}/risk_eval", []),
        )
        for name in extra_names
    }

    rows = build_rows(
        anon_items,
        anon_synth_items,
        anon_risk_items,
        anon_synth_risk_items,
        real_risk_items,
        extras,
    )
    # Spalten: erst Baseline, dann alle DP-Varianten
    write_excel(rows, output, BASE_LABELS + extra_names)
    print(f"Wrote evaluation summary to {output}")


if __name__ == "__main__":
    main()