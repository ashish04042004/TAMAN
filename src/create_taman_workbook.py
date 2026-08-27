from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference, ScatterChart, Series
from openpyxl.styles import Font


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _aqi_bucket(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


def _write_table(ws, start_row: int, headers: list[str], rows: list[list]) -> None:
    for c, h in enumerate(headers, start=1):
        cell = ws.cell(row=start_row, column=c, value=h)
        cell.font = Font(bold=True)
    for r_idx, row in enumerate(rows, start=start_row + 1):
        for c_idx, val in enumerate(row, start=1):
            ws.cell(row=r_idx, column=c_idx, value=val)


def _format_metric_cells(wb: Workbook) -> None:
    metric_sheets = ["Final_Results", "Ablation_Study", "SOTA_Comparison", "Per_Category_Analysis", "Training_Curves"]
    for name in metric_sheets:
        if name not in wb.sheetnames:
            continue
        ws = wb[name]
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
            for cell in row:
                if isinstance(cell.value, (float, int)):
                    header = ws.cell(row=1, column=cell.column).value
                    header = str(header) if header is not None else ""
                    if "R2" in header:
                        cell.number_format = "0.0000"
                    elif "F1" in header:
                        cell.number_format = "0.000"
                    elif any(x in header for x in ("MAE", "RMSE", "loss", "Delta")):
                        cell.number_format = "0.000"
                    elif "improvement" in header.lower() or "%" in header:
                        cell.number_format = "0.00"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "taman_experiment_workbook.xlsx"

    train_csv = root / "data" / "processed" / "train.csv"
    val_csv = root / "data" / "processed" / "val.csv"
    test_csv = root / "data" / "processed" / "test.csv"
    eval_v3 = root / "outputs" / "eval" / "v3" / "test_metrics.json"
    eval_v4 = root / "outputs" / "eval" / "v4" / "test_metrics.json"
    eval_taman = root / "outputs" / "eval" / "taman_r18" / "test_metrics.json"
    hist_taman = root / "outputs" / "train_history_taman_r18.json"
    pred_taman = root / "outputs" / "eval" / "taman_r18" / "test_predictions.csv"
    pred_v3 = root / "outputs" / "eval" / "v3" / "test_predictions.csv"

    train_df = pd.read_csv(train_csv) if train_csv.exists() else pd.DataFrame()
    val_df = pd.read_csv(val_csv) if val_csv.exists() else pd.DataFrame()
    test_df = pd.read_csv(test_csv) if test_csv.exists() else pd.DataFrame()
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True) if len(train_df) + len(val_df) + len(test_df) else pd.DataFrame()

    m_v3 = _read_json(eval_v3)
    m_v4 = _read_json(eval_v4)
    m_t = _read_json(eval_taman)
    history = json.loads(hist_taman.read_text(encoding="utf-8")) if hist_taman.exists() else []
    pred_t = pd.read_csv(pred_taman) if pred_taman.exists() else pd.DataFrame()
    pred3 = pd.read_csv(pred_v3) if pred_v3.exists() else pd.DataFrame()

    # Canonical numbers from TAMAN report PDF tables (user-provided source of truth).
    # Report: "TAMAN_Project_Report (1).pdf", Table 1/2/3.
    report_final = {
        "v1": {"mae": 64.53, "rmse": 93.64, "r2": 0.27, "f1_macro": "N/A", "f1_weighted": "N/A"},
        "v3": {"mae": 46.37, "rmse": 76.14, "r2": 0.55, "f1_macro": 0.435, "f1_weighted": 0.591},
        "v4": {"mae": 48.33, "rmse": 76.29, "r2": 0.54, "f1_macro": 0.404, "f1_weighted": 0.565},
        "taman_r18": {"mae": 15.61, "rmse": 27.32, "r2": 0.94, "f1_macro": 0.653, "f1_weighted": 0.834},
    }

    wb = Workbook()
    wb.remove(wb.active)

    ws = wb.create_sheet("README_Overview")
    ws["A1"] = "AeroSight-AQI TAMAN Experiment Workbook"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A3"] = "Purpose"
    ws["B3"] = "Standard ML experiment workbook for dataset, configs, results, ablations, and charts."
    ws["A4"] = "Project"
    ws["B4"] = "Multimodal AQI regression with v3/v4 baselines and TAMAN-R18 temporal model."
    ws["A5"] = "Source"
    ws["B5"] = "Repository outputs/eval, outputs/train_history, and processed dataset CSVs."
    ws["A6"] = "Reproducibility"
    ws["B6"] = "Seed=42 in config; metrics from saved checkpoints and regenerated evaluation files."
    ws["A7"] = "Metric Reconciliation"
    ws["B7"] = (
        "Final_Results, SOTA_Comparison, and Ablation_Study use canonical numbers copied from TAMAN_Project_Report (1).pdf "
        "(Table 1/2/3). Training_Curves uses outputs/train_history_taman_r18.json validation-per-epoch logs; "
        "therefore curve points need not equal final test metrics."
    )
    _write_table(
        ws,
        9,
        ["Tab", "Purpose"],
        [
            ["Dataset", "Split counts, percentages, AQI bucket distribution, preprocessing notes."],
            ["Experiment_Configs", "Run-level hyperparameters and hardware."],
            ["Final_Results", "Model metrics with formula-based improvement vs baseline."],
            ["Ablation_Study", "Component on/off impact table."],
            ["SOTA_Comparison", "Prior methods vs TAMAN row."],
            ["Per_Category_Analysis", "TAMAN per-AQI bucket error analysis."],
            ["Training_Curves", "Per-epoch TAMAN training and validation metrics."],
            ["Charts", "Charts linked to data tabs."],
            ["Scatter_Data", "Helper data source for the predicted-vs-actual scatter chart."],
        ],
    )

    ws = wb.create_sheet("Dataset")
    n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
    n_total = n_train + n_val + n_test
    split_rows = [
        ["Train", n_train, n_train / n_total if n_total else 0],
        ["Validation", n_val, n_val / n_total if n_total else 0],
        ["Test", n_test, n_test / n_total if n_total else 0],
        ["Total", n_total, 1.0 if n_total else 0],
    ]
    _write_table(ws, 1, ["Split", "Count", "Percent"], split_rows)
    for r in range(2, 6):
        ws.cell(row=r, column=3).number_format = "0.00%"
    ws["A7"] = "AQI bucket distribution (all splits)"
    ws["A7"].font = Font(bold=True)
    if not all_df.empty and "aqi" in all_df.columns:
        bucket_df = all_df.copy()
        bucket_df["bucket"] = bucket_df["aqi"].apply(_aqi_bucket)
        order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
        b_counts = bucket_df["bucket"].value_counts().reindex(order, fill_value=0)
        bucket_rows = [[k, int(v), (int(v) / len(bucket_df)) if len(bucket_df) else 0] for k, v in b_counts.items()]
    else:
        bucket_rows = [[k, None, None] for k in ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]]
    _write_table(ws, 8, ["AQI Bucket", "Count", "Percent"], bucket_rows)
    for r in range(9, 15):
        ws.cell(row=r, column=3).number_format = "0.00%"
    _write_table(
        ws,
        16,
        ["Preprocessing Notes"],
        [
            ["Image transform: train resize/crop/flip + ImageNet normalization; eval resize + normalization."],
            ["Metadata: numeric coercion, z-score standardization using train mean/std."],
            ["Split strategy: AQI-bin stratified train/val/test in prepare_splits.py."],
            ["Temporal TAMAN: seq_len=4 with max frame gap filter on parsed image ids."],
        ],
    )
    _write_table(ws, 22, ["Metadata Fields"], [[x] for x in ["humidity", "temperature", "hour", "is_night", "season_code", "pm25", "pm10"]])

    ws = wb.create_sheet("Experiment_Configs")
    headers = [
        "Run ID",
        "Model Name",
        "Backbone",
        "Sequence Length",
        "Use Metadata",
        "Use LSTM",
        "Use Adaptive Fusion",
        "Optimizer",
        "Learning Rate",
        "Batch Size",
        "Epochs",
        "Loss Function",
        "Hardware",
        "Notes",
    ]
    rows = [
        ["v1", "Baseline multimodal", "ResNet18", 1, "Y", "N", "Y (softmax gate)", "AdamW", "5e-5", 8, 10, "Weighted Huber", "GPU (historical)", "Historical baseline; current ckpt schema mismatch"],
        ["v3", "Multimodal", "ResNet18", 1, "Y", "N", "Y (softmax gate)", "AdamW", "5e-5", 8, 15, "Weighted Huber", "GPU", "Best single-image run"],
        ["v4", "Multimodal", "ResNet50", 1, "Y", "N", "Y (softmax gate)", "AdamW", "5e-5", 8, 15, "Weighted Huber", "GPU", "Negative result vs v3"],
        ["taman_r18", "TAMAN", "ResNet18", 4, "Y", "Y", "Y (sigmoid gate)", "AdamW", "5e-5", 4, 10, "Weighted Huber", "GPU (GTX 1650 class)", "Temporal model"],
    ]
    _write_table(ws, 1, headers, rows)

    ws = wb.create_sheet("Final_Results")
    fr_headers = ["Run ID", "Model", "MAE", "RMSE", "R2", "F1_macro", "F1_weighted", "MAE Improvement vs v3 (%)"]
    fr_rows = [
        ["v1", "Baseline historical", report_final["v1"]["mae"], report_final["v1"]["rmse"], report_final["v1"]["r2"], report_final["v1"]["f1_macro"], report_final["v1"]["f1_weighted"], None],
        ["v3", "Multimodal ResNet18", report_final["v3"]["mae"], report_final["v3"]["rmse"], report_final["v3"]["r2"], report_final["v3"]["f1_macro"], report_final["v3"]["f1_weighted"], 0],
        ["v4", "Multimodal ResNet50", report_final["v4"]["mae"], report_final["v4"]["rmse"], report_final["v4"]["r2"], report_final["v4"]["f1_macro"], report_final["v4"]["f1_weighted"], None],
        ["taman_r18", "TAMAN ResNet18", report_final["taman_r18"]["mae"], report_final["taman_r18"]["rmse"], report_final["taman_r18"]["r2"], report_final["taman_r18"]["f1_macro"], report_final["taman_r18"]["f1_weighted"], None],
    ]
    _write_table(ws, 1, fr_headers, fr_rows)
    ws["A7"] = "Note: v1 F1 fields are N/A because historical v1 artifact did not include CPCB classification metrics."
    ws["H4"] = "=(C3/C4-1)*100"
    ws["H5"] = "=(C3/C5-1)*100"
    ws["H4"].number_format = "0.00"
    ws["H5"].number_format = "0.00"

    ws = wb.create_sheet("Ablation_Study")
    ab_headers = [
        "Ablation ID",
        "Configuration",
        "Image Branch",
        "Metadata Branch",
        "LSTM",
        "Adaptive Fusion",
        "MAE",
        "RMSE",
        "R2",
        "Delta MAE vs previous",
        "Status",
    ]
    ab_rows = [
        ["A1", "CNN only", "Y", "N", "N", "N", 52.84, 81.32, 0.49, None, "From TAMAN report Table 3"],
        ["A2", "CNN + metadata", "Y", "Y", "N", "N", 46.37, 76.14, 0.55, None, "From TAMAN report Table 3"],
        ["A3", "CNN + LSTM", "Y", "N", "Y", "N", 28.72, 49.65, 0.81, None, "From TAMAN report Table 3"],
        ["A4", "CNN + metadata + LSTM (no adaptive fusion)", "Y", "Y", "Y", "N", 20.43, 35.18, 0.90, None, "From TAMAN report Table 3"],
        ["A5", "Full TAMAN", "Y", "Y", "Y", "Y", 15.61, 27.32, 0.94, None, "From TAMAN report Table 3"],
    ]
    _write_table(ws, 1, ab_headers, ab_rows)
    ws["J3"] = "=G2-G3"
    ws["J4"] = "=G3-G4"
    ws["J5"] = "=G4-G5"
    ws["J6"] = "=G5-G6"

    ws = wb.create_sheet("SOTA_Comparison")
    sota_headers = ["Method", "Citation", "MAE", "RMSE", "R2", "F1_macro", "% MAE improvement vs best prior", "Status"]
    sota_rows = [
        [
            "Mondal CNN",
            "Mondal et al. (2024), Scientific Reports, 14(1):1627",
            36.69,
            60.34,
            None,
            0.61,
            None,
            "From TRAQID Table 3 + TAMAN report Table 2",
        ],
        [
            "Kalajdjieski (Inception+MLP)",
            "Kalajdjieski et al. (2020), Remote Sensing 12(24):4142",
            41.07,
            68.28,
            None,
            0.56,
            None,
            "From TRAQID Table 3 + TAMAN report Table 2",
        ],
        [
            "Nilesh et al. (RF + features)",
            "Nilesh et al. (2022), IEEE WF-IoT",
            33.19,
            54.24,
            None,
            0.71,
            None,
            "From TRAQID Table 3 + TAMAN report Table 2",
        ],
        [
            "AQC-Net (best prior)",
            "Zhang et al. (2020), Science of the Total Environment 724:138178",
            31.67,
            52.21,
            None,
            0.74,
            None,
            "From TRAQID Table 3 + TAMAN report Table 2",
        ],
        ["TAMAN-R18 (proposed)", "TAMAN Project Report (2026) Table 2", 15.61, 27.32, 0.94, 0.834, None, "Source of truth from user PDF"],
    ]
    _write_table(ws, 1, sota_headers, sota_rows)
    ws["G6"] = '=IF(COUNT(C2:C5)=0, "", (MIN(C2:C5)/C6-1)*100)'

    ws = wb.create_sheet("Per_Category_Analysis")
    pc_headers = ["AQI Bucket", "Count", "MAE", "RMSE"]
    if not pred_t.empty and {"target_aqi", "pred_aqi"}.issubset(pred_t.columns):
        df = pred_t.copy()
        df["bucket"] = df["target_aqi"].apply(_aqi_bucket)
        rows = []
        for b in ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]:
            s = df[df["bucket"] == b]
            if len(s) == 0:
                rows.append([b, 0, None, None])
            else:
                err = (s["pred_aqi"] - s["target_aqi"]).abs()
                rmse = float(((s["pred_aqi"] - s["target_aqi"]) ** 2).mean() ** 0.5)
                rows.append([b, int(len(s)), float(err.mean()), rmse])
    else:
        rows = [[b, None, None, None] for b in ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]]
    _write_table(ws, 1, pc_headers, rows)
    ws["A10"] = "Confusion matrix can be added below if class-level predictions are exported."

    ws = wb.create_sheet("Training_Curves")
    tc_headers = ["epoch", "train_loss", "val_mae", "val_rmse", "val_r2", "val_f1_macro"]
    tc_rows = []
    for d in history:
        tc_rows.append([d.get("epoch"), d.get("train_loss"), d.get("mae"), d.get("rmse"), d.get("r2"), d.get("f1_macro", d.get("f1"))])
    _write_table(ws, 1, tc_headers, tc_rows if tc_rows else [[None, None, None, None, None, None]])

    ws = wb.create_sheet("Charts")
    ws["A1"] = "Charts referencing workbook data tabs"
    ws["A1"].font = Font(bold=True)

    bar = BarChart()
    bar.title = "Model Comparison: MAE"
    bar.y_axis.title = "MAE"
    data = Reference(wb["Final_Results"], min_col=3, min_row=1, max_row=5)
    cats = Reference(wb["Final_Results"], min_col=1, min_row=2, max_row=5)
    bar.add_data(data, titles_from_data=True)
    bar.set_categories(cats)
    bar.height = 7
    bar.width = 11
    ws.add_chart(bar, "A3")

    l1 = LineChart()
    l1.title = "TAMAN Training Curves"
    l1.y_axis.title = "Metric"
    x = Reference(wb["Training_Curves"], min_col=1, min_row=2, max_row=1 + max(len(tc_rows), 1))
    d = Reference(wb["Training_Curves"], min_col=2, min_row=1, max_col=4, max_row=1 + max(len(tc_rows), 1))
    l1.add_data(d, titles_from_data=True)
    l1.set_categories(x)
    l1.height = 7
    l1.width = 11
    ws.add_chart(l1, "A20")

    bar2 = BarChart()
    bar2.title = "Per-Category MAE (TAMAN)"
    bar2.y_axis.title = "MAE"
    d2 = Reference(wb["Per_Category_Analysis"], min_col=3, min_row=1, max_row=7)
    c2 = Reference(wb["Per_Category_Analysis"], min_col=1, min_row=2, max_row=7)
    bar2.add_data(d2, titles_from_data=True)
    bar2.set_categories(c2)
    bar2.height = 7
    bar2.width = 11
    ws.add_chart(bar2, "N3")

    if not pred3.empty and {"target_aqi", "pred_aqi"}.issubset(pred3.columns):
        s_ws = wb.create_sheet("Scatter_Data")
        _write_table(s_ws, 1, ["target_aqi", "pred_aqi"], pred3[["target_aqi", "pred_aqi"]].head(500).values.tolist())
        scatter = ScatterChart()
        scatter.title = "Predicted vs Actual (v3 sample)"
        scatter.x_axis.title = "Actual AQI"
        scatter.y_axis.title = "Predicted AQI"
        xvalues = Reference(s_ws, min_col=1, min_row=2, max_row=min(501, len(pred3) + 1))
        yvalues = Reference(s_ws, min_col=2, min_row=2, max_row=min(501, len(pred3) + 1))
        series = Series(yvalues, xvalues, title="v3")
        scatter.series.append(series)
        scatter.height = 7
        scatter.width = 11
        ws.add_chart(scatter, "N20")

    _format_metric_cells(wb)

    for sheet in wb.worksheets:
        for col in sheet.columns:
            max_len = 0
            col_letter = col[0].column_letter
            for cell in col:
                val = "" if cell.value is None else str(cell.value)
                if len(val) > max_len:
                    max_len = len(val)
            sheet.column_dimensions[col_letter].width = min(max(12, max_len + 2), 60)

    wb.save(out_path)
    print(f"Workbook created: {out_path}")


if __name__ == "__main__":
    main()
