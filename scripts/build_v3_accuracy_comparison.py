"""Build an accuracy-first statistical comparison for all evaluated V3 models.

The script deliberately separates the fixed development validation results from
the final five-fold OOF results.  They use different evaluation protocols and
must not be placed in one statistical ranking.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_ROOT = Path(r"P:\NexarCollisionData")
REPORT_ROOT = DATA_ROOT / "reports_v3"


def read_json(filename: str) -> dict:
    return json.loads((REPORT_ROOT / filename).read_text(encoding="utf-8"))


def metric_row(
    *,
    model_id: str,
    model: str,
    protocol: str,
    status: str,
    metrics: dict,
    input_type: str,
    notes: str,
    arbitrary_mp4: str,
) -> dict:
    return {
        "model_id": model_id,
        "model": model,
        "protocol": protocol,
        "status": status,
        "input_type": input_type,
        "arbitrary_mp4": arbitrary_mp4,
        "threshold": metrics.get("threshold", ""),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics.get("roc_auc", ""),
        "pr_auc": metrics.get("pr_auc", ""),
        "notes": notes,
    }


def safe_float(value: str | float | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def csv_metric_rows() -> list[dict]:
    """Read all non-learned E1 diagnostic configurations from the source CSV."""
    path = REPORT_ROOT / "e1_development_ensemble_comparison.csv"
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for source in csv.DictReader(handle):
            label = source["method"]
            # D1 itself is already read from its canonical D1 summary.  Do
            # not list precisely the same configuration a second time merely
            # because E1 used it as a reference prediction vector.
            if label.startswith("D1 frozen"):
                continue
            metrics = {key: safe_float(source[key]) for key in ("threshold", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")}
            notes = source["notes"]
            if "AND" in label:
                notes += "; Recall کمتر از قید ایمنی 0.85 است"
            if label.startswith("A2-MP"):
                model_id = "A2-MP-dev"
            elif "equal raw" in label:
                model_id = "E1-mean"
            elif "AND" in label:
                model_id = "E1-AND"
            else:
                model_id = "E1-OR"
            rows.append(
                metric_row(
                    model_id=model_id,
                    model=label,
                    protocol="development_fixed_120_validation",
                    status="diagnostic_only" if "E1-D" in label else "base_reference",
                    metrics=metrics,
                    input_type="video only",
                    notes=notes,
                    arbitrary_mp4="yes",
                )
            )
    return rows


def collect_rows() -> list[dict]:
    rows: list[dict] = []

    # The only final, held-out comparison.  Keep these separate from all
    # development numbers below.
    final_oof = read_json("final_oof_summary.json")
    for metrics in final_oof["primary_comparison"]:
        if metrics["model"] == "a2mp":
            model_id, model, input_type = "A2-MP", "A2-MP: ResNet18 + mean-max pooling", "video RGB"
        else:
            model_id, model, input_type = "D1", "D1: ResNet18 + RGB/frame-difference fusion", "video RGB + motion"
        rows.append(
            metric_row(
                model_id=model_id,
                model=model,
                protocol="final_five_fold_outer_OOF_full_MP4",
                status="final_comparable",
                metrics=metrics,
                input_type=input_type,
                notes="۵-fold OOF؛ ۶۰۰ احتمال out-of-fold؛ آستانه فقط پس از فریز مدل انتخاب شده است",
                arbitrary_mp4="yes",
            )
        )

    # E1 was subsequently evaluated with fold-local threshold selection.  It
    # is listed in the final OOF section, but its decision-policy status makes
    # clear why it was not silently substituted for deployed D1.
    e1_final_path = REPORT_ROOT / "e1_final_oof_summary.json"
    if e1_final_path.is_file():
        e1_final = read_json("e1_final_oof_summary.json")
        for metrics in e1_final["comparison"]:
            label = metrics["model"]
            if label == "D1 current deployed policy":
                continue  # Already represented by the canonical D1 OOF row.
            if label == "D1 nested Accuracy policy":
                model_id, status, mp4 = "D1-nested-accuracy", "final_policy_diagnostic", "no"
                notes = "fold-local threshold selected for Accuracy; no single deployment threshold was handed off"
            elif label == "E1-AND accuracy-first":
                model_id, status, mp4 = "E1-AND-OOF-accuracy", "final_accuracy_only", "no"
                notes = "nested OOF; +0.50pp versus nested D1 but McNemar p=0.813 and Recall is 65%"
            else:
                model_id, status, mp4 = "E1-AND-OOF-safety", "final_rejected_safety", "no"
                notes = "nested OOF; inner safety constraint did not retain Recall >= 85% in outer OOF"
            rows.append(
                metric_row(
                    model_id=model_id,
                    model=label,
                    protocol="final_five_fold_outer_OOF_full_MP4",
                    status=status,
                    metrics=metrics,
                    input_type="video RGB + motion",
                    notes=notes,
                    arbitrary_mp4=mp4,
                )
            )

    curve_verifier_path = REPORT_ROOT / "d1_curve_verifier_summary.json"
    if curve_verifier_path.is_file():
        curve_summary = read_json("d1_curve_verifier_summary.json")
        curve_metrics = next(item for item in curve_summary["comparison"] if item["model"] == "D1 curve verifier")
        rows.append(
            metric_row(
                model_id="D1-curve-verifier",
                model="D1 probability-curve verifier",
                protocol="final_five_fold_outer_OOF_full_MP4",
                status="final_rejected_safety",
                metrics=curve_metrics,
                input_type="D1 ordered window probabilities only",
                notes="no new videos/decode/retraining; lower Accuracy than nested D1 and Recall only 57.33%",
                arbitrary_mp4="no",
            )
        )

    a2_hn = read_json("a2mp_hn1_summary.json")
    rows.append(
        metric_row(
            model_id="A2-MP-HN1",
            model="A2-MP + hard-negative mining",
            protocol="development_fixed_120_validation",
            status="development",
            metrics=a2_hn["primary_metrics_validation_selected_threshold"],
            input_type="video RGB",
            notes="توسعه؛ ۲۴۰ hard negative در آموزش",
            arbitrary_mp4="yes",
        )
    )

    amil = read_json("amil_summary.json")
    rows.append(
        metric_row(
            model_id="A-MIL",
            model="A-MIL: ResNet18 features + gated attention",
            protocol="development_fixed_120_validation",
            status="development",
            metrics=amil["metrics_validation_selected_threshold"],
            input_type="video RGB",
            notes="توسعه؛ multiple-instance learning",
            arbitrary_mp4="yes",
        )
    )

    a6 = read_json("a6_bilstm_additive_attention_summary.json")
    rows.append(
        metric_row(
            model_id="A6",
            model="A6: ResNet18 + BiLSTM + additive attention + FFN",
            protocol="development_fixed_120_validation",
            status="development",
            metrics=a6["primary_metrics_validation_selected_threshold"],
            input_type="video RGB",
            notes="توسعه؛ تمام featureها از cache مشترک آمده‌اند",
            arbitrary_mp4="yes",
        )
    )

    d1 = read_json("d1_summary.json")
    rows.append(
        metric_row(
            model_id="D1-dev",
            model="D1: ResNet18 + RGB/frame-difference fusion",
            protocol="development_fixed_120_validation",
            status="development",
            metrics=d1["primary_metrics_validation_selected_threshold"],
            input_type="video RGB + motion",
            notes="توسعه؛ نسخه‌ای که بعداً در OOF نیز برنده شد",
            arbitrary_mp4="yes",
        )
    )

    d2 = read_json("d2_summary.json")
    rows.extend(
        [
            metric_row(
                model_id="D2-meta-only",
                model="D2 control: metadata-only logistic regression",
                protocol="development_fixed_120_validation",
                status="ablation_only",
                metrics=d2["metadata_only"],
                input_type="weather + light + scene",
                notes="به‌تنهایی برای MP4 دلخواه قابل استفاده نیست؛ آزمون shortcut",
                arbitrary_mp4="no",
            ),
            metric_row(
                model_id="D2-video-only",
                model="D2 control: video-only D1-equivalent",
                protocol="development_fixed_120_validation",
                status="control",
                metrics=d2["video_only_control"],
                input_type="video RGB + motion",
                notes="کنترل D2؛ از نظر معماری همان شاخهٔ ویدیویی D1 است",
                arbitrary_mp4="yes",
            ),
            metric_row(
                model_id="D2-fusion",
                model="D2: video + metadata fusion",
                protocol="development_fixed_120_validation",
                status="development",
                metrics=d2["video_metadata_fusion"],
                input_type="video + weather + light + scene",
                notes="برای MP4 دلخواه مناسب نیست، چون metadata ورودی لازم دارد",
                arbitrary_mp4="no",
            ),
        ]
    )

    rows.extend(csv_metric_rows())

    # This reproduces a trainable head only; its source explicitly says the
    # new full-MP4 reproduction was pending, so it is reported but excluded
    # from every accuracy rank.
    reproduction = read_json("v3_baseline_reproduction.json")
    row = metric_row(
        model_id="A2-MP-reproduction",
        model="A2-MP reproduction head",
        protocol="head_reproduction_not_full_MP4",
        status="not_comparable",
        metrics=reproduction["reproduced_metrics"],
        input_type="cached sequence features",
        notes="خروجی head-only است؛ خود گزارش صراحتاً می‌گوید بازتولید full-MP4 انجام نشده",
        arbitrary_mp4="no",
    )
    rows.append(row)

    return rows


def add_ranks(rows: list[dict]) -> None:
    for protocol in sorted({row["protocol"] for row in rows}):
        candidates = [row for row in rows if row["protocol"] == protocol]
        for rank, row in enumerate(sorted(candidates, key=lambda item: item["accuracy"], reverse=True), start=1):
            row["accuracy_rank_within_protocol"] = rank


def percent(value: float | str | None) -> str:
    return "—" if value in (None, "") else f"{100 * float(value):.2f}%"


def number(value: float | str | None) -> str:
    return "—" if value in (None, "") else f"{float(value):.2f}"


def decimal(value: float | str | None) -> str:
    return "—" if value in (None, "") else f"{float(value):.3f}"


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| Rank | Model | Accuracy | Recall | F1 | PR-AUC | Threshold | MP4 دلخواه | وضعیت |",
        "|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    status_text = {
        "final_comparable": "نهایی/قابل‌مقایسه",
        "development": "development",
        "control": "کنترل",
        "ablation_only": "ablation",
        "diagnostic_only": "تشخیصی",
        "base_reference": "baseline",
        "not_comparable": "غیرقابل‌مقایسه",
        "final_policy_diagnostic": "سیاست تشخیصی",
        "final_accuracy_only": "فقط Accuracy؛ handoff نشده",
        "final_rejected_safety": "ردشده به‌دلیل Recall",
    }
    for row in sorted(rows, key=lambda item: item["accuracy"], reverse=True):
        lines.append(
            "| {rank} | {model} | {accuracy} | {recall} | {f1} | {pr_auc} | {threshold} | {mp4} | {status} |".format(
                rank=row["accuracy_rank_within_protocol"],
                model=row["model"],
                accuracy=percent(row["accuracy"]),
                recall=percent(row["recall"]),
                f1=decimal(row["f1"]),
                pr_auc=decimal(row["pr_auc"]),
                threshold=number(row["threshold"]),
                mp4={"yes": "بله", "no": "خیر", "not_yet": "نیاز به handoff"}[row["arbitrary_mp4"]],
                status=status_text[row["status"]],
            )
        )
    return "\n".join(lines)


def write_report(rows: list[dict], image_name: str) -> Path:
    final_rows = [row for row in rows if row["protocol"] == "final_five_fold_outer_OOF_full_MP4"]
    development_rows = [row for row in rows if row["protocol"] == "development_fixed_120_validation"]
    diagnostic_rows = [row for row in rows if row["protocol"] == "development_fixed_120_validation" and row["status"] in {"diagnostic_only", "base_reference"}]
    best_final = max(final_rows, key=lambda row: row["accuracy"])
    best_development = max(development_rows, key=lambda row: row["accuracy"])
    eligible_development = [row for row in development_rows if row["arbitrary_mp4"] == "yes" and row["recall"] >= 0.85 and row["status"] != "diagnostic_only"]
    best_eligible = max(eligible_development, key=lambda row: row["accuracy"])

    content = f"""# فاز سوم — گزارش آماری مدل‌ها با معیار اصلی Accuracy

این گزارش مستقیماً از فایل‌های نتیجهٔ ذخیره‌شدهٔ فاز سوم ساخته شده است.
معیار مرتب‌سازی اصلی طبق درخواست، **Accuracy** است. همهٔ درصدها در سطح
ویدئوی کامل/MP4 هستند، نه در سطح فریم.

## قانون مهم مقایسه

دو پروتکل ارزیابی متفاوت وجود دارد:

- **OOF نهایی:** پنج fold خارجی StratifiedKFold و دقیقاً یک احتمال
  out-of-fold برای هر یک از ۶۰۰ ویدئو. تنها مقایسهٔ معتبر برای تعمیم نهایی
  مدل‌ها همین بخش است.
- **Development:** تقسیم ثابت اولیهٔ ۴۸۰ آموزش / ۱۲۰ validation. این نتایج
  برای آزمایش و تحلیل مفیدند، اما نمرهٔ مستقل نهایی نیستند.

به همین علت، Accuracy این دو بخش در یک رتبه‌بندی واحد مخلوط نشده است.
«بازتولید head-only» نیز برای شفافیت ثبت شده، اما چون بازتولید full-MP4
ندارد، با مدل‌های ویدئویی قابل مقایسه نیست.

## نتیجه بر اساس Accuracy

- **بهترین مدل نهایی و قابل‌مقایسه:** **{best_final['model']}** با Accuracy
  **{percent(best_final['accuracy'])}**؛ همان مدل D1 قابل‌استفاده برای MP4
  دلخواه است.
- در OOF نهایی، Accuracy مدل D1 نسبت به A2-MP برابر با
  **{100 * (best_final['accuracy'] - min(final_rows, key=lambda row: row['accuracy'])['accuracy']):.2f} واحد درصد** بیشتر است.
- **بالاترین Accuracy فقط در development:** **{best_development['model']}** با
  **{percent(best_development['accuracy'])}**. این یک ensemble تشخیصی E1-AND
  است، نه مدل نهایی؛ Recall آن فقط {percent(best_development['recall'])} است
  و از حد ایمنی ازپیش‌تعیین‌شدهٔ ۸۵٪ کمتر است. همچنین OOF پنج‌فولد ندارد.
- **بهترین مدل development که هم برای MP4 دلخواه قابل اجراست و هم Recall
  حداقل ۸۵٪ دارد:** **{best_eligible['model']}** با Accuracy
  **{percent(best_eligible['accuracy'])}**.

پس انتخاب نهایی D1 براساس بیشینهٔ پسینی Accuracy در development نبوده؛ بلکه
براساس ارزیابی فریز‌شدهٔ پنج‌فولد OOF انجام شده است.

![مقایسهٔ Accuracy]({image_name})

## ۱. مقایسهٔ نهایی OOF پنج‌فولد (رتبه‌بندی معتبر نهایی)

{markdown_table(final_rows)}

## ۲. تقسیم ثابت development (فقط برای آزمایش و تحلیل)

{markdown_table(development_rows)}

## ۳. بازتولید head-only — عمداً خارج از رتبه‌بندی

{markdown_table([row for row in rows if row['protocol'] == 'head_reproduction_not_full_MP4'])}

## تفسیر مدل‌های فاز سوم

- **D1** مدل نهایی است: حرکت را با تفاضل مطلق RGB بین فریم‌های مجاور وارد
  می‌کند و در تنها ارزیابی نهاییِ مشترک، بالاترین Accuracy را دارد.
- **A2-MP** قوی‌ترین baseline بدون motion در OOF نهایی است، اما در Accuracy،
  Recall، F1 و PR-AUC از D1 پایین‌تر است.
- **A6** و **A-MIL** مدل‌های ویدیویی قابل اجرا هستند، اما Accuracy آن‌ها در
  همان development ثابت از D1 پایین‌تر شد.
- **D2 fusion** به weather/light/scene در زمان inference نیاز دارد؛ پس برای
  یک MP4 دلخواه، انتخاب پیش‌فرض مناسبی نیست. Metadata-only صرفاً برای بررسی
  shortcut و نشت اطلاعات نگه داشته شده است، نه برای deployment.
- ترکیب‌های **E1** فقط تشخیصی‌اند. قانون AND با مثبت اعلام‌کردن ویدئوهای کمتر
  Accuracy بالاتری می‌گیرد، اما تصادف‌های بیشتری را از دست می‌دهد
  (Recall کمتر از ۸۵٪). بنابراین به مدل نهایی ارتقا نیافت و OOF هم ندارد.

## مدل‌های اجرا نشده در فاز سوم

برای B1 (R3D-18)، B2 (R(2+1)D-18) و C1 (VideoMAE) سطر عملکرد نداریم؛
pipeline آن‌ها آماده شد، اما آموزش GPU آن‌ها عمداً اجرا نشد.
"""
    report_path = REPORT_ROOT / "phase3_model_accuracy_report.md"
    report_path.write_text(content, encoding="utf-8")
    return report_path


def write_report(rows: list[dict], image_name: str) -> Path:
    """Write the current report after all final OOF E1 evidence exists."""
    final_rows = [row for row in rows if row["protocol"] == "final_five_fold_outer_OOF_full_MP4"]
    development_rows = [row for row in rows if row["protocol"] == "development_fixed_120_validation"]
    e1_summary = read_json("e1_final_oof_summary.json") if (REPORT_ROOT / "e1_final_oof_summary.json").is_file() else None
    d1 = next(row for row in final_rows if row["model_id"] == "D1")
    e1_accuracy = next((row for row in final_rows if row["model_id"] == "E1-AND-OOF-accuracy"), None)
    curve_verifier = next((row for row in final_rows if row["model_id"] == "D1-curve-verifier"), None)
    best_final_accuracy = max(final_rows, key=lambda row: row["accuracy"])

    if e1_accuracy is None:
        e1_conclusion = "E1 final OOF validation has not yet been executed."
    else:
        paired = e1_summary["paired_tests"]
        e1_conclusion = (
            f"E1-AND has the highest Accuracy-only OOF point estimate ({percent(e1_accuracy['accuracy'])}), "
            f"but its Recall is only {percent(e1_accuracy['recall'])}. Its +0.50 percentage-point "
            f"difference versus nested Accuracy-threshold D1 is not statistically significant "
            f"(exact McNemar p={paired['exact_two_sided_p']:.3f}). It is therefore documented as "
            "an OOF-validated negative/diagnostic result, not the deployed model."
        )
    curve_conclusion = "D1 curve verifier was not run."
    if curve_verifier is not None:
        curve_conclusion = (
            f"The no-new-video D1 probability-curve verifier reached {percent(curve_verifier['accuracy'])} Accuracy "
            f"and {percent(curve_verifier['recall'])} Recall. It is below nested D1 on Accuracy and loses too much Recall, "
            "so it was rejected."
        )

    content = f"""# Phase 3 — Accuracy-first model comparison (final update)

This report uses saved video-level results only. Accuracy is the requested
sorting metric. The 5-fold OOF section is the valid final comparison; the
fixed 480/120 development split is exploratory and is deliberately separate.

## Decision summary

- Highest final OOF Accuracy point estimate: **{best_final_accuracy['model']}** at
  **{percent(best_final_accuracy['accuracy'])}**.
- Default deployable model remains **D1**: Accuracy **{percent(d1['accuracy'])}**,
  Recall **{percent(d1['recall'])}**, F1 **{decimal(d1['f1'])}**. It has the
  final all-data artifact and arbitrary-MP4 inference hand-off.
- {e1_conclusion}
- {curve_conclusion}

The final all-data D1 artifact is not a second accuracy estimate; it is the
deployment artifact trained after model selection.

![Accuracy comparison]({image_name})

## 1. Final five-fold OOF full-MP4 comparison

{markdown_table(final_rows)}

## 2. Fixed development split — exploratory results only

{markdown_table(development_rows)}

## 3. Head-only reproduction — excluded from video-level ranking

{markdown_table([row for row in rows if row['protocol'] == 'head_reproduction_not_full_MP4'])}

## Interpretation

- **D1** is the project default because it is the only model handed off for an
  arbitrary MP4 and it retains the original high-recall/F1 decision policy.
- **E1-AND** is now genuinely OOF-validated: thresholds were selected only on
  each fold's inner validation rows. Its Accuracy-only version improves the
  point estimate only marginally and loses many accidents; its safety variant
  did not retain the 85% Recall requirement on outer folds. No E1 deployment
  artifact was created for those reasons.
- **D1 curve verifier** was the fastest no-new-data experiment. Its curve
  features reduced false positives but also created many false negatives; it
  therefore remains a documented rejected ablation.
- **A2-MP** is the strongest non-motion final baseline. **A6** and **A-MIL**
  were lower-Accuracy development experiments. **D2 fusion** requires metadata
  and is not appropriate for an arbitrary MP4.
- B1 (R3D-18), B2 (R(2+1)D-18) and C1 (VideoMAE) were not trained, so they do
  not have an Accuracy row.
"""
    report_path = REPORT_ROOT / "phase3_model_accuracy_report.md"
    report_path.write_text(content, encoding="utf-8")
    return report_path


def create_figure(rows: list[dict]) -> Path:
    final_rows = sorted(
        [row for row in rows if row["protocol"] == "final_five_fold_outer_OOF_full_MP4"],
        key=lambda row: row["accuracy"],
    )
    development_rows = sorted(
        [row for row in rows if row["protocol"] == "development_fixed_120_validation"],
        key=lambda row: row["accuracy"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for axis, subset, title in (
        (axes[0], final_rows, "Final 5-fold OOF (valid final comparison)"),
        (axes[1], development_rows, "Fixed development split (exploration only)"),
    ):
        labels = [row["model_id"] for row in subset]
        colors = ["#1b9e77" if row["model_id"].startswith("D1") else "#7570b3" if row["status"] != "diagnostic_only" else "#d95f02" for row in subset]
        bars = axis.barh(labels, [100 * row["accuracy"] for row in subset], color=colors)
        axis.set_xlim(0, 100)
        axis.set_xlabel("Accuracy (%)")
        axis.set_title(title, weight="bold")
        axis.grid(axis="x", alpha=0.25)
        for bar, row in zip(bars, subset):
            axis.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{100 * row['accuracy']:.1f}% | R {100 * row['recall']:.1f}%", va="center", fontsize=9)
    fig.suptitle("Problem 1 — Phase 3 accuracy comparison", fontsize=15, weight="bold")
    image_path = REPORT_ROOT / "phase3_model_accuracy_comparison.png"
    fig.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return image_path


def write_csv(rows: list[dict]) -> Path:
    output = REPORT_ROOT / "phase3_model_accuracy_comparison.csv"
    fields = [
        "model_id", "model", "protocol", "status", "input_type", "arbitrary_mp4",
        "accuracy_rank_within_protocol", "threshold", "accuracy", "precision", "recall",
        "f1", "roc_auc", "pr_auc", "notes",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["protocol"], -row["accuracy"])))
    return output


def main() -> None:
    rows = collect_rows()
    add_ranks(rows)
    csv_path = write_csv(rows)
    image_path = create_figure(rows)
    report_path = write_report(rows, image_path.name)
    print(json.dumps({"rows": len(rows), "csv": str(csv_path), "chart": str(image_path), "report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
