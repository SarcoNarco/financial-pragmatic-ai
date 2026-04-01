"""Run full evaluation: FinBERT baseline vs custom financial NLP pipeline."""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from financial_pragmatic_ai.analysis.financial_signal_engine import (
    compute_confidence,
    compute_intent_distribution,
    compute_risk_score,
    detect_volatility,
    derive_signal,
)
from financial_pragmatic_ai.analysis.insight_engine import extract_key_drivers
from financial_pragmatic_ai.analysis.market_predictor import predict_market_outlook
from financial_pragmatic_ai.analysis.transcript_analyzer import TranscriptAnalyzer
from financial_pragmatic_ai.evaluation.better_than_fin.metrics import (
    compute_metrics,
    delta_metrics,
    to_numpy_confusion,
)
from financial_pragmatic_ai.evaluation.better_than_fin.utils import (
    SIGNAL_LABELS,
    agreement_rate,
    average_confidence_per_class,
    baseline_sentiment_to_signal,
    build_ground_truth_signals,
    ensure_results_dir,
    explain_our_decision,
    load_evaluation_dataset,
    normalize_confidence_to_percent,
    snippet,
)
from financial_pragmatic_ai.evaluation.better_than_fin.visualize import (
    save_agreement_pie,
    save_class_distribution,
    save_confusion_matrices,
    save_performance_bars,
)


def _normalize_finbert_label(raw_label: str, pred_idx: int) -> str:
    label = str(raw_label).lower()
    if "positive" in label:
        return "positive"
    if "negative" in label:
        return "negative"
    if "neutral" in label:
        return "neutral"

    fallback = {0: "positive", 1: "negative", 2: "neutral"}
    return fallback.get(pred_idx, "neutral")


def run_finbert_baseline(
    texts: List[str],
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 32,
    max_length: int = 128,
) -> Dict[str, List]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    sentiments: List[str] = []
    signals: List[str] = []
    confidences: List[float] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)

            id2label = getattr(model.config, "id2label", {}) or {}
            for row, pred_id_tensor in enumerate(pred_ids):
                pred_id = int(pred_id_tensor.item())
                raw_label = id2label.get(pred_id, str(pred_id))
                sentiment = _normalize_finbert_label(raw_label, pred_id)
                signal = baseline_sentiment_to_signal(sentiment)
                confidence = normalize_confidence_to_percent(float(probs[row, pred_id].item()))

                sentiments.append(sentiment)
                signals.append(signal)
                confidences.append(confidence)

            if (start // batch_size + 1) % 50 == 0:
                print(
                    f"[FinBERT baseline] Processed "
                    f"{min(start + len(batch_texts), len(texts))}/{len(texts)} samples"
                )

    return {
        "sentiments": sentiments,
        "signals": signals,
        "confidences": confidences,
    }


def _safe_analyze(transcript_analyzer: TranscriptAnalyzer, text: str) -> List[dict]:
    with redirect_stdout(io.StringIO()):
        segments = transcript_analyzer.analyze(text)
    return segments


def run_custom_system(texts: List[str]) -> List[Dict]:
    analyzer = TranscriptAnalyzer()
    outputs: List[Dict] = []

    for idx, text in enumerate(texts, start=1):
        segments = _safe_analyze(analyzer, text)
        if not segments:
            segments = [{
                "speaker": "EXECUTIVE",
                "text": text,
                "intent": "GENERAL_UPDATE",
            }]

        score = compute_risk_score(segments)
        signal = derive_signal(score)
        confidence = float(compute_confidence(segments))
        volatility = detect_volatility(segments)
        distribution = compute_intent_distribution(segments)
        prediction = predict_market_outlook(
            signal=signal,
            risk_score=score,
            volatility=volatility,
            intent_distribution=distribution,
        )
        drivers = extract_key_drivers(segments)

        outputs.append({
            "segments": segments,
            "signal": signal,
            "prediction": prediction["prediction"],
            "prediction_explanation": prediction["explanation"],
            "confidence": normalize_confidence_to_percent(confidence),
            "volatility": volatility,
            "score": score,
            "drivers": drivers,
        })

        if idx % 50 == 0:
            print(f"[Our system] Processed {idx}/{len(texts)} samples")

    return outputs


def _build_disagreement_rows(
    df: pd.DataFrame,
    y_true: List[str],
    baseline: Dict[str, List],
    ours: List[Dict],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []
    disagreement_rows = []

    for i in range(len(df)):
        text = str(df.iloc[i]["text"])
        row = {
            "index": i,
            "text": text,
            "text_snippet": snippet(text),
            "ground_truth_signal": y_true[i],
            "finbert_signal": baseline["signals"][i],
            "finbert_sentiment": baseline["sentiments"][i],
            "finbert_confidence": baseline["confidences"][i],
            "our_signal": ours[i]["signal"],
            "our_prediction": ours[i]["prediction"],
            "our_confidence": ours[i]["confidence"],
            "our_volatility": ours[i]["volatility"],
            "our_score": ours[i]["score"],
            "our_explanation": explain_our_decision(ours[i]["segments"], ours[i]["signal"]),
        }
        all_rows.append(row)
        if row["finbert_signal"] != row["our_signal"]:
            disagreement_rows.append(row)

    all_df = pd.DataFrame(all_rows)
    disagreement_df = pd.DataFrame(disagreement_rows)
    return all_df, disagreement_df


def run_evaluation(
    dataset_path: str | Path | None = None,
    max_samples: int | None = None,
    results_dir: str | Path | None = None,
    batch_size: int = 32,
) -> Dict:
    results_path = ensure_results_dir(results_dir)
    df = load_evaluation_dataset(dataset_path=dataset_path, max_samples=max_samples)
    texts = df["text"].tolist()
    y_true = build_ground_truth_signals(df["intent"].tolist())

    print(f"Loaded dataset rows: {len(df)}")
    print("Running FinBERT baseline...")
    baseline = run_finbert_baseline(texts=texts, batch_size=batch_size)

    print("Running custom system...")
    ours = run_custom_system(texts=texts)

    y_pred_finbert = baseline["signals"]
    y_pred_ours = [item["signal"] for item in ours]
    confidence_ours = [item["confidence"] for item in ours]

    finbert_metrics = compute_metrics(y_true=y_true, y_pred=y_pred_finbert, labels=SIGNAL_LABELS)
    our_metrics = compute_metrics(y_true=y_true, y_pred=y_pred_ours, labels=SIGNAL_LABELS)
    deltas = delta_metrics(finbert_metrics, our_metrics)

    agree = agreement_rate(y_pred_finbert, y_pred_ours)
    confidence_comparison = {
        "finbert": average_confidence_per_class(
            y_pred_finbert,
            baseline["confidences"],
            labels=SIGNAL_LABELS,
        ),
        "our_system": average_confidence_per_class(
            y_pred_ours,
            confidence_ours,
            labels=SIGNAL_LABELS,
        ),
    }

    all_df, disagreement_df = _build_disagreement_rows(
        df=df,
        y_true=y_true,
        baseline=baseline,
        ours=ours,
    )

    top10_ours_correct = all_df[
        (all_df["our_signal"] == all_df["ground_truth_signal"])
        & (all_df["finbert_signal"] != all_df["ground_truth_signal"])
    ].copy()
    top10_ours_correct = top10_ours_correct.sort_values(
        by="our_confidence",
        ascending=False,
    ).head(10)

    all_df.to_csv(results_path / "predictions.csv", index=False)
    disagreement_df.to_csv(results_path / "disagreements.csv", index=False)
    top10_ours_correct.to_csv(
        results_path / "top10_ours_correct_finbert_wrong.csv",
        index=False,
    )

    finbert_cm = to_numpy_confusion(finbert_metrics)
    our_cm = to_numpy_confusion(our_metrics)
    save_confusion_matrices(
        labels=SIGNAL_LABELS,
        finbert_cm=finbert_cm,
        our_cm=our_cm,
        output_path=results_path / "confusion_matrices.png",
    )
    save_performance_bars(
        finbert_metrics=finbert_metrics,
        our_metrics=our_metrics,
        output_path=results_path / "accuracy_f1_comparison.png",
    )
    save_class_distribution(
        labels=SIGNAL_LABELS,
        y_true=y_true,
        y_finbert=y_pred_finbert,
        y_ours=y_pred_ours,
        output_path=results_path / "class_distribution.png",
    )
    save_agreement_pie(
        agreement_rate_value=agree,
        output_path=results_path / "agreement_pie.png",
    )

    output = {
        "dataset_rows": len(df),
        "finbert": finbert_metrics,
        "our_system": our_metrics,
        "improvement": deltas,
        "agreement_rate": round(agree, 6),
        "confidence_comparison": confidence_comparison,
        "disagreement_count": int(len(disagreement_df)),
        "top10_ours_correct_finbert_wrong_count": int(len(top10_ours_correct)),
        "results_dir": str(results_path),
    }

    with open(results_path / "summary.json", "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print("\n=== MODEL COMPARISON ===\n")
    print("FinBERT:")
    print(f"- Accuracy: {finbert_metrics['accuracy']:.4f}")
    print(f"- F1: {finbert_metrics['macro_f1']:.4f}")
    print("")
    print("Our System:")
    print(f"- Accuracy: {our_metrics['accuracy']:.4f}")
    print(f"- F1: {our_metrics['macro_f1']:.4f}")
    print("")
    print("Improvement:")
    print(f"- Δ Accuracy: {deltas['accuracy_delta']:.4f}")
    print(f"- Δ F1: {deltas['macro_f1_delta']:.4f}")

    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate custom system vs FinBERT baseline.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to pragmatic_intent_dataset_clean.csv (optional).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit evaluation size for faster runs.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory where outputs (plots/csv/json) are saved.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Baseline FinBERT inference batch size.",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    run_evaluation(
        dataset_path=args.dataset_path,
        max_samples=args.max_samples,
        results_dir=args.results_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
