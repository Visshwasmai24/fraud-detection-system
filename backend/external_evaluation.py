"""Optional external-format compatibility evaluation.

This does NOT become the main dataset and does not retrain the project on an
existing database.

It takes a reproducible sample from an external PaySim-style CSV, maps it
through the existing adapter, builds the same feature schema used by the
main XGBoost model, and evaluates the already-trained model.
"""

from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config
from .data_adapter import from_paysim
from .feature_engineering import build_features


def load_paysim_sample(path: str, sample_rows: int | None = None, seed: int = 42):
    """Load PaySim and optionally take a reproducible stratified sample."""

    required = {
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "isFraud",
    }

    # Read only the columns needed by the adapter.
    df = pd.read_csv(path)

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"PaySim file is missing required columns: {sorted(missing)}"
        )

    if sample_rows is not None:
        if sample_rows < 2:
            raise ValueError("--sample-rows must be at least 2.")

        if len(df) > sample_rows:
            # Keep the original fraud/genuine proportion as closely as
            # possible instead of selecting only fraud rows.
            fraud = df[df["isFraud"] == 1]
            genuine = df[df["isFraud"] == 0]

            fraud_n = round(sample_rows * len(fraud) / len(df))
            fraud_n = min(fraud_n, len(fraud))

            genuine_n = sample_rows - fraud_n
            genuine_n = min(genuine_n, len(genuine))

            # If rounding leaves a small difference, fill it from the
            # remaining class.
            remaining = sample_rows - (fraud_n + genuine_n)

            if remaining > 0:
                extra_fraud = min(remaining, len(fraud) - fraud_n)
                fraud_n += extra_fraud
                remaining -= extra_fraud

            if remaining > 0:
                extra_genuine = min(remaining, len(genuine) - genuine_n)
                genuine_n += extra_genuine

            fraud_sample = fraud.sample(
                n=fraud_n,
                random_state=seed,
            )

            genuine_sample = genuine.sample(
                n=genuine_n,
                random_state=seed,
            )

            df = (
                pd.concat([fraud_sample, genuine_sample])
                .sample(frac=1, random_state=seed)
                .reset_index(drop=True)
            )

    return df


def run(
    path: str,
    sample_rows: int | None = None,
    seed: int = 42,
):
    """Evaluate the existing trained model on an external PaySim sample."""

    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Main trained model not found: {config.MODEL_PATH}. "
            "Train the main project model first."
        )

    raw = load_paysim_sample(
        path,
        sample_rows=sample_rows,
        seed=seed,
    )

    common = from_paysim(raw)
    X = build_features(common)
    y = common.is_fraud.astype(int)

    bundle = joblib.load(config.MODEL_PATH)

    # Use the existing trained XGBoost pipeline.
    raw_probability = bundle["pipeline"].predict_proba(X)[:, 1]

    # Use the threshold selected during the main experiment.
    threshold = float(bundle["decision_threshold"])

    prediction = (raw_probability >= threshold).astype(int)

    result = {
        "source": str(path),
        "rows": int(len(common)),
        "fraud_rows": int(y.sum()),
        "fraud_rate": round(float(y.mean()), 4),
        "sample_rows": sample_rows,
        "seed": seed,
        "adapter_source": common.attrs.get("source"),
        "threshold": round(threshold, 4),

        "precision": round(
            float(precision_score(y, prediction, zero_division=0)),
            4,
        ),

        "recall": round(
            float(recall_score(y, prediction, zero_division=0)),
            4,
        ),

        "f1": round(
            float(f1_score(y, prediction, zero_division=0)),
            4,
        ),

        "pr_auc": round(
            float(average_precision_score(y, raw_probability)),
            4,
        ),

        "roc_auc": round(
            float(roc_auc_score(y, raw_probability)),
            4,
        ),

        "note": (
            "This is an additional external evaluation only. "
            "The model is trained on the project's semi-structured dataset "
            "and is not retrained using PaySim."
        ),
    }

    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the trained fraud model on an external PaySim CSV."
    )

    parser.add_argument(
        "--path",
        required=True,
        help="Path to the PaySim CSV file.",
    )

    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional number of PaySim rows to sample.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible sampling.",
    )

    args = parser.parse_args()

    run(
        path=args.path,
        sample_rows=args.sample_rows,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
