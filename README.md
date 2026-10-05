# Smart Fraud Detection System for Online Payment Transactions

A platform-independent fraud-detection prototype built around **semi-structured JSON/JSONL transaction logs**. The system uses **XGBoost as the single fraud-classification algorithm**, SMOTE for training-data imbalance, calibrated risk scoring, SHAP explanations, temporal evaluation, unseen-pattern testing, analyst feedback, and feature-drift monitoring.

## Core idea

```text
UPI / Wallet / Bank / E-commerce / FinTech transaction logs
                         |
                         v
                 Semi-structured JSON
                         |
                         v
                 Schema validation
                         |
                         v
                    Data Adapter
                         |
                         v
              Common transaction schema
                         |
                         v
               Behavioural features
                         |
                         v
            SMOTE inside training CV only
                         |
                         v
                  XGBoost classifier
                         |
                         v
              Probability calibration
                         |
                         v
             Cost-sensitive threshold
                         |
                         v
                Low / Medium / High
                         |
             +-----------+-----------+
             |                       |
             v                       v
       SHAP explanation       Investigation cases
                                     |
                                     v
                              Analyst feedback
                                     |
                                     v
                              Controlled retraining

                 + Drift monitoring throughout
```

The application does **not** require a fixed external financial database. A new transaction source only needs an adapter mapping into the common schema.

## Main improvements

- Semi-structured, source-independent transaction schema.
- Six overlapping fraud patterns instead of deterministic fraud rules.
- History-derived velocity, recent-amount, device and location features.
- Schema validation and safe defaults for missing optional fields.
- Primary **chronological** train/test split.
- SMOTE is applied only inside CV training folds.
- XGBoost remains the **only fraud classifier**.
- Out-of-fold sigmoid probability calibration.
- Cost-sensitive threshold selection: missed fraud is weighted more heavily than a false alarm.
- SHAP explanations for flagged transactions.
- Model version and feature-schema hash stored with every trained model.
- Analyst decisions and comments retained as labelled feedback.
- Controlled feedback export for explicit retraining; no automatic replacement of the model.
- Feature-distribution drift monitoring using PSI.
- Optional external-format compatibility evaluation without making an existing database the main dataset.
- Temporal vs random split diagnostic.
- Leave-one-fraud-pattern-out evaluation.

## Semi-structured transaction schema

A source can send nested JSON such as:

```json
{
  "transaction_id": "TXN10001",
  "timestamp": "2026-10-05T14:30:00",
  "amount": 650,
  "type": "PAYMENT",
  "sender": {
    "account_id": "ACC1",
    "balance_before": 18000,
    "account_age_days": 900,
    "txn_count_5min": 0,
    "txn_count_1h": 1,
    "txn_count_24h": 2,
    "amount_1h": 650,
    "amount_24h": 650,
    "avg_amount_24h": 325,
    "failed_attempts": 0
  },
  "receiver": {
    "account_id": "ACC2",
    "balance_before": 50000,
    "is_new_receiver": false
  },
  "device_info": {
    "device_id": "DEV1",
    "is_new_device": false,
    "txn_count_24h": 10
  },
  "location": {
    "distance_from_usual_km": 2,
    "geo_mismatch": false
  }
}
```

The adapter maps this into a stable common schema before the ML layer. This makes the classifier independent of the original source's JSON field names.

## Fraud-pattern generator

`backend/generate_data.py` creates sequential account histories and six representative patterns:

- account takeover
- mule drain
- velocity burst
- social engineering
- credential compromise
- device switching

Genuine transactions intentionally overlap with suspicious-looking behaviour so that no single field is a deterministic fraud label.

Default generated dataset: **10,000 transactions** for a fast reproducible demo. For a larger experiment:

```powershell
python -m backend.generate_data --n 30000 --fraud-rate 0.03 --seed 42
```

## Features

The model receives behavioural features such as:

- log transaction amount
- amount-to-balance ratio
- balance-drop ratio
- receiver gain ratio
- balance mismatch
- hour / night / weekend
- account age
- transactions in 5 minutes / 1 hour / 24 hours
- recent transaction volume
- current amount versus recent average
- failed attempts
- device transaction velocity
- minutes since previous transaction
- distance from usual location
- new device / geo mismatch / new receiver
- transaction type indicators

Account IDs, receiver IDs and device IDs are **not** passed directly to XGBoost.

## Training and evaluation

The main evaluation is chronological:

```text
Earlier transactions -> training
Later transactions   -> untouched test
```

SMOTE lives inside the imbalanced-learn pipeline, so synthetic samples are created only from training folds.

Hyperparameter tuning uses cross-validation and average precision. Threshold selection uses out-of-fold calibrated scores and a cost-sensitive policy:

```text
false-negative cost = 5
false-positive cost = 1
```

The saved risk score is calibrated. It should be treated as an estimated risk probability, not a guarantee.

## Run the project

### 1. Create environment

```powershell
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Generate data

```powershell
python -m backend.generate_data
```

### 4. Train

```powershell
python -m backend.train
```

For a larger experiment:

```powershell
python -m backend.generate_data --n 30000 --fraud-rate 0.03 --seed 42
python -m backend.train --n-iter 6 --cv 3 --split temporal
```

### 5. Unseen-pattern test

```powershell
python -m backend.generalization_test
```

### 6. Random-split diagnostic

```powershell
python -m backend.split_compare
```

### 7. Start Flask

Terminal 1:

```powershell
python -m backend.app
```

Backend:

```text
http://localhost:5000
```

### 8. Start React

Terminal 2:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL, normally:

```text
http://localhost:5173
```

## Useful API endpoints

- `GET /api/health` — model/version/schema status
- `GET /api/schema` — common input schema and example
- `POST /api/predict` — real-time scoring
- `POST /api/batch` — CSV/JSON/JSONL batch scoring
- `GET /api/report` — investigation cases
- `POST /api/report/<transaction_id>/decision` — analyst decision/comment
- `GET /api/report/training-labels` — confirmed analyst labels
- `GET /api/report/download` — investigation CSV
- `GET /api/metrics` — model evaluation
- `POST /api/drift/check` — check a recent transaction batch for feature drift
- `GET /api/drift` — latest drift report

## Controlled retraining

Analyst-confirmed cases can be exported:

```powershell
python -m backend.retrain
```

This intentionally **does not automatically replace the model**. Review the feedback first, then retrain explicitly.

## Optional external-format compatibility

The project can map a PaySim-style CSV through the adapter. This is **not the main dataset** and is not required by the application. If you have a benchmark file and want to measure adapter compatibility:

```powershell
python -m backend.external_evaluation --path path\to\benchmark.csv
```

The main paper should continue to present the semi-structured transaction framework as the proposed system.

## Reports generated

- `reports/metrics.json`
- `reports/generalization.json`
- `reports/temporal_evaluation.json`
- `reports/drift_report.json`
- `reports/model_registry.jsonl`
- `reports/pr_curve.png`
- `reports/confusion_matrix.png`
- `reports/shap_importance.png`

## Testing

```powershell
python -m py_compile backend\*.py tests\*.py
python -m pytest -q
```

The test suite covers schema mapping, feature generation, risk bands, reports, model-schema compatibility, generator history, and API behaviour when Flask is installed.

## Important research statement

This is a research/college prototype, not a production banking system. The design removes dependence on a particular existing transaction database and adds temporal evaluation, explainability, calibration, drift monitoring and controlled feedback. Before deployment on a real payment platform, the platform's own labelled transaction data, security controls, latency requirements, privacy requirements and operational costs must be validated.
