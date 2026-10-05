"""Central configuration for the platform-independent fraud framework."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "2.0"
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
for _d in (DATA_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(exist_ok=True)

DEFAULT_JSON_PATH = DATA_DIR / "transactions.jsonl"
MODEL_PATH = MODEL_DIR / "xgb_fraud_model.joblib"
PAYSIM_MODEL_PATH = MODEL_DIR / "xgb_fraud_model_paysim_baseline.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"
PAYSIM_METRICS_PATH = REPORT_DIR / "paysim_baseline_metrics.json"
GENERALIZATION_PATH = REPORT_DIR / "generalization.json"
TEMPORAL_PATH = REPORT_DIR / "temporal_evaluation.json"
DRIFT_PATH = REPORT_DIR / "drift_report.json"
CASES_PATH = REPORT_DIR / "cases.jsonl"
MODEL_REGISTRY_PATH = REPORT_DIR / "model_registry.jsonl"

# These are operational risk bands. The fraud decision threshold is learned from training data.
RISK_LOW_MAX = 0.30
RISK_HIGH_MIN = 0.70
SUGGESTED_ACTION = {
    "Low": "Approve transaction",
    "Medium": "Request additional verification (OTP / call-back)",
    "High": "Hold transaction and send to fraud analyst",
}

TXN_TYPES = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
COMMON_COLUMNS = [
    "transaction_id", "timestamp", "amount", "txn_type", "sender_account_id",
    "receiver_account_id", "device_id", "hour", "is_weekend",
    "sender_balance_before", "sender_balance_after", "receiver_balance_before",
    "receiver_balance_after", "account_age_days", "txn_count_5min", "txn_count_1h",
    "txn_count_24h", "amount_1h", "amount_24h", "avg_amount_24h",
    "failed_attempts", "device_txn_count_24h", "minutes_since_last_txn",
    "distance_from_usual_km", "is_new_device", "geo_mismatch", "is_new_receiver",
    "is_fraud", "fraud_pattern",
]

FEATURE_COLUMNS = [
    "log_amount", "amount_to_balance_ratio", "balance_drop_ratio",
    "receiver_gain_ratio", "balance_mismatch", "hour", "is_night", "is_weekend",
    "account_age_days", "txn_count_5min", "txn_count_1h", "txn_count_24h",
    "log_amount_1h", "log_amount_24h", "amount_vs_avg_24h", "failed_attempts",
    "device_txn_count_24h", "minutes_since_last_txn", "distance_from_usual_km",
    "is_new_device", "geo_mismatch", "is_new_receiver",
] + [f"type_{t}" for t in TXN_TYPES]

REASON_TEXT = {
    "log_amount": "Transaction amount is unusually large",
    "amount_to_balance_ratio": "Transaction consumes a large share of the available balance",
    "balance_drop_ratio": "Payment drains a large share of the sender's balance",
    "receiver_gain_ratio": "Receiver balance increases sharply",
    "balance_mismatch": "Sender balances do not reconcile with the transaction",
    "hour": "Transaction occurs at an unusual time of day",
    "is_night": "Transaction occurs late at night",
    "is_weekend": "Transaction occurs during a weekend period",
    "account_age_days": "Account is relatively new",
    "txn_count_5min": "Several transactions occurred in a short period",
    "txn_count_1h": "Transaction velocity is unusually high",
    "txn_count_24h": "Many transactions occurred in the last 24 hours",
    "log_amount_1h": "Recent transaction volume is high",
    "log_amount_24h": "Recent 24-hour transaction volume is high",
    "amount_vs_avg_24h": "Current amount differs sharply from the user's recent average",
    "failed_attempts": "Several failed attempts occurred before the payment",
    "device_txn_count_24h": "The device has unusual recent transaction activity",
    "minutes_since_last_txn": "Transaction timing is unusually close to the previous transaction",
    "distance_from_usual_km": "Transaction location is far from the usual location",
    "is_new_device": "Payment uses a new or unrecognised device",
    "geo_mismatch": "Location differs from the account's usual location",
    "is_new_receiver": "Money is being sent to a new receiver",
    "type_TRANSFER": "Transfer-type transaction",
    "type_CASH_OUT": "Cash-out transaction",
    "type_PAYMENT": "Payment transaction",
    "type_CASH_IN": "Cash-in transaction",
    "type_DEBIT": "Debit transaction",
}

def feature_hash():
    return hashlib.sha256("|".join(FEATURE_COLUMNS).encode()).hexdigest()[:16]
