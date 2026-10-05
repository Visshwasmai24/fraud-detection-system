"""Run with:  python -m pytest -q"""
from backend.external_evaluation import load_paysim_sample
import io
import json

import pandas as pd
import pytest

from backend import config, report
from backend.data_adapter import AdapterError, from_api, from_paysim, load_bytes
from backend.feature_engineering import build_features
from backend.predict import risk_level

NESTED = {
    "transaction_id": "T1", "timestamp": "2026-09-01T03:15:00", "amount": 5000, "type": "TRANSFER",
    "sender": {"balance_before": 10000, "account_age_days": 100, "txn_count_24h": 3, "failed_attempts": 1},
    "receiver": {"balance_before": 200, "is_new_receiver": True},
    "device_info": {"is_new_device": True}, "location": {"geo_mismatch": False},
}


# ------------------------------------------------------------- data adapter
def test_nested_json_is_flattened_to_common_format():
    df = from_api(NESTED)
    assert list(df.columns) == config.COMMON_COLUMNS
    row = df.iloc[0]
    assert row.hour == 3 and row.is_new_device == 1 and row.is_new_receiver == 1 and row.txn_count_1h == 0
    assert row.sender_balance_after == 5000          # derived: before - amount


def test_jsonl_file_upload():
    text = "\n".join(json.dumps(NESTED | {"transaction_id": f"T{i}"}) for i in range(3))
    df = load_bytes("x.jsonl", text.encode())
    assert len(df) == 3 and df.attrs["source"] == "json"


def test_paysim_csv_is_mapped_with_defaults_and_note():
    ps = pd.DataFrame({
        "step": [1, 26], "type": ["TRANSFER", "PAYMENT"], "amount": [100.0, 20.0],
        "nameOrig": ["C1", "C2"], "oldbalanceOrg": [500.0, 80.0], "newbalanceOrig": [400.0, 60.0],
        "nameDest": ["C9", "M9"], "oldbalanceDest": [0.0, 0.0], "newbalanceDest": [100.0, 0.0],
        "isFraud": [1, 0], "isFlaggedFraud": [0, 0]})
    df = from_paysim(ps)
    assert list(df.hour) == [1, 2]                  # step % 24
    assert df.is_fraud.tolist() == [1, 0]
    assert df.attrs["notes"], "adapter must say which fields were defaulted"
    buf = io.BytesIO(); ps.to_csv(buf, index=False)
    assert load_bytes("paysim.csv", buf.getvalue()).attrs["source"] == "paysim_csv"


def test_bad_inputs_raise_clear_errors():
    with pytest.raises(AdapterError):
        from_api({"amount": "abc", "sender": {"balance_before": 1}})
    with pytest.raises(AdapterError):
        from_api({"amount": 100})                    # no sender balance
    with pytest.raises(AdapterError):
        load_bytes("notes.txt", b"hello")


# --------------------------------------------------------- feature engineering
def test_features_are_complete_and_correct():
    X = build_features(from_api(NESTED))
    assert list(X.columns) == config.FEATURE_COLUMNS
    assert X.isna().sum().sum() == 0
    assert X.loc[0, "balance_drop_ratio"] == pytest.approx(5000 / 10001, rel=1e-3)
    assert X.loc[0, "is_night"] == 1 and X.loc[0, "type_TRANSFER"] == 1


# -------------------------------------------------------------------- risk
@pytest.mark.parametrize("p,level", [(0.0, "Low"), (0.29, "Low"), (0.30, "Medium"),
                                     (0.69, "Medium"), (0.70, "High"), (1.0, "High")])
def test_risk_bands(p, level):
    assert risk_level(p) == level


# ------------------------------------------------------------------ report
def test_report_ranks_by_probability_and_records_decisions(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "CASES_PATH", tmp_path / "cases.jsonl")
    mk = lambda i, p, lv: dict(transaction_id=i, amount=1, txn_type="TRANSFER", probability=p,
                               risk_level=lv, suggested_action="x", reasons=[{"text": "r"}])
    assert report.add_cases([mk("a", .4, "Medium"), mk("b", .95, "High"), mk("c", .01, "Low")], "t") == 2
    rows = report.ranked_cases()
    assert [r["transaction_id"] for r in rows] == ["b", "a"] and rows[0]["rank"] == 1
    assert report.set_decision("b", "confirmed_fraud") and not report.set_decision("zzz", "genuine")
    assert "transaction_id" in report.to_csv(report.ranked_cases())


# --------------------------------------------------------------------- API
needs_model = pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="train the model first")
has_flask = __import__("importlib.util").util.find_spec("flask") is not None
needs_api = pytest.mark.skipif(not has_flask, reason="Flask is not installed in this test environment; install requirements.txt")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "CASES_PATH", tmp_path / "cases.jsonl")
    from backend.app import app
    return app.test_client()


@needs_model
@needs_api
def test_api_flags_suspicious_and_approves_normal(client):
    bad = NESTED | {"amount": 9500, "sender": NESTED["sender"] | {"failed_attempts": 3}}
    good = {"timestamp": "2026-09-01T14:00:00", "amount": 300, "type": "PAYMENT",
            "sender": {"balance_before": 20000, "account_age_days": 800, "txn_count_24h": 2, "failed_attempts": 0},
            "receiver": {"balance_before": 40000, "is_new_receiver": False},
            "device_info": {"is_new_device": False}, "location": {"geo_mismatch": False}}
    r_bad = client.post("/api/predict", json=bad).get_json()
    r_good = client.post("/api/predict", json=good).get_json()
    assert r_bad["risk_level"] in {"Medium", "High"}
    assert r_bad["predicted_fraud"] is True
    assert r_bad["reasons"]
    assert r_bad["risk_score"] >= client.get("/api/health").get_json()["decision_threshold"]
    assert r_good["risk_level"] == "Low"
    assert r_good["predicted_fraud"] is False
    assert len(client.get("/api/report").get_json()) == 1      # only the flagged one is stored


@needs_model
@needs_api
def test_api_validation_and_batch(client):
    assert client.post("/api/predict", json={"amount": 5}).status_code == 400
    assert client.post("/api/batch").status_code == 400
    body = "\n".join(json.dumps(NESTED | {"transaction_id": f"B{i}"}) for i in range(5)).encode()
    res = client.post("/api/batch", data={"file": (io.BytesIO(body), "t.jsonl")},
                      content_type="multipart/form-data").get_json()
    assert res["summary"]["rows"] == 5 and res["summary"]["data_source"] == "json"


# --------------------------------------------------------- model metadata / behavior
def test_external_paysim_sampling_is_reproducible(tmp_path):
    ps = pd.DataFrame({
        "step": list(range(1, 101)),
        "type": ["TRANSFER"] * 100,
        "amount": [100.0] * 100,
        "nameOrig": [f"C{i}" for i in range(100)],
        "oldbalanceOrg": [500.0] * 100,
        "newbalanceOrig": [400.0] * 100,
        "nameDest": [f"D{i}" for i in range(100)],
        "oldbalanceDest": [0.0] * 100,
        "newbalanceDest": [100.0] * 100,
        "isFraud": [1 if i < 10 else 0 for i in range(100)],
        "isFlaggedFraud": [0] * 100,
    })

    path = tmp_path / "paysim.csv"
    ps.to_csv(path, index=False)

    a = load_paysim_sample(
        str(path),
        sample_rows=20,
        seed=42,
    )

    b = load_paysim_sample(
        str(path),
        sample_rows=20,
        seed=42,
    )

    assert len(a) == 20
    assert a.equals(b)
def test_model_bundle_schema_metadata_and_threshold_if_trained():
    if not config.MODEL_PATH.exists():
        pytest.skip("train the model first")
    import joblib
    bundle = joblib.load(config.MODEL_PATH)
    meta = bundle["metadata"]
    assert meta["schema_version"] == config.SCHEMA_VERSION
    assert meta["feature_columns"] == config.FEATURE_COLUMNS
    assert meta["feature_hash"] == config.feature_hash()
    assert 0.0 < bundle["decision_threshold"] < 1.0
    assert meta["source_role"] in {"main_realtime_model", "paysim_baseline"}


def test_incompatible_model_schema_is_rejected(tmp_path):
    import joblib
    from backend.predict import FraudModel, ModelNotTrainedError
    fake = {
        "pipeline": object(),
        "feature_columns": ["wrong_feature"],
        "decision_threshold": 0.5,
        "metadata": {"schema_version": config.SCHEMA_VERSION, "feature_columns": ["wrong_feature"], "feature_hash": "bad"}
    }
    path = tmp_path / "bad.joblib"
    joblib.dump(fake, path)
    with pytest.raises(ModelNotTrainedError, match="feature schema"):
        FraudModel(path)


def test_sequential_generator_has_real_history_features():
    from backend.generate_data import generate
    records = generate(1500, fraud_rate=0.05, seed=123)
    by_account = {}
    for r in records:
        by_account.setdefault(r["sender"]["account_id"], []).append(r)
    # For every account, each transaction's timestamp should be non-decreasing
    # and txn_count_24h must equal the count of earlier transactions in the prior 24h.
    checked = 0
    for account, rows in by_account.items():
        rows = sorted(rows, key=lambda r: r["timestamp"])
        for i, r in enumerate(rows):
            from datetime import datetime, timedelta
            t = datetime.fromisoformat(r["timestamp"])
            expected = 0
            for prev in rows[:i]:
                pt = datetime.fromisoformat(prev["timestamp"])
                if pt >= t - timedelta(hours=24):
                    expected += 1
            assert r["sender"]["txn_count_24h"] == expected
            checked += 1
            if checked >= 100:
                return


@needs_api
def test_real_time_output_uses_risk_score_name_when_model_exists(client):
    if not config.MODEL_PATH.exists():
        pytest.skip("train the model first")
    r = client.post("/api/predict", json=NESTED)
    assert r.status_code == 200
    body = r.get_json()
    assert "risk_score" in body and "risk_level" in body and 0 <= body["risk_score"] <= 1
