"""Source-independent transaction adapter with schema validation and safe defaults."""
from __future__ import annotations
import io, json
from pathlib import Path
from typing import Iterable, Union
import numpy as np
import pandas as pd
from .config import COMMON_COLUMNS

class AdapterError(ValueError): pass

def _pick(df, names: Iterable[str], default=None):
    for n in names:
        if n in df.columns: return df[n]
    return pd.Series([default]*len(df), index=df.index)

def _flag(s):
    if s.dtype == bool: return s.astype(int)
    return s.map(lambda v: 1 if str(v).strip().lower() in {"1","true","yes","y","t"} else 0).astype(int)

def _hour(ts, fallback=12):
    return pd.to_datetime(ts, errors="coerce").dt.hour.fillna(fallback).astype(int)

def _finish(out, notes, source):
    numeric = [c for c in COMMON_COLUMNS if c not in {"transaction_id","timestamp","txn_type","sender_account_id","receiver_account_id","device_id","fraud_pattern"}]
    for c in numeric: out[c] = pd.to_numeric(out[c], errors="coerce")
    required = ["amount","sender_balance_before"]
    if out[required].isna().any().any():
        raise AdapterError("Each transaction requires numeric 'amount' and 'sender.balance_before' (or sender_balance_before).")
    ts = pd.to_datetime(out["timestamp"], errors="coerce")
    out["timestamp"] = ts.fillna(pd.Timestamp("2026-01-01T12:00:00")).dt.strftime("%Y-%m-%dT%H:%M:%S")
    out["hour"] = out["hour"].fillna(_hour(ts)).astype(int).clip(0,23)
    weekend = pd.to_numeric(out["is_weekend"], errors="coerce"); fallback = (pd.to_datetime(out["timestamp"]).dt.dayofweek >= 5).astype(int); out["is_weekend"] = weekend.where(weekend.notna(), fallback).astype(int)
    out["sender_balance_after"] = out["sender_balance_after"].fillna((out["sender_balance_before"]-out["amount"]).clip(lower=0))
    out["receiver_balance_before"] = out["receiver_balance_before"].fillna(0.0)
    out["receiver_balance_after"] = out["receiver_balance_after"].fillna(out["receiver_balance_before"]+out["amount"])
    defaults={"account_age_days":365,"txn_count_5min":0,"txn_count_1h":0,"txn_count_24h":0,"amount_1h":0,"amount_24h":0,"avg_amount_24h":0,"failed_attempts":0,"device_txn_count_24h":0,"minutes_since_last_txn":1440,"distance_from_usual_km":0}
    for c,v in defaults.items(): out[c]=out[c].fillna(v)
    for c in ["is_new_device","geo_mismatch","is_new_receiver"]: out[c]=_flag(out[c].fillna(0))
    out["txn_type"]=out["txn_type"].fillna("PAYMENT").astype(str).str.upper().str.replace(" ","_",regex=False)
    out["transaction_id"]=out["transaction_id"].fillna(pd.Series([f"TXN{i:08d}" for i in range(len(out))],index=out.index)).astype(str)
    out["sender_account_id"]=out["sender_account_id"].fillna("UNKNOWN_SENDER").astype(str)
    out["receiver_account_id"]=out["receiver_account_id"].fillna("UNKNOWN_RECEIVER").astype(str)
    out["device_id"]=out["device_id"].fillna("UNKNOWN_DEVICE").astype(str)
    out["is_fraud"]=pd.to_numeric(out["is_fraud"],errors="coerce").fillna(0).astype(int).clip(0,1)
    out["fraud_pattern"]=out["fraud_pattern"].fillna("none").astype(str)
    out=out[COMMON_COLUMNS].reset_index(drop=True)
    out.attrs["source"]=source; out.attrs["notes"]=notes
    return out

def from_paysim(df):
    need={"step","type","amount","oldbalanceOrg","newbalanceOrig"}; missing=need-set(df.columns)
    if missing: raise AdapterError(f"Not a PaySim file - missing columns: {sorted(missing)}")
    out=pd.DataFrame(index=df.index)
    out["transaction_id"]="PS"+pd.Series(df.index,index=df.index).astype(str)
    out["timestamp"]=(pd.Timestamp("2026-01-01")+pd.to_timedelta(df["step"],unit="h")).astype(str)
    out["amount"]=df["amount"]; out["txn_type"]=df["type"]; out["sender_account_id"]=_pick(df,["nameOrig"],"UNKNOWN")
    out["receiver_account_id"]=_pick(df,["nameDest"],"UNKNOWN"); out["device_id"]="PAYSIM"
    out["hour"]=df["step"]%24; out["is_weekend"]=0
    out["sender_balance_before"]=df["oldbalanceOrg"]; out["sender_balance_after"]=df["newbalanceOrig"]
    out["receiver_balance_before"]=_pick(df,["oldbalanceDest"],0); out["receiver_balance_after"]=_pick(df,["newbalanceDest"],0)
    for c,v in {"account_age_days":365,"txn_count_5min":0,"txn_count_1h":0,"txn_count_24h":0,"amount_1h":0,"amount_24h":0,"avg_amount_24h":0,"failed_attempts":0,"device_txn_count_24h":0,"minutes_since_last_txn":1440,"distance_from_usual_km":0,"is_new_device":0,"geo_mismatch":0,"is_new_receiver":0}.items(): out[c]=v
    out["is_fraud"]=_pick(df,["isFraud"],0); out["fraud_pattern"]=np.where(pd.to_numeric(out["is_fraud"],errors="coerce").fillna(0).astype(int)==1,"paysim_fraud","none")
    return _finish(out,["PaySim is an optional external benchmark only; missing behavioural fields use neutral defaults."],"paysim_csv")

def from_json_records(records: Union[list,dict], source="json"):
    if isinstance(records,dict): records=[records]
    if not isinstance(records,list) or not records or not all(isinstance(x,dict) for x in records): raise AdapterError("Expected a non-empty JSON object/list of objects.")
    df=pd.json_normalize(records,sep="."); out=pd.DataFrame(index=df.index)
    out["transaction_id"]=_pick(df,["transaction_id","txn_id","id"]); out["timestamp"]=_pick(df,["timestamp","time","datetime"],"2026-01-01T12:00:00")
    out["amount"]=_pick(df,["amount","transaction.amount"]); out["txn_type"]=_pick(df,["type","txn_type","transaction_type","transaction.type"],"PAYMENT")
    out["sender_account_id"]=_pick(df,["sender.account_id","sender_account_id","account_id"]); out["receiver_account_id"]=_pick(df,["receiver.account_id","receiver_account_id"])
    out["device_id"]=_pick(df,["device_info.device_id","device_id"]); out["hour"]=_pick(df,["hour"],None)
    out["is_weekend"]=_pick(df,["is_weekend"],None)
    for dest,names in {"sender_balance_before":["sender.balance_before","sender_balance_before","oldbalanceOrg"],"sender_balance_after":["sender.balance_after","sender_balance_after","newbalanceOrig"],"receiver_balance_before":["receiver.balance_before","receiver_balance_before","oldbalanceDest"],"receiver_balance_after":["receiver.balance_after","receiver_balance_after","newbalanceDest"],"account_age_days":["sender.account_age_days","account_age_days"],"txn_count_5min":["sender.txn_count_5min","txn_count_5min"],"txn_count_1h":["sender.txn_count_1h","txn_count_1h"],"txn_count_24h":["sender.txn_count_24h","txn_count_24h"],"amount_1h":["sender.amount_1h","amount_1h"],"amount_24h":["sender.amount_24h","amount_24h"],"avg_amount_24h":["sender.avg_amount_24h","avg_amount_24h"],"failed_attempts":["sender.failed_attempts","failed_attempts"],"device_txn_count_24h":["device_info.txn_count_24h","device_txn_count_24h"],"minutes_since_last_txn":["sender.minutes_since_last_txn","minutes_since_last_txn"],"distance_from_usual_km":["location.distance_from_usual_km","distance_from_usual_km"]}.items(): out[dest]=_pick(df,names)
    out["is_new_device"]=_flag(_pick(df,["device_info.is_new_device","is_new_device"],0)); out["geo_mismatch"]=_flag(_pick(df,["location.geo_mismatch","geo_mismatch"],0)); out["is_new_receiver"]=_flag(_pick(df,["receiver.is_new_receiver","is_new_receiver"],0))
    out["is_fraud"]=_pick(df,["is_fraud","isFraud"],0); out["fraud_pattern"]=_pick(df,["fraud_pattern"],"none")
    if out["hour"].isna().all(): out["hour"]=_hour(pd.to_datetime(out["timestamp"],errors="coerce"))
    return _finish(out,[],source)

def from_generic_csv(df): return from_json_records(df.to_dict(orient="records"),source="generic_csv")
def from_api(payload): return from_json_records(payload,source="api")

def _read_json_text(text):
    text=text.strip()
    if not text: raise AdapterError("The file is empty.")
    try: return json.loads(text)
    except json.JSONDecodeError:
        rows=[]
        for i,line in enumerate(text.splitlines(),1):
            if not line.strip(): continue
            try: rows.append(json.loads(line))
            except json.JSONDecodeError as e: raise AdapterError(f"Invalid JSON on line {i}: {e.msg}") from e
        return rows

def load_bytes(filename,data):
    suffix=Path(filename).suffix.lower()
    if suffix==".csv":
        df=pd.read_csv(io.BytesIO(data)); return from_paysim(df) if {"oldbalanceOrg","newbalanceOrig","step"}.issubset(df.columns) else from_generic_csv(df)
    if suffix in {".json",".jsonl",".ndjson"}: return from_json_records(_read_json_text(data.decode("utf-8")),source="json")
    raise AdapterError(f"Unsupported file type '{suffix}'. Use .csv, .json or .jsonl")

def load_path(path):
    path=Path(path)
    if not path.exists(): raise AdapterError(f"File not found: {path}")
    return load_bytes(path.name,path.read_bytes())
