"""Controlled retraining helper using analyst-confirmed live cases.

It never replaces the model automatically. It exports confirmed feedback to a
semi-structured JSONL file and then requires an explicit --train command.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from . import config,report
from .data_adapter import from_json_records,load_path

def export_feedback(out=None):
    cases=report.confirmed_training_cases(); out=Path(out or (config.DATA_DIR/"analyst_feedback.jsonl")); records=[]
    for c in cases:
        tx=c.get("normalized_transaction")
        if not tx: continue
        tx=dict(tx); tx["is_fraud"]=1 if c["analyst_decision"]=="confirmed_fraud" else 0; tx["fraud_pattern"]=c.get("fraud_pattern","analyst_confirmed") if c["analyst_decision"]=="confirmed_fraud" else "analyst_genuine"
        records.append(tx)
    with open(out,"w",encoding="utf-8") as f:
        for r in records:f.write(json.dumps(r)+"\n")
    return {"confirmed_cases":len(cases),"exported":len(records),"path":str(out),"next_step":"Review the feedback file, merge it with approved training data, then run backend.train explicitly."}
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--out"); a=ap.parse_args(); print(json.dumps(export_feedback(a.out),indent=2))
