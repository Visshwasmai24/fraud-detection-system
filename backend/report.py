"""Investigation cases and analyst feedback persistence."""
from __future__ import annotations
import csv,io,json,threading
from datetime import datetime,timezone
from .config import CASES_PATH
_lock=threading.Lock()
def _read():
    if not CASES_PATH.exists():return []
    with open(CASES_PATH,encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]
def _write(rows):
    CASES_PATH.parent.mkdir(exist_ok=True)
    tmp=CASES_PATH.with_suffix('.tmp'); tmp.write_text(''.join(json.dumps(r)+"\n" for r in rows),encoding='utf-8'); tmp.replace(CASES_PATH)
def add_cases(results,source):
    flagged=[r for r in results if r["risk_level"]!="Low"]
    if not flagged:return 0
    now=datetime.now(timezone.utc).isoformat(); added=0
    with _lock:
        rows=_read(); by_id={r["transaction_id"]:r for r in rows}
        for r in flagged:
            old=by_id.get(r["transaction_id"],{}); by_id[r["transaction_id"]]={**old,"transaction_id":r["transaction_id"],"amount":r["amount"],"txn_type":r["txn_type"],"risk_score":r.get("risk_score", r.get("probability", 0.0)),"risk_level":r["risk_level"],"suggested_action":r["suggested_action"],"reasons":[x["text"] for x in r["reasons"]],"source":source,"flagged_at":old.get("flagged_at",now),"updated_at":now,"analyst_decision":old.get("analyst_decision","pending"),"analyst_comment":old.get("analyst_comment",""),"model_version":r.get("model_version","unknown"),"normalized_transaction":r.get("normalized_transaction")}; added+=1 if r["transaction_id"] not in {x["transaction_id"] for x in rows} else 0
        _write(list(by_id.values()))
    return added
def ranked_cases(limit=None):
    with _lock: rows=_read()
    rows.sort(key=lambda r:r.get("risk_score",0),reverse=True)
    for i,r in enumerate(rows,1):r["rank"]=i
    return rows[:limit] if limit else rows
def set_decision(transaction_id,decision,comment=""):
    if decision not in {"pending","confirmed_fraud","genuine"}:raise ValueError("decision must be pending, confirmed_fraud or genuine")
    with _lock:
        rows=_read(); hit=False
        for r in rows:
            if r["transaction_id"]==transaction_id:r["analyst_decision"]=decision;r["analyst_comment"]=comment; r["decision_at"]=datetime.now(timezone.utc).isoformat();hit=True
        if hit:_write(rows)
    return hit
def confirmed_training_cases():
    with _lock:return [r for r in _read() if r.get("analyst_decision") in {"confirmed_fraud","genuine"}]
def clear():
    with _lock:_write([])
def to_csv(rows):
    buf=io.StringIO(); cols=["rank","transaction_id","risk_score","risk_level","suggested_action","amount","txn_type","reasons","analyst_decision","analyst_comment","source","flagged_at","decision_at","model_version"]; w=csv.DictWriter(buf,fieldnames=cols);w.writeheader()
    for r in rows:w.writerow({c:(" | ".join(r.get("reasons",[])) if c=="reasons" else r.get(c,"")) for c in cols})
    return buf.getvalue()
