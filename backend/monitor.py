"""Lightweight feature-distribution drift monitoring using PSI."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from . import config

def _psi(expected,actual,bins=10):
    expected=np.asarray(expected,float); actual=np.asarray(actual,float)
    if expected.size==0 or actual.size==0:return 0.0
    cuts=np.unique(np.quantile(expected,np.linspace(0,1,bins+1)))
    if len(cuts)<3:return 0.0
    cuts[0]-=1e-9; cuts[-1]+=1e-9
    e,_=np.histogram(expected,bins=cuts); a,_=np.histogram(actual,bins=cuts); e=(e+0.5)/(e.sum()+0.5*len(e)); a=(a+0.5)/(a.sum()+0.5*len(a))
    return float(np.sum((a-e)*np.log(a/e)))

def build_report(reference, X_current):
    rows=[]
    for c,meta in reference.items():
        if c not in X_current:continue
        s=pd.to_numeric(X_current[c],errors="coerce").fillna(0).astype(float)
        # Reconstruct a reference distribution from quantile points for a stable approximate PSI.
        q=np.asarray(meta.get("quantiles",[]),float)
        if len(q)>=2:
            rng=np.random.default_rng(7); ref=np.interp(rng.random(max(1000,len(s))),np.linspace(0,1,len(q)),q)
            psi=_psi(ref,s)
        else: psi=0.0
        rows.append({"feature":c,"psi":round(psi,4),"current_mean":round(float(s.mean()),4),"reference_mean":round(float(meta.get("mean",0)),4),"status":"high" if psi>=0.25 else "moderate" if psi>=0.10 else "stable"})
    high=sum(r["status"]=="high" for r in rows); moderate=sum(r["status"]=="moderate" for r in rows)
    return {"status":"drift_detected" if high else "monitor" if moderate else "stable","high_drift_features":high,"moderate_drift_features":moderate,"features":sorted(rows,key=lambda x:x["psi"],reverse=True)}

def save(report): config.DRIFT_PATH.write_text(json.dumps(report,indent=2),encoding="utf-8")
def load(): return json.loads(config.DRIFT_PATH.read_text()) if config.DRIFT_PATH.exists() else None
