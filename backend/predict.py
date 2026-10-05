"""Real-time fraud scoring with calibrated risk scores and SHAP explanations."""
from __future__ import annotations
import hashlib, json, joblib, numpy as np, pandas as pd
from . import config
from .feature_engineering import build_features
class ModelNotTrainedError(RuntimeError): pass

def risk_level(score,low_max=config.RISK_LOW_MAX,high_min=config.RISK_HIGH_MIN):
    if score<low_max:return "Low"
    if score<high_min:return "Medium"
    return "High"
def _safe_record(row: pd.Series) -> dict:
    raw=row.to_dict()
    return json.loads(json.dumps(raw, default=lambda x: x.item() if hasattr(x, "item") else str(x)))

class FraudModel:
    def __init__(self,path=config.MODEL_PATH):
        if not path.exists(): raise ModelNotTrainedError("No trained model found. Run: python -m backend.generate_data then python -m backend.train")
        bundle=joblib.load(path); meta=bundle.get("metadata",{}); expected=list(config.FEATURE_COLUMNS); actual=list(bundle.get("feature_columns",[])); expected_hash=config.feature_hash()
        if meta.get("schema_version") and meta["schema_version"]!=config.SCHEMA_VERSION: raise ModelNotTrainedError(f"Model schema {meta['schema_version']} is incompatible with API schema {config.SCHEMA_VERSION}. Retrain the model.")
        if actual!=expected or meta.get("feature_hash",expected_hash)!=expected_hash: raise ModelNotTrainedError("Model feature schema is incompatible with the current application. Retrain the model.")
        self.pipeline=bundle["pipeline"]; self.calibrator=bundle.get("calibrator"); self.features=actual; self.threshold=float(bundle["decision_threshold"]); self.low_max=float(bundle.get("risk_low_max",config.RISK_LOW_MAX)); self.high_min=float(bundle.get("risk_high_min",config.RISK_HIGH_MIN)); self.model_version=meta.get("model_version","unversioned")
        self.info={k:bundle.get(k) for k in ("trained_on","trained_at")}; self.info.update({"model_role":meta.get("source_role","unknown"),"schema_version":meta.get("schema_version"),"feature_hash":meta.get("feature_hash"),"calibration":meta.get("calibration","none"),"model_version":self.model_version}); self._explainer=None
    @property
    def explainer(self):
        if self._explainer is None:
            import shap; self._explainer=shap.TreeExplainer(self.pipeline.named_steps["xgb"])
        return self._explainer
    def _calibrate(self,raw):
        raw=np.asarray(raw)
        if self.calibrator is None:return raw
        p=np.clip(raw,1e-6,1-1e-6); z=np.log(p/(1-p)).reshape(-1,1); return self.calibrator.predict_proba(z)[:,1]
    def _reasons(self,X_row,shap_row,top_k=4):
        order=np.argsort(-shap_row); reasons=[]
        for i in order:
            if shap_row[i]<=0.05: break
            f=self.features[i]; reasons.append({"feature":f,"text":config.REASON_TEXT.get(f,f),"value":round(float(X_row.iloc[i]),3),"impact":round(float(shap_row[i]),3)})
            if len(reasons)>=top_k: break
        return reasons
    def score(self,common,explain="flagged",explain_limit=300):
        X=build_features(common)[self.features]; raw=self.pipeline.predict_proba(X)[:,1]; proba=self._calibrate(raw); levels=[risk_level(p,self.low_max,self.high_min) for p in proba]
        want=np.zeros(len(X),dtype=bool)
        if explain=="all":want[:]=True
        elif explain=="flagged":want=np.array([x!="Low" for x in levels])
        idx=np.where(want)[0]
        if len(idx)>explain_limit: idx=idx[np.argsort(-proba[idx])[:explain_limit]]
        shap_vals={}
        if len(idx):
            sv=self.explainer.shap_values(X.iloc[idx]); shap_vals={int(i):sv[k] for k,i in enumerate(idx)}
        out=[]
        for i in range(len(X)):
            row=common.iloc[i]; reasons=self._reasons(X.iloc[i],shap_vals[i]) if i in shap_vals else []
            score=round(float(proba[i]),4); level=levels[i]
            out.append({"transaction_id":row.transaction_id,"amount":float(row.amount),"txn_type":row.txn_type,"hour":int(row.hour),"risk_score":score,"risk_level":level,"suggested_action":config.SUGGESTED_ACTION[level],"predicted_fraud":bool(score>=self.threshold),"reasons":reasons,"reasons_label":"Why this was flagged" if level!="Low" else "Factors that raised the score (not enough to flag)","actual_label":int(row.is_fraud) if "is_fraud" in common.columns else None,"model_version":self.model_version,"normalized_transaction":_safe_record(row)})
        return out
