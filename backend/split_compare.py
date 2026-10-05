"""Diagnostic random-holdout comparison; primary evaluation remains temporal."""
from __future__ import annotations
import json,joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score,roc_auc_score
from . import config
from .data_adapter import load_path
from .feature_engineering import build_features
from .train import build_pipeline,calibrate,threshold_metrics

def run():
    bundle=joblib.load(config.MODEL_PATH); common=load_path(config.DEFAULT_JSON_PATH); X=build_features(common); y=common.is_fraud.astype(int)
    tr,te=train_test_split(common.index,test_size=.30,stratify=y,random_state=42)
    model=build_pipeline(42,**bundle["best_params"]); model.fit(X.loc[tr],y.loc[tr]); raw=model.predict_proba(X.loc[te])[:,1]; proba=calibrate(bundle.get("calibrator"),raw) if bundle.get("calibrator") else raw; threshold=float(bundle["decision_threshold"]); m=threshold_metrics(y.loc[te],proba,threshold)
    out={"type":"random_holdout_comparison","metrics":m,"pr_auc":round(float(average_precision_score(y.loc[te],proba)),4),"roc_auc":round(float(roc_auc_score(y.loc[te],proba)),4),"note":"This is a secondary diagnostic using the production model configuration and threshold. The primary evaluation is chronological to simulate future transactions."}; config.TEMPORAL_PATH.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2)); return out
if __name__=="__main__":run()
