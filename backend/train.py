"""Leakage-aware training for the single-XGBoost fraud classifier.

Main evaluation is chronological: past transactions train the model and later
transactions form the untouched test set. SMOTE is applied inside CV only.
A small probability-calibration layer is fitted on out-of-fold XGBoost scores;
it calibrates the score but is not a second fraud classifier.
"""
from __future__ import annotations
import argparse, hashlib, json, math, time, warnings
import joblib, matplotlib
import numpy as np, pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from xgboost import XGBClassifier
from . import config
from .data_adapter import from_paysim, load_path
from .feature_engineering import build_features
warnings.filterwarnings("ignore", category=UserWarning)
PARAM_GRID={"n_estimators":[80,120,160,200],"max_depth":[3,4,5],"learning_rate":[.03,.05,.1],"subsample":[.8,1.0],"colsample_bytree":[.8,1.0],"min_child_weight":[1,3]}
THRESHOLDS=[.20,.30,.35,.40,.45,.50,.55,.60,.70,.80]
FP_COST=1.0; FN_COST=5.0

def load_common(source,path,sample,seed):
    if source=="paysim":
        if not path: raise SystemExit("--path is required with --source paysim")
        raw=pd.read_csv(path)
        if sample and len(raw)>sample:
            fraud=raw[raw.isFraud==1]; rest=raw[raw.isFraud==0].sample(max(sample-len(fraud),1),random_state=seed); raw=pd.concat([fraud,rest]).sample(frac=1,random_state=seed)
        return from_paysim(raw.reset_index(drop=True))
    return load_path(path or config.DEFAULT_JSON_PATH)

def temporal_indices(common,test_size=.30):
    ts=pd.to_datetime(common.timestamp,errors="coerce"); order=ts.sort_values(kind="stable").index; cut=max(1,min(len(order)-1,int(len(order)*(1-test_size))))
    return order[:cut],order[cut:]

def random_indices(common,test_size=.30,seed=42):
    return train_test_split(common.index,test_size=test_size,stratify=common.is_fraud,random_state=seed)

def build_pipeline(seed,**params):
    return ImbPipeline([("smote",SMOTE(random_state=seed)),("xgb",XGBClassifier(eval_metric="logloss",tree_method="hist",n_jobs=1,random_state=seed,**params))])

def _logit(p):
    p=np.clip(np.asarray(p),1e-6,1-1e-6); return np.log(p/(1-p))

def fit_calibrator(oof_scores,y):
    cal=LogisticRegression(solver="lbfgs",random_state=42); cal.fit(_logit(oof_scores).reshape(-1,1),np.asarray(y)); return cal

def calibrate(cal,raw): return np.clip(cal.predict_proba(_logit(raw).reshape(-1,1))[:,1],0,1)

def threshold_metrics(y,proba,t):
    pred=(np.asarray(proba)>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel(); precision=precision_score(y,pred,zero_division=0); recall=recall_score(y,pred,zero_division=0); f1=f1_score(y,pred,zero_division=0); cost=FP_COST*fp+FN_COST*fn
    return {"threshold":round(float(t),4),"precision":round(float(precision),4),"recall":round(float(recall),4),"f1":round(float(f1),4),"accuracy":round(float(accuracy_score(y,pred)),4),"false_positive_rate":round(float(fp/(fp+tn)) if fp+tn else 0,4),"false_negative_rate":round(float(fn/(fn+tp)) if fn+tp else 0,4),"flagged":int(pred.sum()),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),"expected_cost":round(float(cost),2)}

def choose_threshold(y,proba):
    grid=np.unique(np.r_[np.linspace(.05,.95,181),THRESHOLDS]); rows=[threshold_metrics(y,proba,t) for t in grid]
    # Primary policy: minimize expected operational error with FN five times FP.
    best=min(rows,key=lambda r:(r["expected_cost"],-r["recall"],-r["f1"]))
    return float(best["threshold"]),rows,best

def reference_profile(X):
    ref={}
    for c in X.columns:
        s=pd.to_numeric(X[c],errors="coerce").fillna(0).astype(float); q=np.unique(np.quantile(s,np.linspace(0,1,11)))
        if len(q)<2: q=np.array([s.min()-1,s.max()+1])
        ref[c]={"mean":float(s.mean()),"std":float(s.std()),"quantiles":[float(x) for x in q]}
    return ref

def save_plots(y,proba,thr,cm,shap_imp):
    out=config.REPORT_DIR; prec,rec,_=precision_recall_curve(y,proba)
    plt.figure(figsize=(5,4)); plt.plot(rec,prec); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"Precision-Recall curve (PR-AUC={average_precision_score(y,proba):.3f})"); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(out/"pr_curve.png",dpi=140); plt.close()
    plt.figure(figsize=(4,3.6)); plt.imshow(cm,cmap="Blues");
    for i in range(2):
        for j in range(2): plt.text(j,i,f"{cm[i,j]:,}",ha="center",va="center",color="white" if cm[i,j]>np.max(cm)/2 else "black")
    plt.xticks([0,1],["Genuine","Fraud"]); plt.yticks([0,1],["Genuine","Fraud"]); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion matrix (threshold={thr:.3f})"); plt.tight_layout(); plt.savefig(out/"confusion_matrix.png",dpi=140); plt.close()
    top=shap_imp[:12][::-1]; plt.figure(figsize=(6,4)); plt.barh([x["feature"] for x in top],[x["mean_abs_shap"] for x in top]); plt.xlabel("Mean |SHAP value|"); plt.title("What drives model decisions"); plt.tight_layout(); plt.savefig(out/"shap_importance.png",dpi=140); plt.close()

def run_training(source="json",path=None,sample=None,n_iter=3,cv=3,seed=42,test_size=.30,scoring="average_precision",split="temporal",verbose=True,save=True):
    t0=time.time(); log=print if verbose else (lambda *a,**k:None); common=load_common(source,path,sample,seed)
    if common.is_fraud.nunique()<2: raise ValueError("Training data must contain both genuine and fraud transactions.")
    log(f"Source: {common.attrs.get('source')} | rows={len(common):,} | fraud={int(common.is_fraud.sum()):,} ({common.is_fraud.mean():.2%})")
    X=build_features(common); y=common.is_fraud.astype(int)
    idx_tr,idx_te=temporal_indices(common,test_size) if split=="temporal" else random_indices(common,test_size,seed)
    X_tr,X_te=X.loc[idx_tr],X.loc[idx_te]; y_tr,y_te=y.loc[idx_tr],y.loc[idx_te]
    if y_tr.nunique()<2 or y_te.nunique()<2: raise ValueError("Both train and test periods must contain genuine and fraud examples. Increase dataset size or adjust test_size.")
    log(f"Split={split}: train={len(X_tr):,} test={len(X_te):,} | train period {common.loc[idx_tr,'timestamp'].min()} -> {common.loc[idx_tr,'timestamp'].max()} | test period {common.loc[idx_te,'timestamp'].min()} -> {common.loc[idx_te,'timestamp'].max()}")
    skf=StratifiedKFold(n_splits=cv,shuffle=True,random_state=seed)
    search=RandomizedSearchCV(build_pipeline(seed),{f"xgb__{k}":v for k,v in PARAM_GRID.items()},n_iter=n_iter,scoring=scoring,cv=skf,random_state=seed,n_jobs=1,refit=True)
    log(f"Tuning XGBoost: {n_iter} configs x {cv}-fold CV..."); search.fit(X_tr,y_tr); best=search.best_estimator_; params={k.replace("xgb__",""):(v.item() if hasattr(v,"item") else v) for k,v in search.best_params_.items()}
    oof_raw=cross_val_predict(build_pipeline(seed,**params),X_tr,y_tr,cv=skf,method="predict_proba",n_jobs=-1)[:,1]
    calibrator=fit_calibrator(oof_raw,y_tr); oof=calibrate(calibrator,oof_raw); threshold,threshold_rows,policy=choose_threshold(y_tr,oof)
    raw_test=best.predict_proba(X_te)[:,1]; proba=calibrate(calibrator,raw_test); pred=(proba>=threshold).astype(int); cm=confusion_matrix(y_te,pred,labels=[0,1]); tn,fp,fn,tp=(int(v) for v in cm.ravel())
    import shap; xgb=best.named_steps["xgb"]; sx=X_te.sample(min(1500,len(X_te)),random_state=seed); sv=shap.TreeExplainer(xgb).shap_values(sx); mean_abs=np.abs(sv).mean(axis=0); shap_imp=sorted([{"feature":f,"mean_abs_shap":round(float(v),4)} for f,v in zip(X_te.columns,mean_abs)],key=lambda d:d["mean_abs_shap"],reverse=True)
    at=threshold_metrics(y_te,proba,threshold); at["confusion_matrix"]={"tn":at.pop("tn"),"fp":at.pop("fp"),"fn":at.pop("fn"),"tp":at.pop("tp")}
    metrics={"schema_version":config.SCHEMA_VERSION,"trained_at":time.strftime("%Y-%m-%d %H:%M:%S"),"data_source":common.attrs.get("source"),"source_role":"paysim_baseline" if source=="paysim" else "main_realtime_model","rows":int(len(common)),"fraud_rows":int(y.sum()),"train_rows":int(len(X_tr)),"test_rows":int(len(X_te)),"test_fraud":int(y_te.sum()),"cv":{"folds":cv,"n_iter":n_iter,"scoring":scoring,"best_score":round(float(search.best_score_),4)},"best_params":params,"decision_threshold":round(threshold,4),"threshold_policy":{"type":"cost_sensitive","false_positive_cost":FP_COST,"false_negative_cost":FN_COST,"training_selection":policy},"at_decision_threshold":at,"pr_auc":round(float(average_precision_score(y_te,proba)),4),"roc_auc":round(float(roc_auc_score(y_te,proba)),4),"threshold_analysis":threshold_rows,"risk_band_distribution":risk_band_table(y_te,proba),"shap_importance":shap_imp,"split":{"type":split,"test_size":test_size,"seed":seed,"train_start":str(common.loc[idx_tr,'timestamp'].min()),"train_end":str(common.loc[idx_tr,'timestamp'].max()),"test_start":str(common.loc[idx_te,'timestamp'].min()),"test_end":str(common.loc[idx_te,'timestamp'].max())},"feature_reference":reference_profile(X_tr),"notes":["Primary evaluation uses a chronological train/test split.","SMOTE is applied only inside training cross-validation folds.","The displayed risk score is calibrated and should not be interpreted as a guarantee.","The semi-structured dataset is generated for research; production deployment requires validation on the target platform labels."],"limitations_note":"Synthetic semi-structured data is used for development; external real-world validation remains necessary before production use."}
    if save:
        model_version=f"xgb-{int(time.time())}"
        model_path=config.PAYSIM_MODEL_PATH if source=="paysim" else config.MODEL_PATH; metrics_path=config.PAYSIM_METRICS_PATH if source=="paysim" else config.METRICS_PATH
        meta={"schema_version":config.SCHEMA_VERSION,"common_columns":list(config.COMMON_COLUMNS),"feature_columns":list(X.columns),"feature_hash":config.feature_hash(),"classifier":"XGBoost","imbalance_method":"SMOTE","calibration":"Platt-style sigmoid calibration on out-of-fold XGBoost scores","source_role":metrics["source_role"],"trained_source":metrics["data_source"],"model_version":model_version}
        bundle={"pipeline":best,"calibrator":calibrator,"feature_columns":list(X.columns),"best_params":params,"decision_threshold":threshold,"risk_low_max":config.RISK_LOW_MAX,"risk_high_min":config.RISK_HIGH_MIN,"trained_on":metrics["data_source"],"trained_at":metrics["trained_at"],"metadata":meta}
        joblib.dump(bundle,model_path); metrics_path.write_text(json.dumps(metrics,indent=2)); save_plots(y_te,proba,threshold,cm,shap_imp)
        with open(config.MODEL_REGISTRY_PATH,"a",encoding="utf-8") as f: f.write(json.dumps({"version":model_version,"trained_at":metrics["trained_at"],"model_path":str(model_path),"schema_version":config.SCHEMA_VERSION,"pr_auc":metrics["pr_auc"],"roc_auc":metrics["roc_auc"],"threshold":threshold,"split":split})+"\n")
    if verbose:
        log(f"Threshold={threshold:.3f} | Precision={at['precision']:.3f} Recall={at['recall']:.3f} F1={at['f1']:.3f} PR-AUC={metrics['pr_auc']:.3f} ROC-AUC={metrics['roc_auc']:.3f}"); log(f"Confusion matrix TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}"); log(f"Saved model -> {model_path if save else 'disabled'}"); log(f"Done in {time.time()-t0:.0f}s")
    return metrics

def risk_band_table(y,proba):
    bands={"Low":proba<config.RISK_LOW_MAX,"Medium":(proba>=config.RISK_LOW_MAX)&(proba<config.RISK_HIGH_MIN),"High":proba>=config.RISK_HIGH_MIN}; y=np.asarray(y); return {k:{"genuine":int(((y==0)&m).sum()),"fraud":int(((y==1)&m).sum())} for k,m in bands.items()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--source",choices=["json","paysim"],default="json"); ap.add_argument("--path"); ap.add_argument("--sample",type=int); ap.add_argument("--n-iter",type=int,default=3); ap.add_argument("--cv",type=int,default=3); ap.add_argument("--scoring",default="average_precision"); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--test-size",type=float,default=.30); ap.add_argument("--split",choices=["temporal","random"],default="temporal"); a=ap.parse_args(); run_training(a.source,a.path,a.sample,a.n_iter,a.cv,a.seed,a.test_size,a.scoring,a.split)
if __name__=="__main__": main()
