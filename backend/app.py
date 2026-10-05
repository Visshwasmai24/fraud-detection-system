"""Flask API for source-independent fraud scoring and investigation."""
from __future__ import annotations
import json,uuid
from flask import Flask,Response,jsonify,request,send_from_directory
from flask_cors import CORS
from . import config,report,monitor
from .data_adapter import AdapterError,from_api,load_bytes
from .predict import FraudModel,ModelNotTrainedError
from .feature_engineering import build_features
app=Flask(__name__); app.config["MAX_CONTENT_LENGTH"]=25*1024*1024; CORS(app)
_model=None
def _mtime():return config.MODEL_PATH.stat().st_mtime if config.MODEL_PATH.exists() else None
def get_model():
    global _model; mt=_mtime()
    if _model is None or getattr(_model,"_mtime",None)!=mt:_model=FraudModel();_model._mtime=mt
    return _model
@app.errorhandler(AdapterError)
def adapter_error(e):return jsonify(error=str(e)),400
@app.errorhandler(ModelNotTrainedError)
def no_model(e):return jsonify(error=str(e)),503
@app.errorhandler(413)
def too_large(e):return jsonify(error="Upload is too large. Maximum file size is 25 MB."),413
@app.get("/api/health")
def health():
    try:
        m=get_model();return jsonify(status="ok",model_loaded=True,**m.info,decision_threshold=round(m.threshold,4),risk_bands={"low_max":m.low_max,"high_min":m.high_min})
    except ModelNotTrainedError as e:return jsonify(status="no_model",model_loaded=False,hint=str(e))
@app.get("/api/schema")
def schema():return jsonify(schema_version=config.SCHEMA_VERSION,common_columns=config.COMMON_COLUMNS,feature_columns=config.FEATURE_COLUMNS,transaction_example={"transaction_id":"TXN1","timestamp":"2026-10-05T14:30:00","amount":650,"type":"PAYMENT","sender":{"account_id":"ACC1","balance_before":18000,"account_age_days":900,"txn_count_5min":0,"txn_count_1h":1,"txn_count_24h":2,"amount_1h":650,"amount_24h":650,"avg_amount_24h":325,"failed_attempts":0},"receiver":{"account_id":"ACC2","balance_before":50000,"is_new_receiver":False},"device_info":{"device_id":"DEV1","is_new_device":False,"txn_count_24h":10},"location":{"distance_from_usual_km":2,"geo_mismatch":False}})
@app.post("/api/predict")
def predict():
    payload=request.get_json(silent=True)
    if payload is None:return jsonify(error="Send a JSON object or list of transaction objects."),400
    single=isinstance(payload,dict)
    if single: payload={**payload,"transaction_id":payload.get("transaction_id") or "LIVE-"+uuid.uuid4().hex[:8].upper()}
    elif isinstance(payload,list): payload=[{**p,"transaction_id":p.get("transaction_id") or "LIVE-"+uuid.uuid4().hex[:8].upper()} for p in payload]
    else:return jsonify(error="JSON body must be a transaction object or a list."),400
    common=from_api(payload); results=get_model().score(common,explain="all")
    if request.args.get("save","true").lower()!="false":report.add_cases(results,"api")
    return jsonify(results[0] if single else results)
@app.post("/api/batch")
def batch():
    f=request.files.get("file")
    if f is None:return jsonify(error="Upload a .csv, .json or .jsonl file in the 'file' field."),400
    common=load_bytes(f.filename or "upload.jsonl",f.read()); results=get_model().score(common,explain="flagged"); added=report.add_cases(results,f"batch:{f.filename}")
    counts={lv:sum(r["risk_level"]==lv for r in results) for lv in ("Low","Medium","High")}; labelled=[r for r in results if r["actual_label"] is not None]
    summary={"filename":f.filename,"rows":len(results),"data_source":common.attrs.get("source"),"notes":common.attrs.get("notes",[]),"risk_counts":counts,"added_to_report":added}
    if labelled and any(r["actual_label"]==1 for r in labelled):
        tp=sum(r["predicted_fraud"] and r["actual_label"]==1 for r in labelled);fp=sum(r["predicted_fraud"] and r["actual_label"]==0 for r in labelled);fn=sum((not r["predicted_fraud"]) and r["actual_label"]==1 for r in labelled);summary["quick_metrics"]={"precision":round(tp/(tp+fp),4) if tp+fp else 0,"recall":round(tp/(tp+fn),4) if tp+fn else 0,"tp":tp,"fp":fp,"fn":fn}
    X=build_features(common); reference=None
    try: reference=json.loads(config.METRICS_PATH.read_text()).get("feature_reference") if config.METRICS_PATH.exists() else None
    except Exception: reference=None
    if reference: summary["drift"] = monitor.build_report(reference,X)
    top=sorted([r for r in results if r["risk_level"]!="Low"],key=lambda r:r["risk_score"],reverse=True)[:50]
    return jsonify(summary=summary,top_flagged=top)
@app.get("/api/report")
def get_report():return jsonify(report.ranked_cases(request.args.get("limit",type=int)))
@app.get("/api/report/download")
def download_report():return Response(report.to_csv(report.ranked_cases()),mimetype="text/csv",headers={"Content-Disposition":"attachment; filename=investigation_report.csv"})
@app.post("/api/report/<txn_id>/decision")
def decide(txn_id):
    body=request.get_json(silent=True) or {}
    try:ok=report.set_decision(txn_id,body.get("decision",""),body.get("comment",""))
    except ValueError as e:return jsonify(error=str(e)),400
    return jsonify(ok=True) if ok else (jsonify(error="Case not found"),404)
@app.get("/api/report/training-labels")
def training_labels():return jsonify(count=len(report.confirmed_training_cases()),cases=report.confirmed_training_cases())
@app.delete("/api/report")
def clear_report():report.clear();return jsonify(ok=True)
@app.get("/api/metrics")
def metrics():
    if not config.METRICS_PATH.exists():return jsonify(error="No metrics yet. Run: python -m backend.generate_data then python -m backend.train"),404
    data=json.loads(config.METRICS_PATH.read_text());
    for path,key in [(config.GENERALIZATION_PATH,"generalization"),(config.TEMPORAL_PATH,"temporal_evaluation"),(config.DRIFT_PATH,"drift")]:
        if path.exists():data[key]=json.loads(path.read_text())
    return jsonify(data)
@app.post("/api/drift/check")
def drift_check():
    payload=request.get_json(silent=True)
    if payload is None:return jsonify(error="Send a transaction object or list."),400
    common=from_api(payload); reference=json.loads(config.METRICS_PATH.read_text()).get("feature_reference") if config.METRICS_PATH.exists() else None
    if not reference:return jsonify(error="No training reference profile found. Retrain the model first."),409
    result=monitor.build_report(reference,build_features(common)); result["rows_checked"]=len(common); monitor.save(result); return jsonify(result)
@app.get("/api/drift")
def drift():return jsonify(monitor.load() or {"status":"not_run","message":"Run /api/drift/check with a recent batch."})
@app.get("/api/plots/<name>")
def plots(name):
    if name not in {"pr_curve.png","confusion_matrix.png","shap_importance.png"}:return jsonify(error="Unknown plot"),404
    if not (config.REPORT_DIR/name).exists():return jsonify(error="Run training first"),404
    return send_from_directory(config.REPORT_DIR,name)
if __name__=="__main__":print("Fraud Detection API running on http://localhost:5000");app.run(host="0.0.0.0",port=5000,debug=False)
