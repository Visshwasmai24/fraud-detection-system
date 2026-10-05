"""Behavioural feature engineering. No transaction-ID fields are fed to XGBoost."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import FEATURE_COLUMNS, TXN_TYPES

def build_features(common: pd.DataFrame) -> pd.DataFrame:
    df=common.copy(); before=df.sender_balance_before.astype(float); after=df.sender_balance_after.astype(float); amount=df.amount.astype(float)
    rb=df.receiver_balance_before.astype(float); ra=df.receiver_balance_after.astype(float)
    cash_in=df.txn_type.eq("CASH_IN")
    X=pd.DataFrame(index=df.index)
    X["log_amount"]=np.log1p(amount)
    X["amount_to_balance_ratio"]=(amount/(before+1)).clip(0,5)
    X["balance_drop_ratio"]=((before-after)/(before+1)).clip(-1,1)
    X["receiver_gain_ratio"]=((ra-rb)/(ra+1)).clip(-1,1)
    expected=np.where(cash_in,before+amount,before-amount)
    X["balance_mismatch"]=np.log1p(np.abs(after-expected))
    X["hour"]=df.hour.astype(float); X["is_night"]=((df.hour>=23)|(df.hour<5)).astype(float); X["is_weekend"]=df.is_weekend.astype(float)
    X["account_age_days"]=df.account_age_days.astype(float)
    X["txn_count_5min"]=df.txn_count_5min.astype(float); X["txn_count_1h"]=df.txn_count_1h.astype(float); X["txn_count_24h"]=df.txn_count_24h.astype(float)
    X["log_amount_1h"]=np.log1p(df.amount_1h.astype(float).clip(lower=0)); X["log_amount_24h"]=np.log1p(df.amount_24h.astype(float).clip(lower=0))
    avg=df.avg_amount_24h.astype(float); X["amount_vs_avg_24h"]=(amount/(avg+1)).clip(0,50)
    X["failed_attempts"]=df.failed_attempts.astype(float).clip(0,50); X["device_txn_count_24h"]=df.device_txn_count_24h.astype(float).clip(0,1000)
    X["minutes_since_last_txn"]=df.minutes_since_last_txn.astype(float).clip(0,10080); X["distance_from_usual_km"]=df.distance_from_usual_km.astype(float).clip(0,20000)
    for c in ["is_new_device","geo_mismatch","is_new_receiver"]: X[c]=df[c].astype(float)
    for t in TXN_TYPES: X[f"type_{t}"]=df.txn_type.eq(t).astype(float)
    X=X[FEATURE_COLUMNS].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    return X
