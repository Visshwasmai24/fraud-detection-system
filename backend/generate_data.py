"""Generate reusable semi-structured transaction logs with realistic overlap and history-derived features."""
from __future__ import annotations
import argparse,json
from collections import deque
from datetime import datetime,timedelta
import numpy as np
from .config import DEFAULT_JSON_PATH
PATTERNS=["account_takeover","mule_drain","velocity_burst","social_engineering","credential_compromise","device_switch"]
PATTERN_SHARE=np.array([.25,.20,.18,.12,.12,.13])

def _hour(rng,p=.1): return int(rng.choice([23,0,1,2,3,4])) if rng.random()<p else int(np.clip(round(rng.normal(14,4.5)),5,22))
def _genuine(rng,balance):
    t=str(rng.choice(["PAYMENT","TRANSFER","CASH_OUT","CASH_IN","DEBIT"],p=[.46,.30,.10,.08,.06]))
    amount=float(rng.lognormal(6.5,1.05));
    if rng.random()<.06 and t in {"PAYMENT","TRANSFER"}: amount=float(rng.uniform(.35,.9)*max(balance,1))
    if t!="CASH_IN": amount=min(amount,.95*max(balance,1))
    return dict(ttype=t,amount=max(amount,1),new_device=rng.random()<.08,geo=rng.random()<.06,new_receiver=rng.random()<(.25 if t=="PAYMENT" else .38),failed=int(rng.poisson(.10)),hour=_hour(rng,.10),receiver_before=float(rng.lognormal(8.4,1.15)))

def _fraud(rng,balance,pattern,recent_count):
    # Deliberately overlap fraud and genuine distributions: no single field determines the label.
    d=dict(receiver_before=float(rng.lognormal(8.3,1.2)))
    if pattern=="account_takeover":
        d.update(ttype=str(rng.choice(["TRANSFER","CASH_OUT","PAYMENT"],p=[.5,.3,.2])),amount=rng.uniform(.25,.95)*max(balance,1),new_device=rng.random()<.62,geo=rng.random()<.42,new_receiver=rng.random()<.55,failed=int(rng.poisson(.8)),hour=_hour(rng,.35))
    elif pattern=="mule_drain":
        d.update(ttype=str(rng.choice(["TRANSFER","CASH_OUT"],p=[.65,.35])),amount=rng.uniform(.35,.98)*max(balance,1),new_device=rng.random()<.25,geo=rng.random()<.18,new_receiver=rng.random()<.78,failed=int(rng.poisson(.3)),hour=_hour(rng,.38))
    elif pattern=="velocity_burst":
        d.update(ttype=str(rng.choice(["TRANSFER","PAYMENT"],p=[.6,.4])),amount=float(rng.lognormal(5.7,.95)),new_device=rng.random()<.25,geo=rng.random()<.12,new_receiver=rng.random()<.58,failed=int(rng.poisson(1.1)),hour=_hour(rng,.25))
    elif pattern=="social_engineering":
        d.update(ttype="TRANSFER",amount=float(rng.lognormal(7.4,.9)),new_device=rng.random()<.28,geo=rng.random()<.16,new_receiver=rng.random()<.88,failed=int(rng.poisson(.15)),hour=_hour(rng,.12))
    elif pattern=="credential_compromise":
        d.update(ttype=str(rng.choice(["PAYMENT","TRANSFER"],p=[.55,.45])),amount=float(rng.lognormal(6.9,1.0)),new_device=rng.random()<.25,geo=rng.random()<.38,new_receiver=rng.random()<.45,failed=int(rng.poisson(1.4)),hour=_hour(rng,.28))
    else:
        d.update(ttype=str(rng.choice(["TRANSFER","PAYMENT"],p=[.65,.35])),amount=float(rng.lognormal(6.4,.85)),new_device=True,geo=rng.random()<.55,new_receiver=rng.random()<.55,failed=int(rng.poisson(.7)),hour=_hour(rng,.2))
    if d["ttype"]!="CASH_IN": d["amount"]=min(d["amount"],.995*max(balance,1))
    return d

def generate(n=30000,fraud_rate=.03,seed=42):
    rng=np.random.default_rng(seed); n_acc=max(700,n//8); n_dev=max(900,n//5); base=datetime(2026,9,1)
    balances=rng.lognormal(9.2,1.1,n_acc).clip(800,400000); ages=rng.lognormal(5.8,.9,n_acc).clip(5,3000).astype(int)
    last=[base-timedelta(days=int(rng.integers(1,20))) for _ in range(n_acc)]
    histories=[deque() for _ in range(n_acc)]; receivers=[set() for _ in range(n_acc)]; devices=[set() for _ in range(n_acc)]; device_counts=[{} for _ in range(n_acc)]
    usual_lat=rng.normal(17.38,.25,n_acc); usual_lon=rng.normal(78.48,.25,n_acc)
    records=[]
    for i in range(n):
        fraud=bool(rng.random()<fraud_rate); pattern=str(rng.choice(PATTERNS,p=PATTERN_SHARE)) if fraud else "none"
        if pattern=="velocity_burst":
            cand=[j for j,h in enumerate(histories) if len(h)>=4]; a=int(rng.choice(cand if cand else np.arange(n_acc)))
        else: a=int(rng.integers(n_acc))
        b=int(rng.integers(n_acc-1)); b=b+1 if b>=a else b
        now=last[a]+timedelta(minutes=int(rng.integers(2,240))); cutoff=now-timedelta(hours=24); h=histories[a]
        while h and h[0][0]<cutoff: h.popleft()
        recent=list(h); c5=sum(t>=now-timedelta(minutes=5) for t,_ in recent); c1=sum(t>=now-timedelta(hours=1) for t,_ in recent); c24=len(recent)
        amt24=sum(v for t,v in recent); avg24=(amt24/c24) if c24 else 0; last_gap=(now-recent[-1][0]).total_seconds()/60 if recent else 1440
        d=_fraud(rng,float(balances[a]),pattern,c24) if fraud else _genuine(rng,float(balances[a]))
        new_receiver=bool(d["new_receiver"] or b not in receivers[a]); new_device=bool(d["new_device"] or not devices[a])
        device_id=f"DEV{int(rng.integers(1000,999999))}" if new_device else str(rng.choice(tuple(devices[a])))
        lat=float(usual_lat[a]+rng.normal(0,.02)); lon=float(usual_lon[a]+rng.normal(0,.02))
        if d["geo"]: lat=float(rng.uniform(8,35)); lon=float(rng.uniform(68,97))
        # Haversine approximation in km for feature realism.
        dist=float(111*np.sqrt((lat-usual_lat[a])**2+((lon-usual_lon[a])*np.cos(np.radians(lat)))**2))
        amount=round(float(max(d["amount"],1)),2); bal=float(balances[a]); after=round(bal+amount if d["ttype"]=="CASH_IN" else max(bal-amount,0),2); rb=round(float(d["receiver_before"]),2)
        ts=now.isoformat()
        records.append({"transaction_id":f"TXN{i:08d}","timestamp":ts,"amount":amount,"type":d["ttype"],"sender":{"account_id":f"ACC{a:05d}","balance_before":round(bal,2),"balance_after":after,"account_age_days":int(ages[a]),"txn_count_5min":c5,"txn_count_1h":c1,"txn_count_24h":c24,"amount_1h":round(sum(v for t,v in recent if t>=now-timedelta(hours=1)),2),"amount_24h":round(amt24,2),"avg_amount_24h":round(avg24,2),"failed_attempts":int(d["failed"]),"minutes_since_last_txn":round(last_gap,2)},"receiver":{"account_id":f"ACC{b:05d}","balance_before":rb,"balance_after":round(rb+amount,2),"is_new_receiver":new_receiver},"device_info":{"device_id":device_id,"is_new_device":new_device,"txn_count_24h":int(device_counts[a].get(device_id,0))},"location":{"country":"IN","latitude":round(lat,6),"longitude":round(lon,6),"distance_from_usual_km":round(dist,2),"geo_mismatch":bool(d["geo"])},"is_fraud":int(fraud),"fraud_pattern":pattern})
        last[a]=now; h.append((now,amount)); receivers[a].add(b); devices[a].add(device_id); device_counts[a][device_id]=device_counts[a].get(device_id,0)+1; balances[a]=after
    rng.shuffle(records); return records

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=30000); ap.add_argument("--fraud-rate",type=float,default=.03); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--out",default=str(DEFAULT_JSON_PATH)); a=ap.parse_args()
    records=generate(a.n,a.fraud_rate,a.seed); Path=__import__('pathlib').Path
    with open(a.out,'w',encoding='utf-8') as f:
        for r in records:f.write(json.dumps(r)+"\n")
    fraud=[r for r in records if r["is_fraud"]]; print(f"Wrote {len(records):,} transactions to {a.out}"); print(f"Fraud: {len(fraud):,} ({len(fraud)/len(records):.2%})")
    for p in PATTERNS: print(f"  {p:<24} {sum(r['fraud_pattern']==p for r in fraud):,}")
if __name__=='__main__': main()
