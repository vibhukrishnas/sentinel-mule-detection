"""Generate a self-contained Colab notebook (valid .ipynb JSON) for SENTINEL PS2."""
import json
from pathlib import Path

def md(text):  return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}
def code(text):return {"cell_type": "code", "metadata": {}, "execution_count": None,
                       "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}

cells = []

cells.append(md("""# 🛡️ SENTINEL — PS2 Mule-Account Detection (Colab)

Self-contained pipeline: **hygiene → leakage detection → leakage-sensitivity sweep →
model tournament → calibrated final model → explainable report → export artifacts.**

> Built fundamentals-first. The headline isn't a 0.99 score — it's that we **detect and
> remove the leakage** that fakes a 0.99, and report the number we can defend.

**How to use:** Runtime → Run all. Upload `DataSet.csv` when prompted. Artifacts zip
downloads at the end. (CPU runtime is fine — GPU does nothing for tabular boosting.)
"""))

cells.append(md("## 1 · Install dependencies"))
cells.append(code("""
!pip -q install lightgbm xgboost shap scikit-learn pandas numpy
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
print("ready")
"""))

cells.append(md("## 2 · Upload `DataSet.csv`\n(Or mount Drive: see the commented lines.)"))
cells.append(code("""
from google.colab import files
up = files.upload()                      # pick DataSet.csv
CSV = next(iter(up))
# --- Drive alternative ---
# from google.colab import drive; drive.mount('/content/drive')
# CSV = '/content/drive/MyDrive/DataSet.csv'
print("using:", CSV)
"""))

cells.append(md("""## 3 · Preprocess + leakage blocking
Drops dead/sparse columns, ordinal-encodes categoricals, adds deduped missingness
flags, and removes leakage: **F3912** (a 96%-aligned label proxy) plus any feature with
a value-bucket ≥20% fraud (22× the 0.89% base rate) — the non-monotonic leaks that
fool boosters but hide from monotonic AUC. Bank-hinted features are never auto-removed."""))
cells.append(code("""
TARGET="F3924"
HINTS={"F115","F321","F527","F531","F670","F1692","F2082","F2122","F2582","F2678",
       "F2737","F2956","F3043","F3836","F3887","F3889","F3891","F3894"}
LEAK_BLOCK=["F3912"]; LEAK_RATE=0.20; LEAK_MIN_N=10; MIN_NONNA=30

def detect_bucket_leaks(X,y,rate=LEAK_RATE,min_n=LEAK_MIN_N):
    leaks=[]; yv=y.values
    for c in X.columns:
        if c.split("__")[0] in HINTS: continue
        d=pd.DataFrame({"v":X[c].round(3),"y":yv}).dropna(subset=["v"])
        if d.empty: continue
        g=d.groupby("v")["y"]; cnt,mean=g.count(),g.mean(); sel=cnt>=min_n
        if sel.any() and mean[sel].max()>=rate: leaks.append(c)
    return leaks

def preprocess(csv, apply_bucket_leaks=True, verbose=True):
    df=pd.read_csv(csv,index_col=0,low_memory=False)
    y=df[TARGET].astype(int); X=df.drop(columns=[TARGET])
    X=X.drop(columns=[c for c in LEAK_BLOCK if c in X.columns])
    na=X.isna().mean(); nn=X.notna().sum()
    X=X.drop(columns=sorted(set(na[na>0.97].index)|set(nn[nn<MIN_NONNA].index)))
    X=X.drop(columns=X.columns[X.nunique(dropna=True)<=1])
    na2=X.isna().mean(); flag=na2[na2>0.30].index
    mf=pd.DataFrame({f"{c}__ismissing":X[c].isna().astype("int8") for c in flag},index=X.index)
    mf=mf.loc[:,~mf.T.duplicated()]
    for c in X.select_dtypes(include="object").columns:
        cats={v:i for i,v in enumerate(sorted(X[c].dropna().unique(),key=str))}
        X[c]=X[c].map(cats).astype("float32")
    X=pd.concat([X.astype("float32"),mf],axis=1)
    bl=detect_bucket_leaks(X,y) if apply_bucket_leaks else []
    X=X.drop(columns=bl)
    if verbose: print(f"X={X.shape}, pos={int(y.sum())} ({y.mean()*100:.2f}%), "
                      f"blocked F3912 + {len(bl)} bucket-leaks")
    return X,y,bl

X,y,bucket_leaks=preprocess(CSV)
pw=float((y==0).sum()/(y==1).sum())
"""))

cells.append(md("""## 4 · Leakage-sensitivity sweep (the honest-performance curve)
Strip features at falling fraud-rate thresholds; where PR-AUC **flattens** is genuine
behavioral signal. The steep early drop is leakage leaving."""))
cells.append(code("""
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_validate
Xf,_,_=preprocess(CSV, apply_bucket_leaks=False, verbose=False)
cv3=StratifiedKFold(3,shuffle=True,random_state=42)
def cvpr(Xm):
    m=lgb.LGBMClassifier(n_estimators=400,learning_rate=0.04,num_leaves=31,subsample=0.8,
        colsample_bytree=0.6,reg_lambda=1.0,min_child_samples=20,scale_pos_weight=pw,
        n_jobs=-1,random_state=42,verbose=-1)
    r=cross_validate(m,Xm,y,cv=cv3,scoring={"pr":"average_precision","roc":"roc_auc"},n_jobs=1)
    return r["test_pr"].mean(),r["test_pr"].std(),r["test_roc"].mean()
print(f"{'thr':>6}{'n_leaks':>9}{'n_feat':>8}{'PR-AUC':>9}{'ROC':>7}")
for thr in [1.01,0.30,0.20,0.10,0.05,0.03,0.02]:
    lk=[] if thr>1 else detect_bucket_leaks(Xf,y,rate=thr)
    pr,sd,roc=cvpr(Xf.drop(columns=lk))
    print(f"{('none' if thr>1 else f'{thr:.2f}'):>6}{len(lk):>9}{Xf.shape[1]-len(lk):>8}{pr:>9.3f}{roc:>7.3f}")
print(f"baseline PR-AUC={y.mean():.4f}")
"""))

cells.append(md("## 5 · Model tournament (leakage-removed, PR-AUC primary)"))
cells.append(code("""
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=2,random_state=42)
models={
 "LogReg":Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler()),
   ("c",LogisticRegression(max_iter=2000,C=0.1,class_weight="balanced"))]),
 "RandomForest":Pipeline([("i",SimpleImputer(strategy="median")),
   ("c",RandomForestClassifier(n_estimators=400,min_samples_leaf=2,max_features="sqrt",
   class_weight="balanced_subsample",n_jobs=-1,random_state=42))]),
 "LightGBM":lgb.LGBMClassifier(n_estimators=600,learning_rate=0.03,num_leaves=31,subsample=0.8,
   colsample_bytree=0.6,reg_lambda=1.0,min_child_samples=20,scale_pos_weight=pw,
   n_jobs=-1,random_state=42,verbose=-1),
 "XGBoost":xgb.XGBClassifier(n_estimators=600,learning_rate=0.03,max_depth=4,subsample=0.8,
   colsample_bytree=0.6,reg_lambda=1.0,min_child_weight=5,scale_pos_weight=pw,
   tree_method="hist",eval_metric="aucpr",n_jobs=-1,random_state=42)}
res=[]
for n,m in models.items():
    r=cross_validate(m,X,y,cv=cv,scoring={"pr":"average_precision","roc":"roc_auc"},
                     return_train_score=True,n_jobs=1)
    res.append({"model":n,"PR_AUC":r["test_pr"].mean(),"std":r["test_pr"].std(),
                "ROC_AUC":r["test_roc"].mean(),"train_PR":r["train_pr"].mean()})
    print(f"{n:14s} PR-AUC {res[-1]['PR_AUC']:.3f}±{res[-1]['std']:.3f} ROC {res[-1]['ROC_AUC']:.3f}")
lb=pd.DataFrame(res).sort_values("PR_AUC",ascending=False); display(lb)
WINNER=lb.iloc[0]["model"]; print("winner:",WINNER)
"""))

cells.append(md("## 6 · Calibrate winner + honest 80/20 holdout + threshold dial"))
cells.append(code("""
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
def mk(n):
    return {k:v for k,v in models.items()}[n] if n in ("LightGBM","XGBoost") else models[n]
Xtr,Xho,ytr,yho=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
clf=CalibratedClassifierCV(mk(WINNER),method="sigmoid",cv=5).fit(Xtr,ytr)
p=clf.predict_proba(Xho)[:,1]
print(f"HOLDOUT  PR-AUC {average_precision_score(yho,p):.3f} | "
      f"ROC-AUC {roc_auc_score(yho,p):.3f} | Brier {brier_score_loss(yho,p):.4f}")
tt=[]
for t in [0.3,0.5,0.7,0.8,0.9,0.95]:
    pred=(p>=t); tp=int(((pred)&(yho==1)).sum()); fp=int(((pred)&(yho==0)).sum())
    tt.append({"thr":t,"alerts":tp+fp,"mules_caught":tp,"false_alarms":fp,
               "precision":round(tp/(tp+fp),3) if tp+fp else 0,"recall":round(tp/yho.sum(),3)})
display(pd.DataFrame(tt))
final=CalibratedClassifierCV(mk(WINNER),method="sigmoid",cv=5).fit(X,y)
"""))

cells.append(md("## 7 · Explainability — top SHAP drivers for a flagged account"))
cells.append(code("""
import shap
base=mk(WINNER).fit(X,y)
ex=shap.TreeExplainer(base)
idx=y[y==1].index[0]                       # a real mule
row=X.loc[[idx]]
sv=ex.shap_values(row); sv=sv[1] if isinstance(sv,list) else sv
sv=np.asarray(sv).reshape(X.shape[1])
top=np.argsort(-np.abs(sv))[:8]
print(f"Account #{idx} | risk prob = {final.predict_proba(row)[0,1]:.1%} | label = MULE")
for i in top:
    print(f"  {X.columns[i]:18s} value={row.iloc[0,i]!s:>10}  shap={sv[i]:+.3f}  "
          f"{'raises' if sv[i]>0 else 'lowers'} risk")
"""))

cells.append(md("## 8 · Export artifacts (model + metrics) and download"))
cells.append(code("""
import joblib, json, os
os.makedirs("artifacts",exist_ok=True)
joblib.dump(final,"artifacts/sentinel_model.joblib")
joblib.dump(base,"artifacts/base_model.joblib")
lb.to_json("artifacts/tournament_results.json",orient="records",indent=2)
json.dump({"winner":WINNER,"holdout_pr_auc":float(average_precision_score(yho,p)),
           "holdout_roc_auc":float(roc_auc_score(yho,p)),
           "bucket_leaks_removed":len(bucket_leaks)},
          open("artifacts/holdout_metrics.json","w"),indent=2)
!cd /content && zip -qr sentinel_artifacts.zip artifacts
from google.colab import files; files.download("sentinel_artifacts.zip")
print("exported ->", os.listdir("artifacts"))
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 0}

out = Path(__file__).resolve().parent.parent / "colab"; out.mkdir(exist_ok=True)
(out / "SENTINEL_PS2_Colab.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote colab/SENTINEL_PS2_Colab.ipynb with", len(cells), "cells")
