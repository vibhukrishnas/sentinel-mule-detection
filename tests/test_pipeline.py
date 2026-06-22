"""Smoke tests — guard the load-bearing invariants. Run: pytest -q (from project root)."""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
ART = ROOT / "artifacts"


def test_leakage_blocked_from_matrix():
    """The two confirmed leak features must NOT be in the modeling matrix."""
    import pandas as pd
    X = pd.read_parquet(ART / "X_clean.parquet")
    assert "F3912" not in X.columns, "F3912 (label proxy) leaked into the matrix"
    assert "F2230" not in X.columns, "F2230 (month-stamp leak) leaked into the matrix"


def test_bank_hint_features_retained():
    """Genuine bank-listed features must survive cleaning."""
    import pandas as pd
    X = pd.read_parquet(ART / "X_clean.parquet")
    for f in ["F115", "F2678", "F3891"]:
        assert f in X.columns, f"bank-listed feature {f} was wrongly dropped"


def test_engine_scores_demo_accounts():
    """Engine returns a valid 0-100 score and a non-trivial report for real accounts."""
    from sentinel import SentinelEngine
    import pandas as pd
    eng = SentinelEngine()
    demo = pd.read_parquet(ART / "demo_accounts.parquet")
    sc = eng.score(demo.iloc[0])
    assert 0 <= sc["risk_score"] <= 100
    assert 0.0 <= sc["probability"] <= 1.0
    assert sc["band"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    rep = eng.report(demo.iloc[0], account_id="test")
    assert "INVESTIGATION REPORT" in rep and "Risk Score" in rep


def test_mules_score_higher_than_legit_on_average():
    """Sanity: real mules should score higher than legit accounts."""
    from sentinel import SentinelEngine
    import pandas as pd
    eng = SentinelEngine()
    X = pd.read_parquet(ART / "demo_accounts.parquet")
    yv = pd.read_parquet(ART / "demo_targets.parquet")["target"]
    mule = [eng.score(X.loc[i])["risk_score"] for i in X.index[yv.values == 1]]
    legit = [eng.score(X.loc[i])["risk_score"] for i in X.index[yv.values == 0]]
    assert sum(mule) / len(mule) > sum(legit) / len(legit)


def test_honest_metrics_present_and_sane():
    """Reported CV PR-AUC must be strong but NOT a leaky ~1.0."""
    he = json.loads((ART / "honest_eval.json").read_text())
    pr = he["leaderboard"][0]["pr_auc"]
    assert 0.6 < pr < 0.98, f"PR-AUC {pr} is implausible (leak?) or too weak"


def test_boi_headline_trains_on_boi_only():
    """INTEGRITY: the BOI headline matrix is exactly the 9,082 BOI accounts / 81 mules —
    no external rows were ever concatenated in. (Guards integrity rules #1 & #2.)"""
    import pandas as pd
    X = pd.read_parquet(ART / "X_clean.parquet")
    y = pd.read_parquet(ART / "y.parquet")
    assert X.shape[0] == 9082, f"BOI matrix has {X.shape[0]} rows (expected 9082) — external rows merged?"
    assert int(y.iloc[:, 0].sum()) == 81, "BOI positive count changed — external positives merged in?"
    boi = ART / "boi" / "metrics.json"
    if boi.exists():
        m = json.loads(boi.read_text())
        assert m.get("dataset") == "BOI"
        assert m["phases"]["0a_baseline"]["leak_removal"]["n_positives"] == 81


def test_no_external_metric_labelled_as_boi():
    """INTEGRITY: every external/graph/auditor metrics file must carry a non-BOI dataset
    tag — an external number can never be presented as the BOI result. (Rule #3.)"""
    checks = {
        ART / "external" / "baf" / "metrics.json": "BAF",
        ART / "graph" / "amlsim" / "metrics.json": "AMLSim",
    }
    for path, expect in checks.items():
        if path.exists():
            tag = str(json.loads(path.read_text()).get("dataset", ""))
            assert expect in tag and tag != "BOI", f"{path} is not clearly tagged '{expect}' (got {tag!r})"


def test_oof_predictions_are_boi_only():
    """INTEGRITY: saved BOI OOF predictions are 9,082 rows, all tagged dataset=BOI."""
    import pandas as pd
    p = ART / "boi" / "oof_predictions.csv"
    if p.exists():
        df = pd.read_csv(p)
        assert len(df) == 9082, f"OOF has {len(df)} rows (expected 9082 BOI accounts)"
        assert (df["dataset"] == "BOI").all(), "non-BOI rows present in BOI OOF predictions"
