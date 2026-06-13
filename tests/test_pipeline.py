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
