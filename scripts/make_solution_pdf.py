"""
Build SOLUTION_APPROACH_PS2.pdf — the actual Phase-1 hackathon deliverable
(CyberShield Hackathon, BOI x IIT-H; upload a solution-approach PDF by 15 Jun 2026).

Assembled from the real artifacts/figures this project produced. fpdf2 core fonts are
latin-1, so text is sanitised (Rs for the rupee glyph, ASCII dashes, etc.).
"""
import re
from pathlib import Path
from fpdf import FPDF
from fpdf.enums import XPos, YPos

ROOT = Path(__file__).resolve().parent.parent   # scripts/ -> repo root
FIG = ROOT / "figures"
NAVY, RED, GREY = (31, 59, 110), (200, 30, 45), (90, 90, 90)


def san(s: str) -> str:
    rep = {"₹": "Rs ", "–": "-", "—": "-", "≥": ">=", "≤": "<=",
           "→": "->", "×": "x", "≈": "~", "“": '"', "”": '"',
           "’": "'", "‘": "'", "🛡": "", "✅": "", "\U0001f3af": "",
           "±": "+/-", "•": "-"}
    for k, v in rep.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "replace").decode("latin-1")


class PDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8); self.set_text_color(*GREY)
        self.cell(0, 8, san("SENTINEL - Suspicious / Mule Account Risk Engine | CyberShield Hackathon PS2"), align="L")
        self.ln(10)

    def footer(self):
        self.set_y(-12); self.set_font("Helvetica", "I", 8); self.set_text_color(*GREY)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    def mc(self, *a, **k):
        self.set_x(self.l_margin); self.multi_cell(*a, **k)

    def h1(self, t):
        self.set_font("Helvetica", "B", 15); self.set_text_color(*NAVY)
        self.ln(2); self.mc(0, 8, san(t)); self.ln(1)
        self.set_draw_color(*RED); self.set_line_width(0.6)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y()); self.ln(3)

    def h2(self, t):
        self.set_font("Helvetica", "B", 11.5); self.set_text_color(*RED)
        self.ln(1); self.mc(0, 6, san(t)); self.set_text_color(0, 0, 0)

    def body(self, t):
        self.set_font("Helvetica", "", 10); self.set_text_color(20, 20, 20)
        self.mc(0, 5.2, san(t)); self.ln(1)

    def bullet(self, t):
        self.set_font("Helvetica", "", 10); self.set_text_color(20, 20, 20)
        self.set_x(self.l_margin); self.cell(5); self.multi_cell(0, 5.2, san("- " + t))

    def figure(self, name, w=150, caption=""):
        p = FIG / name
        if p.exists():
            self.image(str(p), x=(self.w - w) / 2, w=w)
            if caption:
                self._fign = getattr(self, "_fign", 0) + 1   # auto-number; no duplicates
                caption = re.sub(r"^Fig\s*\d+\.\s*", "", caption)
                self.set_font("Helvetica", "I", 8.5); self.set_text_color(*GREY)
                self.set_x(self.l_margin)
                self.multi_cell(0, 4.5, san(f"Fig {self._fign}. " + caption), align="C"); self.ln(2)


pdf = PDF(); pdf.set_auto_page_break(True, margin=15); pdf.set_margins(18, 15, 18)

# ---------- COVER ----------
pdf.add_page(); pdf.ln(30)
pdf.set_font("Helvetica", "B", 30); pdf.set_text_color(*NAVY)
pdf.mc(0, 14, "SENTINEL", align="C")
pdf.set_font("Helvetica", "B", 15); pdf.set_text_color(*RED)
pdf.mc(0, 9, san("Leakage-Proof, Explainable Mule-Account Detection"), align="C"); pdf.ln(1)
pdf.set_font("Helvetica", "BI", 12); pdf.set_text_color(40, 40, 40)
pdf.mc(0, 7, san('"Catch the mules. Trust the score. Defend every number."'), align="C"); pdf.ln(3)
# honest-differentiator hook (clean single line, no heavy banner)
pdf.set_font("Helvetica", "BI", 10.5); pdf.set_text_color(*NAVY)
pdf.set_x(pdf.l_margin); pdf.mc(0, 6, san("Most teams will show '99%' - that is data leakage. We detect it, "
       "remove it, and report a number we can defend."), align="C"); pdf.ln(4)
pdf.set_font("Helvetica", "", 12); pdf.set_text_color(40, 40, 40)
pdf.mc(0, 7, san("CyberShield Hackathon 2026  -  Bank of India x IIT Hyderabad\n"
    "Problem Statement 2: AI/ML Classification of Suspicious Mule Accounts\nSolution Approach"), align="C")
pdf.ln(16)
pdf.set_font("Helvetica", "", 11); pdf.set_text_color(*GREY)
pdf.mc(0, 6, san("Team: Probe Rockerz  (individual participation)\n"
    "Participant: Vibhu Krishna S\n"
    "Institute: SRM Easwari Engineering College   |   Date: 14/06/2026"), align="C")
pdf.ln(12)
pdf.set_draw_color(*NAVY); pdf.set_fill_color(245, 246, 250); pdf.set_font("Helvetica", "B", 11)
pdf.set_text_color(*NAVY)
pdf.mc(0, 6.5, san("One line: SENTINEL is the leakage-proof DETECTION & CONTAINMENT core of a real-time "
    "mule-account engine - it scores any account 0-100 in ~35 ms, explains every alert in plain English, "
    "recommends a containment action (monitor / hold / escalate), and removes the data leakage that fakes a "
    "perfect score - so we report the number we can defend."),
    align="C", fill=True, border=1)

# ---------- 1. PROBLEM ----------
pdf.add_page(); pdf.h1("1. Problem Understanding")
pdf.set_font("Helvetica", "BI", 10.5); pdf.set_text_color(*RED)
pdf.mc(0, 5.6, san("Mule accounts are the conduit for nearly all cyber-enabled financial fraud in India - "
    "fraud that runs into thousands of crores a year - which is why RBI made it a national priority and "
    "launched MuleHunter.AI. The bottleneck is no longer 'can we detect it' but 'can we detect it FAST, "
    "explain it, and stop the money before it disperses'."))
pdf.ln(2); pdf.set_text_color(0, 0, 0)
pdf.body("Mule accounts receive, move and launder fraudulent funds across banking channels. "
         "Rule-based monitoring is reactive, brittle, and drowns analysts in false positives. "
         "PS2 asks us to learn behavioural / transactional patterns from the provided data to "
         "flag suspicious & mule accounts, with anomaly detection, predictive risk scoring and "
         "intelligent alert generation.")
pdf.body("The dataset is an account-level SNAPSHOT, not a transaction stream, so 'real-time' here means "
         "on-demand low-latency scoring of an account, not stream processing - we are explicit about this.")
pdf.h2("The deeper problem: containment, not just detection")
pdf.body("Once funds enter a mule chain they disperse across accounts, channels and banks in MINUTES. So the "
         "real objective in PS2's words is to PREVENT CIRCULATION of fraudulent proceeds - speed, linkage and "
         "a decisive action (hold / escalate) matter more than yet another anomaly flag. SENTINEL is built as "
         "a detect -> explain -> ACT containment loop, not a passive classifier.")
pdf.h2("Why this matters now (regulatory relevance)")
pdf.body("Mule-account fraud is a declared national priority: RBI's MuleHunter.AI, NPCI's mule-account pilot "
         "and FATF cyber-fraud guidance all target exactly this. SENTINEL aligns with that direction and adds "
         "what those are not publicly known for: explainable alerts and provable leakage-free scoring. "
         "Primary users are bank fraud/AML analysts and transaction-monitoring & cyber-fraud desks; outputs "
         "feed risk officers and ultimately protect defrauded customers.")
pdf.h2("The trap most teams will fall into")
pdf.body("The dataset is riddled with TARGET LEAKAGE. A naive XGBoost scores PR-AUC 1.000 / ROC-AUC 1.000 "
         "- impossible for genuine fraud detection on 81 anonymised positives; the model is simply reading "
         "the answer. A team that submits that 0.99 gets dismantled by one judge question: 'why is fraud "
         "detection perfect?'. Our entire approach is built to NOT fall into this trap.")

# ---------- 2. DATASET ----------
pdf.add_page(); pdf.h1("2. Dataset - Detailed Description")
pdf.body("Source: DataSet.csv (provided in the portal). Shape: 9,082 rows (one per bank account) x 3,924 "
         "feature columns (F1..F3923), plus the target F3924. Features are anonymised behavioural / "
         "transactional attributes.")
pdf.h2("Target variable (F3924)")
pdf.body("Binary: 1 = suspicious/mule, 0 = legitimate. 81 mules vs 9,001 legitimate = 0.89% prevalence "
         "(1:111 imbalance). Only 81 positives, so overfitting and metric fragility are the dominant risks; "
         "accuracy is meaningless (predicting 'all clean' scores 99.1%). We use PR-AUC, recall@precision and "
         "calibration instead.")
pdf.h2("Feature inventory")
pdf.set_font("Helvetica", "B", 9); pdf.set_fill_color(*NAVY); pdf.set_text_color(255, 255, 255)
pdf.set_x(pdf.l_margin)
pdf.cell(30, 7, " Type", border=1, fill=True); pdf.cell(22, 7, "Count", border=1, fill=True)
pdf.cell(0, 7, " Examples / notes", border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font("Helvetica", "", 9); pdf.set_text_color(20, 20, 20)
for t, c, e in [("Float", "3,876", "continuous behavioural / transactional aggregates & ratios"),
                ("Integer", "40", "counts and binary flags"),
                ("Categorical", "8", "F3889 tenure (G365D/L90D..), F3891 occupation, F3894 ~ age")]:
    pdf.set_x(pdf.l_margin)
    pdf.cell(30, 6, " " + t, border=1); pdf.cell(22, 6, " " + c, border=1)
    pdf.cell(0, 6, " " + san(e), border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(2)
pdf.h2("Data quality & missingness")
for b in ["27.6% of all cells are missing (NaN). ~89 columns are fully populated; ~1,138 columns are >50% "
          "empty; 281 columns are constant and ~390 hyper-sparse - all dropped as dead weight.",
          "Missingness is itself predictive: incomplete profiles correlate with mule behaviour, so we add "
          "(deduped) is-missing indicator features.",
          "We use models that consume NaN natively (LightGBM) - no blind imputation that would distort signal."]:
    pdf.bullet(b)
pdf.h2("Bank-listed features (named in the brief)")
pdf.body("The brief identifies 18 features the bank uses for fraud detection: F115, F321, F527, F531, F670, "
         "F1692, F2082, F2122, F2582, F2678, F2737, F2956, F3043, F3836, F3887, F3889, F3891, F3894. We treat "
         "these as trusted domain signals and NEVER auto-remove them; several separate the classes (e.g. "
         "F2678, F2956, F670).")
pdf.h2("Data integrity: leakage in the raw data (critical)")
pdf.body("Profiling revealed the dataset is contaminated with TARGET LEAKAGE: F3912 (a flag ~96% aligned with "
         "the label), F2230 (a month-stamp that is 100% fraud for 3 of its 4 values, capturing all 81 mules), "
         "and ~585 value/range-leak features. We detect and exclude these (Sections 3-4).")
pdf.h2("How the data is used")
pdf.body("After hygiene + leakage removal + missingness flags, the modelling matrix is 9,082 x 2,965. We "
         "evaluate with repeated stratified 5x2 cross-validation (robust given only 81 positives) and a "
         "held-out set used once. Categorical encoding is target-free; all learned steps are leakage-safe.")

# ---------- 3. USP ----------
pdf.add_page(); pdf.h1("3. Our Unique Selling Proposition")
pdf.body("Anyone can fit a classifier on this data. Four things make SENTINEL different - and we will "
         "defend every one of them in front of any judge:")
pdf.h2("USP 1  -  We find the leakage everyone else ships")
pdf.body("Most teams will proudly present a ~1.000 score. We built an automated Data Integrity Auditor, "
         "caught ~585 leak features, and report a DEFENSIBLE 0.885 with the audit to prove it. When the "
         "judge asks 'why is your fraud model perfect?', we are the team with the answer.")
pdf.h2("USP 2  -  A 100%-precision watchlist on day one")
pdf.body("precision@50 = 100%: our 50 highest-risk accounts are all real mules. That is a ready-to-action "
         "watchlist a fraud desk can use immediately - not a research metric.")
pdf.h2("USP 3  -  Every alert explains itself")
pdf.body("Score + severity band + plain-English SHAP reasons + peer-deviation + a one-click investigation "
         "report + recommended action. An analyst acts in minutes; nothing is a black box.")
pdf.h2("USP 4  -  We speak the bank's language: rupees and capacity")
pdf.body("A rupee-cost engine turns the threshold into money (~Rs 1.7 cr saved / 9,082 accounts) and frames "
         "the operating point as an analyst-capacity decision - exactly how a risk officer thinks.")
pdf.ln(1); pdf.set_font("Helvetica", "BI", 10.5); pdf.set_text_color(*NAVY)
pdf.mc(0, 5.4, san("How we play: we don't win with a flashier model - we win by mastering the fundamentals, "
       "hunting the leakage others miss, and reporting numbers we can stand behind. Dominate the problem, "
       "respect the data, play it straight."))

# ---------- 3. APPROACH ----------
pdf.add_page(); pdf.h1("4. Our Approach & Architecture")
pdf.body("SENTINEL is a layered detect -> explain -> ACT containment engine. We built and validated the CORE "
         "on the provided data; the diagram marks honestly what is built today vs the Phase-2 layer (live "
         "feeds + graph mule-rings). We never claim to have built what we have not.")
pdf.figure("arch_diagram.png", 182,
           "Target architecture. SOLID = built & validated on the provided dataset (hygiene + leak auditor "
           "-> calibrated scoring -> score/SHAP -> containment action -> API/dashboard). DASHED = Phase-2: "
           "live transaction/alert/regulatory feeds + a graph engine for mule-RING detection.")
pdf.h2("Differentiator: a Data Integrity Auditor (rigour, applied)")
pdf.body("We built an automated scanner that scores every feature on FOUR leak signatures and removes the "
         "offenders before modelling:")
for b in ["Label-proxy: a value ~equal to the target (e.g. F3912, 96% aligned with the label).",
          "Exact-value bucket: a discrete value that is hyper-pure fraud (e.g. F2230 is a month-stamp - "
          "'Oct25'=100% legit, 'Sep/Nov/Dec25'=100% fraud, capturing ALL 81 mules) - invisible to monotonic AUC.",
          "Continuous range/decile leaks (caught features the bucket scan missed).",
          "Univariate-AUC near-perfect single-feature rankers, gated by mule coverage."]:
    pdf.bullet(b)
pdf.body("Bank-listed features (the 18 named in the brief) are never auto-removed. Result: ~585 leak "
         "features blocked, leaving a defensible 2,965-feature matrix.")
pdf.h2("Why LightGBM + class weighting (not deep learning, not SMOTE)")
pdf.body("Gradient-boosted trees consume the 27.6% NaN natively, capture non-monotonic interactions, and "
         "train in seconds. With only 81 positives, deep learning overfits and SMOTE manufactures noise in "
         "~3,000 dimensions - so we use class weights + probability calibration instead.")
pdf.h2("Containment action tiers (preventing circulation)")
pdf.body("The score maps to a tiered ACTION, so the output is a decision, not just a number: 0-39 LOW -> "
         "monitor; 40-69 MEDIUM -> enhanced watch / soft-hold high-value transfers; 70-89 HIGH -> hold "
         "outbound + same-day analyst review; 90-100 CRITICAL -> freeze + escalate to L2 + SAR. Thresholds are "
         "configurable to the bank's risk appetite and analyst capacity.")
pdf.h2("How we compare")
pdf.body("vs RBI MuleHunter.AI / NPCI pilots: same regulatory target, but we add explainable alerts and "
         "PROVABLE leakage-free scoring (not publicly documented for those). vs traditional rule engines: we "
         "catch the non-monotonic, networked patterns rules miss. vs black-box ML / teams chasing 0.99: every "
         "score is explained and every number is leakage-audited - we trade a fake 1.0 for a defensible 0.885.")

# ---------- 3. RESULTS ----------
pdf.add_page(); pdf.h1("5. Results (honest, measured, leakage-removed)")
pdf.h2("Headline result a judge cannot knock down")
for b in ["precision@50 = 100%: the 50 highest-risk accounts are ALL 50 real mules - zero false positives "
          "in a 0.89%-mule population.",
          "PR-AUC 0.885 +/- 0.055, ROC-AUC 0.979 (tuned LightGBM, repeated 5x2 cross-validation) - ~99x the "
          "0.0089 random baseline. Calibrated (CV Brier 0.0022). Scoring + explanation ~35 ms.",
          "Honest range 0.81-0.89: even deleting the entire near-label feature block, a leak-paranoid FLOOR "
          "holds at ~0.81. We report the range, not a cherry-picked point."]:
    pdf.bullet(b)
pdf.figure("03_score_distribution.png", 145,
           "Fig 1. Risk-score distribution: legitimate accounts pile up near 0, mules at ~100 (with an honest hard tail).")
pdf.figure("01_pr_curve.png", 120,
           "Fig 2. Precision-Recall (out-of-fold): perfect precision to ~70% recall, then the hard tail. AP=0.89.")

pdf.add_page(); pdf.h1("5. Results (continued)")
pdf.figure("04_confusion_matrix.png", 110,
           "At a low operating threshold: 70/81 mules caught, only 22 false alarms across 9,001 legit accounts.")
pdf.figure("08_leakage_sensitivity.png", 140,
           "Fig 4. Leakage-sensitivity: score collapses 0.998 -> ~0.86 as leaks are removed, then PLATEAUS - "
           "proving the remaining signal is genuine, not one hidden leak.")

# ---------- 5. SEE IT WORK ----------
pdf.add_page(); pdf.h1("6. See It Work - Live System Output")
pdf.body("These are REAL outputs from our running engine for mule account #9003 (not mockups) - the exact "
         "score, SHAP reasons, investigation report and API response an analyst/system receives.")
pdf.h2("Analyst workflow (how it is used)")
pdf.body("Open the dashboard -> accounts ranked by risk -> click a flagged account -> read its score, the "
         "plain-English SHAP reasons and peer-deviation -> trigger the recommended containment action "
         "(hold / escalate) with the auto-generated report attached as evidence. The screens below are the "
         "exact steps in that flow.")
pdf.figure("shot_scoring_card.png", 165,
           "Fig 5. Real-time scoring card: a calibrated risk score, severity band, the SHAP drivers behind it, "
           "and a recommended action.")
pdf.figure("shot_api.png", 120,
           "Fig 6. The live /score REST API response (FastAPI) - drop-in for any monitoring system.")
pdf.add_page(); pdf.h1("6. See It Work (continued)")
pdf.figure("shot_investigation.png", 175,
           "Fig 7. The one-click, auto-generated investigation report - analyst-ready, with confidence caveats "
           "and the leakage note built in.")

# ---------- 4. IMPACT + ALERTS ----------
pdf.add_page(); pdf.h1("7. Business Impact & Intelligent Alerts")
pdf.h2("Rupee-cost decision engine (banks act on money, not PR-AUC)")
pdf.body("Translating the threshold into money (assumptions, configurable: avg mule loss Rs 2,50,000; "
         "analyst review Rs 400; false-positive harm Rs 5,000): at a practical operating point we catch "
         "85-89% of mules for a net ~Rs 1.7 crore saved per 9,082-account population. Net savings is flat "
         "across thresholds, so the operating point is analyst-capacity-bound - the bank chooses the "
         "recall/alert-volume trade-off it can staff.")
pdf.figure("09_cost_curve.png", 135, "Fig 5. Net Rs-savings and recall vs alert threshold.")
pdf.h2("Intelligent, explainable alerts (PS2's 'intelligent alert generation')")
pdf.body("Every alert carries: a 0-100 risk score + severity band (LOW/MEDIUM/HIGH/CRITICAL), the top "
         "SHAP-driven reasons in plain English, the account's deviation vs its peer cohort, a recommended "
         "action, and an auto-generated investigation report - so an analyst acts in minutes, not hours.")
pdf.h2("Go-to-market & sustainability (B2B / regulated)")
pdf.body("Users & buyers: PSU & private banks' fraud, AML and transaction-monitoring desks (buyer: CRO / "
         "Head of Fraud), plus payment bodies (NPCI) and regulator programmes (RBIH).")
pdf.body("Deployment: on-prem / bank-VPC - data never leaves the bank - integrating with core banking and "
         "the existing transaction-monitoring + case-management systems; scoring on-demand via the REST API.")
pdf.body("Adoption path: single-bank pilot on the bank's own book (prove precision@50) -> production scoring "
         "-> multi-bank consortium / federated intelligence - the same distribution model RBIH uses for "
         "MuleHunter.AI.")
pdf.body("Revenue & sustainability: per-bank annual licence + a per-account-scored usage tier; the shared "
         "cross-bank intelligence layer is regulator / consortium-funded. The cost to a bank is a small "
         "fraction of the fraud prevented (~Rs 1.7 cr per 9,082 accounts above). Acquisition is pulled by "
         "regulator credibility and a measurable ROI, with a low-friction pilot on the bank's existing data.")

# ---------- 8. MARKET / BUSINESS / TRACTION ----------
def table(headers, rows, widths, hh=7, rh=6):
    pdf.set_x(pdf.l_margin); pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_fill_color(*NAVY); pdf.set_text_color(255, 255, 255)
    for h, w in zip(headers, widths):
        pdf.cell(w, hh, " " + san(h), border=1, fill=True)
    pdf.ln(hh)
    pdf.set_font("Helvetica", "", 8.5); pdf.set_text_color(20, 20, 20)
    for r in rows:
        pdf.set_x(pdf.l_margin)
        for c, w in zip(r, widths):
            pdf.cell(w, rh, " " + san(str(c)), border=1)
        pdf.ln(rh)
    pdf.ln(2)

pdf.add_page(); pdf.h1("8. Market, Business Model & Traction")
pdf.h2("Market opportunity (illustrative, bottom-up)")
pdf.body("Bottom-up estimates with explicit assumptions; exact market value to be finalised from RBI Annual "
         "Report fraud statistics, I4C / NCRP and industry AML-tech reports.")
table(["Tier", "Definition", "Size (illustrative)"],
      [["TAM", "All Indian banks + payment cos needing mule/AML scoring", "~Rs 300-350 cr/yr"],
       ["SAM", "Scheduled commercial + payment + SFBs (~120, real-time-capable)", "~Rs 120 cr/yr @ ~Rs 1 cr ACV"],
       ["SOM", "3-year capture: design partners -> ~20 banks", "~Rs 18-20 cr ARR"]],
      [18, 110, 46])
pdf.body("Context: RBI-reported bank frauds run into tens of thousands of crores a year and mule accounts are "
         "the conduit - this is regulatory-mandated spend, not discretionary.")
pdf.h2("Business model & unit economics (illustrative)")
table(["Metric", "Value*", "Basis / assumption"],
      [["ACV / bank / yr", "Rs 0.8-1.2 cr", "enterprise licence + per-account usage tier"],
       ["Gross margin", "~80%", "software-led; cloud/compute + support"],
       ["CAC / bank", "~Rs 20-30 L", "enterprise sales + pilot + integration"],
       ["LTV (5-yr)", "~Rs 3.5 cr", "ACV x margin x retention"],
       ["LTV : CAC", "~12-15x", "well above the 3x benchmark"],
       ["CAC payback", "< 6 months", "post-landing"]],
      [34, 28, 112])
table(["Year", "Banks (cum.)", "ARR*", "Driver"],
      [["Y1", "3 (design partners)", "~Rs 1.5 cr", "discounted pilots; prove ROI on real books"],
       ["Y2", "10", "~Rs 8 cr", "regulator-backed referrals (RBIH / NPCI)"],
       ["Y3", "22", "~Rs 18-20 cr", "+ consortium shared-intelligence revenue"]],
      [14, 40, 28, 92])
pdf.set_font("Helvetica", "I", 8); pdf.set_text_color(*GREY)
pdf.mc(0, 4.5, san("*Illustrative & assumption-driven - to be calibrated during paid pilots; not a forecast."))
pdf.set_text_color(0, 0, 0)
pdf.h2("Traction: technical validation today, customer validation next (honest)")
pdf.body("WHAT WE HAVE (technical traction): a working end-to-end prototype, CI smoke tests (5/5), a "
         "deployable FastAPI service, a live Streamlit dashboard, reproducible results and a ranked watchlist "
         "(precision@50 = 100%). WHAT WE DO NOT YET HAVE: customers, paid pilots or LOIs - all traction so far "
         "is technical, and we say so plainly.")
for b in ["Gate 1 (0-3 mo): 1-2 design-partner banks; run SENTINEL on their historical book; validate "
          "precision@50 on real labels; target 2-3 letters of intent.",
          "Gate 2 (3-9 mo): RBIH / NPCI engagement + regulatory sandbox; first paid pilot.",
          "Gate 3 (9-18 mo): production deployment at 1-2 banks; published case study; consortium pilot."]:
    pdf.bullet(b)
pdf.body("Why this is winnable, not wishful: a hard regulatory mandate (RBI MuleHunter.AI), a low-friction "
         "pilot that runs on the bank's own data on-prem, and a measurable rupee ROI - the three levers that "
         "shorten enterprise-bank sales cycles.")

# ---------- 9. FEASIBILITY / ROADMAP ----------
pdf.add_page(); pdf.h1("9. Feasibility, Prototype Roadmap & Honesty")
pdf.h2("Already working (reproducible)")
for b in ["End-to-end pipeline (preprocess -> auditor -> model -> finalize -> insights -> scoring).",
          "Real-time FastAPI /score + /report service (~35 ms) and a Streamlit analyst dashboard.",
          "Deliverables: scored predictions for all 9,082 accounts + a ranked watchlist with reasons.",
          "Model card, smoke tests (5/5 passing), pinned requirements, and a self-contained Colab notebook."]:
    pdf.bullet(b)
pdf.ln(1); pdf.set_font("Helvetica", "B", 9.5); pdf.set_text_color(*NAVY)
pdf.mc(0, 5.2, san("Reproducible end-to-end - run it yourself.   Repo: [GitHub link]   |   "
                   "Live demo: [demo URL]   |   demo video: [link]"))
pdf.set_text_color(0, 0, 0)
pdf.h2("Phase-2 prototype plan (1 Jul - 17 Aug)")
for b in ["GRAPH mule-RING detection (the standout upgrade): ingest shared device / IP / beneficiary links + "
          "circular fund-flow to catch networks, not just isolated accounts.",
          "Multi-feed fusion: live transaction streams + fraud / txn-monitoring alerts + govt cyber-fraud "
          "tickets + RBI/regulatory feeds, scored in real time to PREVENT CIRCULATION.",
          "Confirm with the bank which near-label features are pre- vs post-event (locks the exact number); "
          "plug in real cost figures for a per-channel cost-optimal threshold.",
          "Federated, privacy-preserving cross-bank intelligence; drift monitoring + scheduled leak re-audit.",
          "Hardening: streaming + batch scoring, alert-workflow / case-management integration, audit logging."]:
    pdf.bullet(b)
pdf.h2("What we explicitly do NOT claim (maturity signal)")
for b in ["Not a transaction stream (snapshot data) - we say so plainly.",
          "Anonymised features -> semantic labels in narratives are inferred, not bank-confirmed.",
          "81 positives -> estimates carry real variance; we report a range and a confidence interval.",
          "Decision-support for analysts, not an autonomous freeze authority."]:
    pdf.bullet(b)
pdf.ln(2); pdf.set_font("Helvetica", "B", 10.5); pdf.set_text_color(*NAVY)
pdf.mc(0, 5.5, san("Bottom line: most teams will present a fake 1.000. We present a defensible 0.885, a "
    "100%-precision watchlist, a rupee impact, and the leakage audit that proves we earned it."))

(ROOT / "docs").mkdir(exist_ok=True)
out = ROOT / "docs" / "SOLUTION_APPROACH_PS2.pdf"
pdf.output(str(out))
print(f"Wrote {out} ({out.stat().st_size//1024} KB, {pdf.page_no()} pages)")
