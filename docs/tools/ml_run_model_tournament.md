# ml_run_model_tournament: Phase 3 Master Model Tournament & Anti-Overfit Tuner

> **Tool Name:** `ml_run_model_tournament`  
> **Module Source:** `src/ml_mcp/engine/tournament_orchestrator.py` / `src/ml_mcp/tools.py`  
> **Class Implementation:** `TournamentOrchestrator`  
> **Layer:** Phase 3: Model Training & Refinement (Master Orchestrator)

---

## ১. পেছনের গল্প (The Human Story: "The Overfitting Mirage")

মেশিন লার্নিংয়ে সবচেয়ে বড় বিপর্যয় ঘটে যখন একটি মডেল তার ট্রেইনিং ডেটাসেটে **৯৯.৮% রোক-এউসি (ROC-AUC)** অর্জন করে সবার প্রশংসা কুড়ায়, কিন্তু টেস্ট বা লাইভ ডেটায় গিয়ে ৫৫%-এ ক্র্যাশ করে।

কেন এমন হয়?
সাধারণ হাইপারপ্যারামিটার অপ্টিমাইজেশন (GridSearch, RandomSearch বা স্ট্যান্ডার্ড Optuna) শুধু ভ্যালিডেশন স্কোরের সর্বোচ্চ সংখ্যা খোঁজে। তারা মডেলের ট্রেইনিং স্কোর আর ভ্যালিডেশন স্কোরের মধ্যে তৈরি হওয়া **"জেনারেলিজেশন গ্যাপ" (Generalization Gap)** কিংবা ৫টি ফোল্ডের মধ্যে তৈরি হওয়া ভ্যারিয়েন্স ($\sigma_{	ext{CV}}$) হিসাব করে না। এর ফলে অপটুনা প্রায়ই এমন সব হাইপারপ্যারামিটার পছন্দ করে যা ডেটার মেমোরাইজেশন তৈরি করে।

`ml_run_model_tournament` তৈরি করা হয়েছে এই মিথ্যা মোহের অবসান ঘটাতে। এটি কেবল মডেলের টুর্নামেন্ট চালায় না, বরং **পেনাল্টি-যুক্ত অবজেক্টিভ ফাংশন** দিয়ে অপটুনা টিউন করে ওভারফিটিং শূন্যে নামিয়ে আনে।

---

## ২. কেন এটি ক্রিটিক্যাল? (Operational Invariants & Architecture)

`TournamentOrchestrator` তিনটি প্রধান স্তম্ভের ওপর প্রতিষ্ঠিত:
1. **Multi-Model 5-Fold Stratified Tournament:** একই ডেটায় LightGBM, XGBoost, CatBoost, Random Forest এবং L2-Regularized Logistic Regression/Ridge চালিয়ে বাস্তব ভ্যালিডেশন স্কোর তুলনা করে।
2. **Honest Anti-Overfit Bayesian Tuning:** অপটুনা টিউনিংয়ের সময় প্রতিটি ট্রায়ালে ট্রেইনিং স্কোর ও ভ্যালিডেশন স্কোরের পার্থক্য মাপা হয়। যদি ট্রেইনিং স্কোর ভ্যালিডেশন স্কোরের চেয়ে বেশি হয়, তবে অবজেক্টিভ স্কোর থেকে কঠোর পেনাল্টি কাটা হয়।
3. **KISS Stacking Gate (Keep It Simple, Stupid):** স্ট্যাকিং এনসেম্বল ট্রেইন করা হলেও তা সরাসরি গ্রহণ করা হয় না। যদি স্ট্যাকিং এনসেম্বল একক সেরা চ্যাম্পিয়ন মডেলের চেয়ে কমপক্ষে ১.৫% ($+0.015$) বেশি ভালো না করে, তবে প্রোডাকশন জটিলতা বাঁচাতে একক চ্যাম্পিয়ন মডেলকেই বহাল রাখা হয়।

---

## ৩. গাণিতিক ভিত্তি (Mathematical Foundations)

### ক. পেনাল্টি-যুক্ত ফিটনেস ফাংশন (Penalized Generalization Objective)
$$\mathcal{F}_{	ext{penalized}}(	heta) = \overline{	ext{Score}}_{	ext{val}}(	heta) - \lambda \cdot \max\left(0, \overline{	ext{Score}}_{	ext{train}}(	heta) - \overline{	ext{Score}}_{	ext{val}}(	heta)ight) - \gamma \cdot \sigma_{	ext{CV}}(	heta)$$
যেখানে:
- $\overline{	ext{Score}}_{	ext{val}}$: ৫-ফোল্ড ক্রস-ভ্যালিডেশনের গড় স্কোর।
- $\lambda$: ওভারফিটিং পেনাল্টি গুণাঙ্ক (ডিফল্ট $\lambda = 1.0$)।
- $\sigma_{	ext{CV}}$: ফোল্ডগুলোর স্কোরের স্ট্যান্ডার্ড ডেভিয়েশন।
- $\gamma$: ভ্যারিয়েন্স পেনাল্টি ফ্যাক্টর ($\gamma = 0.5$)।

### খ. KISS Stacking ডিসিশন রুল
$$	ext{Deploy}(	ext{Stacking}) = egin{cases} 	ext{True} & 	ext{if } 	ext{Score}_{	ext{stack}} \ge 	ext{Score}_{	ext{champion}} + 0.015 \ 	ext{False} & 	ext{otherwise (Keep Champion)} \end{cases}$$

---

## ৪. MCP Tool Interface & Input Parameters

```python
async def ml_run_model_tournament(
    csv_path: str,
    target_column: str,
    task_type: Literal["classification", "regression"] = "classification",
    primary_metric: str = "pr_auc",
    n_splits: int = 5,
    tune_trials: int = 20,
    overfit_penalty_lambda: float = 1.0,
    view: Literal["compact", "detailed"] = "compact",
) -> Dict[str, Any]
```

---

## ৫. এক্সিকিউটিভ সার্টিফিকেট কার্ড (Receipt Output Example)

```text
========================================================================================
🏆 ML-MCP PHASE 3: MODEL TOURNAMENT & ANTI-OVERFIT REFINEMENT
Champion Architecture : lightgbm
Tuned Validation Score: 0.8842 PR-AUC
Training Score        : 0.9015 PR-AUC
Generalization Gap    : 0.0173 (1.73% -> Status: HEALTHY_MINIMAL_GAP)
Cross-Val Stability   : +/- 0.0092 across 5 folds
========================================================================================

🏁 1. BASELINE TOURNAMENT STANDINGS:
  • #1 lightgbm            : 0.8650 PR-AUC (+/- 0.0110)
  • #2 xgboost             : 0.8590 PR-AUC (+/- 0.0125)
  • #3 catboost            : 0.8540 PR-AUC (+/- 0.0130)
  • #4 random_forest       : 0.8210 PR-AUC (+/- 0.0150)
  • #5 logistic_regression : 0.7420 PR-AUC (+/- 0.0180)

🎯 2. ANTI-OVERFIT BAYESIAN TUNING (20 Trials):
  • Raw Validation PR-AUC  : 0.8842
  • Penalized Study Metric : 0.8623 (Penalized for 0.0173 gap and 0.0092 CV std)
  • Overfit Verdict        : ✅ SAFE FOR PRODUCTION (No memoization detected)

🧱 3. KISS STACKING ENSEMBLE GATE:
  • Stacking Blend Score   : 0.8890 PR-AUC
  • Delta vs Champion      : +0.0048 (< 0.0150 KISS threshold)
  • Gate Decision          : 🛡️ REJECT STACKING (Deploy single Champion LightGBM for speed)

👉 READY FOR PHASE 4: Run ml_certify_safety_and_decisions!
========================================================================================
```
