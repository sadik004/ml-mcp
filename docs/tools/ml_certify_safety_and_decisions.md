# ml_certify_safety_and_decisions: Phase 4 Master Safety, Calibration & Decision Orchestrator

> **Tool Name:** `ml_certify_safety_and_decisions`  
> **Module Source:** `src/ml_mcp/engine/safety_orchestrator.py` / `src/ml_mcp/tools.py`  
> **Class Implementation:** `SafetyOrchestrator`  
> **Layer:** Phase 4: Validation & Safety (Master Orchestrator)

---

## ১. পেছনের গল্প (The Human Story: "The Danger of Naive 0.50 Thresholds")

অধিকাংশ মেশিন লার্নিং মডেল প্রোডাকশনে ফেইল করে অ্যালগরিদমের ভুলের জন্য নয়, বরঞ্চ **বিজনেস ও সেফটি পলিসির ভুলের জন্য**।

বাস্তব উদাহরণ:
ধরা যাক একটি হাসপাতালে রোগীর সেপসিস বা হার্ট অ্যাটাক ডিটেকশন মডেল বসানো হয়েছে। ডিফল্টভাবে পাইথন ক্লাসিফায়াররা `prediction = probability >= 0.50` হিসাব করে।  
এখন, একজন অসুস্থ রোগীকে সুস্থ বলে ছেড়ে দিলে (False Negative) তার মৃত্যুর ঝুঁকি তৈরি হয় ($FN 	ext{ cost} = \$250$ বা জীবনহানি)। অপরদিকে, একজন সুস্থ রোগীকে সতর্কতার সাথে অতিরিক্ত টেস্ট করালে (False Positive) মাত্র \$5 খরচে রক্ত পরীক্ষা করা যায় ($FP 	ext{ cost} = \$5$)।  
এই অসমান বাস্তব খরচের দুনিয়ায় ৫০% কাট-অফ দিয়ে সিদ্ধান্ত নেওয়া একটি অপরাধমূলক বোকামি! এখানে অপটিমাল থ্রেশহোল্ড হওয়া উচিত $p^* pprox 0.08$ বা ৮%।

একইভাবে, মডেল যদি কোনো অদ্ভুত নতুন রোগীর ক্ষেত্রে আত্মবিশ্বাসের সাথে ৯৯% ভুল ডায়াগনোসিস দেয় (Out-of-Distribution), তবে পুরো সিস্টেম ধসে পড়তে পারে।

`ml_certify_safety_and_decisions` হলো সেই **মাস্টার সেফটি ও ডিসিশন সার্টিফিকেট**, যা প্রোডাকশন রিলিজের আগে মডেলের প্রবাবিলিটি ক্যালিব্রেট করে, প্রকৃত অর্থনৈতিক ডলারে লস কমায়, ৯৫% কনফর্মাল গ্যারান্টি দেয় এবং অ্যানোমালি কাট-অফ নির্ধারণ করে।

---

## ২. কেন এটি ক্রিটিক্যাল? (Operational Invariants & Architecture)

`SafetyOrchestrator` পাঁচটি কঠোর সেফটি ইঞ্জিন একসাথে চালায়:
1. **Probability Calibration (Platt / Beta Scaling):** মডেলের অনুমিত সম্ভাবনাকে বাস্তব ফ্রিকোয়েন্সির সাথে মেলায় এবং Expected Calibration Error (ECE) হ্রাস করে।
2. **Decision Curve Analysis (Asymmetric Cost-Loss $p^*$):** ফলস পজিটিভ ($FP$) এবং ফলস নেগেটিভ ($FN$)-এর প্রকৃত ব্যবসায়িক খরচের ওপর ভিত্তি করে গাণিতিকভাবে সর্বনিম্ন খরচের কাট-অফ ($p^*$) বের করে এবং নাইভ ৫০% কাট-অফের তুলনায় মোট কত ডলার সেভ হলো তা পরিমাপ করে।
3. **TreeSHAP Decision Drivers:** ট্রি-মডেলের প্রতিটি সিদ্ধান্তের শীর্ষ ৫টি বৈশ্বিক ও স্থানীয় ফিচার অ্যাট্রিবিউশন ব্যাখ্যা করে।
4. **Mondrian Conformal Risk Control:** ডিপেন্ডেন্ট বা ইন্ডিপেন্ডেন্ট ডেটাতে নির্দিষ্ট $lpha=0.05$ কনফিডেন্স লেভেলে গাণিতিক ৯৫% কভারেজ গ্যারান্টি প্রদান করে।
5. **Helmholtz Energy / Isolation Forest OOD Boundary:** প্রোডাকশনে আসা নতুন কোয়েরিগুলো মডেলের চেনা ট্রেইনিং বাউন্ডারির বাইরে কিনা তা যাচাই করার জন্য একটি অ্যানোমালি কাট-অফ স্কোর নির্ধারণ করে।

---

## ৩. গাণিতিক ভিত্তি (Mathematical Foundations)

### ক. Expected Calibration Error (ECE)
$$	ext{ECE} = \sum_{m=1}^{M} rac{|B_m|}{N} \left| 	ext{acc}(B_m) - 	ext{conf}(B_m) ight|$$

### খ. Asymmetric Cost-Loss Optimization ($p^*$)
$$\mathcal{L}(p) = 	ext{cost}_{FP} \cdot FP(p) + 	ext{cost}_{FN} \cdot FN(p)$$
$$p^* = rg\min_{p \in [0, 1]} \mathcal{L}(p)$$

### গ. কনফর্মাল প্রেডিকশন কভারেজ (Conformal Coverage Guarantee)
$$P(Y_{n+1} \in \mathcal{C}(X_{n+1})) \ge 1 - lpha \quad (	ext{typically } 95\%)$$

---

## ৪. MCP Tool Interface & Input Parameters

```python
async def ml_certify_safety_and_decisions(
    csv_path: str,
    target_column: str,
    cost_fp: float = 5.0,
    cost_fn: float = 250.0,
    model_path: Optional[str] = None,
    view: Literal["compact", "detailed"] = "compact",
) -> Dict[str, Any]
```

---

## ৫. এক্সিকিউটিভ সার্টিফিকেট কার্ড (Receipt Output Example)

```text
========================================================================================
🛡️ ML-MCP PHASE 4: SAFETY, CALIBRATION & DECISION CERTIFICATE
Optimal Operating Cutoff (p*)  : 0.1420 (vs Naive 0.5000)
Net Financial Loss Reduction   : $14,250.00 Saved (68.4% False Positive Drop)
========================================================================================

🎯 1. PROBABILITY CALIBRATION (ECE):
  • Raw Expected Calibration Error (ECE) : 14.80%
  • Calibrated ECE (Platt/Beta Scaled)   : 2.15% (Aligned with physical truth)

💰 2. REAL-WORLD FINANCIAL IMPACT:
  • Naive 0.50 Threshold Total Loss      : $22,450.00 (FP=90, FN=88)
  • Optimal p* Cutoff Total Loss         : $8,200.00 (FP=40, FN=24)
  • Net Dollar Savings                   : $14,250.00

🔍 3. TOP-5 TREESHAP DECISION DRIVERS:
  • #1 transaction_amount   (Mean |SHAP| = 0.4120)
  • #2 debt_to_income_ratio (Mean |SHAP| = 0.2850)
  • #3 outlier_l2_manifold  (Mean |SHAP| = 0.1940)
  • #4 hour_sin             (Mean |SHAP| = 0.1120)
  • #5 account_age_months   (Mean |SHAP| = 0.0890)

📐 4. CONFORMAL RISK GUARANTEES (Mondrian Set):
  • Target Confidence Coverage           : 95.0%
  • Empirical Realized Coverage          : 95.2% (Guaranteed mathematically)
  • Pure Singletons                      : 94.1% of samples uniquely classified

⚡ 5. OOD DETECTOR SAFETY THRESHOLD:
  • Anomaly Boundary Threshold           : -0.420 (Queries exceeding boundary flagged as wild OOD)

========================================================================================
👉 READY FOR PHASE 5: ONNX Microsecond Optimization, FastAPI & Docker!
========================================================================================
```
