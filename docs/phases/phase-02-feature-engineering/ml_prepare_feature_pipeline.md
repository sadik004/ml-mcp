# ml_prepare_feature_pipeline: Phase 2 Master Feature Engineering Pipeline

> **Tool Name:** `ml_prepare_feature_pipeline`  
> **Module Source:** `src/ml_mcp/engine/feature_orchestrator.py` / `src/ml_mcp/tools.py`  
> **Class Implementation:** `FeaturePipelineOrchestrator`  
> **Layer:** Phase 2: Feature Engineering & Preprocessing (Master Orchestrator)

---

## ১. পেছনের গল্প (The Human Story: "From Raw Columns to Signal Goldmine")

মেশিন লার্নিং মডেল কখনো কোনো জাদুকর নয়; কাঁচা ডেটাতে যদি গাণিতিক প্যাটার্ন বা ফিজিক্যাল ইন্টারঅ্যাকশন লুকানো না থাকে, বিশ্বের সেরা নিউরাল নেটওয়ার্কও আন্ডারফিট করবে।

উদাহরণস্বরূপ:
একটি ই-কমার্স ফ্রড সিস্টেমে `transaction_hour` আছে। কিন্তু সাধারণ ট্রি-মডেল বুঝতে পারে না যে রাত ২৩:৫৯ এবং রাত ০০:০১ আসলে মাত্র ২ মিনিটের ব্যবধান! তারা ধরে নেয় ২৩ এবং ০-এর মাঝে বিশাল গ্যাপ। সাইন ও কোসাইন ট্র্যান্সফর্মেশন দিলে এটি একটি গোল বৃত্তাকার ক্লক ডোমেনে পরিণত হয়।

আবার, একজন ইউজারের ক্রেডিট কার্ড ট্রানজাকশন ভলিউম তার স্বাভাবিক গড়ের চেয়ে ৩ গুণ বড় কিনা—তা ম্যানুয়ালি লিখতে গেলে কোড মেসি হয়ে যায়। 

`ml_prepare_feature_pipeline` হলো ফেজ ২-এর সেই **মাস্টার একক কমান্ড**, যা সম্পূর্ণ অটোমেটেড উপায়ে ফিচার তৈরি করে, মিসিং ভ্যালু সামলায়, স্কেলিং করে, ইমব্যালান্স ক্লাস ব্যালান্স করে এবং বাজে অপ্রয়োজনীয় ফিচার ছেঁটে ফেলে (pruning)।

---

## ২. কেন এটি ক্রিটিক্যাল? (Operational Invariants & Architecture)

`FeaturePipelineOrchestrator` একটি ডিফেন্সিভ, প্রোডাকশন-গ্রেড প্রিপ্রসেসিং পাইপলাইন তৈরি করে:
1. **Automatic Temporal & Cyclical Harmonics:** তারিখ বা সময়ের কলাম সনাক্ত করে সাইন-কোসাইন ক্লকিং ($\sin(2\pi t/T), \cos(2\pi t/T)$), ডে-অব-উইক এবং মাসভিত্তিক পিরিয়ডিক ভেরিয়েবল তৈরি করে।
2. **Latent Manifold Outlier L2-Norm:** আইসোলেশন ফরেস্ট বা লো-র‍্যাঙ্ক মেনিম্যাক্স ম্যানিফোল্ডে ডেটার ডিস্ট্যান্স পরিমাপ করে নতুন অ্যানোমালি-স্কোর ফিচার যোগ করে।
3. **ExploreKit Group-By Aggregations:** ক্যাটাগরিক্যাল কলামের ওপর নিউমেরিক কলামের গড়, স্ট্যান্ডার্ড ডেভিয়েশন ও অনুপাত হিসাব করে ইন্টারঅ্যাকশন ফিচার তৈরি করে।
4. **Defensive Scaling & Imputation:** মিডিয়ান এবং মোড ইম্পিউটার দিয়ে নাল ভ্যালু ডিফেন্সিভলি ভরাট করে, এবং আউটলায়ার থেকে বাঁচতে `RobustScaler` প্রয়োগ করে।
5. **Cost-Sensitive Class Balancing:** কোনো আর্টিফিশিয়াল নয়েজি রো যোগ না করে ক্লাস ওয়েট ম্যাট্রিক্স তৈরি করে ($w_c = rac{N}{K \cdot N_c}$)।
6. **Permutation Importance Feature Pruner:** গ্র্যাডিয়েন্ট পারমুটেশন টেস্ট চালিয়ে যেসব ফিচার মডেলে নয়েজ তৈরি করে তাদের বাদ দেয়।

---

## ৩. গাণিতিক ভিত্তি (Mathematical Foundations)

### ক. সার্কুলার হারমোনিক এনকোডিং (Cyclical Fourier Basis)
$$f_{\sin}(t) = \sin\left(rac{2\pi \cdot t}{T}ight), \quad f_{\cos}(t) = \cos\left(rac{2\pi \cdot t}{T}ight)$$

### খ. মেনিম্যাক্স ম্যানিফোল্ড আউটলায়ার ডিস্ট্যান্স
$$d_{\mathcal{M}}(x) = \min_{z \in \mathcal{M}} \|x - z\|_2$$

### গ. সুষম ইনভার্স ক্লাস ওয়েটিং
$$w_c = rac{N}{2 \cdot N_c} \quad 	ext{for } c \in \{0, 1\}$$

---

## ৪. MCP Tool Interface & Input Parameters

```python
async def ml_prepare_feature_pipeline(
    csv_path: str,
    target_column: str,
    task_type: Literal["classification", "regression"] = "classification",
    enable_synthesis: bool = True,
    enable_pruning: bool = True,
    view: Literal["compact", "detailed"] = "compact",
) -> Dict[str, Any]
```

---

## ৫. এক্সিকিউটিভ সার্টিফিকেট কার্ড (Receipt Output Example)

```text
========================================================================================
✨ ML-MCP PHASE 2: FEATURE ENGINEERING & PREPROCESSING RECEIPT
Original Shape   : (10000, 12)
Transformed Shape: (10000, 24)
Synthesized      : 15 New Rich Domain Signals
Pruned Features  : 3 Zero-Importance / Noise Features Removed
Retained         : 24 High-Impact Feature Dimensions
========================================================================================

⚙️ 1. AUTOMATIC FEATURE SYNTHESIS:
  • Cyclical Harmonics  : hour_sin, hour_cos, day_of_week_sin, day_of_week_cos
  • Financial Ratios    : debt_to_income_ratio, utilization_to_limit_ratio
  • Latent Manifold     : outlier_l2_dist_manifold

⚖️ 2. DEFENSIVE IMPUTATION & SCALING:
  • Missing Values Handled : Median / Mode Imputer Fitted
  • Outlier Resistance     : Scikit-Learn RobustScaler Applied
  • Inverse Class Weights  : {0: 0.53, 1: 8.92}

✂️ 3. NOISE PRUNING:
  • Dropped 3 uninformative noise features to prevent overfitting.

👉 READY FOR PHASE 3: Run ml_run_model_tournament!
========================================================================================
```
