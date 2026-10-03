# ml_preflight_audit: Phase 1 Master Pre-Flight Hygiene Orchestrator

> **Tool Name:** `ml_preflight_audit`  
> **Module Source:** `src/ml_mcp/engine/preflight_auditor.py` / `src/ml_mcp/tools.py`  
> **Class Implementation:** `PreflightAuditor`  
> **Layer:** Phase 1: Data Audit & Hygiene (Master Orchestrator)

---

## ১. পেছনের গল্প (The Human Story: "The 30-Second Pre-Flight Check")

একটি রকেট বা আধুনিক যাত্রীবাহী জেট ওড়ানোর আগে পাইলট কখনো ম্যানুয়ালি এক হাজারটা সুইচ আলাদা করে পরীক্ষা করেন না; বরঞ্চ একটি অটোমেটেড "Pre-Flight Master Built-In Test (BIT)" রান করেন। যদি কোথাও কোনো প্রেসার লিক বা হাইড্রোলিক ফেইলিউর থাকে, রকেট লঞ্চ প্যাডেই লক হয়ে যায়।

মেশিন লার্নিং পাইপলাইনে ফেজ ১-এর আলাদা ৫-৬টি টুল (`ml_audit_dataset`, `ml_detect_target_leakage`, `ml_check_collinearity`, `ml_detect_label_errors`, `ml_verify_constraints`) ম্যানুয়ালি রান করতে করতে জুনিয়র ইঞ্জিনিয়াররা কোনো না কোনো স্টেপ মিস করেন। ফলাফল: লাইভ প্রোডাকশনে ডেটা লিকেজ, করাপ্ট লেবেল কিংবা কন্ডিশন নাম্বার বিস্ফোরণ।

`ml_preflight_audit` হলো সেই **মাস্টার সিঙ্গেল-পাস প্রি-ফ্লাইট চেক**, যা এক ক্লিকে সম্পূর্ণ ডেটা স্যুটের পোস্ট-মর্টেম ও লিগ্যাল স্বাস্থ্য সার্টিফিকেট তৈরি করে।

---

## ২. কেন এটি ক্রিটিক্যাল? (Operational Invariants & Architecture)

`PreflightAuditor` পাঁচটি জটিল ইঞ্জিনকে একই পাইপলাইনে সিঙ্ক্রোনাইজ করে:
1. **Immutable SHA-256 Lineage Hash:** কাঁচা ডেটাসেটের প্রতি বাইটের ক্রিপ্টোগ্রাফিক হ্যাশ তৈরি করে, যাতে মডেল ট্রেইনিংয়ের সময় ডেটা ড্রিফট বা করাপশন তৎক্ষণাৎ ধরা পড়ে।
2. **Missingness & Outlier Profile:** কলামভিত্তিক নাল পার্সেন্টেজ এবং ট্রিমিং বাষ্পীভবন পরিমাপ।
3. **Leakage Shield (Normalized PPS & Cramér's V):** মেজরিটি-ক্লাস স্বাভাবিকীকৃত Predictive Power Score (PPS) এবং ক্র্যামার্স ভি দিয়ে নিখুঁতভাবে ভবিষ্যদ্বাণীমূলক ছদ্ম-ফিচার (যেমন `leak_loan_id` বা `approved_timestamp`) সনাক্ত করে।
4. **Belsley Collinearity Index:** VIF > 10 এবং Condition Number $\kappa > 30$ হলে স্বয়ংক্রিয়ভাবে ফিচার তালিকা লাল দাগ দেয়।
5. **Cleanlab Confident Learning:** লেবেল নয়েজ এবং ভুল গ্রাউন্ড ট্রুথ ফ্ল্যাগ করে।
6. **Domain Constraint Verifier:** নেতিবাচক বয়স, অসম্ভব রেভিনিউ ইত্যাদির মতো ডোমেন ভায়োলেশন বের করে।

---

## ৩. গাণিতিক ভিত্তি (Mathematical Foundations)

### ক. SHA-256 ডেটা লিনিয়েজ
$$\mathcal{H} = 	ext{SHA256}(	ext{bytes}(X \mathbin{\Vert} y))$$

### খ. Normalized Predictive Power Score (PPS)
অসম ক্লাসের ক্ষেত্রে ভুল এলার্ম ঠেকাতে বেসলাইন এক্যুরেসি বাদ দিয়ে নরমালাইজেশন:
$$	ext{PPS}_{	ext{norm}}(X_i 	o y) = \max\left(0, rac{	ext{Acc}(T) - 	ext{Baseline}(y)}{1 - 	ext{Baseline}(y)}ight)$$
যেখানে $	ext{Baseline}(y) = \max_{c} P(y = c)$।

### গ. Belsley Condition Number
কলাম ম্যাট্রিক্স $X$-এর সিঙ্গুলার ভ্যালুর সর্বোচ্চ ও সর্বনিম্ন অনুপাত:
$$\kappa(X) = rac{\sigma_{\max}(X)}{\sigma_{\min}(X)}$$
যদি $\kappa > 30$ হয়, তবে ম্যাট্রিক্সটি মারাত্মক মাল্টিকোলিনিয়ার।

---

## ৪. MCP Tool Interface & Input Parameters

```python
async def ml_preflight_audit(
    csv_path: str,
    target_column: str,
    task_type: Literal["classification", "regression"] = "classification",
    dataset_name: str = "Dataset",
    view: Literal["compact", "detailed"] = "compact",
) -> Dict[str, Any]
```

### প্যারামিটার বিবরণ:
- `csv_path` *(str)*: মূল ট্রেইনিং ডেটাসেটের অ্যাবসোলিউট বা রিলেটিভ ফাইল পাথ।
- `target_column` *(str)*: প্রেডিকশন টার্গেট কলামের নাম।
- `task_type` *(Literal["classification", "regression"])*: কাজের ধরন।
- `dataset_name` *(str)*: সার্টিফিকেটে প্রদর্শনের জন্য ডেটাসেটের নাম।
- `view` *(Literal["compact", "detailed"])*: এলএলএম কনটেক্সট টোকেন বাঁচাতে `"compact"` (ডিফল্ট), অথবা সম্পূর্ণ ডাম্পের জন্য `"detailed"`।

---

## ৫. এক্সিকিউটিভ সার্টিফিকেট কার্ড (Receipt Output Example)

```text
========================================================================================
📋 ML-MCP PHASE 1 PRE-FLIGHT AUDIT CERTIFICATE: Financial_Fraud_Dataset
SHA-256 Lineage: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
Overall Status : ⚠️ ACTION REQUIRED (Leakage Alert: 1, Label Errors: 4)
========================================================================================

🚨 1. TARGET LEAKAGE RADAR:
  • leak_approval_id (PPS: 0.998, Method: pps) -> ❌ ACTION: Drop feature immediately!

⚠️ 2. MULTICOLLINEARITY (VIF / Condition Index):
  • max_vif_feature (VIF: 14.2) -> Consider ridge regularization or PCA reduction.

🧹 3. LABEL QUALITY (Confident Learning):
  • 4 noisy labels detected in target column.

🛡️ 4. DOMAIN CONSTRAINTS:
  • 0 constraint violations detected.

👉 READY FOR PHASE 2: Run ml_prepare_feature_pipeline!
========================================================================================
```
