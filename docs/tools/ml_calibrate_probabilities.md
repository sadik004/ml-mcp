# ml_calibrate_probabilities: Deep-Dive Architectural Guide & Reference

> **Tool Name:** `ml_calibrate_probabilities`  
> **Module Source:** `src/ml_mcp/engine/calibrator.py` / `src/ml_mcp/tools.py`  
> **Class Implementation:** `ProbabilityCalibrator`  
> **Layer:** AI Safety, Probability Calibration & Reliability Engineering

---

## ১. মানুষের গল্পের মতো পেছনের ইতিহাস (The Human Story: "The Overconfident Liar")

আবহাওয়াবিদ ও ব্যাংকের লোন অফিসারের বাস্তব জীবনের একটি দারুণ উদাহরণ চিন্তা করুন:
- একজন আবহাওয়াবিদ টিভিতে এসে বললেন: *"কাল বৃষ্টি হওয়ার সম্ভাবনা ৮০%।"*
- আপনি যদি সারা বছর এমন দিনগুলোকে পর্যবেক্ষণ করেন যেগুলোতে আবহাওয়াবিদ "৮০% সম্ভাবনা" বলেছিলেন—এবং দেখেন যে বাস্তবে সেই ১০০টি দিনের মধ্যে **ঠিক ৮০টি দিনই বৃষ্টি হয়েছে**, তবে আপনি বলবেন আবহাওয়াবিদ একজন **নিখুঁত ক্যালিব্রেটেড (Well-Calibrated)** মানুষ।
- কিন্তু যদি দেখা যায় যেসব দিনে তিনি ৮০% বৃষ্টি বলেছিলেন, বাস্তবে বৃষ্টি হয়েছে মাত্র ৩০টি দিনে—তবে তিনি একজন চরম **ওভার-কনফিডেন্ট মিথ্যুক (Overconfident Predictor)**!

মেশিন লার্নিংয়ের আধুনিক মডেলগুলো (যেমন: **XGBoost, CatBoost, LightGBM, Random Forest এবং Deep Neural Networks**) ঠিক এই মারাত্মক রোগে আক্রান্ত!
- একটি ব্যাংক ফ্রড ডিটেকশন মডেল হয়তো কোনো লেনদেন দেখে বলল: `Fraud Probability = 0.90` (৯০% নিশ্চিত)।
- কিন্তু বাস্তবে দেখা যায়, যেসব ট্রানজেকশনে মডেল ০.৯০ সম্ভাবনা দিয়েছিল, সেগুলোর মধ্যে প্রকৃত ফ্রডের হার মাত্র **৪০%**!
- মডেলের সফটম্যাক্স (Softmax) বা ট্রি-লিফ ফ্র্যাকশন স্কোর আসলে **True Empirical Probability** নয়। গ্রেডিয়েন্ট বুস্টিং বা ট্রি-মডেলগুলো লস অপটিমাইজ করতে গিয়ে প্রেডিকশনগুলোকে এক্সট্রিম কর্নারে (০ অথবা ১ এর কাছে) ঠেলে দেয়।

**প্রোডাকশন ইঞ্জিনিয়ারদের সংকট:**
আপনার বিজনেসে যদি এমন নিয়ম থাকে: *"যেসব লোন অ্যাপ্লিকেশনের ডিফল্ট রিস্ক ৭০%-এর বেশি, সেগুলো অটোমেটিক রিজেক্ট করো আর বাকিগুলোতে লোন দাও"*—তাহলে আনক্যালিব্রেটেড মডেল ব্যবহার করলে আপনার ব্যাংক হাজার হাজার ভালো কাস্টমার হারাবে এবং ভুল ক্যালকুলেশনে কোটি কোটি টাকার লোকসান গুনবে।

**এই সমস্যার গাণিতিক সমাধান হলো `ml_calibrate_probabilities`:**  
এটি মডেলের কাঁচা বিকৃত স্কোরগুলোকে পোস্ট-প্রসেসিংয়ের মাধ্যমে ফিল্টার করে রিয়েল-ওয়ার্ল্ড ট্রু ফ্রিকোয়েন্সির সাথে নিখুঁতভাবে সারিবদ্ধ (Align) করে দেয়। ফলে মডেল যখন বলবে **৮০% সম্ভাবনা**, তখন বাস্তবেও তার নির্ভুলতার হার হবে **ঠিক ৮০%**!

---

## ২. এটা আসলে কী কাজ করে? (Core Mission)

সহজ কথায়: **এটি মডেলের মুখের ফাঁকা বুলি (Overconfidence) দূর করে সত্যবাদী সম্ভাব্যতা (True Posterior Probability) উপহার দেয়।**

এটি একটি স্বয়ংক্রিয় ৩-ধাপের পাইপলাইন চালায়:
1. **প্রি-ক্যালিব্রেশন এরর অডিট (Baseline Audit):** মূল মডেলের ওপর Out-Of-Fold ক্রস-ভ্যালিডেশন চালিয়ে এর **Brier Score** এবং **Expected Calibration Error (ECE)** পরিমাপ করে দেখে মডেল কতটা মিথ্যে বলছে।
2. **অটোমেটিক মেথড সিলেকশন (Platt Scaling vs Isotonic):** ডেটাসেটের আকার যদি ছোট হয় (< ১০০০ স্যাম্পল), এটি ওভারফিটিং রোধ করতে প্যারামেট্রিক **Platt Scaling (Sigmoid)** বেছে নেয়। আর ডেটাসেট বড় হলে ( $\ge ১০০০$ স্যাম্পল), নন-প্যারামেট্রিক **Isotonic Regression** সক্রিয় করে।
3. **লিক-ফ্রি ক্যালিব্রেটেড মডেল তৈরি (Fitted Pipeline):** `CalibratedClassifierCV` ব্যবহার করে ডেটা লিকেজ ছাড়াই ক্যালিব্রেটেড মডেল তৈরি করে এবং ক্যালিব্রেশনের পর ব্রায়ার স্কোর ও ECE কতটা কমল (`brier_score_lift`, `ece_lift`) তা রিপোর্টে তুলে ধরে।

---

## ৩. নোটবুকের ঠিক কোন কোড সেলের পর এটি কাজ করবে? (Pipeline Placement)

```mermaid
flowchart TD
    C1["Cell 1: ml_benchmark_models (চ্যাম্পিয়ন মডেল নির্বাচন)"] --> C2["Cell 2: ml_tune_hyperparameters (অপটুনা টিউনিং)"]
    C2 --> C3["Cell 3: ml_create_ensemble (স্ট্যাকিং এনসেম্বল)"]
    C3 --> C4["Cell 4: ml_conformal_risk_control (রিস্ক গ্যারান্টি বাউন্ডিং)"]
    C4 --> C5["🎯 Cell 5: [EXACTLY HERE] ml_calibrate_probabilities"]
    C5 --> C6["Cell 6: ml_tune_threshold_and_errors (কাটঅফ থ্রেশহোল্ড টিউনিং)"]
    C6 --> C7["Cell 7: ml_optimize_inference (ONNX / INT8 সার্ভিং)"]
```

### 🎯 সুনির্দিষ্ট নিয়ম:
> **এটি সবসময় মডেল ট্রেনিং/টিউনিংয়ের পরে এবং ডিসিশন থ্রেশহোল্ড টিউনিং (`ml_tune_threshold_and_errors`)-এর ঠিক পূর্বে বসবে।**

### কেন আগে বা পরে বসানো যাবে না? (Engineering Reason)
- **মডেল ট্রেনিংয়ের আগে কেন নয়?** কারণ এটি কোনো নতুন মডেল ট্রেইন করে না, বরং বিদ্যমান প্রি-ট্রেইনড মডেলের আউটপুট প্রবাবিলিটিকে পোস্ট-প্রসেস করে।
- **থ্রেশহোল্ড টিউনিংয়ের পরে কেন নয়?** যদি আপনার প্রবাবিলিটিই বিকৃত বা ভুল থাকে, তবে সেই ভুল প্রবাবিলিটির ওপর ভিত্তি করে কোনো অপটিমাল বিজনেস ডিসিশন থ্রেশহোল্ড ($F_\beta$ Cutoff) বের করা অসম্ভব! আগে সম্ভাব্যতাকে খাঁটি বানাতে হবে, তারপর ডিসিশন থ্রেশহোল্ড কাটতে হবে।

---

## ৪. প্যারামিটার পরিচিতি (The Exact Parameters)

| প্যারামিটার | টাইপ | রিকোয়ার্ড? | ডিফল্ট | বিবরণ |
| :--- | :---: | :---: | :---: | :--- |
| **`csv_path`** | `string` | **হ্যাঁ** | - | ট্রেন্ড ও ভ্যালিডেশন ডেটাসেটের পাথ। |
| **`target_column`** | `string` | **হ্যাঁ** | - | যে টার্গেট কলামের সম্ভাব্যতা ক্যালিব্রেট করতে হবে। |
| **`method`** | `string` বা `null` | না | `null` | ক্যালিব্রেশন মেথড: `"sigmoid"` (Platt Scaling), `"isotonic"`, `"temperature"` (Guo et al. ICML 2017), অথবা `null` (স্বয়ংক্রিয় নির্বাচন)। |
| **`model_name`** | `string` | না | `"lightgbm"` | মডেল অ্যালগরিদম (`"lightgbm"`, `"xgboost"`, `"catboost"`, `"random_forest"`)। |
| **`model_path`** | `string` বা `null` | না | `null` | প্রাক-প্রশিক্ষিত সেভ করা `.joblib` মডেলের পাথ। |
| **`alpha`** | `number` | না | `0.10` | কনফরমাল প্রেডিকশন মার্জিনাল এরর বাজেট ($1 - \alpha = 90\%$ গ্যারান্টিড কভারেজ)। |

---

## ৫. ইঞ্জিন ভেতরের আর্কিটেকচারাল রহস্য (Internal Engine Secrets)

`ProbabilityCalibrator` ইঞ্জিনের আধুনিক মাল্টিক্লাস সিমপ্লেক্স ও কনফরমাল পাইপলাইন:

```mermaid
flowchart TD
    A["Raw Model + Validation Data"] --> B["1. Pre-Audit: Brier Score & Adaptive-Quantile ECE (Roelofs 2022)"]
    B --> C{"Calibration Mode Selection"}
    C -->|method='temperature'| D["TemperatureScaler: ArgMin Cross-Entropy over T > 0"]
    C -->|method='sigmoid' or N < 1000| E["Platt Scaling: CalibratedClassifierCV(method='sigmoid')"]
    C -->|method='isotonic' or N >= 1000| F["Isotonic Regression: Piecewise Step Fit"]
    D --> G["2. Multiclass Simplex Normalization: sum(p_i) == 1.0 (Kull et al. 2019)"]
    E --> G
    F --> G
    G --> H["3. Conformal Prediction Alpha Guard (Angelopoulos 2023)"]
    H --> I["4. Post-Audit: Brier Lift & Reliability Metrics in CalibrationReportDTO"]
```

### থিওরিটিক্যাল ভিত্তি ও আধুনিক গবেষণা (2017–2023 Foundations):

#### ১. Roelofs et al. (NeurIPS 2022) — Adaptive-Quantile Binning ECE
ঐতিহ্যবাহী ফিক্সড ১০-বিন ইকুয়াল-উইডথ ECE ডেটাসেটের আকারের ওপর কৃত্রিম বায়াস তৈরি করে। আধুনিক `calculate_adaptive_ece` প্রতি বিনে সমসংখ্যক স্যাম্পল (Quantile Bins) নিশ্চিত করে ট্রু মিসক্যালিব্রেশন রেট পরিমাপ করে:
$$\text{ECE}_{\text{adaptive}} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

#### ২. Kull, Perello-Nieto, Flach et al. (NeurIPS 2019) — Multiclass Simplex Projection
আইসোটোনিক বা প্ল্যাট স্কেলিং মাল্টিক্লাসে ওয়ান-ভার্সেস-রেস্ট ফর্মে চলে। এর ফলে প্রোবাবিলিটির যোগফল ১.০ না হয়ে ($\sum p_i \ne 1.0$) ভায়োলেশন ঘটে। আমাদের ইঞ্জিন সফটম্যাক্স প্রজেকশন প্রয়োগ করে গ্যারান্টি দেয় যে প্রতিটি রো-এর জন্য:
$$\sum_{k=1}^K p_{ik} = 1.0 \quad \forall i$$

#### ৩. Guo, Pleiss, Sun, & Weinberger (ICML 2017) — Temperature Scaling
মডার্ন নিউরাল নেটওয়ার্ক ও বুস্টিং ট্রি মডেলে কনফিডেন্স ওভারফিটিং কমাতে একটি সিঙ্গেল বাউন্ডেড প্যারামিটার $T > 0$ অপ্টিমাইজ করা হয়:
$$\hat{p}_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

#### ৪. Angelopoulos & Bates (2021–2023) — Conformal Prediction Alpha Guard
ডিস্ট্রিবিউশন-ফ্রি স্প্লিট কনফরমাল প্রেডিকশন প্রয়োগ করে নন-কনফরমিটি কোয়ান্টাইল $\hat{q} = \text{Quantile}_{1-\alpha}(1 - \hat{P}(Y_i \mid X_i))$ নির্ণয় করা হয়, যা টেস্ট স্যাম্পলের ওপর গাণিতিকভাবে $1 - \alpha$ মার্জিনাল কভারেজ প্রদান করে।

---

## ৬. ব্যবহারের প্র্যাকটিক্যাল উদাহরণ (Usage Example via MCP)

### ইনপুট রিকোয়েস্ট:
```json
{
  "csv_path": "data/loan_default_data.csv",
  "target_column": "is_default",
  "method": "temperature",
  "alpha": 0.10
}
```

### আউটপুট রেসপন্স (CalibrationReportDTO):
```json
{
  "method": "temperature",
  "temperature": 1.482,
  "pre_brier_score": 0.2145,
  "post_brier_score": 0.0812,
  "brier_score_lift": 0.1333,
  "is_well_calibrated": true,
  "pre_ece": 0.184,
  "post_ece": 0.042,
  "adaptive_ece": 0.038,
  "ece_lift": 0.142,
  "conformal_alpha": 0.10,
  "conformal_coverage": 0.9125,
  "status": "completed"
}
```

---

### কী আউটপুট পাওয়া গেল?
`ml_calibrate_probabilities` মডেলের অতিরিক্ত ওভারকনফিডেন্ট প্রেডিকশনকে টেম্পারেচার স্কেলিং ($T=1.482$) দিয়ে ট্রু সম্ভাবনায় নামিয়ে এনেছে। এডাপ্টিভ ECE ১৮.৪% থেকে কমে ৩.৮%-এ নেমেছে এবং কনফরমাল কভারেজ ৯১.২৫% নিশ্চিত হয়েছে!
