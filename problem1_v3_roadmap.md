# مسئلهٔ ۱ — نقشهٔ راه فاز سوم (V3)

## 0. هدف سند

این سند برنامهٔ اجرایی فاز سوم پروژهٔ تشخیص تصادف را تعریف می‌کند. V3 جایگزین V2 نیست؛ بلکه بر مبنای خروجی‌های واقعی V2، تحلیل خطا، جلوگیری از نشت داده، کاهش اختلاف آموزش و استنتاج، و آزمایش مدل‌های مکمل ساخته می‌شود.

هدف اصلی همچنان ثابت است:

```text
ورودی: یک فایل MP4 کامل
خروجی:
- احتمال وجود تصادف یا نزدیک‌به‌تصادف
- برچسب دودویی:
  0 = بدون تصادف
  1 = دارای تصادف یا نزدیک‌به‌تصادف
```

هشدار پیش از تصادف، پیش‌بینی زمان رخداد و localization زمانی، اهداف تکمیلی هستند و معیار اصلی انتخاب مدل V3 نیستند.

---

# 1. نقطهٔ شروع V3

## 1.1 نتایج مرجع V2

دادهٔ توسعهٔ فعلی:

- ۶۰۰ ویدئو
- ۳۰۰ مثبت
- ۳۰۰ منفی
- ۴۸۰ ویدئوی train
- ۱۲۰ ویدئوی validation
- تقسیم در سطح ویدئو
- ۱۶ فریم RGB از پنجره‌های ۵ ثانیه‌ای
- ارزیابی اصلی در سطح MP4 کامل

بهترین نتیجهٔ فعلی full-MP4:

| مدل | Aggregation | F1 | Recall | Precision | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| A2 اصلی | max | 0.738 | 0.867 | 0.642 | 0.724 | 0.708 |
| A2-MP | top3_mean | 0.738 | 0.867 | 0.642 | 0.740 | 0.723 |

Confusion matrix فعلی:

```text
TP = 52
TN = 31
FP = 29
FN = 8
```

نتیجهٔ پایه برای V3:

- مدل منتخب موقت: `A2-MP`
- معیار پایهٔ مقایسه: full-MP4 Accident F1 = 0.738
- Recall پایه: 0.867
- PR-AUC پایه: 0.723
- مشکل اصلی: false positive زیاد
- مشکل دوم: temporal localization محدود
- محدودیت علمی: انتخاب مدل، aggregation و threshold روی همان validation ثابت انجام شده است

## 1.2 سؤال‌های اصلی V3

V3 باید پاسخ دهد:

1. آیا مشکل اصلی با اصلاح داده و hard-negative mining بهتر حل می‌شود یا با مدل پیچیده‌تر؟
2. آیا مدل‌های motion-aware خطاهای RGB-only را کاهش می‌دهند؟
3. آیا metadata اطلاعات مکمل واقعی می‌دهد یا shortcut ایجاد می‌کند؟
4. آیا مدل‌های temporal مانند BiLSTM نسبت به pooling/GRU سود معنادار دارند؟
5. آیا مدل‌های 3D و transformer ویدئویی نسبت به هزینهٔ اجرا بهبود واقعی می‌دهند؟
6. آیا ensemble مدل‌های دارای خطاهای مکمل، عملکرد full-MP4 را بهتر می‌کند؟
7. آیا نتیجهٔ نهایی روی cross-validation نیز پایدار است؟

---

# 2. قواعد غیرقابل‌تغییر V3

## 2.1 ویژگی‌های ممنوع

متغیرهای زیر هرگز ورودی مدل نیستند:

```text
time_of_event
time_of_alert
time_to_accident
video_id
video_path
file_name
split
label
هر ویژگی مشتق‌شده‌ای که برچسب را لو دهد
```

استفادهٔ مجاز از متغیرهای زمانی:

| متغیر | استفادهٔ مجاز |
|---|---|
| time_of_event | ساخت برچسب و پنجره‌های آموزشی، تحلیل خطا، localization تشخیصی |
| time_of_alert | تحلیل پژوهشی و تعریف آزمایش early-risk؛ نه ورودی مدل |
| time_to_accident | تحلیل Kaggle-style یا test protocol؛ نه ورودی مدل |

## 2.2 تقسیم داده

- split قبل از ساخت پنجره و augmentation انجام می‌شود.
- تمام فریم‌ها، کلیپ‌ها و پنجره‌های یک ویدئو فقط در یک split قرار می‌گیرند.
- duplicate و near-duplicateها باید در یک split باقی بمانند.
- در صورت وجود `trip_id`، `driver_id`، `camera_id`، `route_id` یا شناسه مشابه، split باید group-aware باشد.
- validation هیچ augmentation تصادفی ندارد.
- threshold فقط با predictionهای validation تنظیم می‌شود.
- test نهایی یا cross-validation نباید برای tuning روزمره مصرف شود.

## 2.3 تعریف خروجی

هر مدل، حتی اگر در سطح کلیپ آموزش ببیند، باید در نهایت خروجی video-level تولید کند:

```text
MP4
  → مجموعه پنجره‌ها
  → probability هر پنجره
  → aggregation
  → video probability
  → threshold
  → binary label
```

## 2.4 اصل مقایسهٔ منصفانه

برای هر ablation فقط یک عامل تغییر کند:

- داده ثابت، مدل تغییر کند؛ یا
- مدل ثابت، sampling تغییر کند؛ یا
- مدل و داده ثابت، aggregation تغییر کند.

تغییر هم‌زمان چند عامل، نتیجه را غیرقابل‌تفسیر می‌کند.

---

# 3. استفاده از Kaggle و تفاوت آن با مسئلهٔ کوئرا

## 3.1 هدف اصلی رقابت Kaggle

رقابت اصلی Nexar برای پیش‌بینی زودهنگام خطر طراحی شده است. ارزیابی آن بر احتمال تصادف در فاصله‌های زمانی پیش از رخداد، از جمله حدود ۵۰۰، ۱۰۰۰ و ۱۵۰۰ میلی‌ثانیه قبل از حادثه تأکید دارد.

در مقابل، هدف فعلی پروژهٔ کوئرا ساده‌تر است:

```text
آیا کل MP4 شامل تصادف یا نزدیک‌به‌تصادف است؟
```

بنابراین:

- معماری‌ها، sampling، transfer learning، sliding window و ensemble از Kaggle قابل اقتباس‌اند.
- تعریف label و معیار Kaggle نباید بدون تغییر وارد پروژهٔ کوئرا شود.
- در مدل اصلی کوئرا دیدن لحظهٔ تصادف مجاز است.
- در آزمایش تکمیلی early-risk دیدن لحظه و پس از رخداد ممنوع است.

## 3.2 نکات قابل‌انتقال از راهکار VideoMAEv2 منتشرشده

یک راهکار منتشرشده برای Nexar با VideoMAEv2-giant گزارش کرده است:

- استفاده از ۱۶ فریم
- sliding window
- stride کوتاه
- مثبت‌کردن پنجره‌هایی که انتهایشان نزدیک رخداد است
- undersampling نمونه‌های منفی
- temperature scaling
- استفاده از backbone ویدئویی ازپیش‌آموزش‌دیده

این ایده‌ها در V3 به شکل تعدیل‌شده استفاده می‌شوند:

1. sliding-window sampling برای کاهش mismatch آموزش و inference
2. multi-positive windows برای هر ویدئوی مثبت
3. negative window sampling کنترل‌شده
4. hard-negative mining
5. استفاده از model-specific preprocessing
6. calibration پس از آموزش
7. مقایسهٔ model familyهای مکمل

اما `VideoMAEv2-giant` برای سخت‌افزار فعلی بسیار سنگین است. در V3 ابتدا VideoMAE-base موجود یا نسخه کوچک‌تر عملی آزمایش می‌شود.

## 3.3 الگوی BADAS-Open

مدل BADAS-Open از یک foundation model زمانی، attentive aggregation و MLP head استفاده می‌کند و بر تهدید ego-centric تمرکز دارد. این مدل نشان می‌دهد که:

- نمایش زمانی قوی می‌تواند مفید باشد.
- attentive aggregation ارزش آزمایش دارد.
- near-missها باید به‌عنوان مثبت حفظ شوند.
- false positive ناشی از رخدادهای نامرتبط یک مسئلهٔ مهم است.

V3 قرار نیست BADAS را از ابتدا بازتولید کند، اما یک آزمایش اختیاری transfer/evaluation برای BADAS-Open یا feature extraction از آن می‌تواند پس از تکمیل مسیر اصلی انجام شود.

---

# 4. ساختار کلی V3

V3 به ۱۲ مرحله تقسیم می‌شود:

| مرحله | عنوان | اولویت |
|---|---|---|
| V3-0 | Freeze و Audit مبنا | اجباری |
| V3-1 | Cross-validation و پروتکل ارزیابی | اجباری |
| V3-2 | بازطراحی dataset و window sampling | اجباری |
| V3-3 | Hard-negative mining | اجباری |
| V3-4 | Baselineهای قوی و aggregation | اجباری |
| V3-5 | A6: BiLSTM + additive attention | پیشنهادی |
| V3-6 | D1: motion / frame difference | اجباری |
| V3-7 | D2: metadata fusion | پیشنهادی |
| V3-8 | B1/B2: مدل‌های 3D | در صورت GPU |
| V3-9 | C1: VideoMAE | در صورت GPU |
| V3-10 | E1: ensemble و calibration | پس از مدل‌های مکمل |
| V3-11 | ارزیابی نهایی، گزارش و بستهٔ inference | اجباری |

---

# 5. مرحله V3-0 — Freeze و Audit مبنا

## هدف

ایجاد یک نقطهٔ شروع کاملاً قابل‌بازتولید قبل از هر تغییر.

## اقدامات

### 5.1 تثبیت artifactهای V2

موارد زیر immutable شوند:

```text
video_manifest_v2.csv
metadata_split_v1.csv
sequence_manifest_v2_multipos.csv
frame_cache_index_v2_multipos.csv
A2 checkpoint
A2-MP checkpoint
full-MP4 validation predictions
threshold curve
calibration parameters
error analysis
```

برای هر فایل ثبت شود:

```text
path
size
SHA-256
created_at
source_notebook
git_commit
```

### 5.2 ساخت registry آزمایش‌ها

فایل:

```text
experiments_v3_registry.csv
```

ستون‌ها:

```text
run_id
stage
model_id
dataset_version
split_version
window_version
feature_version
augmentation_version
checkpoint_path
config_path
git_commit
status
primary_metric
primary_value
notes
```

### 5.3 بازتولید A2-MP

A2-MP یک‌بار از صفر با همان seed و config اجرا شود.

شرط عبور:

- اختلاف F1 حداکثر 0.01
- اختلاف PR-AUC حداکثر 0.01
- confusion matrix قابل توضیح
- checksum split و manifest یکسان

### 5.4 Audit نشت داده

برای تمام featureها و manifestها:

- جست‌وجوی ستون‌های ممنوع
- بررسی featureهای مشتق‌شده از timestamp رخداد
- بررسی نام فایل و مسیر
- بررسی overlap ویدئوها
- بررسی duplicate و near-duplicate
- بررسی preprocessing cache برای مخلوط‌شدن splitها

### 5.5 Audit توزیع

مقایسهٔ train و validation برای:

- duration
- FPS
- resolution
- codec
- weather
- light_conditions
- scene
- موقعیت نسبی رخداد
- فاصلهٔ alert تا event
- تعداد پنجره تولیدشده برای هر ویدئو

## خروجی‌ها

```text
v3_baseline_reproduction_report.md
v3_leakage_audit.csv
v3_distribution_audit.csv
experiments_v3_registry.csv
artifacts_v2_checksums.csv
```

## Gate

V3-1 فقط وقتی آغاز شود که baseline بازتولید شود و leakage آشکار وجود نداشته باشد.

---

# 6. مرحله V3-1 — پروتکل ارزیابی و Cross-validation

## هدف

جلوگیری از بیش‌برازش روی validation ثابت ۱۲۰تایی.

## 6.1 دو سطح ارزیابی

### Development split

برای آزمایش سریع:

```text
train = 480 videos
validation = 120 videos
```

### Final model comparison

برای ۲ تا ۴ مدل برتر:

```text
Stratified Group K-Fold
K = 5
```

اگر group identifier وجود ندارد:

```text
Stratified K-Fold در سطح video_id
```

اما duplicate groupها همچنان باید گروه‌بندی شوند.

## 6.2 معیارها

معیار اصلی:

```text
Accident F1 در سطح MP4
```

قید ایمنی:

```text
Accident Recall >= حداقل از پیش تعیین‌شده
```

پیشنهاد اولیه:

```text
minimum_recall = 0.85
```

معیارهای ثانویه:

- PR-AUC
- ROC-AUC
- Precision
- Accuracy
- specificity
- balanced accuracy
- Brier score
- ECE
- inference time per MP4

## 6.3 Confidence interval

برای مدل‌های نهایی:

- bootstrap 95% CI برای F1
- bootstrap 95% CI برای Recall
- bootstrap 95% CI برای PR-AUC
- paired bootstrap برای اختلاف دو مدل
- McNemar test برای اختلاف برچسب‌های نهایی

## 6.4 انتخاب threshold

برای هر fold:

1. checkpoint با PR-AUC یا validation loss در threshold مستقل انتخاب شود.
2. پس از پایان training، threshold روی validation fold تنظیم شود.
3. predictionهای out-of-fold ذخیره شوند.
4. threshold نهایی از OOF predictionها تعیین شود.

## خروجی‌ها

```text
cv_folds_v3.csv
evaluation_protocol_v3.md
metric_functions_test.ipynb
bootstrap_results_v3.csv
```

---

# 7. مرحله V3-2 — بازطراحی Window Sampling

## هدف

کاهش اختلاف میان event-centered training و sliding-window inference.

## 7.1 خانوادهٔ datasetهای V3

سه dataset variant ساخته شود.

### DS-V3-A — مرجع V2

همان multi-positive فعلی برای مقایسه.

### DS-V3-B — Sliding-window aligned

برای هر MP4 آموزشی، پنجره‌ها مشابه inference ساخته شوند:

```text
window_length = 5s
stride = 2.5s
```

برای ویدئوی مثبت، پنجره‌ها به سه گروه تقسیم شوند:

#### Positive-core

پنجره‌ای که رخداد را شامل می‌شود یا انتهای آن بسیار نزدیک رخداد است.

برای مسئلهٔ binary detection:

```text
event داخل پنجره
یا
window_end در بازه [event - 1.5s, event + 1.0s]
```

این محدوده باید با ablation تأیید شود.

#### Positive-context

پنجره‌های قبل از رخداد که احتمالاً نشانهٔ خطر دارند:

```text
window_end بین time_of_alert و time_of_event
```

این پنجره‌ها نباید بدون بررسی مستقیم مثبت قطعی فرض شوند.

دو سیاست مقایسه شود:

- label سخت مثبت
- soft label بر اساس فاصله تا رخداد

نمونه soft label:

```text
event-containing = 1.0
alert-to-event = 0.7 تا 1.0
قبل از alert = 0.0 یا ignore
```

#### In-video negatives

پنجره‌های همان ویدئوی مثبت که از رخداد فاصله دارند.

این‌ها برای کاهش shortcut بسیار مهم‌اند؛ اما ممکن است حاوی مقدمهٔ خطر یا aftermath باشند.

قواعد پیشنهادی:

```text
window_end < time_of_alert - margin
```

و برای جلوگیری از aftermath:

```text
window_start > time_of_event + 3s
```

پنجره‌های مبهم در ابتدا `ignore` شوند.

### DS-V3-C — MIL video bags

هر MP4 یک bag از پنجره‌ها است:

```text
bag = [clip_1, clip_2, ..., clip_n]
video_label = 0/1
```

در این variant نیاز نیست به تک‌تک کلیپ‌های مثبت label قطعی بدهیم.

روش‌های aggregation قابل آزمایش:

- max
- top-k mean
- noisy-or
- gated attention
- log-sum-exp
- transformer/attention over clip embeddings

این variant از نظر علمی با هدف video-level هماهنگ‌تر است.

## 7.2 sampling منفی

برای هر ویدئوی بدون تصادف:

- پنجره‌های یکنواخت از کل ویدئو
- پنجره‌های تصادفی با seed ثابت
- پنجره‌های hard-negative از مدل قبلی
- حفظ تنوع زمانی و محیطی

تعداد پنجره‌های منفی باید بر اساس نسبت مثبت/منفی تعیین شود، نه صرفاً بیشترین تعداد ممکن.

پیشنهاد شروع:

```text
positive windows : random negative windows : hard negatives
1 : 1 : 1
```

سپس ablation:

```text
1 : 2 : 1
1 : 1 : 2
```

## 7.3 جلوگیری از dominance ویدئوهای بلند

اگر یک ویدئو پنجره‌های بیشتری دارد، نباید وزن بیشتری در loss بگیرد.

روش‌ها:

- تعداد ثابت پنجره در هر epoch برای هر ویدئو
- WeightedRandomSampler در سطح video
- loss averaging ابتدا در سطح video و سپس batch
- cap پنجره‌ها

## 7.4 ذخیرهٔ provenance

برای هر sequence:

```text
sequence_id
video_id
split
window_start
window_end
window_length
relative_event_position
distance_to_alert
distance_to_event
label_policy
hard_label
soft_label
sample_weight
is_hard_negative
source_model
sampling_seed
```

ستون‌های زمانی فقط در manifest و analysis هستند و وارد tensor مدل نمی‌شوند.

## خروجی‌ها

```text
sequence_manifest_v3_sliding.csv
bag_manifest_v3_mil.csv
window_label_policy_v3.md
window_distribution_report_v3.csv
```

## Gate

قبل از training:

- visualization حداقل ۵۰ پنجره مثبت
- visualization حداقل ۵۰ hard negative
- بررسی دستی پنجره‌های ambiguous
- بررسی balance در سطح video و clip

---

# 8. مرحله V3-3 — Hard-negative Mining

## هدف

کاهش ۲۹ false positive فعلی و آموزش مدل روی صحنه‌های دشوار.

## 8.1 استخراج hard negative اولیه

A2-MP روی تمام پنجره‌های ویدئوهای منفی train اجرا شود.

برای هر ویدئو ذخیره شود:

```text
top_1 window
top_3 windows
probability
timestamps
scene metadata
```

پنجره‌هایی با شرایط زیر hard negative هستند:

```text
true video label = 0
predicted window probability بالا
```

پیشنهاد threshold اولیه:

```text
p >= 0.6
```

اگر تعداد کم بود، top-k per video استفاده شود.

## 8.2 دسته‌بندی hard negativeها

برای تحلیل دستی حداقل این گروه‌ها ساخته شوند:

- dense traffic
- braking
- close vehicle
- intersection
- camera shake
- speed bump
- glare
- night
- rain
- occlusion
- lane change
- parked vehicle
- irrelevant accident outside ego path

## 8.3 چرخهٔ mining

### Round 1

- استخراج با A2-MP
- آموزش مجدد A2-MP-HN1
- ارزیابی full-MP4

### Round 2

فقط اگر Round 1 سودمند بود:

- استخراج hard negativeهای باقی‌مانده با A2-MP-HN1
- جلوگیری از تکرار نمونه‌ها
- آموزش A2-MP-HN2

حداکثر دو round برای جلوگیری از overfitting.

## 8.4 وزن نمونه‌ها

hard negativeها می‌توانند وزن بیشتر داشته باشند:

```text
normal negative weight = 1.0
hard negative weight = 1.5
```

مقایسه شود با:

- oversampling بدون وزن
- focal loss
- online hard example mining

## 8.5 معیار موفقیت

Hard-negative mining پذیرفته می‌شود اگر:

- FP حداقل ۱۰٪ نسبی کاهش یابد
- Recall کمتر از 0.85 نشود
- F1 یا PR-AUC بهبود یابد
- خطاها فقط به گروه دیگری منتقل نشده باشند

## خروجی‌ها

```text
hard_negatives_round1.csv
hard_negatives_round2.csv
hard_negative_contact_sheets/
hard_negative_analysis.md
```

---

# 9. مرحله V3-4 — Baselineهای قوی و Aggregation

## هدف

قبل از مدل‌های سنگین، بهترین نتیجهٔ ممکن از backbone فعلی گرفته شود.

## مدل‌های این مرحله

### A2-V3

```text
ResNet18
→ frame embeddings
→ mean + max concatenation
→ MLP head
```

### A2-TopK

```text
frame/window logits
→ top-k mean
```

مقادیر:

```text
k = 1, 2, 3, 5
```

### A3-Gated Attention

```text
frame embeddings
→ gated temporal attention
→ video feature
→ classifier
```

### A-MIL

```text
clip embeddings
→ attention MIL
→ video probability
```

## aggregationهای full-MP4

مقایسه شود:

- max
- mean
- top2_mean
- top3_mean
- top5_mean
- noisy-or
- log-sum-exp
- learned attention over windows

## قواعد

- learned aggregation فقط روی train آموزش می‌بیند.
- aggregation parameterها نباید روی همان validation بیش از حد جست‌وجو شوند.
- تمام probabilityهای window-level ذخیره شوند.
- inference speed ثبت شود.

## خروجی‌ها

```text
aggregation_ablation_v3.csv
full_mp4_window_predictions_v3.parquet
mil_attention_visualization/
```

---

# 10. مرحله V3-5 — A6: BiLSTM + Additive Attention + FFN

## هدف

بررسی اینکه مدل زمانی دوطرفه نسبت به GRU و pooling مزیت واقعی دارد یا خیر.

## معماری

```text
Input [B, T, 3, H, W]
→ shared ResNet18 encoder
→ frame features [B, T, 512]
→ 1-layer BiLSTM
→ temporal states [B, T, 2H]
→ additive attention
→ attended video feature
→ FFN
→ binary logit
```

## تنظیمات شروع

```text
T = 16
hidden_size = 128
num_layers = 1
bidirectional = True
dropout_head = 0.3
optimizer = AdamW
encoder_lr = 1e-5
temporal_head_lr = 1e-4
gradient_clip = 1.0
```

## ablation محدود

فقط این موارد مقایسه شوند:

1. hidden size 128 در برابر 256
2. encoder frozen در برابر unfreeze آخرین block
3. additive attention در برابر final-state pooling

از جست‌وجوی وسیع hyperparameter پرهیز شود.

## تحلیل الزامی

- attention weight روی timestampها
- entropy توجه
- مقایسه attention با زمان رخداد
- بررسی اینکه attention روی glare یا dashboard متمرکز نشده باشد
- تحلیل FP/FN نسبت به A2-MP

## شرط پذیرش

A6 فقط زمانی وارد مدل‌های نهایی می‌شود که:

- full-MP4 F1 بهتر شود، یا
- PR-AUC بهتر و خطاها مکمل باشند، یا
- Recall بهتر بدون افزایش شدید FP ایجاد شود.

---

# 11. مرحله V3-6 — D1: Motion / Frame Difference

## هدف

افزودن اطلاعات حرکت بدون هزینهٔ optical flow کامل.

## 11.1 baseline motion

برای فریم‌های متوالی:

```text
diff_t = abs(frame_t - frame_(t-1))
```

گزینه‌های نمایش:

- RGB difference
- grayscale difference
- thresholded motion map
- normalized difference
- stacked differences

## 11.2 معماری‌های D1

### D1-A: Early fusion

```text
RGB + frame difference
→ channel concatenation
→ encoder اصلاح‌شده
```

مزیت: تعامل زودهنگام appearance و motion  
عیب: نیاز به تغییر لایه اول pretrained CNN

### D1-B: Two-stream lightweight

```text
RGB stream: ResNet18
Motion stream: ResNet18 کوچک یا CNN سبک
→ concatenate features
→ temporal pooling/attention
→ classifier
```

### D1-C: Late probability fusion

```text
RGB model probability
Motion-only model probability
→ weighted average / logistic regression
```

برای شروع D1-B یا D1-C ترجیح دارد، چون backbone RGB حفظ می‌شود.

## 11.3 motion-only baseline

قبل از fusion، مدل motion-only ساخته شود تا مشخص شود motion به تنهایی چه چیزی یاد می‌گیرد.

مقایسهٔ اجباری:

```text
RGB only
Motion only
RGB + Motion
```

## 11.4 خطرهای motion

- لرزش دوربین
- دست‌انداز
- motion blur
- تغییر نور
- برف پاک‌کن
- عبور نزدیک اشیا بدون تصادف

این موارد hard negative motion هستند.

## 11.5 ablation زمانی

```text
16 frames / 5s
16 frames / 3s
32 frames / 5s
```

اگر GPU محدود است، ابتدا:

```text
16 frames / 5s
```

## 11.6 optical flow

optical flow کامل فقط اگر frame difference بهبود معنادار نشان داد.

ترتیب:

1. frame difference
2. compressed flow یا TV-L1 روی subset
3. optical flow کامل فقط در صورت توجیه

## خروجی‌ها

```text
motion_cache_index_v3.csv
motion_ablation_v3.csv
motion_error_analysis.md
```

---

# 12. مرحله V3-7 — D2: Metadata Fusion

## هدف

بررسی ارزش اطلاعات محیطی بدون leakage.

## ویژگی‌های مجاز

```text
weather
light_conditions
scene
```

ویژگی‌های ممنوع:

```text
time_of_event
time_of_alert
time_to_accident
video_id
path
```

## آزمایش‌های اجباری

### D2-1 Metadata only

مدل‌های سبک:

- Logistic Regression
- CatBoost یا LightGBM در صورت مجازبودن کتابخانه
- MLP کوچک با categorical embeddings

هدف: تشخیص shortcut.

### D2-2 Video only

مدل منتخب RGB یا RGB+motion.

### D2-3 Video + Metadata

```text
video embedding
+
weather embedding
+
light embedding
+
scene embedding
→ concatenate
→ LayerNorm
→ MLP fusion head
→ logit
```

## unknown category

encoder باید:

```text
UNK
MISSING
```

را پشتیبانی کند.

## Metadata dropout

برای جلوگیری از وابستگی بیش از حد:

```text
metadata_dropout = 0.1 تا 0.3
```

## شرط استفادهٔ نهایی

اگر metadata برای MP4 دلخواه در دسترس نیست:

- مدل اصلی باید video-only باقی بماند.
- D2 فقط مدل benchmark روی دیتاست است.
- خروجی production نباید به metadata دستی وابسته باشد.

## بررسی bias

گزارش عملکرد جداگانه بر اساس:

- شب/روز
- باران/صاف
- highway/urban/suburban
- categoryهای کم‌نمونه

## Gate

اگر metadata-only عملکرد غیرعادی بالا داشت، پیش از fusion باید shortcut یا distribution bias بررسی شود.

---

# 13. مرحله V3-8 — B1 و B2: مدل‌های 3D

این مرحله نیازمند CUDA GPU است.

## B1: R3D-18

```text
torchvision.models.video.r3d_18
pretrained weights
```

## B2: R(2+1)D-18

```text
torchvision.models.video.r2plus1d_18
pretrained weights
```

B2 مدل اصلی 3D پیشنهادی است و B1 baseline مقایسه‌ای.

## ورودی اولیه

بر اساس transform رسمی weight:

```text
[B, C, T, H, W]
T = 16
```

resolution و normalization از خود weights خوانده شود.

## schedule آموزش

### Stage 1

- freeze backbone
- train classifier head

### Stage 2

- unfreeze final stage
- backbone LR کوچک
- mixed precision

### Stage 3

فقط در صورت بهبود:

- unfreeze یک stage بیشتر

## تنظیمات شروع

```text
batch_size = 2 تا 8
gradient_accumulation = در صورت نیاز
head_lr = 1e-4
backbone_lr = 1e-5
weight_decay = 1e-4
early_stopping = PR-AUC
```

## ablation sampling

حداقل:

```text
16 frames from 5s
16 frames from 3s
```

اگر منابع اجازه داد:

```text
32 frames from 5s
```

## شرط توقف

اگر مدل 3D:

- نسبت به A2-MP-HN بهبود ندارد،
- بسیار کندتر است،
- یا خطاهای مکمل ندارد،

وارد ensemble نمی‌شود.

---

# 14. مرحله V3-9 — C1: VideoMAE

## هدف

استفاده از representation زمانی pretrained و مقایسه با CNN/RNN و 3D CNN.

## C1-0 Preflight

موارد ثبت‌شونده:

- Python
- PyTorch
- transformers
- CUDA
- GPU model
- VRAM
- RAM
- disk
- processor config
- model config
- input shape
- output shape
- inference time
- peak memory

## C1-1 انتخاب checkpoint

گزینهٔ فعلی:

```text
MCG-NJU/videomae-base-finetuned-kinetics
```

قبل از اجرا بررسی شود:

- checkpoint واقعاً برای video classification قابل استفاده است
- تعداد labels قابل جایگزینی است
- frame count
- image size
- tubelet size
- mean/std
- sampling assumptions

در صورت دسترسی، یک checkpoint کوچک‌تر نیز مقایسه شود:

```text
VideoMAE-small یا معادل عملی
```

VideoMAEv2-giant منتشرشده برای Kaggle فقط مرجع معماری/روش است و baseline عملی این پروژه نیست.

## C1-2 Frozen embeddings

- encoder freeze
- processor رسمی
- ذخیره فقط pooled embedding
- cache resumable
- ثبت دقیق pooling method

روش pooling باید یکی از این‌ها و صریح باشد:

- CLS token
- mean token pooling
- official pooler output

هر سه نباید بدون ablation مخلوط شوند.

## C1-3 Linear/MLP head

Baseline:

```text
LayerNorm
→ Dropout
→ Linear
```

سپس:

```text
LayerNorm
→ Linear
→ GELU
→ Dropout
→ Linear
```

## C1-4 Partial fine-tuning

فقط GPU:

1. classifier
2. آخرین transformer block
3. حداکثر دو block آخر

تنظیم اولیه:

```text
batch_size = 1 یا 2
gradient_accumulation = 8
mixed_precision = fp16 یا bf16
backbone_lr = 1e-5
head_lr = 1e-4
```

## C1-5 Kaggle-inspired window training

یک variant جداگانه:

- sliding windows
- ۱۶ فریم
- پنجره‌های نزدیک رخداد
- negative undersampling کنترل‌شده

اما برای هدف کوئرا، event-containing windows نیز مثبت هستند.

سه label policy مقایسه شود:

1. event-containing only
2. event + alert-to-event
3. MIL video bag بدون label قطعی کلیپ

## C1-6 full-MP4 inference

```text
5s windows
stride = 2.5s
16 frames
official processor
window probabilities
aggregation
video probability
```

aggregation:

- max
- top2_mean
- top3_mean
- learned attention در صورت امکان

## C1-7 خطا و explainability

- highest-probability windows
- attention rollout یا saliency فقط برای نمونه محدود
- رخداد داخل top window؟
- مقایسه با A2 و D1
- runtime

## C1-8 BADAS-Open اختیاری

پس از C1 اصلی:

- اجرای inference روی subset
- feature extraction در صورت دسترسی
- مقایسه zero-shot/frozen
- عدم ادعای مقایسه منصفانه مگر preprocessing و protocol یکسان باشد

این مرحله optional است و نباید اجرای مدل‌های اصلی را متوقف کند.

---

# 15. مرحله V3-10 — E1: Ensemble و Calibration

## شرط ورود

حداقل دو مدل باید:

- به‌تنهایی قابل قبول باشند
- خطاهای کاملاً یکسان نداشته باشند
- probability قابل استفاده تولید کنند

مدل‌های کاندید:

```text
A2-MP-HN
A6
D1 RGB+motion
D2 video+metadata
B2 R(2+1)D
C1 VideoMAE
```

## 15.1 تحلیل مکمل‌بودن

برای هر جفت مدل:

- correlation probability
- disagreement rate
- مشترک‌بودن FP
- مشترک‌بودن FN
- Jaccard خطاها
- conditional accuracy on disagreements

## 15.2 روش‌های ensemble

### Baseline

```text
mean probability
```

### Weighted average

وزن‌ها با OOF predictionها تعیین شوند.

### Logistic stacking

ورودی:

```text
prob_model_1
prob_model_2
...
```

مدل:

```text
Logistic Regression با regularization
```

### Rank average

اگر calibration مدل‌ها متفاوت باشد.

## 15.3 ممنوعیت

- تنظیم وزن روی همان ۱۲۰ validation و گزارش همان نتیجه به‌عنوان نتیجه نهایی
- استفاده از label test
- ensemble مدل ضعیف صرفاً برای افزایش تعداد
- ترکیب probability خام و calibrated بدون پروتکل مشخص

## 15.4 Calibration

مقایسه:

- temperature scaling
- Platt scaling
- isotonic regression فقط با داده کافی

Calibration باید cross-fitted یا مبتنی بر OOF باشد.

گزارش:

- ECE
- Brier score
- reliability diagram
- threshold خام و calibrated

## معیار پذیرش

ensemble پذیرفته می‌شود اگر:

- OOF F1 بهتر شود
- Recall قید را پاس کند
- bootstrap difference امیدوارکننده باشد
- inference cost قابل دفاع باشد

---

# 16. مرحله V3-11 — ارزیابی نهایی و بستهٔ Inference

## 16.1 انتخاب مدل نهایی

ترتیب معیار:

```text
1. Accident F1
2. Recall constraint
3. PR-AUC
4. پایداری cross-validation
5. calibration
6. inference time
7. سادگی و قابلیت بازتولید
```

## 16.2 گزارش نهایی

برای هر مدل finalist:

- mean ± std در ۵ fold
- 95% CI
- threshold
- confusion matrix OOF
- PR curve
- ROC curve
- calibration curve
- error taxonomy
- inference time
- model size
- peak RAM/VRAM

## 16.3 تحلیل slice

نتیجه جداگانه برای:

- light condition
- weather
- scene
- مدت ویدئو
- موقعیت زمانی رخداد
- فاصله alert تا event
- رخداد کوچک/بزرگ
- glare
- night
- rain
- highway
- near-miss در برابر collision، اگر label تفکیکی قابل اعتماد وجود دارد

## 16.4 inference pipeline نهایی

```text
validate MP4
→ read duration/FPS
→ build sliding windows
→ sample frames
→ model-specific preprocessing
→ batch inference
→ window probabilities
→ aggregate
→ calibrate
→ threshold
→ output
```

خروجی JSON پیشنهادی:

```json
{
  "video_probability": 0.82,
  "calibrated_probability": 0.79,
  "label": 1,
  "threshold": 0.43,
  "confidence": "high",
  "aggregation": "top3_mean",
  "top_window": {
    "start": 14.5,
    "end": 19.5,
    "probability": 0.91
  },
  "windows": []
}
```

## 16.5 localization تکمیلی

برای positive validation:

- فاصله مرکز top window تا `time_of_event`
- Recall within ±1s
- Recall within ±2s
- Recall within ±5s

این معیارها تشخیصی‌اند و معیار اصلی مدل نیستند.

## 16.6 بسته نهایی

```text
configs/
models/
manifests/
notebooks/
src/
reports/
inference/
tests/
requirements.txt
environment.yml
README.md
MODEL_CARD.md
DATA_CARD.md
```

---

# 17. ترتیب دقیق اجرای V3

## مسیر CPU-first

1. V3-0: freeze و audit
2. V3-1: پروتکل ارزیابی و foldها
3. V3-2: sliding-aligned dataset و MIL manifest
4. V3-3: hard-negative mining با A2-MP
5. V3-4: aggregation و MIL baseline
6. V3-6: D1 frame difference
7. V3-7: D2 metadata ablation
8. V3-5: A6 در صورت زمان کافی
9. shortlist مدل‌ها
10. cross-validation مدل‌های سبک
11. E1 سبک
12. گزارش

## مسیر GPU

پس از آماده‌شدن داده و baseline:

1. B1: R3D-18
2. B2: R(2+1)D-18
3. C1-0: VideoMAE preflight
4. C1-2: frozen embeddings
5. C1-3: linear/MLP head
6. C1-4: partial fine-tuning
7. full-MP4 evaluation
8. complementary error analysis
9. E1 ensemble
10. cross-validation finalists

---

# 18. اولویت‌بندی مدل‌ها

| اولویت | مدل | دلیل |
|---:|---|---|
| 1 | A2-MP + hard-negative mining | کم‌هزینه‌ترین اصلاح مستقیم مشکل FP |
| 2 | D1 RGB + frame difference | افزودن منبع اطلاعاتی مکمل |
| 3 | Attention MIL | هماهنگ با label سطح ویدئو |
| 4 | D2 metadata fusion | آزمایش مکمل و کم‌هزینه |
| 5 | A6 BiLSTM + additive attention | تکمیل مقایسه temporal |
| 6 | B2 R(2+1)D-18 | مدل ویدئویی pretrained متناسب |
| 7 | B1 R3D-18 | baseline 3D |
| 8 | C1 VideoMAE | بالقوه قوی، اما پرهزینه |
| 9 | BADAS-Open evaluation | آزمایش اختیاری foundation model |
| 10 | E1 ensemble | فقط پس از اثبات مکمل‌بودن |

---

# 19. آزمایش‌های حداقلی و آزمایش‌های امتیازی

## حداقل لازم

- بازتولید baseline
- leakage audit
- sliding-aligned sampling
- hard-negative mining
- A2-MP-HN
- D1 frame difference
- full-MP4 evaluation
- ۵-fold برای دو مدل برتر
- گزارش خطا

## پیشنهادشده

- MIL attention
- D2 metadata
- A6
- R(2+1)D-18
- VideoMAE frozen + partial fine-tune
- ensemble OOF

## امتیازی/پژوهشی

- soft labels از alert-to-event
- early-warning auxiliary task
- BADAS-Open comparison
- optical flow
- multi-task event localization
- uncertainty-based abstention
- dashboard تحلیل پنجره‌ها
- model interpretability

---

# 20. آزمایش تکمیلی Early-risk بدون مخلوط‌شدن با هدف اصلی

این بخش یک track مستقل است.

## تعریف

```text
ورودی فقط تا پیش از event
خروجی احتمال خطر
```

پنجره‌های مجاز مثبت:

```text
window_end <= time_of_event - 0.5s
```

استفاده از `time_of_alert`:

- ساخت target یا وزن آموزشی
- ارزیابی زمان هشدار
- نه ورودی مدل

معیارها:

- AP at 0.5s
- AP at 1.0s
- AP at 1.5s
- mean time-to-accident
- false positive rate

نتیجه این track نباید با F1 مسئله باینری اصلی مخلوط شود.

---

# 21. تست‌های الزامی کد

## Data tests

- هیچ video_id بین splitها مشترک نیست
- هیچ sequence از یک ویدئو در دو split نیست
- ستون ممنوع وارد feature tensor نشده
- frame order صحیح است
- timestampها صعودی‌اند
- پنجره‌ها از duration خارج نیستند
- label policy درست اجرا شده
- validation deterministic است

## Model tests

- shape ورودی/خروجی
- mask فریم نامعتبر
- gradient فقط برای لایه‌های intended
- frozen layers واقعاً frozen
- probability finite
- checkpoint save/load equality
- batch size 1 inference

## Evaluation tests

- threshold فقط از validation
- metric implementation با sklearn تطبیق دارد
- aggregation deterministic است
- calibration روی داده آموزش‌دیده fit نشده
- OOF هر نمونه دقیقاً یک prediction دارد

---

# 22. ساختار پیشنهادی فایل‌ها

```text
problem1_v3/
├── configs/
│   ├── data_v3.yaml
│   ├── a2_hn.yaml
│   ├── a6_bilstm.yaml
│   ├── d1_motion.yaml
│   ├── d2_metadata.yaml
│   ├── b1_r3d18.yaml
│   ├── b2_r2plus1d18.yaml
│   ├── c1_videomae.yaml
│   └── e1_ensemble.yaml
├── manifests/
│   ├── video_manifest_v3.csv
│   ├── cv_folds_v3.csv
│   ├── sequence_manifest_v3_sliding.csv
│   ├── bag_manifest_v3_mil.csv
│   └── hard_negatives_v3.csv
├── notebooks/
│   ├── 30_v3_baseline_audit.ipynb
│   ├── 31_v3_evaluation_protocol.ipynb
│   ├── 32_v3_window_dataset.ipynb
│   ├── 33_v3_hard_negative_mining.ipynb
│   ├── 34_v3_mil_baselines.ipynb
│   ├── 35_v3_a6_bilstm.ipynb
│   ├── 36_v3_d1_motion.ipynb
│   ├── 37_v3_d2_metadata.ipynb
│   ├── 38_v3_b1_b2_3d_models.ipynb
│   ├── 39_v3_c1_videomae.ipynb
│   ├── 40_v3_ensemble.ipynb
│   └── 41_v3_final_evaluation.ipynb
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   └── utils/
├── models/
├── predictions/
├── reports/
├── figures/
└── tests/
```

---

# 23. جدول تصمیم‌گیری مدل نهایی

| شرط | تصمیم |
|---|---|
| A2-HN بهتر شد و D1 بهتر نشد | A2-HN مدل اصلی |
| D1 F1/PR-AUC بهتر و runtime قابل قبول | D1 مدل اصلی |
| A6 فقط recall را بالا برد ولی FP زیاد شد | ensemble candidate یا رد |
| D2 فقط روی metadata موجود خوب است | benchmark مکمل، نه مدل عمومی |
| B2 بهبود کوچک ولی هزینه بسیار زیاد دارد | فقط برای ensemble یا رد |
| C1 بهبود پایدار در CV دارد | finalist |
| C1 فقط validation ثابت را بهتر کرد | نیازمند CV؛ انتخاب نشود |
| دو مدل خطاهای مکمل دارند | E1 |
| ensemble فقط روی validation بهتر است | رد تا OOF/CV |
| هیچ مدل پیچیده‌ای baseline را پایدار بهتر نکرد | مدل ساده‌تر انتخاب شود |

---

# 24. معیار پایان V3

V3 زمانی کامل است که:

1. leakage audit پاس شده باشد.
2. baseline بازتولید شده باشد.
3. hard-negative mining اجرا شده باشد.
4. حداقل یک مدل motion-aware اجرا شده باشد.
5. مدل‌های درخواستی A6، D1، D2، B1، B2، C1 و E1 طبق محدودیت سخت‌افزار اجرا یا با دلیل مستند block شده باشند.
6. full-MP4 inference برای همه finalistها انجام شده باشد.
7. حداقل دو مدل برتر cross-validation شده باشند.
8. threshold و calibration از OOF یا validation معتبر حاصل شده باشد.
9. گزارش FP/FN و slice analysis آماده باشد.
10. یک pipeline inference قابل اجرای مجدد تحویل داده شود.

---

# 25. اطلاعاتی که پیش از اجرای کامل باید مشخص شود

این موارد مانع نوشتن Roadmap نیستند، اما قبل از اجرای بعضی مراحل باید پاسخ داده شوند:

1. آیا به Google Colab GPU، Kaggle GPU یا GPU محلی دسترسی وجود دارد؟
2. حداکثر VRAM قابل دسترس چقدر است؟
3. آیا امکان استفاده از کل ۱۵۰۰ ویدئوی train وجود دارد یا فعلاً ۶۰۰ نمونه ثابت می‌ماند؟
4. آیا شناسه‌ای مانند trip، driver، route یا camera در داده خام وجود دارد؟
5. حداقل Recall مورد قبول نهایی دقیقاً چند است؟
6. زمان مجاز inference برای هر MP4 چقدر است؟
7. آیا هدف ارائه فقط notebook است یا pipeline اجرایی/CLI نیز لازم است؟
8. آیا مدل‌های gated مانند BADAS-Open قابل دانلود و استفاده هستند؟

تا زمان تعیین این موارد، پیش‌فرض V3:

```text
development set = همان ۶۰۰ ویدئو
minimum recall = 0.85
CPU-first
GPU stages = Colab/Kaggle در صورت دسترسی
offline inference
```

---

# 26. منابع

## منابع پروژه

- `problem1_v2_roadmap.md`
- `phase2_v2_results_analysis.md`
- `c1_videomae_pipeline.md`
- صورت مسئلهٔ پروژهٔ دوم کوئرا

## منابع خارجی

- Kaggle leaderboard:
  https://www.kaggle.com/competitions/nexar-collision-prediction/leaderboard

- Kaggle competition:
  https://www.kaggle.com/competitions/nexar-collision-prediction/

- Nexar dataset:
  https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction

- Dataset and challenge paper:
  https://arxiv.org/abs/2503.03848

- VideoMAEv2 Nexar solution model card:
  https://huggingface.co/zhiyaowang/VideoMaev2-giant-nexar-solution

- BADAS-Open:
  https://huggingface.co/nexar-ai/BADAS-Open

---

# 27. خلاصهٔ اجرایی

اولین کار V3 نباید اجرای VideoMAE یا مدل 3D باشد. ترتیب صحیح:

```text
Audit
→ Cross-validation protocol
→ Sliding-aligned dataset
→ Hard-negative mining
→ Strong RGB/MIL baseline
→ Motion
→ Metadata/BiLSTM
→ 3D models
→ VideoMAE
→ Ensemble
→ Final CV and inference package
```

مهم‌ترین فرضیهٔ V3 این است که بهبود کیفیت نمونه‌سازی و hard negatives می‌تواند پیش از افزایش پیچیدگی مدل، false positiveها را کاهش دهد. مدل‌های سنگین فقط زمانی ارزش دارند که روی full-MP4 و ارزیابی group-safe، بهبود پایدار یا خطاهای مکمل نشان دهند.
