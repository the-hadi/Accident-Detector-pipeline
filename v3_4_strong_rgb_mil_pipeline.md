# V3-4 — Strong RGB / MIL pipeline

> وضعیت: این سند مسیر کم‌ریسک و قابل‌ادامهٔ مرحلهٔ V3-4 را مشخص می‌کند. ابتدا
> `A2-MP-HN1` اجرا می‌شود؛ سپس فقط در صورت عبور از gate، attention-MIL روی همان
> feature cache توسعه می‌یابد. هیچ مدل سه‌بعدی یا transformer در این مرحله اجرا
> نمی‌شود.

## مسئله و خروجی معتبر

ورودی نهایی همیشه یک MP4 کامل است. مدل درون آن پنجره‌های 5 ثانیه‌ای با stride
2.5 ثانیه می‌بیند، برای هر پنجره probability می‌سازد و سپس آن‌ها را به یک
video-level probability تجمیع می‌کند. `time_of_event`، `time_of_alert`، مسیر،
شناسه و split هرگز ورودی tensor یا feature مدل نیستند.

## V3-4A — A2-MP-HN1 (اولین اجرای این مرحله)

معماری ثابت و سبک است:

```text
16 RGB frames / 5 s
  -> frozen ImageNet ResNet18
  -> [16, 512] frame features
  -> mean + max pooling
  -> MLP logit
```

تنها تغییر نسبت به A2-MP: نمونه‌سازی V3 و وزن‌دهی hard-negativeها است؛ پس نتیجه
قابل تفسیر خواهد بود.

### مجموعهٔ آموزش

| گروه | انتخاب | برچسب / وزن loss |
|---|---|---|
| positive core | همهٔ پنجره‌های `positive_core` در train | 1 / وزن balance‌شده |
| normal negative | انتخاب seed=42 و round-robin از negative-videoها | 0 / 1.0 |
| hard negative R1 | 240 ردیف `hard_negatives_round1.csv` | 0 / 1.5 |

تعداد normal negative با تعداد positive core برابر است. وزن positive به‌صورت
خودکار محاسبه می‌شود تا مجموع contribution مثبت و منفی در loss برابر بماند؛
بنابراین 240 hard negative، تعادل کلاس را خراب نمی‌کنند. سقف سه hard negative
برای هر ویدئو از V3-3 حفظ می‌شود.

### feature cache قابل ادامه

نیازهای این اجرای A2-MP-HN1 شامل پنجره‌های train منتخب و همهٔ پنجره‌های
validation برای ارزیابی کامل MP4 است. cache زیر به‌ترتیب اولویت پر می‌شود:

1. cache کامل V3-3 همهٔ پنجره‌های منفیِ train را دارد. بنابراین 240 hard
   negative **و** 603 negative عادیِ انتخاب‌شده از آن reuse می‌شوند؛ برای آن‌ها
   decode دوباره انجام نمی‌گیرد.
2. فقط sequenceهای باقیمانده از MP4 decode می‌شوند.
3. بعد از هر 25 batch فایل partial به‌صورت atomic ذخیره می‌شود.
4. cache فقط وقتی قابل استفاده است که checksum manifest، checkpoint و نسخهٔ
   preprocessing دقیقاً یکسان باشند.

بنابراین قطع اجرا حداکثر کار چند batch آخر را تکرار می‌کند. هم‌زمان فقط یک
kernel باید این notebook را اجرا کند؛ اجرای موازی روی یک cache مجاز نیست.

### preprocessing ثابت

- 16 timestamp یکنواخت، endpoint=False
- RGB (نه BGR)
- letterbox با replicated-edge padding تا `224x320`
- ImageNet mean/std
- ResNet18 با `IMAGENET1K_V1` و encoder فریز

این قرارداد با A2-MP و V3-3 یکسان است. cache قدیمی V2 فقط برای sequenceهای
کاملاً یکسان قابل reuse است؛ در عمل پنجره‌های sliding جدید متفاوت‌اند، پس
استفادهٔ نادرست از cache قدیمی ممنوع است.

## آموزش، توقف و انتخاب مدل

- head از checkpoint A2-MP شروع می‌شود؛ encoder همچنان فریز است.
- optimizer: AdamW، فقط head.
- checkpoint هر epoch با **validation PR-AUC video-level** انتخاب می‌شود؛ این
  معیار threshold-free است و از تنظیم مکرر threshold هنگام early stopping
  جلوگیری می‌کند.
- پس از freeze شدن بهترین checkpoint، threshold از predictionهای validation
  برای بیشینه‌کردن Accident F1 با قید Recall >= 0.85 انتخاب می‌شود.
- aggregationهای `max`، `mean`، `top2/3/5_mean`، `noisy_or` و `logsumexp` فقط
  مقایسه و ثبت می‌شوند. learned attention در variant جداگانهٔ A-MIL خواهد بود.

ارزیابی این notebook فقط development split ثابت 480/120 است. در ارزیابی نهایی
5-fold، mining، انتخاب hard negative و آموزش باید درون outer-train همان fold
تکرار شوند؛ فایل R1 فعلی نباید برای outer validation به‌عنوان دادهٔ آموزشی یا
تنظیم استفاده شود.

## خروجی‌های مورد انتظار

```text
P:\NexarCollisionData\manifests_v3\a2mp_hn1_train_windows.csv
P:\NexarCollisionData\processed_v3\a2mp_hn1_features_partial.pt
P:\NexarCollisionData\processed_v3\a2mp_hn1_features.pt
P:\NexarCollisionData\models_v3\a2mp_hn1_frozen_best.pt
P:\NexarCollisionData\models_v3\a2mp_hn1_training_history.csv
P:\NexarCollisionData\predictions_v3\a2mp_hn1_validation_window_predictions.*
P:\NexarCollisionData\predictions_v3\a2mp_hn1_validation_video_predictions.csv
P:\NexarCollisionData\reports_v3\a2mp_hn1_aggregation_ablation.csv
P:\NexarCollisionData\reports_v3\a2mp_hn1_summary.json
```

## Gate برای A-MIL و مدل‌های بعدی

`A2-MP-HN1` تنها در صورت داشتن F1 حداقل baseline full-MP4، Recall >= 0.85،
و افزایش‌ندادن شدید false positiveها، finalist محسوب می‌شود. اگر سود نداشت نیز
نتیجه مهم است: hard-negative weighting ثبت می‌شود و به D1 motion می‌رویم؛
hyperparameter search گسترده انجام نمی‌شود.

برای A-MIL، همین cache تا هرجا sequenceهای مشترک دارد reuse می‌شود؛ فقط
windowهای اضافهٔ موردنیاز bagها decode می‌شوند. A-MIL pipeline جداگانه پیش از
اجرای آن ساخته خواهد شد.
