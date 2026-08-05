# B1/B2 — R3D-18 و R(2+1)D-18 staged pipeline

> وضعیت اجرا: این سند طراحی اجرای مدل‌های سه‌بعدی است، نه اجازهٔ اجرا. ایجاد
> notebook، نصب/دانلود وزن‌ها و هر آموزش فقط بعد از تأیید صریح کاربر انجام
> می‌شود.

## هدف و مرز مسئله

ورودی نهایی یک MP4 کامل است و خروجی فقط احتمال و برچسب «دارای تصادف / بدون
تصادف» خواهد بود. نزدیک‌به‌تصادف، در نبود برچسب معتبر مستقل، کلاس مثبت نیست و
فقط می‌تواند hard negative یا مورد تحلیل خطا باشد.

مدل‌های مورد مقایسه:

| شناسه | مدل | نقش |
|---|---|---|
| B1 | `r3d_18` با وزن Kinetics-400 | baseline سه‌بعدی |
| B2 | `r2plus1d_18` با وزن Kinetics-400 | مدل سه‌بعدی اصلی |

هر دو مدل باید با A2-MP-HN و D1 در سطح MP4 کامل مقایسه شوند؛ نتیجهٔ clip-level
به‌تنهایی برای انتخاب مدل کافی نیست.

## الگوی اجرای کم‌فشار، بدون کاهش پروتکل

```text
سیستم محلی / VS Code (CPU)
  1. تولید و اعتبارسنجی manifestهای V3 و cache فریم
  2. اجرای testهای داده و ساخت بستهٔ انتقالی
  3. دریافت checkpoint/prediction و اجرای گزارش/تحلیل محلی

سیستم دارای CUDA GPU
  4. دانلود وزن pretrained و forward smoke test
  5. head-only training و سپس partial fine-tuning در صورت عبور gate
  6. full-MP4 window inference و ذخیرهٔ predictionها
```

بنابراین روی سیستم فعلی هیچ آموزش 3D شروع نمی‌شود. این کار کیفیت را کم نمی‌کند؛
فقط decode، داده و گزارش از آموزش GPU جدا می‌شوند و هر اجرای پرهزینه resumable
خواهد بود.

## ورودی ثابت و قابل‌بازتولید

- منبع V3 پس از مرحلهٔ V3-2: `sequence_manifest_v3_sliding.csv`؛ MIL در صورت
  نیاز از `bag_manifest_v3_mil.csv` استفاده می‌کند.
- هر window: ۵ ثانیه، ۱۶ فریم RGB با ترتیب زمانی صعودی.
- شکل ورودی مدل: `[B, C, T, H, W]`، یعنی `[B, 3, 16, H, W]`.
- resize، crop و normalization دقیقاً از transforms رسمی وزن torchvision خوانده
  می‌شوند؛ فریم‌ها با ImageNet transform مربوط به ResNet18 دوباره normalize
  نمی‌شوند.
- validation، full-MP4 inference و sampling در foldها deterministic هستند.
- مسیر فایل، `video_id`، `time_of_event`، `time_of_alert`، `time_to_accident` و
  هر ویژگی مشتق‌شده از آن‌ها هرگز وارد tensor مدل نمی‌شوند.

## C0 — بستهٔ انتقالی و preflight محلی

بدون دانلود مدل یا آموزش، موارد زیر بررسی و ثبت می‌شوند:

1. checksum manifest، split/folds و cache فریم؛
2. نبودن overlap ویدئو بین splitها و صعودی‌بودن timestampها؛
3. خواندن درست دو sequence و تبدیل آن‌ها به `[3, 16, H, W]`؛
4. فایل config شامل seed، window policy، label policy و نسخهٔ preprocessing؛
5. فضای دیسک مقصد و فهرست artifactهایی که باید از GPU برگردند.

خروجی‌های پیشنهادی:

```text
problem1_v3/manifests/transfer_manifest_b1_b2.csv
problem1_v3/configs/b1_r3d18.yaml
problem1_v3/configs/b2_r2plus1d18.yaml
problem1_v3/reports/b1_b2_preflight.md
```

## C1 — GPU smoke test

فقط روی GPU، ابتدا یک forward بدون gradient برای دو sequence اجرا می‌شود و ثبت
می‌گردد:

- نسخه‌های Python، PyTorch و torchvision؛ مدل GPU، VRAM، RAM و فضای دیسک؛
- نام دقیق weight و transforms رسمی آن؛
- شکل ورودی/خروجی، زمان هر sequence و peak VRAM؛
- صحت forward با batch size 1.

حداقل عملی: GPU با ۸GB VRAM، batch size 1 و gradient accumulation. پیشنهاد
راحت‌تر: ۱۲GB یا بیشتر. در صورت نرسیدن به حافظه، ابتدا فقط resolution و batch
طبق transform رسمی/مستند weight تنظیم می‌شوند؛ window، تعداد فریم و policy
ارزیابی تغییر نمی‌کنند مگر به‌عنوان ablation مستقل و ثبت‌شده.

## C2 — B1: R3D-18 baseline

ترتیب اجرای کم‌ریسک:

1. freeze کردن backbone و آموزش classifier head؛
2. ارزیابی validation و full-MP4 با پنجرهٔ ۵ ثانیه/stride ۲.۵ ثانیه؛
3. فقط در صورت عبور gate، unfreeze آخرین stage با learning rate کوچک؛
4. early stopping بر اساس PR-AUC و انتخاب threshold فقط روی validation/fold.

تنظیمات شروع:

```text
batch_size = 2 (یا 1 با accumulation)
head_lr = 1e-4
backbone_lr = 1e-5
weight_decay = 1e-4
mixed_precision = fp16 یا bf16
```

## C3 — B2: R(2+1)D-18

همان داده، split، sampling، policy و پروتکل B1 را دارد. تنها متغیر تغییرکرده
معماری B2 است؛ بنابراین مقایسهٔ B1/B2 منصفانه می‌ماند. B2 مدل اصلی 3D است و
نباید پیش از ثبت نتیجهٔ B1 با hyperparameter متفاوت اجرا شود.

## C4 — full-MP4 inference و aggregation

برای هر MP4:

```text
MP4 → windows پنج‌ثانیه‌ای با stride ۲.۵ ثانیه
    → ۱۶ فریم → probability هر window
    → aggregation → probability و label ویدئو
```

aggregationهای `max`، `top2_mean`، `top3_mean`، `top5_mean` و `mean` فقط با
پیش‌بینی‌های validation یا OOF انتخاب می‌شوند. همهٔ probabilityهای window-level
و زمان inference هر MP4 ذخیره خواهند شد.

## Gateهای توقف و پذیرش

| Gate | شرط |
|---|---|
| شروع آموزش | C0 و C1 بدون خطا و دادهٔ V3 معتبر باشد |
| ادامهٔ fine-tune | head-only از مدل تصادفی بهتر و از نظر runtime عملی باشد |
| ورود به ensemble | full-MP4 F1/PR-AUC بهبود داشته باشد یا خطاهای مکمل اثبات شود |
| توقف | بهبود پایدار نسبت به A2-MP-HN ندارد، بسیار کند است، یا FP/FN مکمل ندارد |

مدل منتخب باید Recall تصادف حداقل ۰.۸۵ را پاس کند و معیار اصلی انتخاب آن
Accident F1 در سطح MP4 است. PR-AUC، calibration، زمان inference و تحلیل FP/FN
معیارهای ثانویه‌اند.

## Artifactهای اجباری برای بازگشت از GPU

```text
checkpoint به‌همراه config و seed
predictionهای window-level و video-level
metricهای validation/fold و curveهای threshold
full-MP4 metrics، زمان inference و peak VRAM
فهرست transforms و نسخهٔ weight
گزارش خطاهای FP/FN
```

این artifactها باید در `P:\NexarCollisionData` یا بستهٔ انتقالی معادل ذخیره
شوند تا گزارش، calibration، ensemble و ارزیابی نهایی در VS Code محلی قابل انجام
باشد.

## دستور تأیید برای شروع

برای شروع هر بخش کافی است دقیقاً یکی از این‌ها را تأیید کنید:

```text
«C0 B1/B2 را بساز»       ← فقط preflight و فایل‌های انتقالی، بدون دانلود مدل
«C1 B1/B2 را اجرا کن»    ← فقط smoke test روی GPU آماده
«B1 را پیاده‌سازی کن»    ← notebook اجرایی B1
«B2 را پیاده‌سازی کن»    ← پس از ثبت نتیجهٔ B1
```
