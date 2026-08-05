# C1 — VideoMAE staged pipeline

> وضعیت اجرا: این سند فقط نقشهٔ اجرای C1 است. ایجاد notebook اجرایی، نصب
> کتابخانه، دانلود checkpoint و آموزش، هر کدام پس از تأیید صریح کاربر انجام
> می‌شوند.

## الگوی اجرای کم‌فشار V3

کیفیت و پروتکل ارزیابی نباید برای سبک‌کردن اجرا تغییر کند. سبک‌سازی فقط با
تفکیک کارها انجام می‌شود:

```text
VS Code محلی (CPU)
  → کنترل manifest، checksum، نمونهٔ ورودی و گزارش‌ها
  → آماده‌سازی فایل‌های قابل‌انتقال و اجرای تحلیل نتایج

GPU موقت/remote
  → دانلود checkpoint، استخراج embedding یا آموزش
  → ذخیرهٔ checkpoint و predictionهای window/video

VS Code محلی (CPU)
  → full-MP4 evaluation، calibration، تحلیل FP/FN و گزارش
```

- دادهٔ آموزشی V3 پس از ساخته‌شدن `DS-V3-B` یا `DS-V3-C` منبع اصلی C1 است.
  `sequence_manifest_v2_multipos.csv` فقط baseline/smoke test مقایسه‌ای باقی
  می‌ماند.
- فایل‌های انتقالی باید شامل manifest، split/fold، config، checksum و در صورت
  نیاز cache فریم باشند؛ MP4های خام فقط وقتی منتقل می‌شوند که cache کافی نباشد.
- هر اجرای GPU باید همان seed، policy برچسب، تعداد ۱۶ فریم و full-MP4 sliding
  window پنج‌ثانیه‌ای با stride ۲.۵ ثانیه را ثبت کند. `time_of_event` فقط برای
  ساخت پنجره و ارزیابی است، نه feature مدل.
- اگر GPU در دسترس نباشد، فقط preflight و smoke test بسیار کوچک مجاز است؛
  آموزش کامل روی CPU عمداً شروع نمی‌شود.

## هدف

مدل C1 باید با ورودی یک MP4 کامل، احتمال رخ‌دادن تصادف را برگرداند. time_of_event فقط برای ساخت پنجرهٔ آموزشی و تحلیل خطا استفاده می‌شود و هرگز ورودی مدل نیست.

Checkpoint اولیه:

    MCG-NJU/videomae-base-finetuned-kinetics

این checkpoint در پیکربندی پایه، ۱۶ فریم، ورودی ۲۲۴×۲۲۴ و hidden size برابر ۷۶۸ دارد. جزئیات processor و پیکربندی باید در اجرای preflight مستقیماً از checkpoint ثبت شوند؛ بنابراین preprocessing به حدس یا نرمال‌سازی ResNet وابسته نیست.

## داده و قواعد ثابت

- در V3: `sequence_manifest_v3_sliding.csv` و/یا `bag_manifest_v3_mil.csv` پس
  از تأیید gate مرحلهٔ V3-2.
- برای smoke test یا بازتولید V2: `sequence_manifest_v2_multipos.csv`، شامل
  ۱۴۴۰ sequence متوازن و validation ثابت V2-W2.
- cache ورودی باید در manifest دقیقاً ثبت شود؛ cache فعلی V2 فقط تا پیش از
  ساخته‌شدن cache V3 استفاده می‌شود.
- full-MP4 inference: پنجرهٔ ۵ ثانیه، stride برابر ۲٫۵ ثانیه، ۱۶ فریم یکنواخت.
- split و metadata فعلی تغییر نمی‌کنند.
- مسیر فایل، video_id، time_of_event و time_to_accident ویژگی مدل نیستند.

## مرحلهٔ C1-0 — preflight و smoke test

Notebook: notebooks/22_c1_videomae_preflight.ipynb

1. ثبت Python، PyTorch، CUDA، VRAM، RAM و فضای آزاد.
2. بررسی ۱۵۶۰ sequence و ۲۴٬۹۶۰ فریم cache‌شده.
3. بررسی نصب transformers، accelerate و safetensors.
4. دانلود checkpoint و processor فقط با اجازهٔ صریح در config.
5. اجرای forward بدون gradient روی ۲ sequence و ثبت شکل دقیق pixel_values، شکل embedding، زمان هر sequence و peak VRAM یا RAM.

در CPU فعلی، این مرحله فقط smoke test است؛ اجرای کامل فعال نمی‌شود.

## مرحلهٔ C1-1 — preprocessing سازگار با checkpoint

Processor رسمی همان checkpoint تنها مرجع resize، crop، ترتیب رنگ و normalization است.

- ورودی هر sequence: ۱۶ فریم RGB مرتب‌شده از cache فعلی.
- خروجی processor باید از نظر تعداد فریم، اندازه، dtype و channel order با config مدل assert شود.
- augmentation فقط برای train و فقط به‌صورت زمانی/تصویری سازگار روی کل sequence اعمال می‌شود.
- validation و full-MP4 inference کاملاً deterministic هستند.

در شروع، فریم‌های ۲۲۴×۳۲۰ موجود را در حافظه به processor می‌دهیم؛ cache جدید فقط در صورتی ساخته می‌شود که preflight نشان دهد تبدیل on-the-fly گلوگاه واقعی است.

## مرحلهٔ C1-2 — frozen VideoMAE features

هدف این مرحله جداکردن هزینهٔ encoder از آموزش classifier است.

- encoder: VideoMAEModel یا بخش videomae از مدل classification.
- encoder کاملاً freeze است.
- از embeddingهای token-level حجیم cache نمی‌کنیم.
- فقط بردار pooled هر sequence با شکل [768] ذخیره می‌شود.
- cache به شکل resumable ذخیره می‌شود تا توقف اجرا موجب شروع دوباره نشود.

خروجی‌های پیشنهادی:

    P:\NexarCollisionData\processed_v2\videomae_base_frozen_features_v2_multipos.pt
    P:\NexarCollisionData\processed_v2\videomae_base_frozen_features_v2_multipos_partial.pt

## مرحلهٔ C1-3 — frozen linear head

روی featureهای [768]:

    LayerNorm → Dropout → Linear(768, 1)

- loss: BCEWithLogitsLoss
- کلاس‌ها در سطح sequence متوازن‌اند؛ class weight در baseline استفاده نمی‌شود.
- checkpoint با validation PR-AUC در threshold ثابت ۰٫۵ انتخاب می‌شود.
- threshold فقط پس از پایان آموزش و با معیار F1، تحت قید Recall حادثه، تنظیم می‌شود.

این مرحله روی CPU هم شدنی است؛ اما فقط بعد از موفقیت preflight.

## مرحلهٔ C1-4 — partial fine-tuning

فقط با CUDA GPU انجام می‌شود.

ترتیب امن:

1. classifier head
2. آخرین transformer block + layer norm + classifier
3. در صورت بهبود واقعی، حداکثر دو block آخر

تنظیم اولیه:

    batch_size = 1 یا 2
    gradient_accumulation = 8
    mixed_precision = fp16 یا bf16
    backbone_lr = 1e-5
    head_lr = 1e-4
    early_stopping = validation PR-AUC

در نبود CUDA، این مرحله عمداً block می‌شود تا سیستم چند روز درگیر نشود.

## مرحلهٔ C1-5 — full-MP4 evaluation

همان پروتکل A2:

1. decode پنجره‌های ۵ ثانیه‌ای از MP4 کامل.
2. ۱۶ فریم یکنواخت در هر پنجره.
3. processor رسمی VideoMAE.
4. احتمال هر پنجره.
5. مقایسهٔ max، top2_mean، top3_mean و mean.
6. انتخاب با Accident F1 و قید minimum Recall.

گزارش اجباری:

- Accuracy، precision، recall، F1
- ROC-AUC و PR-AUC
- confusion matrix
- زمان inference برای هر MP4
- تحلیل FP/FN

## مرحلهٔ C1-6 — تصمیم و ensemble

فقط اگر C1 نسبت به A2 یا D1/D2 خطاهای مکمل داشته باشد، وارد E1 می‌شود. ensemble باید روی probabilityهای calibrated و ترجیحاً out-of-fold ساخته شود؛ نه صرفاً با تنظیم وزن روی همان validation ثابت.

## گیت‌های اجرایی

| گیت | شرط عبور |
|---|---|
| Preflight | processor و model بدون خطا، شکل ورودی/خروجی ثبت‌شده |
| CPU smoke test | زمان و مصرف حافظه قابل‌قبول برای دو sequence |
| Frozen full run | CUDA موجود باشد، یا برآورد CPU واقعاً عملی باشد |
| Partial fine-tune | CUDA GPU با VRAM کافی |
| انتخاب نهایی | full-MP4 F1 بهتر، Recall قابل‌قبول، و زمان inference ثبت‌شده |

با سیستم فعلی، C1-0 و فقط head آموزشی پس از آماده‌شدن embeddingهای استخراج‌شده
قابل اجرا هستند؛ استخراج کامل C1-2 و C1-4 بدون GPU نباید آغاز شوند.
