# V3-3 — A2-MP hard-negative mining pipeline

> وضعیت اجرا: این فایل مسیر اجرایی V3-3 را مشخص می‌کند. مدل جدیدی دانلود یا
> آموزش داده نمی‌شود؛ فقط checkpoint ثابت A2-MP روی پنجره‌های منفی train
> امتیازدهی می‌شود. اجرای کامل پس از تأیید notebook انجام می‌شود.

## هدف

هدف کاهش false positiveهای مدل فعلی است، بدون دست‌زدن به validation یا استفاده
از time_of_event و time_of_alert به‌عنوان feature مدل.

ورودی mining فقط پنجره‌های sequence manifest V3 است که split=train،
window_role=negative_video و video_label=0 دارند. پس هیچ نمونهٔ validation و
هیچ ویدئوی مثبت وارد mining نمی‌شود.

## اجرای کم‌فشار و قابل‌ادامه

manifest منفی train → decode شانزده فریم هر پنجره با preprocessing دقیق A2-MP
→ frozen ResNet18 embedding → ذخیرهٔ resumable feature cache → mean-max head و
probability → hard-negative selection و contact sheet

- cache ویژگی‌ها بر اساس sequence_id ذخیره می‌شود؛ اگر اجرا قطع شود، فقط
  sequenceهای باقی‌مانده پردازش می‌شوند.
- MP4 یا frameهای V2 بازنویسی نمی‌شوند.
- cache به‌جای ذخیرهٔ JPEGهای موقت، فقط featureهای [16, 512] را نگه می‌دارد؛
  بنابراین فضای کمتری مصرف می‌کند و آموزش HN بعدی سریع‌تر خواهد بود.
- decode و inference به صورت batch انجام می‌شوند، اما به دلیل CPU فعلی، اجرای
  کامل می‌تواند چند ساعت طول بکشد. زمان هر checkpoint در progress ثبت می‌شود.

## قرارداد فنی ثابت

- checkpoint: resnet18_meanmax_pooling_frozen_multipos_best.pt
- window: 5 seconds، 16 uniform timestamps
- preprocessing: RGB، letterbox 224×320، replicated edge padding، ImageNet mean/std
- encoder: ResNet18_Weights.IMAGENET1K_V1، frozen
- head: A2-MP mean + max pooling head

هر تغییری در این قرارداد یک ablation جدا است و نباید در mining فعلی مخلوط شود.

## مراحل اجرایی

### HN-0 — Preflight

- checksum checkpoint و manifest
- شمارش پنجره‌های منفی train
- forward روی دو sequence
- ثبت زمان و حافظه

### HN-1 — Resumable feature cache

- هر sequence کامل فقط وقتی معتبر است که feature shape آن [16, 512] باشد.
- هر ۲۵ batch cache موقت atomically ذخیره می‌شود.
- sequence ناقص یا decode-failed در failure report ثبت می‌شود و بی‌صدا حذف
  نمی‌شود.

### HN-2 — Scoring

features [16, 512] → frozen A2-MP head → positive_probability

خروجی شامل timestamp، path، metadata، decode status و checkpoint signature است.

### HN-3 — Selection

اولویت انتخاب:

1. همهٔ پنجره‌های منفی با p >= 0.60
2. اگر تعداد کافی نبود، top-k پنجره از هر ویدئو

k و threshold در خروجی ثبت می‌شوند. یک ویدئوی منفی نباید با تعداد زیادی پنجرهٔ
تقریباً تکراری غالب شود؛ حداکثر سه نمونهٔ انتخاب‌شده از هر ویدئو در مرحلهٔ HN1
وارد training می‌شود.

### HN-4 — Review و taxonomy

برای پنجره‌های انتخاب‌شده contact sheet و CSV تولید می‌شود. دسته‌بندی دستی:

dense traffic، braking، close vehicle، intersection، camera shake، speed bump،
glare، night، rain، occlusion، lane change، parked vehicle و other.

تا پیش از بازبینی، taxonomy=unreviewed است؛ نام‌گذاری خودکار به‌عنوان حقیقت
ذخیره نمی‌شود.

## خروجی‌ها

- P:\NexarCollisionData\processed_v3\a2mp_hn_train_negative_features_partial.pt
- P:\NexarCollisionData\processed_v3\a2mp_hn_train_negative_features.pt
- P:\NexarCollisionData\predictions_v3\a2mp_train_negative_window_scores_v3.csv
- P:\NexarCollisionData\manifests_v3\hard_negatives_round1.csv
- P:\NexarCollisionData\reports_v3\hard_negative_mining_summary_v3.json
- P:\NexarCollisionData\reports_v3\hard_negative_review_queue_v3.csv

## Gate برای ورود به A2-MP-HN1

- همهٔ scoreها مربوط به train negative باشند.
- checkpoint و preprocessing در report ثبت شده باشند.
- failureها در CSV ثبت شده باشند.
- selected hard negativeها از نظر window/time معتبر باشند.
- validation در mining دیده نشده باشد.

مدل A2-MP-HN1 فقط در مرحلهٔ بعد، با weight پیشنهادی ۱٫۵ برای hard negative و
ارزیابی full-MP4 ثابت، آموزش داده می‌شود.
