# D1 — RGB + frame-difference motion pipeline

> وضعیت: طراحی اجرا آماده است. این سند قبل از ساخت notebook و اجرای طولانی تهیه
> شده تا قرارداد داده، مقایسه و cache روشن باشند.

## هدف

مدل‌های RGB-only فعلی صحنه را می‌بینند، اما تغییر بین فریم‌ها را به‌صورت صریح
نمی‌بینند. D1 باید اطلاعات motion سبک را اضافه کند تا مواردی مانند تکان ناگهانی
دوربین، جابه‌جایی سریع خودروها، ضربه، motion blur و تغییر ناگهانی تصویر بهتر
تشخیص داده شوند.

هدف همچنان ثابت است:

```text
ورودی: MP4 کامل
خروجی: احتمال وجود تصادف در کل MP4
```

این مدل پیش‌بینی پیش از تصادف یا تخمین زمان تصادف نیست.

## قرارداد ثابت داده

- V3 development split ثابت 480/120 باقی می‌ماند.
- train windowها دقیقاً همان manifest `a2mp_hn1_train_windows.csv` هستند:
  603 positive core، 603 negative عادی و 240 hard negative R1.
- full-MP4 validation شامل تمام 1768 پنجرهٔ sliding از 120 ویدئو است.
- timestamp رخداد، alert، نام فایل، مسیر، video_id و metadata ورودی مدل نیستند.
- `time_of_event` فقط پیش‌تر برای ساخت windowهای آموزشی استفاده شده و در tensor
  D1 حضور ندارد.

بنابراین در ablation D1 تنها عامل جدید motion است.

## ورودی motion

برای یک پنجرهٔ 16 فریمی RGB:

```text
M_0 = 0
M_t = abs(RGB_t - RGB_(t-1))    برای t=1..15
```

مراحل قطعی پیش از difference:

1. decode timestampهای ثابت 16تایی
2. BGR → RGB
3. letterbox replicated-edge تا 224×320
4. absolute difference روی RGBهای uint8
5. ImageNet normalization برای ارسال به encoder فریز

هیچ augmentation تصادفی در validation اعمال نمی‌شود. در نسخهٔ اول D1، frame
difference خام و Optical Flow استفاده نمی‌شود؛ optical flow در صورت نیاز ablation
بعدی و مستقل خواهد بود.

## معماری D1a: frozen two-stream feature fusion

```text
RGB sequence [16, 3, H, W]     Motion sequence [16, 3, H, W]
          ↓                                  ↓
frozen ResNet18 (ImageNet)            همان frozen ResNet18
          ↓                                  ↓
RGB features [16, 512]              motion features [16, 512]
          ↓                                  ↓
mean + max pooling                  mean + max pooling
          └──────── concatenate [2048] ─────┘
                         ↓
                 LayerNorm + Dropout + Linear
                         ↓
                      window logit
```

دو branch وزن encoder مشترک دارند؛ فقط head fusion آموزش می‌بیند. بنابراین
پارامترهای قابل‌آموزش کم می‌ماند و نتیجه با A2-MP قابل مقایسه است.

## cache و اجرای قابل ادامه

### RGB featureها

از cache کامل `a2mp_hn1_features.pt` reuse می‌شوند؛ RGB دوباره decode یا encode
نمی‌شود.

### motion featureها

برای همان 3214 sequence لازم در A2-MP-HN1 ساخته می‌شوند:

```text
603 positive core train
603 normal negative train
240 hard negative train
1768 validation sliding windows
```

فقط `[16,512]` motion feature ذخیره می‌شود، نه JPEGهای موقت یا float imageهای
بزرگ. فایل partial هر 25 batch به‌صورت atomic ذخیره می‌شود و شامل checksum
manifest، checksum RGB cache و preprocessing version است. اجرای بعدی فقط
sequenceهای باقیمانده را پردازش می‌کند.

هر failure با sequence_id، path، window و دلیل در CSV ثبت می‌شود؛ بی‌صدا حذف
نمی‌شود.

## آموزش و ارزیابی

- optimizer: AdamW، فقط fusion head.
- loss weightها دقیقاً از manifest A2-MP-HN1 خوانده می‌شوند؛ hard negative وزن
  1.5 و positive balance weight حفظ می‌شود.
- checkpoint با PR-AUC video-level و aggregation ثابت `top3_mean` انتخاب می‌شود.
- پس از انتخاب checkpoint، threshold برای بیشینه‌کردن F1 با قید Recall >= 0.85
  فقط از validation development گرفته می‌شود.
- `max`، `mean`، `top2_mean`، `top3_mean` و `top5_mean` به‌عنوان aggregation
  ablation ثبت می‌شوند؛ primary comparison همان `top3_mean` باقی می‌ماند.

## خروجی‌ها

```text
processed_v3/d1_motion_features_partial.pt
processed_v3/d1_motion_features.pt
models_v3/d1_rgb_motion_fusion_frozen_best.pt
models_v3/d1_rgb_motion_fusion_training_history.csv
predictions_v3/d1_validation_window_predictions.csv
predictions_v3/d1_validation_video_predictions.csv
reports_v3/d1_motion_decode_failures.csv
reports_v3/d1_aggregation_ablation.csv
reports_v3/d1_summary.json
```

## تحلیل اجباری

پس از اجرا، D1 باید با A2-MP و A2-MP-HN1 مقایسه شود:

- F1، Recall، Precision، PR-AUC و confusion matrix در سطح MP4
- تغییر FP و FN، به‌ویژه شب، باران، ترافیک و لرزش دوربین
- زمان feature extraction و inference
- montage نمونه‌هایی که D1 درست کرده اما A2-MP اشتباه داشته است، و برعکس

## Gate

D1 فقط وقتی finalist می‌شود که حداقل یکی از موارد زیر برقرار باشد:

- full-MP4 F1 از A2-MP بهتر شود و Recall >= 0.85 باقی بماند؛
- PR-AUC بهتر و خطاهای FP/FN مکمل داشته باشد؛
- Recall بالاتر با افزایش کنترل‌شدهٔ false positive ایجاد کند.

اگر عبور نکند، به‌عنوان ablation ثبت می‌شود. نتیجهٔ آن برای ensemble یا انتخاب
مدل نهایی بدون OOF/CV استفاده نمی‌شود.
