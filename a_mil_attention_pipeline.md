# V3-4B — Attention MIL pipeline

> وضعیت: پایپ‌لاین آماده‌سازی شده است. اجرای feature extraction یا training آن
> تا پایان `A2-MP-HN1` آغاز نمی‌شود تا دو job سنگین CPU هم‌زمان روی MP4ها کار
> نکنند.

## هدف

مدل A-MIL به‌جای برچسب‌دادن جداگانه به هر کلیپ، یک **bag از پنجره‌های یک MP4**
را با یک برچسب video-level یاد می‌گیرد:

```text
MP4
  -> 5s sliding windows
  -> frozen ResNet18 frame features [window, 16, 512]
  -> clip embedding [window, 512]
  -> gated attention MIL
  -> video logit / probability
```

پس `time_of_event` و `time_of_alert` فقط برای ساخت manifest و تحلیل باقی
می‌مانند؛ نه feature مدل و نه label کلیپ.

## قرارداد داده و split

- split توسعه ثابت است: 480 video train و 120 video validation.
- هیچ window از یک video بین دو split پخش نمی‌شود.
- bag هر video فقط از sequenceهای همان video تشکیل می‌شود.
- validation augmentation تصادفی ندارد.
- training با label سطح video انجام می‌شود؛ `hard_label` پنجره به مدل داده
  نمی‌شود.
- در 5-fold نهایی، bag sampling و hard-negative selection باید داخل outer-train
  همان fold دوباره ساخته شوند.

## sampling سبک و متوازن برای CPU

### Train bags

برای هر video حداکثر 8 پنجره با `seed=42` انتخاب می‌شود. برای جلوگیری از اینکه
bag مثبت شواهد حادثه را از دست بدهد، حداکثر سه `positive_core` در bag مثبت حفظ
می‌شود؛ در bag منفی نیز حداکثر سه hard negative R1 حفظ می‌شود. باقی ظرفیت bag
از پنجره‌های با پوشش یکنواخت همان MP4 پر می‌شود. اگر ویدئو کمتر از 8 پنجره داشته
باشد، همان همهٔ پنجره‌ها استفاده می‌شوند.

این یک تصمیم **sample-construction** است، نه feature یا ورودی مدل: attention-MIL
فقط RGB featureها و video-level label را می‌بیند و timestamp رخداد را نمی‌بیند.
policy و شمار core/hard-negativeها در manifest ذخیره می‌شوند تا در گزارش شفاف
باشد.

### Validation / full MP4

تمام پنجره‌های sliding validation در bag می‌مانند؛ پس معیار نهایی همچنان روی
ورودی MP4 کامل است، نه یک crop نزدیک حادثه.

### hard negatives

negative bagها از همان پنجره‌های کامل ویدئو ساخته می‌شوند. windowهایی که در
`hard_negatives_round1.csv` هستند یک بار در training bag حفظ می‌شوند و هنگام
نیاز جایگزین یک انتخاب یکنواخت می‌گردند؛ اما همچنان فقط label کل ویدئو به MIL
داده می‌شود. تعداد windowهای hard-negative و source آن‌ها در manifest ثبت
می‌شود.

## feature cache قابل reuse

اولویت featureها:

1. cache کامل V3-3 برای train-negative windows.
2. cache A2-MP-HN1 برای positive-core و validationهایی که با A-MIL مشترک‌اند.
3. فقط sequenceهای باقی‌مانده از MP4 decode می‌شوند.

هر feature باید دقیقاً `[16, 512]` و با قرارداد زیر باشد:

- RGB
- letterbox replicated edge تا 224×320
- ImageNet normalization
- ResNet18 `IMAGENET1K_V1` فریز

partial cache با checksum manifest، لیست bagها، checkpoint و preprocessing
version کنترل می‌شود. هیچ دو process نباید هم‌زمان روی یک partial cache بنویسند.

## معماری پیشنهادی

برای هر window:

```text
z_j = mean(frame_features_j)  # [512]
```

سپس attention gated:

```text
h_j = tanh(V z_j) ⊙ sigmoid(U z_j)
a_j = softmax(wᵀ h_j)        # فقط درون همان video bag
z_bag = Σ a_j z_j
logit = FFN(z_bag)
```

تنظیمات شروع:

```text
attention_dim = 128
dropout = 0.30
optimizer = AdamW(lr=1e-3, weight_decay=1e-4)
batch_size = 16 bags
epochs = 40
early stopping = validation PR-AUC
```

## خروجی‌ها و تحلیل

```text
mil_bag_manifest_v3.csv
amil_features_partial.pt
amil_features.pt
amil_frozen_resnet18_best.pt
amil_training_history.csv
amil_validation_video_predictions.csv
amil_attention_weights.csv
amil_attention_contact_sheets/
amil_summary.json
```

برای هر video validation باید probability، prediction، attention weights و
sequence_id/window start/end ذخیره شود. این برای بررسی FP/FN و بررسی تمرکز
اشتباه attention روی glare، dashboard یا overlay ضروری است.

## Gate تصمیم

A-MIL فقط در صورت عبور از موارد زیر وارد shortlist می‌شود:

- F1 حادثه در full-MP4 حداقل برابر A2-MP-HN1 باشد،
- Recall حادثه حداقل 0.85 باشد،
- یا PR-AUC بهتر و FP/FN آن مکمل مدل A2-MP-HN1 باشد.

در غیر این صورت، نتیجه رد نمی‌شود؛ به‌عنوان ablation منفی ثبت می‌شود و مرحلهٔ
D1 motion اولویت بالاتری خواهد داشت.
