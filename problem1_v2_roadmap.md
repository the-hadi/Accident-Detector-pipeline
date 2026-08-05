# مسئلهٔ ۱ — نقشهٔ راه نسخهٔ دوم (V2)

این فایل مرجع مراحل بهبود مدل تشخیص تصادف است. V1 را حذف نمی‌کنیم؛ V1 baseline قابل‌مقایسهٔ ماست و V2 باید با همان split ثابت ارزیابی شود.

## وضعیت فعلی V1

- ۶۰۰ ویدئوی train متوازن: ۳۰۰ مثبت و ۳۰۰ منفی.
- split ثابت در سطح ویدئو: ۴۸۰ train و ۱۲۰ validation.
- ۸ فریم یکنواخت از کل هر ویدئوی حدوداً ۴۰ ثانیه‌ای استخراج شده است.
- baseline: ResNet18 fine-tune‌شده با میانگین‌گیری احتمال فریم‌ها.
- بهترین F1 validation با آستانهٔ ۰.۳۵: حدود ۰.۷۳۰؛ accuracy حدود ۰.۶۴۲.
- ResNet18 + GRU روی ۸ فریم پراکنده بهتر از baseline نبود.

محدودیت اصلی V1: ۸ فریم از کل ویدئو ممکن است صحنه یا نشانه‌های نزدیک به تصادف را نبینند. همچنین برچسب‌دادن جداگانه به فریم‌ها می‌تواند label noise ایجاد کند.

## تعریف رسمی مسئله، V1 و V2

**هدف اصلی پروژه (فرض تثبیت‌شده):** تشخیص آفلاین اینکه آیا یک MP4 شامل تصادف/نزدیک‌به‌تصادف است یا نه.

```text
input: MP4 کامل
output: probability of accident + binary label
```

پیش‌بینی زمان تقریبی رخداد یک خروجی تکمیلیِ ارزشمند است، اما معیار اصلی انتخاب مدل نیست. «هشدار پیش از تصادف» یک مسئلهٔ جداگانه است و فقط در توسعهٔ بعدی بررسی می‌شود؛ در آن حالت مدل حق ندارد فریم‌های بعد از رخداد را ببیند.

این تصمیم مستقیماً با صورت مسئله سازگار است: برچسب از وجود `time_of_event` ساخته می‌شود و خروجی نهایی برای کل MP4، `0 = بدون تصادف` یا `1 = دارای تصادف` است.

تعریف‌ها:

- **V1:** baseline فریم‌محور. ۸ فریم پراکنده از کل ویدئو، آموزش در سطح فریم، و میانگین‌گیری احتمال‌ها برای تصمیم ویدئویی.
- **V2:** pipeline دنباله‌محور و event-centered. یک پنجرهٔ زمانی کوتاه از هر ویدئو، ۱۶ فریم مرتب، یک label برای کل sequence، و مدل video-level.
- Window variantها فقط تنظیم پنجره درون V2 هستند، نه نسخه‌های مستقل سیستم:

| نام | پنجرهٔ مثبت |
|---|---|
| V2-W1 | [-4s, +1s] |
| V2-W2 | [-3s, +2s] |
| V2-W3 | [-2s, +3s] |

## اصل‌های غیرقابل‌تغییر

1. برچسب در سطح ویدئو/sequence تعریف می‌شود:

   ```python
   label = 0 if pd.isna(time_of_event) else 1
   ```

2. مسیر فایل، نام فایل، شمارهٔ ویدئو، `time_of_event` و `time_to_accident` نباید ویژگی ورودی مدل باشند.
3. split باید پیش از انتخاب پنجره‌ها و augmentation ثابت شود.
4. تمام فریم‌های یک ویدئو فقط در یک split قرار می‌گیرند.
5. validation بدون augmentation تصادفی و با timestampهای ثابت اجرا می‌شود.
6. هیچ ویدئوی خراب یا نمونهٔ حذف‌شده‌ای نباید بی‌صدا حذف شود؛ دلیل آن در manifest ذخیره می‌شود.

## مرحلهٔ ۱ — ساخت manifest مرکزی و Data Audit

یک DataFrame مرکزی بسازیم که هر سطر آن یک ویدئو است. نام پیشنهادی:

```text
video_manifest_v2.csv
```

ستون‌های لازم:

```text
video_id
video_path
label
time_of_event
duration
fps
frame_count
width
height
weather
light_conditions
scene
split
is_valid
error_reason
```

برای تمام ۶۰۰ ویدئو بررسی شود:

- آیا MP4 باز می‌شود و حداقل یک فریم دارد؟
- آیا FPS، resolution و duration معتبرند؟
- آیا `time_of_event` برای نمونهٔ مثبت در بازهٔ واقعی ویدئو قرار دارد؟
- آیا ویدئو بسیار کوتاه، سیاه یا خراب است؟
- آیا duration، FPS، resolution یا codec بین کلاس‌ها تفاوت مشکوک دارد؟
- آیا فایل‌های تکراری یا تقریباً مشابه وجود دارند؟

روش عملی duplicate detection:

- **exact duplicate:** هش SHA-256 فایل MP4.
- **near duplicate:** perceptual hash از ۳ تا ۵ فریم یکنواخت هر ویدئو و مقایسهٔ فاصلهٔ Hamming.
- **اختیاری:** شباهت embedding فریم‌ها با encoder ازپیش‌آموزش‌دیده.

هر گروه مشابه باید در یک split باقی بماند؛ threshold تشابه و اعضای گروه در manifest ثبت می‌شوند.

**معیار پایان:** فایل manifest کامل است و تعداد و علت نمونه‌های نامعتبر گزارش شده است.

### نتیجهٔ بررسی اولیهٔ ۶۰۰ ویدئوی منتخب

بررسی انجام‌شده روی دادهٔ فعلی:

- ۶۰۰ از ۶۰۰ ویدئو باز شدند و FPS/frame count معتبر داشتند.
- ۳۰۰ نمونهٔ مثبت همگی `time_of_event` غیرتهی دارند.
- هر ۳۰۰ زمان رخداد در بازهٔ duration واقعی MP4 قرار دارند.
- `time_of_event` برحسب ثانیه از ابتدای ویدئو ثبت شده و مقادیر اعشاری دارد؛ بازهٔ مشاهده‌شده ۳.۹۲۱ تا ۵۶.۸ ثانیه است.
- duration ویدئوها ۱۵ تا ۶۰ ثانیه و FPSها تقریباً ۲۳.۶ تا ۳۱ هستند؛ بنابراین sampling باید برحسب timestamp انجام شود، نه frame index ثابت.

ستون schema تنها یک `time_of_event` برای هر نمونهٔ مثبت دارد؛ V2 فعلاً فرض «یک رخداد برچسب‌خورده در هر ویدئو» را استفاده می‌کند.

## مرحلهٔ ۲ — تثبیت و بررسی split

همان split فعلی حفظ شود:

```text
train: 480 videos (240 positive, 240 negative)
validation: 120 videos (60 positive, 60 negative)
```

فایل پیشنهادی:

```text
metadata_split_v1.csv
```

بررسی‌های لازم:

- video_id بین train و validation اشتراک ندارد.
- اگر duplicate/perceptual duplicate وجود دارد، هر گروه فقط در یک split است.
- در صورت وجود trip_id/drive_id/driver_id باید از GroupShuffleSplit استفاده شود.

**معیار پایان:** split در تمام آزمایش‌های V2 ثابت و قابل‌بازتولید است.

## مرحلهٔ ۳ — ساخت پنجرهٔ زمانی V2

### پنجرهٔ مثبت

baseline پیشنهادی V2-W2:

```text
[time_of_event - 3s, time_of_event + 2s]
```

این بازه باید به محدودهٔ واقعی ویدئو محدود شود.

نسخه‌های قابل مقایسه:

| نسخه | پنجرهٔ مثبت | هدف |
|---|---|---|
| V2-W1 | [-4s, +1s] | زمینهٔ بیشتر پیش از حادثه |
| V2-W2 | [-3s, +2s] | baseline متوازن |
| V2-W3 | [-2s, +3s] | تاکید بیشتر بر خود حادثه |

اگر هدف «هشدار پیش از حادثه» باشد، فریم‌های پس از رخداد نباید وارد پنجره شوند؛ مثلاً `[-5s, -0.5s]`.

### پنجرهٔ منفی

- طول پنجره دقیقاً برابر پنجرهٔ مثبت باشد: ۵ ثانیه.
- انتخاب با seed ثابت برای validation انجام شود.
- موقعیت نسبی پنجرهٔ منفی باید تا حد ممکن مشابه توزیع موقعیت رخداد در مثبت‌ها باشد تا shortcut زمانی ایجاد نشود.
- شروع/پایان پنجره و seed در manifest ذخیره شود.

### سیاست مرزهای ویدئو

طول پنجرهٔ هدف ۵ ثانیه است. برای نمونه‌ای که event نزدیک ابتدا/انتهاست، ابتدا پنجره را با حفظ طول ۵ ثانیه shift می‌دهیم. اگر کل ویدئو کوتاه‌تر از ۵ ثانیه باشد، padding یا exclusion انجام می‌شود و دلیل آن در manifest ثبت می‌شود؛ هیچ clamp خامی که طول نمونه‌ها را ناهمسان کند مجاز نیست.

**معیار پایان:** فایل `sequence_manifest_v2.csv` شامل یک ردیف برای هر sequence و اطلاعات window_start/window_end، window_policy، sampling_seed و timestampهای انتخاب‌شده است.

## مرحلهٔ ۴ — نمونه‌برداری و پردازش فریم

### نمونه‌برداری

- شروع V2: ۱۶ فریم یکنواخت از پنجرهٔ ۵ ثانیه‌ای.
- آزمایش‌های بعدی: T = 8، 16، 24، 32.
- temporal jitter فقط برای train؛ validation کاملاً ثابت.
- timestampها با واحد **ثانیه از ابتدای ویدئو** در manifest ذخیره می‌شوند. sampling پایه:

  ```python
  timestamps = np.linspace(window_start, window_end, num=16, endpoint=False)
  ```

- برای seeking تصادفی، PyAV یا decord نسبت به OpenCV ارجح است؛ اگر OpenCV استفاده شود، خطای seek/read و fallback آن ثبت می‌شود.
- هر نمونه باید یک sequence باشد:

```text
X.shape = [T, C, H, W]
y = video-level label
```

### کنترل کیفیت فریم

- ثبت frame mask در صورت نخواندن فریم.
- بررسی فریم‌های کاملاً سیاه/سفید.
- استفاده از blur score فقط برای شناسایی خرابی شدید، نه حذف خودکار motion blur.
- بررسی تکراری‌بودن فریم‌های متوالی برای اطمینان از درست‌بودن decode/seeking.

فریم نامعتبر در مدل باید mask شود:

- mean pooling: مجموع featureهای معتبر تقسیم بر تعداد featureهای معتبر.
- attention pooling: logit فریم نامعتبر برابر `-inf` پیش از softmax.
- فریم جایگزین‌شده با فریم قبلی نباید weight یک فریم واقعی را بگیرد.

### پردازش قطعی

- OpenCV BGR باید به RGB تبدیل شود.
- حاشیه/overlay/dashboard فقط پس از مشاهدهٔ نمونه‌ها و با crop ثابت و محافظه‌کارانه حذف شود.
- نسبت تصویر dashcam باید تا حد ممکن حفظ شود. گزینه‌های قابل آزمایش: center crop مربعی، letterbox به 224×224، یا ورودی عریض مانند 224×320. crop مربعی فقط پس از مشاهدهٔ نمونه‌ها و اطمینان از حذف‌نشدن رخدادهای کناری استفاده شود.
- نرمال‌سازی ImageNet برای ResNet18:

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

- frame cache قبل از augmentation ذخیره شود.

**معیار پایان:** ۹۶۰۰ فریم خامِ پردازش‌شده (۶۰۰ × ۱۶) و manifest دقیق آن‌ها آماده باشد.

## مرحلهٔ ۵ — augmentation سازگار در سطح sequence

فقط برای train و با پارامتر مشترک برای تمام فریم‌های یک sequence:

- crop ملایم مشترک
- brightness/contrast کم و مشترک
- optional horizontal flip؛ با و بدون flip مقایسه شود
- temporal jitter، frame dropping محدود یا variable stride

نسخهٔ اولیه augmentation فقط شامل crop ملایم، brightness/contrast ملایم، flip اختیاری و temporal jitter است. Gaussian noise و JPEG compression در ablation جداگانه اضافه می‌شوند، نه به‌صورت پیش‌فرض.

transformهای geometric باید برای تمام فریم‌ها دقیقاً یکسان باشند. transformهای photometric می‌توانند مشترک یا با تغییر زمانی بسیار ملایم باشند، اما هرگز نباید پرش مصنوعی میان فریم‌ها ایجاد کنند.

نباید استفاده شود:

- vertical flip
- rotation شدید
- crop شدید
- frame shuffle
- sequence reversal
- augmentation مستقل برای هر فریم

**معیار پایان:** validation کاملاً deterministic و train دارای augmentation ملایمِ sequence-consistent است.

## مرحلهٔ ۶ — مدل‌ها و ترتیب آزمایش

همهٔ مدل‌ها باید در سطح sequence/video خروجی بدهند:

1. ResNet18 encoder + mean temporal pooling + linear classifier.
2. ResNet18 encoder + max temporal pooling + linear classifier.
3. ResNet18 encoder + GRU / BiGRU + classifier.
4. ResNet18 encoder + temporal attention + classifier.
5. ResNet18 encoder + temporal BiLSTM + additive attention pooling + FFN.

### نتیجهٔ مهم از survey و ترتیب درست مقایسه

مقالهٔ مرور HAR تأکید می‌کند که اضافه‌کردن RNN/LSTM همیشه بهتر از pooling نیست؛ در بعضی مطالعات، temporal pooling روی ویژگی‌های CNN از LSTM روی featureهای از پیش‌آموزش‌دیده بهتر عمل کرده است. بنابراین mean/max pooling باید baselineهای جدی V2 باشند، نه فقط مرحله‌ای برای عبور به GRU.

همچنین یک نسخهٔ سبک از ایدهٔ **Temporal Segment Network (TSN)** قابل آزمایش است:

- پنجرهٔ ۵ ثانیه‌ای را به چند segment تقسیم کنیم.
- از هر segment یک یا چند فریم نمونه‌گیری کنیم.
- feature یا logit segmentها را با mean/attention pooling به یک پیش‌بینی ویدئویی تبدیل کنیم.

این روش پوشش زمانی بهتری می‌دهد و با دادهٔ محدود، معمولاً سبک‌تر از مدل‌های 3D است.

### Temporal Attention و Multiple Instance Learning

برای حادثه، همهٔ ۱۶ فریم پنجره ارزش یکسان ندارند. بنابراین پس از mean/max pooling، مدل سبک و اولویت‌دار بعدی این است:

```text
ResNet18 frame features [B, T, 512]
        ↓
Temporal attention weights [B, T, 1]
        ↓
weighted temporal sum
        ↓
binary classifier
```

این مدل دو مزیت دارد:

- مدل می‌تواند روی timestampهای نزدیک حادثه وزن بیشتری بگذارد.
- weightهای attention قابل نمایش‌اند و برای تحلیل/ارائه مفید هستند.

برای مسئله‌ای که رخداد ممکن است فقط در چند فریم دیده شود، دو baseline دیگر نیز باید مقایسه شوند:

- **Top-k pooling:** میانگین بزرگ‌ترین k logit/احتمال فریم‌ها، مثلاً `k=3`.
- **Mean + max concatenation:** ترکیب میانگین و بیشینهٔ featureها پیش از classifier.

Attention و top-k pooling شکل‌های ساده‌ای از Multiple Instance Learning هستند: ویدئوی مثبت یک bag است که احتمالاً تنها چند فریم مهم دارد.

### معماری پیشرفته: ResNet18 + BiLSTM + Additive Attention + FFN

این معماری به‌عنوان آزمایش A6 در V2 اضافه می‌شود:

```text
RGB sequence [B, T, 3, H, W]
        ↓
Shared ResNet18 encoder
        ↓
frame features [B, T, 512]
        ↓
1-layer BiLSTM (hidden size 128 یا 256)
        ↓
temporal states [B, T, 2H]
        ↓
additive attention pooling
        ↓
video feature [B, 2H]
        ↓
FFN / MLP head
        ↓
binary logit
```

در additive attention، مدل برای هر timestamp یک score یاد می‌گیرد و بعد با softmax، ترکیب وزن‌دار stateهای BiLSTM را می‌سازد. وزن هر فریم همراه با timestamp قابل نمایش است و برای تحلیل خطا/ارائه مفید خواهد بود.

**محدودیت:** BiLSTM برای طبقه‌بندی یک clip کامل مناسب است، چون فریم‌های قبل و بعد را می‌بیند. برای هشدار آنلاینِ پیش از تصادف مناسب نیست؛ در آن حالت از LSTM/GRU یک‌طرفه استفاده می‌کنیم.

برای مدل‌های زمانی، encoder باید دست‌کم تا حدی fine-tune شود؛ freeze کامل encoder فقط baseline سبک است.

مدل‌های سنگین‌تر در صورت وجود GPU:

- VideoMAE-small
- X3D
- R(2+1)D / 3D ResNet
- I3D یا SlowFast ازپیش‌آموزش‌دیده روی داده‌های ویدئویی بزرگ مانند Kinetics

### R(2+1)D-18 به‌عنوان نخستین مدل 3D

اگر مدل‌های سبک V2 کافی نبودند و GPU در دسترس بود، نخستین مدل 3D پیشنهادی **R(2+1)D-18 pretrained** است:

```text
input: [B, 3, 16, 112, 112]
```

- کانولوشن مکانی و زمانی را جدا می‌کند و برای ۱۶ فریم مناسب است.
- ابتدا بخش‌های ابتدایی freeze و فقط head/final stage آموزش داده شود.
- batch size حدود ۴ تا ۸، mixed precision و early stopping استفاده شود.
- R3D-18 یک baseline 3D مقایسه‌ای است.

SlowFast و VideoMAE مدل‌های سطح بالاترند، نه نخستین آزمایش V2. VideoMAE با ۶۰۰ ویدئو باید بسیار محافظه‌کارانه fine-tune شود: ابتدا head، سپس فقط ۱ تا ۲ block آخر با learning rate کوچک. قبل از پیاده‌سازی VideoMAE باید نام دقیق checkpoint، dataset پیش‌آموزش، تعداد فریم، tubelet size، resolution و preprocessing رسمی همان checkpoint در config ثبت شود.

برای 3D CNN، ۱۶ فریم در پنجرهٔ ۵ ثانیه‌ای فقط حدود ۳.۲ FPS است و ممکن است حرکت سریع را از دست بدهد. همهٔ مدل‌ها باید یک بخش زمانی یکسان از ویدئو را ببینند، اما تعداد فریم و stride می‌تواند با معماری سازگار باشد؛ مثلاً ۱۶ فریم از ۲ تا ۳ ثانیه یا ۳۲ فریم از ۵ ثانیه برای R(2+1)D.

### سطح‌بندی آزمایش‌ها با ۶۰۰ ویدئو

| سطح | آزمایش‌ها |
|---|---|
| Required | A0 (V1)، A1 mean pooling، A2 attention، A3 GRU یا GRU+attention، B2 R(2+1)D-18 pretrained در صورت GPU |
| Recommended | mean+max، top-k، metadata ablation، VideoMAE partial fine-tune در صورت GPU مناسب |
| Optional / پژوهشی | BiLSTM+additive attention، SlowFast، optical flow کامل، ensemble، multi-clip inference پیشرفته |

BiLSTM در roadmap باقی می‌ماند، اما اختیاری است؛ recurrent اصلی V2، GRU+attention است تا تعداد مدل‌های مشابه با دادهٔ محدود بی‌دلیل زیاد نشود.

**معیار پایان:** هر مدل با manifest و split یکسان اجرا و مقایسه شده است.

## جریان حرکتی و multi-clip inference (آزمایش اختیاری پس از RGB V2)

مقاله نشان می‌دهد که معماری‌های دو-جریانی، ظاهر RGB و حرکت را جداگانه مدل می‌کنند:

```text
RGB stream + motion stream (optical flow یا motion representation) → fusion → classifier
```

این ایده برای تصادف منطقی است، چون تغییر ناگهانی سرعت/جهت یا لرزش دوربین نشانهٔ مهمی است. اما محاسبهٔ optical flow پرهزینه و زمان‌بر است؛ بنابراین نباید اولین آزمایش V2 باشد.

ترتیب پیشنهادی:

1. RGB sequence V2 را کامل و ارزیابی کنیم.
2. یک ablation سبک با اختلاف RGB فریم‌های متوالی یا optical flow کم‌حجم بسازیم.
3. فقط اگر بهبود معنادار داشت، RGB و motion را با late fusion ترکیب کنیم.

پیش از optical flow کامل، نمایش سبک حرکت را آزمایش کنیم:

```text
frame_difference[t] = RGB[t] - RGB[t - 1]
```

آزمایش‌ها باید به‌ترتیب `RGB only`، `RGB + frame difference` و سپس در صورت لزوم `RGB + optical flow` باشند. لرزش dashcam می‌تواند motion stream را به false positive حساس کند؛ پس اثر آن باید با تحلیل خطا بررسی شود.

برای inference روی کل ویدئو نیز می‌توان چند clip ثابت از قسمت‌های مختلف ویدئو برداشت و احتمال clipها را میانگین یا با attention تجمیع کرد. timestampهای این clipها در validation باید ثابت باشند.

## انتقال یادگیری و محدودیت محاسباتی

مقاله بر transfer learning در ویدئو تأکید می‌کند. در پروژهٔ ما:

- ResNet18 ImageNet-pretrained یک transfer-learning baseline مناسب است.
- اگر GPU در دسترس شد، encoder ویدئوییِ Kinetics-pretrained را با learning rate کم fine-tune می‌کنیم.
- در محیط محدود، feature cache کردن خروجی encoder و آموزش pooling/GRU سبک قابل‌دفاع است؛ اما freeze کامل encoder نباید تنها نسخهٔ مدل زمانی باشد.
- optical flow و 3D-CNNها باید با هزینهٔ اجرا، زمان inference و افزایش واقعی F1 مقایسه شوند.

Normalization، input resolution و crop به backbone وابسته است و نباید یک transform ثابت برای همهٔ مدل‌ها تحمیل شود:

- ResNet18: ImageNet preprocessing.
- R(2+1)D/R3D: transform رسمی pretrained weights.
- VideoMAE: processor/checkpoint رسمی همان مدل.

## مرحلهٔ ۷ — ارزیابی و انتخاب مدل

برای validation ثابت گزارش شود:

- Accuracy
- Precision حادثه
- Recall حادثه
- F1 حادثه
- ROC-AUC
- PR-AUC
- Confusion matrix

معیار انتخاب مدل:

```text
Primary: Accident F1
Safety constraint: Accident Recall
Secondary: PR-AUC
Descriptive: Accuracy and inference speed
```

مدلی با F1 بالاتر فقط زمانی انتخاب می‌شود که Recall حادثه افت ناموجهی نسبت به baseline نداشته باشد. مقدار حداقل Recall باید پیش از مقایسهٔ نهایی در config مشخص شود.

آستانهٔ تصمیم فقط روی validation تنظیم شود و هر دو مورد گزارش شوند:

- معیارها در threshold = 0.5
- بهترین F1 در threshold منتخب validation

این اعداد **development metrics** هستند، نه تخمین کاملاً بی‌طرفانهٔ عملکرد نهایی؛ چون hyperparameterها و threshold با همین validation انتخاب می‌شوند. برای ادعای نهایی باید cross-validation یا test مستقل استفاده شود.

ارزیابی دو سطح دارد:

- **clip-level:** آیا پنجرهٔ ۵ ثانیه‌ای حادثه را دارد؟
- **video-level:** آیا کل MP4 شامل حادثه است؟ این خروجی اصلی مسئله است.

برای probability قابل‌استفاده در inference، علاوه بر F1 باید Brier score، Expected Calibration Error، reliability diagram و در صورت نیاز temperature scaling بررسی شود. نسخهٔ سادهٔ uncertainty نیز ثبت می‌شود: احتمال در بازهٔ `[0.4, 0.6]` به‌عنوان `uncertain` گزارش شود.

تحلیل خطا شامل نمونه‌های زیر باشد:

- True positive
- True negative
- ۲۰ false positive با بیشترین confidence
- ۲۰ false negative با کمترین probability

و بررسی شرایط: شب، باران، ترافیک، توقف ناگهانی، لرزش دوربین و موقعیت رخداد.

برای هر مدل جدید، علاوه بر نتیجهٔ کلی، مشخص شود بهبود از کدام منبع آمده است:

- نمونه‌برداری زمانی V2
- pooling در برابر GRU/attention
- RGB در برابر RGB+motion
- encoder frozen در برابر fine-tuned
- یک clip در برابر multi-clip inference

### پایداری آماری نتیجه

validation ثابت ۱۲۰ ویدئویی برای توسعه و مقایسهٔ منصفانه حفظ می‌شود. اما چون داده کم است، برای ۲ یا ۳ مدل نهایی بهتر است **Stratified Group K-Fold** با `K=5` انجام شود (در صورت وجود گروه‌ها، group-aware). در گزارش نهایی مدل‌های برتر، میانگین و انحراف معیار F1 ثبت شود:

```text
F1 = mean ± std
```

bootstrap confidence interval برای F1/Recall/PR-AUC و bootstrap difference یا McNemar test برای مقایسهٔ V1 و V2، خروجی‌های ارزشمند تکمیلی هستند.

## تنظیمات آموزش و loss

چون کلاس‌ها در سطح ویدئو متوازن‌اند، loss شروع باید ساده باشد:

```text
BCEWithLogitsLoss یا CrossEntropyLoss معمولی
```

class weight فقط زمانی استفاده شود که هزینهٔ false negative به‌صورت آگاهانه بیشتر در نظر گرفته شود، نه برای اصلاح imbalance.

آزمایش‌های بعدی، پس از baseline loss:

- label smoothing ملایم
- focal loss برای نمونه‌های سخت

برای encoderهای pretrained:

1. freeze encoder و آموزش head.
2. unfreeze آخرین block با backbone LR حدود `1e-5` تا `3e-5` و head LR حدود `1e-4` تا `1e-3`.
3. فقط اگر validation بهبود داشت، blockهای بیشتر باز شوند.

از AdamW، early stopping، gradient clipping و mixed precision در صورت GPU استفاده شود.

checkpoint با validation loss یا PR-AUC در threshold ثابت ۰.۵ انتخاب می‌شود. threshold فقط **پس از** پایان آموزش و روی validation برای F1 تنظیم می‌شود؛ checkpoint selection نباید هم‌زمان با جست‌وجوی threshold انجام شود.

## بازتولیدپذیری، tracking و بودجهٔ اجرا

برای هر run ثبت شود:

```text
run_id
git_commit_hash
config.yaml
environment.txt
split_seed
window_seed
augmentation_seed
best_checkpoint.pt
last_checkpoint.pt
metrics.json
confusion_matrix.png
threshold_curve.csv
```

seedهای Python، NumPy، PyTorch CPU/CUDA، DataLoader worker و تنظیمات deterministic cuDNN باید در config ذخیره شوند.

برنامهٔ اولیهٔ حافظه برای Colab:

| خانواده | batch size شروع | نکته |
|---|---:|---|
| ResNet sequence | 8–16 | mixed precision در GPU |
| R(2+1)D | 2–8 | gradient accumulation در صورت OOM |
| VideoMAE | 1–4 | gradient accumulation + mixed precision |

### سخت‌افزار فعلی

محیط فعلی VS Code روی CPU اجرا می‌شود: PyTorch هیچ CUDA GPUای پیدا نکرده و `nvidia-smi` نیز در دسترس نیست. VS Code فقط محیط ویرایش/اجرای کد است و به‌خودی‌خود GPU فراهم نمی‌کند.

بنابراین در محیط محلی، مراحل Required V2 (data audit، sequence sampling، ResNet+pooling/attention و GRU سبک) اجرا می‌شوند. R(2+1)D، VideoMAE، SlowFast و optical flow سنگین فقط در صورت استفاده از Google Colab یا محیط دارای GPU وارد برنامه می‌شوند.

## ماتریس آزمایش V2 (ablation)

همهٔ آزمایش‌ها باید با manifest، split، window و معیار ثابت مقایسه شوند:

| شناسه | مدل | ورودی |
|---|---|---|
| A0 | ResNet18 V1 | ۸ فریم پراکنده (مرجع فعلی) |
| A1 | ResNet18 + mean pooling | ۱۶ فریم نزدیک رخداد |
| A2 | ResNet18 + max / mean+max pooling | ۱۶ فریم نزدیک رخداد |
| A3 | ResNet18 + temporal attention | ۱۶ فریم نزدیک رخداد |
| A4 | ResNet18 + GRU | ۱۶ فریم نزدیک رخداد |
| A5 | ResNet18 + GRU + attention | ۱۶ فریم نزدیک رخداد |
| A6 | ResNet18 + BiLSTM + additive attention + FFN | ۱۶ فریم نزدیک رخداد |
| B1 | R3D-18 pretrained | کلیپ ۱۶ فریمی |
| B2 | R(2+1)D-18 pretrained | کلیپ ۱۶ فریمی |
| C1 | VideoMAE frozen / partial fine-tune | کلیپ ۱۶ فریمی |
| D1 | بهترین مدل + frame difference | RGB + motion سبک |
| D2 | بهترین مدل + metadata | video + weather/light/scene |
| E1 | ensemble | فقط مدل‌های مکمل برتر |

Ensemble فقط زمانی مجاز است که مدل‌ها به‌تنهایی قابل‌قبول باشند و خطاهای همسان نداشته باشند. وزن‌های ensemble فقط روی validation انتخاب می‌شوند.

## مرحلهٔ ۸ — metadata به‌عنوان آزمایش جداگانه

مدل‌های مستقل زیر مقایسه شوند:

```text
Video only
Metadata only (weather, light_conditions, scene)
Video + metadata
```

هدف: تشخیص اینکه metadata shortcut ایجاد نمی‌کند. `time_of_event` و `time_to_accident` هرگز ورودی مدل نیستند.

### Meta-data fusion با مدل ویدئویی

fusion فقط پس از ارزیابی video-only اجرا می‌شود:

```text
video feature از attention/BiLSTM
        +
weather embedding + light_conditions embedding + scene embedding
        ↓
concatenate
        ↓
FFN / MLP fusion head
        ↓
binary logit
```

قواعد لازم:

- فقط `weather`، `light_conditions` و `scene` بررسی شوند.
- `time_of_event`، `time_of_alert` و `time_to_accident` ورودی مدل نیستند.
- metadata باید در زمان inference واقعاً در دسترس باشد؛ اگر برای ویدئوی جدید نداریم، مدل نهایی عملی باید video-only باشد یا metadata با روشی مستقل استخراج شود.
- سه ablation اجباری گزارش شوند: `video only`، `metadata only` و `video + metadata`.
- اگر metadata-only غیرعادی خوب بود، احتمال shortcut/bias بررسی شود.

### وضعیت بررسی‌شدهٔ metadata این دیتاست

در فایل‌های metadata موجود:

- train: `weather` برای ۱۴۹۸ از ۱۵۰۰ نمونه و `light_conditions`/`scene` برای همهٔ نمونه‌ها موجود است.
- test: هر سه ستون برای همهٔ ۱۳۴۴ نمونه موجود هستند.
- train شامل `Snow` و test شامل `Fog` است؛ بنابراین encoder دسته‌ای باید unknown category را پشتیبانی کند.

بااین‌حال برای یک MP4 دلخواه خارج از دیتاست، این metadata به‌طور خودکار وجود ندارد. پس مدل اصلی و قابل‌استفاده در inference باید **video-only** باشد؛ metadata fusion فقط یک آزمایش مکمل روی دیتاست است، مگر اینکه در محصول واقعی روش مستقلی برای تأمین این اطلاعات داشته باشیم.

## inference نهایی بدون `time_of_event`

در آموزش، `time_of_event` فقط برای ساخت positive clip استفاده می‌شود. در MP4 جدید این مقدار وجود ندارد؛ بنابراین inference نهایی باید sliding-window باشد:

```text
MP4 کامل
  ↓
5-second windows با stride ثابت (مثلاً 2.5 seconds)
  ↓
predict probability for every clip
  ↓
aggregate to video probability
  ↓
calibrated threshold → label + confidence state
```

قواعد پیشنهادی:

1. ابتدا ویدئو validate و duration آن خوانده می‌شود.
2. windowهای ۵ ثانیه‌ای کل ویدئو ساخته می‌شوند؛ برای ویدئوی کوتاه سیاست padding/exclusion همان manifest اعمال می‌شود.
3. preprocessing در inference deterministic و دقیقاً مطابق مدل منتخب است.
4. probability سطح ویدئو با `max` یا `top-k mean` از probability کلیپ‌ها ساخته می‌شود؛ روش تجمیع باید روی validation انتخاب و ثابت شود.
5. خروجی شامل `video_probability`، label، confidence state، probability هر window و window با بیشترین احتمال است.

به‌عنوان خروجی تکمیلی، مرکز window با بیشترین احتمال را زمان تقریبی رخداد گزارش می‌کنیم. برای positive validation می‌توان خطای زمانی را با `time_of_event` سنجید:

```text
MAE زمان رخداد
Recall within ±1s / ±2s / ±5s
```

این localization معیار اصلی انتخاب مدل نیست، اما ارزش تحلیلی/امتیازی دارد.

## ترتیب اجرای پیشنهادی

1. Data audit و ساخت `video_manifest_v2.csv`.
2. freeze کردن split و بررسی duplicate/leakage.
3. ساخت `sequence_manifest_v2.csv` با پنجرهٔ V2-W2 `[-3s, +2s]`.
4. استخراج ۱۶ فریم از هر window و cache قطعی.
5. baseline sequence جدید: ResNet18 + mean pooling.
6. max/mean+max pooling، top-k pooling و temporal attention با همان داده و split.
7. GRU + attention و BiLSTM + additive attention + FFN فقط پس از baselineهای pooling.
8. در صورت نیاز و GPU: R(2+1)D-18 pretrained.
9. در صورت نیاز و GPU مناسب: VideoMAE با fine-tuning محافظه‌کارانه.
10. threshold tuning، تحلیل خطا، و در نهایت cross-validation مدل‌های برتر.

## خروجی‌های مورد انتظار V2

```text
video_manifest_v2.csv
metadata_split_v1.csv
sequence_manifest_v2.csv
processed_v2/...
models_v2/...
v2_training_history.csv
v2_validation_predictions.csv
v2_error_analysis.csv
run_config.yaml
metrics.json
inference_windows.csv
```

## تصمیم‌های باز پیش از پیاده‌سازی V2

تصمیم‌های تثبیت‌شده: هدف اصلی، offline video-level accident detection است؛ metadata fusion فقط آزمایش مکمل است؛ محیط محلی GPU ندارد.

پیش از پیاده‌سازی فقط این موارد باید در manifest/data audit روشن شوند:

1. آیا شناسه‌های group مانند `trip_id`، `driver_id`، `camera_id` یا `route_id` وجود دارند؟
2. حداقل Recall قابل‌قبول برای انتخاب مدل نهایی چه مقدار است؟
3. آیا در مراحل سنگین‌تر به Google Colab دارای GPU دسترسی خواهیم داشت یا V2 باید کاملاً CPU-friendly بماند؟

## منبع تکمیلی بررسی‌شده

Sharma et al. (2022), *A Review of Deep Learning-based Human Activity Recognition on Benchmark Video Datasets*.

این مقاله مخصوص تصادف نیست و نباید اعداد accuracy آن با دیتاست Nexar مقایسه شود. ارزش آن برای پروژهٔ ما در تأیید و مقایسهٔ معماری‌های video-level است: CNN+pooling، CNN+RNN/LSTM، دو-جریانی RGB/flow، 3D-CNN، Temporal Segment Network و transfer learning ویدئویی.
