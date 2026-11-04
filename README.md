# Posture Guard

[Türkçe](#türkçe) | [English](#english)

---

## Türkçe

Posture Guard, web kameranızla duruşunuzu takip eder ve kamburlaştığınızda hafif bir sesle uyarır. Görüntü yalnızca bilgisayarınızda işlenir, kaydedilmez ve hiçbir yere gönderilmez.

**Gereksinimler:** macOS (ses için `afplay` kullanılır), Python 3 ve web kamerası.

### Çalıştırma

```bash
chmod +x run.sh   # yalnızca ilk seferde gerekir
./run.sh
```

İlk çalıştırmada `.venv` yoksa oluşturulur ve bağımlılıklar kurulur. macOS kamera izni isterse onaylayın. İzin penceresi çıkmazsa *Sistem Ayarları > Gizlilik ve Güvenlik > Kamera* bölümünden terminal uygulamanıza izin verin.

### Kalibrasyon

Uygulama kalibrasyon olmadan da çalışır, ama kişisel kalibrasyon daha hassas sonuç verir.

1. Kameraya bakacak şekilde dik oturun. Kalibrasyon yaklaşık bir saniye sürer, bu yüzden tuşa basmadan önce hazır olun.
2. **C** tuşuna basın.
3. Mavi kart `KALİBRASYON: 30/30` değerine ulaşana kadar dik kalın.

Kalibrasyon `calibration.json` dosyasına kaydedilir ve sonraki açılışlarda otomatik yüklenir. Oturma düzeniniz veya kamera konumunuz değişirse yeniden kalibre edin.

### Kısayollar

| Tuş | İşlev |
| --- | --- |
| `C` | Kalibre et |
| `M` | Sesi kapat / aç |
| `T` | Sesi test et |
| `H` | Sade görünüm / tam görünüm |
| `Q` veya `ESC` | Çıkış |

Kısayollar, açıklamalarıyla birlikte sağ üstteki kartta da listelenir. Sade görünümde bu kart gizlenir.

### Ekrandaki renkler

| Renk | Anlamı |
| --- | --- |
| Yeşil | Duruşunuz iyi |
| Sarı | Dikkat: duruşunuz bozuluyor veya uyarı süresi işliyor |
| Kırmızı | Kambur duruş süresi eşiği aştı, ses uyarısı devrede |
| Mavi | Kalibrasyon sürüyor |
| Gri | Kamerada kimse görünmüyor |

### Seçenekler

| Seçenek | Varsayılan | Açıklama |
| --- | --- | --- |
| `--camera N` | `0` | Kamera indeksi |
| `--sound tink\|purr\|synth` | `tink` | Uyarı sesi |
| `--volume V` | `0.15` | Ses seviyesi (0.0-1.0) |
| `--threshold SN` | `2.5` | Uyarıdan önceki kambur duruş süresi (saniye) |
| `--cooldown SN` | `3.5` | Uyarı tekrar aralığı (saniye) |
| `--headless` | kapalı | Penceresiz çalışır: yalnızca konsol kaydı ve sesli uyarı |

Örnek: `./run.sh --sound purr --volume 0.3 --threshold 4`

`--headless` modunda klavye kısayolları yoktur; `Ctrl+C` ile çıkılır. Bu modda kalibrasyon yapılamaz, ancak pencereli modda kaydedilen kalibrasyon kullanılır.

### Testler

```bash
.venv/bin/python -m unittest
```

---

## English

Posture Guard watches your posture through the webcam and plays a soft sound when you slouch. Video is processed only on your computer. It is never saved or sent anywhere.

**Requirements:** macOS (sound uses `afplay`), Python 3 and a webcam.

### Running

```bash
chmod +x run.sh   # first time only
./run.sh
```

On first run, `.venv` is created if missing and the dependencies are installed. Allow camera access if macOS asks. If no prompt appears, enable your terminal app under *System Settings > Privacy & Security > Camera*.

### Calibration

The app works without calibration, but a personal baseline is more accurate.

1. Sit upright facing the camera. Calibration takes about a second, so get ready before pressing the key.
2. Press **C**.
3. Stay upright until the blue card reaches `KALİBRASYON: 30/30`.

The baseline is saved to `calibration.json` and loaded automatically on later runs. Recalibrate if your seat or camera position changes.

### Shortcuts

| Key | Action |
| --- | --- |
| `C` | Calibrate |
| `M` | Mute / unmute |
| `T` | Test the sound |
| `H` | Simple / full view |
| `Q` or `ESC` | Quit |

The shortcuts are also listed with descriptions on the card at the top right. Simple view hides this card.

### On-screen colors

| Color | Meaning |
| --- | --- |
| Green | Good posture |
| Yellow | Warning: posture is slipping or the grace period is running |
| Red | Slouching past the threshold, sound alert active |
| Blue | Calibrating |
| Gray | Nobody in view |

### Options

| Option | Default | Description |
| --- | --- | --- |
| `--camera N` | `0` | Camera index |
| `--sound tink\|purr\|synth` | `tink` | Alert sound |
| `--volume V` | `0.15` | Volume (0.0-1.0) |
| `--threshold SEC` | `2.5` | Slouch time before the alert (seconds) |
| `--cooldown SEC` | `3.5` | Time between repeated alerts (seconds) |
| `--headless` | off | Runs without a window: console log and sound alerts only |

Example: `./run.sh --sound purr --volume 0.3 --threshold 4`

In `--headless` mode there are no keyboard shortcuts; press `Ctrl+C` to quit. Calibration isn't available in this mode, but a baseline saved from the windowed mode is used.

### Tests

```bash
.venv/bin/python -m unittest
```
