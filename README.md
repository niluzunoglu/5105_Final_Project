# BLM5105 Doğal Dil İşleme Dersi Final Projesi

Bu repository, **5105 Doğal Dil İşleme Dersi** için hazırlanan final projesini içermektedir. Proje, **haber özetleri ve başlık arasındaki anlam ilişkileri** üzerine odaklanmaktadır ve modellerin etkinliğini değerlendirmek için hem otomatik hem de insan değerlendirmelerini içermektedir.

## Proje Özeti
Bu projenin ana amacı, en az üç farklı Türkçe GPT modelinin iki görevdeki performansını karşılaştırmaktır:

1. **Başlıktan Özet Çıkarma**: Verilen bir başlığa dayanarak detaylı bir özet oluşturmak.
2. **Özetlerden Başlık Oluşturma**: Verilen bir özete dayanarak kısa ve anlamlı bir başlık oluşturmak.

### Temel Özellikler
- **Türkçe GPT modellerinin** doğal dil işleme görevlerinde kullanımı.
- Değerlendirme:
  - **500 örnek** üzerinde otomatik metriklerle.
  - **50 örnek** üzerinde insan değerlendirmesiyle.
- Modellerin performanslarının belirli metriklerle ve nitel analizlerle karşılaştırılması.

## Depo Yapısı

```
5105_Final_Project/
├── logs/                 # Loglar
├── outputs/              # Çıktıları içeren excel dosyaları
├── ModelTrainer.py       # Model eğitimi ve evaluationı için oluşturulan sınıf
├── DatasetHandler.py     # Veri ön işleme ve değerlendirme için oluşturulan sınıf
├── Logger.py             # Loglama kütüphanesi
├── README.md             # Proje dökümantasyonu
```



### Dizin Detayları
- **logs/**: Çalışma süresince oluşturulan log dosyalarını içerir.
- **outputs/**: Model değerlendirme çıktılarının yer aldığı excel dosyalarını içerir.
- **results/**: Otomatik değerlendirme sonuçlarını ve insan geri bildirim analizlerini içerir.
- **ModelTrainer.py**: Model eğitimi ve değerlendirmesi için oluşturulan sınıf.
- **DatasetHandler.py**: Veri ön işleme ve değerlendirme görevlerini gerçekleştiren sınıf.
- **Logger.py**: Loglama işlemleri için kullanılan özel kütüphane.

## Gereksinimler

Bu projeyi çalıştırmak için:

- Python 3.8 veya üzeri
- Hugging Face Transformers
- TensorFlow veya PyTorch
- `requirements.txt` dosyasında listelenen ek bağımlılıklar

## Kurulum Talimatları

1. Repoyu klonlayın:
   ```bash
   git clone https://github.com/niluzunoglu/5105_Final_Project.git
   cd 5105_Final_Project
   ```

2. Gerekli bağımlılıkları yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

3. Veri setini indirerek `data/` dizinine yerleştirin.

4. Ön işleme betiklerini çalıştırın:
   ```bash
   python scripts/preprocess_data.py
   ```

5. Sağlanan notebook'lar veya betikler ile modelleri eğitin veya ince ayar yapın.

6. Modelleri değerlendirin:
   ```bash
   python scripts/evaluate_models.py
   ```

## Değerlendirme Metrikleri

Proje hem otomatik hem de insan değerlendirmesini kullanır:

1. **Otomatik Metrikler**:
   - BLEU
   - ROUGE
   - METEOR

2. **İnsan Değerlendirmesi**:
   - Akıcılık
   - Uygunluk
   - Tutarlılık

## Sonuçlar
Değerlendirme sonuçları `results/` dizininde bulunabilir. Detaylı performans karşılaştırmaları ve çıkarımlar proje raporunda sunulmuştur.

## Gelecek Çalışmalar
- Değerlendirme kapsamına daha fazla Türkçe dil modeli eklemek.
- Çok dilli özetleme ve başlık oluşturma görevlerini keşfetmek.
- Üretim kalitesini artırmak için pekiştirmeli öğrenme yöntemlerini uygulamak.

## Katkıda Bulunanlar
- **A. Nil Uzunoğlu**
- **Mehmet Taştan**  

  Yıldız Teknik Üniversitesi
  Hesaplamalı Anlambilim Dersi, 2024

Sorularınız veya katkılarınız için lütfen iletişime geçin.

---
