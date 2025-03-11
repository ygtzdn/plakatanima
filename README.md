# Plaka Tanıma Sistemi

## Proje Açıklaması
Bu proje, kamera görüntüsünden araç plakalarını gerçek zamanlı olarak tespit edip okuyan bir sistemdir. PySide6 tabanlı kullanıcı arayüzü ile kolay kullanım sağlar.

## Özellikler
- Gerçek zamanlı plaka tespiti ve okuma
- Birden fazla kamera desteği
- Tespit edilen plakaların takibi
- Tespit istatistikleri gösterimi
- Ayarlanabilir tespit parametreleri

## Gereksinimler
Projeyi çalıştırmak için aşağıdaki kütüphanelerin yüklü olması gerekmektedir:
- Python 3.10 (OCR sistemi üst sürümlerde sorun çıkarabiliyor)
- PySide6
- OpenCV (cv2)
- NumPy

## Kurulum
1. Python 3.10'un yüklü olduğundan emin olun
   ```
   python --version
   ```

2. Gerekli kütüphaneleri yükleyin:
   ```
   pip install -r requirements.txt
   ```

2. `plate_detection.py` ve `camera_processing.py` modüllerini oluşturun veya projeye dahil edin.

3. `models/abc.pt` model dosyasını uygun konuma yerleştirin.

## Kullanım
Programı başlatmak için: 
   ```
   python main.py
   ```
   Açılması biraz zaman alabilir.
