import sys
import cv2
from collections import Counter
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPlainTextEdit,
                               QDoubleSpinBox, QSpinBox, QVBoxLayout, QHBoxLayout, QPushButton,
                               QGroupBox, QFormLayout, QCheckBox, QComboBox)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap

from plate_detection import PlateDetector
from camera_processing import SimpleSort

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plaka Tanıma Arayüzü")

        # Ana widget ve genel layout
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Sol panel: Kamera görüntüsü (üst) ve konsol (alt)
        left_panel = QVBoxLayout()

        # Kamera görüntüsünün gösterileceği alan
        self.camera_label = QLabel("Kamera Görüntüsü", self)
        self.camera_label.setFixedSize(640, 480)
        self.camera_label.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(self.camera_label)

        # Konsol: Tespit sonuçlarının yazılacağı alan
        self.console = QPlainTextEdit(self)
        self.console.setReadOnly(True)
        self.console.setFixedHeight(150)
        left_panel.addWidget(self.console)

        main_layout.addLayout(left_panel)

        # Sağ panel: Parametre ayarlarının yapılabileceği kontrol paneli
        right_panel = QGroupBox("Ayarlar", self)
        right_layout = QFormLayout(right_panel)

        # Kamera seçimi için ComboBox
        self.camera_combo = QComboBox(self)
        self.scan_cameras()  # Mevcut kameraları tara
        right_layout.addRow("Kamera Seçimi:", self.camera_combo)

        # Güven eşiği ayarı
        self.confidence_spin = QDoubleSpinBox(self)
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.7)
        right_layout.addRow("Güven Eşiği:", self.confidence_spin)

        # Tarama sıklığı ayarı (kaç frame'de tespit yapılacağı)
        self.scan_frequency_spin = QSpinBox(self)
        self.scan_frequency_spin.setRange(1, 100)
        self.scan_frequency_spin.setValue(5)
        right_layout.addRow("Tarama Sıklığı:", self.scan_frequency_spin)

        # Minimum tespit (track onay için)
        self.min_hits_spin = QSpinBox(self)
        self.min_hits_spin.setRange(1, 10)
        self.min_hits_spin.setValue(3)
        right_layout.addRow("Min Hits:", self.min_hits_spin)

        # Maksimum kayıp frame sayısı
        self.max_missing_spin = QSpinBox(self)
        self.max_missing_spin.setRange(1, 20)
        self.max_missing_spin.setValue(5)
        right_layout.addRow("Max Missing:", self.max_missing_spin)

        # Kayıt hafızası boyutu ayarı
        self.history_size_spin = QSpinBox(self)
        self.history_size_spin.setRange(5, 50)
        self.history_size_spin.setValue(30)
        right_layout.addRow("Kayıt Hafızası:", self.history_size_spin)

        # İstatistik yazma sıklığı ayarı
        self.stats_frequency_spin = QSpinBox(self)
        self.stats_frequency_spin.setRange(1, 50)
        self.stats_frequency_spin.setValue(5)
        right_layout.addRow("İstatistik Sıklığı:", self.stats_frequency_spin)
        
        # Plaka okumalarını konsola yazdırma seçeneği
        self.show_readings_checkbox = QCheckBox("Her Plaka Okumasını Göster", self)
        self.show_readings_checkbox.setChecked(False)  # Varsayılan olarak kapalı
        right_layout.addRow(self.show_readings_checkbox)

        # Parametreleri güncelleme butonu
        self.update_button = QPushButton("Parametreleri Güncelle", self)
        right_layout.addRow(self.update_button)

        main_layout.addWidget(right_panel)

        # Kamera yakalama ayarları
        self.active_camera = 0  # Varsayılan olarak ilk kamera
        self.cap = cv2.VideoCapture(self.active_camera)
        self.frame_count = 0

        # PlateDetector ve SimpleSort (takip) modüllerini başlat
        self.plate_detector = PlateDetector(
            "models/abc.pt",
            confidence_threshold=self.confidence_spin.value(),
            scan_frequency=self.scan_frequency_spin.value()
        )
        self.tracker = SimpleSort(
            max_missing=self.max_missing_spin.value(),
            min_hits=self.min_hits_spin.value(),
            iou_threshold=0.3
        )

        # Son 30 plaka bilgisini tutmak ve tespit sayısı
        self.history_size = 30
        self.stats_frequency = 5
        self.detection_history = []
        self.total_detections = 0
        self.current_plate_text = None
        self.show_readings = False  # Plaka okumalarını gösterme durumu

        # Kamera görüntüsünü düzenli olarak güncellemek için timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Yaklaşık 33 FPS

        # Parametre güncelleme butonuna tıklama
        self.update_button.clicked.connect(self.update_parameters)

    def scan_cameras(self):
        """Sistemdeki kameraları tarar ve ComboBox'a ekler"""
        available_cameras = []
        
        # 10 kameraya kadar kontrol et (0-9 indeksli)
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        
        # Hiç kamera bulunamazsa uyarı göster
        if not available_cameras:
            self.camera_combo.addItem("Kamera bulunamadı", -1)
            self.console.appendPlainText("UYARI: Hiçbir kamera bulunamadı!")
            return
        
        # Bulunan kameraları ComboBox'a ekle
        for cam_id in available_cameras:
            self.camera_combo.addItem(f"Kamera {cam_id}", cam_id)
        
        self.console.appendPlainText(f"{len(available_cameras)} adet kamera bulundu.")
        
        # Eğer sadece bir kamera varsa bile görünür olsun
        if len(available_cameras) == 1:
            self.console.appendPlainText("Cihazınızda yalnızca bir kamera tespit edildi.")

    def change_camera(self, camera_id):
        """Aktif kamerayı değiştirir"""
        # Geçerli kamerayı kapat
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        # Yeni kamerayı aç
        self.cap = cv2.VideoCapture(camera_id)
        if self.cap.isOpened():
            self.active_camera = camera_id
            self.console.appendPlainText(f"Kamera {camera_id} aktif edildi.")
            return True
        else:
            self.console.appendPlainText(f"HATA: Kamera {camera_id} açılamadı!")
            # Eski kamerayı tekrar açmayı dene
            self.cap = cv2.VideoCapture(self.active_camera)
            return False

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        self.frame_count += 1

        # Plaka tespiti: YOLO ile kutu belirleme
        detections = self.plate_detector.detect(frame, self.frame_count)
        # Takip algoritması ile tespitleri güncelle
        confirmed_tracks = self.tracker.update(detections)
        if confirmed_tracks:
            # İlk onaylanmış track'i kullanarak OCR işlemi
            track_id, bbox = confirmed_tracks[0]
            plate_text = self.plate_detector.ocr_plate(frame, bbox)
            if plate_text:
                # İsteğe bağlı olarak her plakayı konsola yazdır
                if self.show_readings:
                    self.console.appendPlainText(f"Okunan Plaka: {plate_text}")
                
                # Plaka tespiti başarılı - hafızaya ekle
                self.detection_history.append(plate_text)
                # Hafıza boyutu kontrolü
                if len(self.detection_history) > self.history_size:
                    self.detection_history.pop(0)
                
                # Her yeni tespit sayılmalı
                self.total_detections += 1
                
                # Her 5 okumada bir istatistik yazdır
                if self.total_detections % self.stats_frequency == 0:
                    self.print_stats()
                
                # Yeni bir plaka tespit edildiyse
                if self.current_plate_text is None or self.current_plate_text != plate_text:
                    self.current_plate_text = plate_text
                
                # Tespit edilen kutuyu ve plaka metnini ekrana çiz
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, plate_text, (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            # Eğer plaka takip edilemiyorsa, süreci sıfırla
            if self.current_plate_text is not None:
                self.console.appendPlainText("Plaka kaybedildi. Süreç yeniden başlatılıyor.")
                # Plaka kaybedildiğinde geçmiş temizleniyor
                self.detection_history = []
                self.console.appendPlainText("Kayıt hafızası temizlendi.")
            self.current_plate_text = None

        # İşlenmiş frame'i arayüzde görüntüle
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(qt_image))

    def print_stats(self):
        if not self.detection_history:
            return
        
        # Hafızadaki plakaları say
        counter = Counter(self.detection_history)
        
        # En çok tespit edilen plakayı bul
        most_common_plate, count = counter.most_common(1)[0]
        
        # Tespit oranını hesapla
        detection_rate = (count / len(self.detection_history)) * 100
        
        # İstatistik metnini oluştur ve konsola yazdır
        stat_text = f"İstatistik: Son {len(self.detection_history)} okumada en çok okunan plaka: {most_common_plate} ({count} kez) - Oran: {detection_rate:.2f}%"
        self.console.appendPlainText(stat_text)

    def update_parameters(self):
        # Kontrol panelindeki parametreleri alıp ilgili modüllere aktar
        confidence = self.confidence_spin.value()
        frequency = self.scan_frequency_spin.value()
        min_hits = self.min_hits_spin.value()
        max_missing = self.max_missing_spin.value()
        history_size = self.history_size_spin.value()
        stats_frequency = self.stats_frequency_spin.value()
        show_readings = self.show_readings_checkbox.isChecked()
        
        # Kamera seçimini al ve kontrolünü yap
        selected_camera = self.camera_combo.currentData()
        
        # Eğer seçilen kamera varsa ve şu anki aktif kameradan farklıysa değiştir
        if selected_camera is not None and selected_camera != -1 and selected_camera != self.active_camera:
            if self.change_camera(selected_camera):
                self.frame_count = 0  # Kamera değiştiğinde frame sayacını sıfırla
                self.detection_history = []  # Tespit geçmişini temizle
                self.current_plate_text = None
        
        self.plate_detector.confidence_threshold = confidence
        self.plate_detector.scan_frequency = frequency
        self.tracker.min_hits = min_hits
        self.tracker.max_missing = max_missing
        self.history_size = history_size
        self.stats_frequency = stats_frequency
        self.show_readings = show_readings  # Plaka gösterme durumunu güncelle

        # Güncelleme mesajına kamera bilgisini ekle
        update_message = (
            f"Parametreler güncellendi: Kamera={self.active_camera}, Güven Eşiği={confidence}, Tarama Sıklığı={frequency}, "
            f"Min Hits={min_hits}, Max Missing={max_missing}, Kayıt Hafızası={history_size}, "
            f"İstatistik Sıklığı={stats_frequency}"
        )
        
        # Plaka okuma gösterimi hakkında bilgi ekle
        if show_readings:
            update_message += ", Her plaka okuması gösterilecek"
        else:
            update_message += ", Sadece istatistikler gösterilecek"
            
        self.console.appendPlainText(update_message)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
