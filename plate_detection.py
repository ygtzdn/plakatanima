import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR

class PlateDetector:
    def __init__(self, yolo_model_path, confidence_threshold=0.7, scan_frequency=5):
        """
        :param yolo_model_path: YOLO modelinin dosya yolu
        :param confidence_threshold: YOLO tespitlerinde güven eşiği
        :param scan_frequency: Kaç frame’de bir tespit yapılacağı
        """
        self.model = YOLO(yolo_model_path)
        self.confidence_threshold = confidence_threshold
        self.scan_frequency = scan_frequency
        self.ocr = PaddleOCR(use_angle_cls=True, lang='tr')  # PaddleOCR başlatılıyor

    def detect(self, frame, frame_count):
        """
        Belirli frame’lerde YOLO tespiti yapar ve tespit edilen kutuları döner.
        :param frame: İşlenecek görüntü
        :param frame_count: Geçerli frame sayısı
        :return: [(x1, y1, x2, y2), ...] listesi
        """
        if frame_count % self.scan_frequency != 0:
            return []
        results = self.model(frame, conf=self.confidence_threshold)
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Her satır: [x1, y1, x2, y2]
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                h, w = frame.shape[:2]
                # Görüntü sınırları içinde kalması için düzeltme:
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue
                detections.append((x1, y1, x2, y2))
        return detections

    def ocr_plate(self, frame, bbox):
        """
        Verilen bounding box bölgesinde OCR işlemi yapar.
        :param frame: Orijinal görüntü
        :param bbox: (x1, y1, x2, y2)
        :return: Tespit edilen plaka metni (string) veya boş string
        """
        x1, y1, x2, y2 = bbox
        plate_region = frame[y1:y2, x1:x2]
        result = self.ocr.ocr(plate_region, cls=True)
        plate_texts = []
        for line in result:
            if line:
                for word_info in line:
                    if len(word_info) >= 2:
                        text = word_info[1][0].upper()  # Metni al ve büyük harfe çevir
                        clean_text = ''.join(c for c in text if c.isalnum())
                        if clean_text:
                            plate_texts.append(clean_text)
        if plate_texts:
            return plate_texts[0]
        return ""
