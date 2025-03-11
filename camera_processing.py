from collections import Counter

class SimpleSort:
    """
    Basitleştirilmiş bir SORT takip algoritması.
    IoU tabanlı eşleştirme yaparak tespitleri track eder.
    """
    def __init__(self, max_missing=5, min_hits=3, iou_threshold=0.3):
        self.tracks = {}   # track_id -> {'bbox': (x1, y1, x2, y2), 'hits': int, 'missing': int}
        self.next_id = 0
        self.max_missing = max_missing
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

    def update(self, detections):
        """
        Gelen tespitleri mevcut track’lerle eşleştirir.
        :param detections: [(x1, y1, x2, y2), ...]
        :return: Onaylanmış (confirmed) track’ların listesi [(track_id, bbox), ...]
        """
        updated_tracks = {}
        used_detections = set()
        confirmed_tracks = []

        for track_id, track in self.tracks.items():
            best_iou = 0
            best_det = None
            best_det_idx = None
            for i, det in enumerate(detections):
                if i in used_detections:
                    continue
                iou = self.compute_iou(track['bbox'], det)
                if iou > best_iou:
                    best_iou = iou
                    best_det = det
                    best_det_idx = i
            if best_iou > self.iou_threshold:
                track['bbox'] = best_det
                track['hits'] += 1
                track['missing'] = 0
                used_detections.add(best_det_idx)
            else:
                track['missing'] += 1

            if track['hits'] >= self.min_hits:
                confirmed_tracks.append((track_id, track['bbox']))
            if track['missing'] <= self.max_missing:
                updated_tracks[track_id] = track

        for i, det in enumerate(detections):
            if i not in used_detections:
                updated_tracks[self.next_id] = {'bbox': det, 'hits': 1, 'missing': 0}
                self.next_id += 1

        self.tracks = updated_tracks
        return confirmed_tracks

    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
