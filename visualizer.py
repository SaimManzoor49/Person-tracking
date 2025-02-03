# visualizer.py
import cv2

class Visualizer:
    @staticmethod
    def draw_results(frame, results):
        for bbox, obj_id in results:
            x1, y1, x2, y2 = map(int, bbox)
            if obj_id is not None:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {obj_id}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, "Waiting...", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        cv2.putText(frame, f"Detections: {len([r for r in results if r[1] is not None])}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return frame