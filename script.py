import argparse
from collections import defaultdict, deque
import cv2
import numpy as np
from inference.models.utils import get_roboflow_model

import supervision as sv
from rule import SheetLogger, SpeedRuleEngine


# ROI
SOURCE = np.array([[924, 285],[1088, 286], [1676, 1048], [272, 1048]], dtype=np.int32)

# Bird-view
TARGET_WIDTH = 11.61
TARGET_HEIGHT = 155.23

TARGET = np.array([
    [0.0, 0.0],
    [TARGET_WIDTH, 0.0],
    [TARGET_WIDTH, TARGET_HEIGHT],
    [0.0, TARGET_HEIGHT],
], dtype=np.float32)

LINE_P1 = (520, 754)
LINE_P2 = (1433, 749)
CROSS_COOLDOWN_SEC = 3.0


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.m = cv2.getPerspectiveTransform(
            source.astype(np.float32),
            target.astype(np.float32)
        )

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        out = cv2.perspectiveTransform(pts, self.m)
        return out.reshape(-1, 2)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_video_path", required=True, type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    fps = video_info.fps

    model = get_roboflow_model("yolov8n-640")
    byte_track = sv.ByteTrack(frame_rate=fps)

    thickness = sv.calculate_optimal_line_thickness(video_info.resolution_wh)
    text_scale = 0.45
    text_thickness = 1

    box_annotator = sv.BoxAnnotator(
        thickness=thickness,
        color_lookup=sv.ColorLookup.TRACK
    )
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=text_thickness,
        text_position=sv.Position.BOTTOM_CENTER,
        color_lookup=sv.ColorLookup.TRACK
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=fps * 2,
        position=sv.Position.BOTTOM_CENTER,
        color_lookup=sv.ColorLookup.TRACK
    )

    frame_generator = sv.get_video_frames_generator(args.source_video_path)

    polygon_zone = sv.PolygonZone(SOURCE)
    view_transformer = ViewTransformer(SOURCE, TARGET)

    line_img = np.array([LINE_P1, LINE_P2], dtype=np.float32)
    line_bird = view_transformer.transform_points(line_img)
    LINE_Y = float(np.mean(line_bird[:, 1]))
    print("LINE_Y (bird-view) =", LINE_Y)

    WINDOW_SECONDS = 1.0
    MIN_POINTS = 8
    MAX_KMH = 180

    tracks = defaultdict(lambda: deque(maxlen=int(fps * WINDOW_SECONDS)))

    prev_y = defaultdict(lambda: None)
    last_cross_time = defaultdict(lambda: 0.0)

    sheet_logger = SheetLogger()
    rule_engine = SpeedRuleEngine(sheet_logger, speed_limit=70, cooldown_sec=3)

    for frame in frame_generator:
        result = model.infer(frame)[0]
        detections = sv.Detections.from_inference(result)
        detections = detections[polygon_zone.trigger(detections)]
        detections = byte_track.update_with_detections(detections=detections)

        points = detections.get_anchors_coordinates(
            anchor=sv.Position.BOTTOM_CENTER
        )
        points = view_transformer.transform_points(points)

        labels = []
        for tracker_id, (x, y) in zip(detections.tracker_id, points):
            if tracker_id is None:
                labels.append("")
                continue

            tracks[tracker_id].append((float(x), float(y)))

            if len(tracks[tracker_id]) < MIN_POINTS:
                labels.append(f"#{tracker_id}")
                prev_y[tracker_id] = y
                continue

            arr = np.array(tracks[tracker_id], dtype=np.float32)

            y_start = arr[0, 1]
            y_end = arr[-1, 1]
            dt = (len(arr) - 1) / fps
            if dt <= 0:
                labels.append(f"#{tracker_id}")
                prev_y[tracker_id] = y
                continue

            v_mps = (y_end - y_start) / dt
            v_kmh = abs(v_mps) * 3.6

            py = prev_y[tracker_id]
            if py is not None and py < LINE_Y <= y:
                now = cv2.getTickCount() / cv2.getTickFrequency()
                if now - last_cross_time[tracker_id] > CROSS_COOLDOWN_SEC:
                    last_cross_time[tracker_id] = now
                    try:
                        rule_engine.push_if_overspeed(
                            str(tracker_id),
                            float(v_kmh)
                        )
                    except Exception as e:
                        print("[SHEET ERROR]", e)

            prev_y[tracker_id] = y

            if 0 < v_kmh < MAX_KMH:
                labels.append(f"#{tracker_id} {v_kmh:.1f} km/h")
            else:
                labels.append(f"#{tracker_id}")

        annotated = frame.copy()
        cv2.line(annotated, LINE_P1, LINE_P2, (0, 255, 0), 2)
        annotated = trace_annotator.annotate(annotated, detections)
        annotated = box_annotator.annotate(annotated, detections)
        annotated = label_annotator.annotate(annotated, detections, labels)

        cv2.imshow("Annotated Frame", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
