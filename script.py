import argparse
from collections import defaultdict, deque
import cv2
import numpy as np
from inference.models.utils import get_roboflow_model

import supervision as sv

SOURCE = np.array([[272, 1048], [924, 285], [1088, 286], [1676, 1048]], dtype=np.int32)

TARGET_WIDTH = 11.61
TARGET_HEIGHT = 155.23

TARGET = np.array([
    [0.0, 0.0],                 # TL
    [TARGET_WIDTH, 0.0],         # TR
    [TARGET_WIDTH, TARGET_HEIGHT],# BR
    [0.0, TARGET_HEIGHT],        # BL
], dtype=np.float32)


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation using Inference and Supervision"
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    model = get_roboflow_model("yolov8n-640")

    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    thickness =  sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

    frame_generator = sv.get_video_frames_generator(args.source_video_path)

    polygon_zone = sv.PolygonZone(SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    WINDOW_SECONDS = 1.0
    tracks = defaultdict(lambda: deque(maxlen=int(video_info.fps * WINDOW_SECONDS)))
    MAX_KMH = 180

    for frame in frame_generator:
        # ---------- detect ----------
        result = model.infer(frame)[0]
        detections = sv.Detections.from_inference(result)
        detections = detections[polygon_zone.trigger(detections)]
        detections = byte_track.update_with_detections(detections=detections)

        # ---------- transform ----------
        points = detections.get_anchors_coordinates(
            anchor=sv.Position.BOTTOM_CENTER
        )
        points = view_transformer.transform_points(points)

        # ---------- speed ----------
        labels = []
        for tracker_id, (x, y) in zip(detections.tracker_id, points):
            tracks[tracker_id].append((float(x), float(y)))

            if len(tracks[tracker_id]) < 5:
                labels.append(f"#{tracker_id}")
                continue

            arr = np.array(tracks[tracker_id], dtype=np.float32)
            dif = arr[1:] - arr[:-1]
            dist = np.linalg.norm(dif, axis=1)  # meters / frame
            v_kmh = np.median(dist) * video_info.fps * 3.6

            if 0 < v_kmh < MAX_KMH:
                labels.append(f"#{tracker_id} {v_kmh:.1f} km/h")
            else:
                labels.append(f"#{tracker_id}")

        # ---------- draw ----------
        annotated_frame = frame.copy()
        annotated_frame = sv.draw_polygon(
            annotated_frame, polygon=SOURCE, color=sv.Color.RED
        )
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

        cv2.imshow("Annotated Frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()