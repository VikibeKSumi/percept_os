from pathlib import Path
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import time
import torch

def run(job: dict, ctx: dict) -> dict:
    logger = ctx["logger"]
    paths = ctx["paths"]
    timer = ctx["timer"]

    logger.info("Starting REALTIME pipeline (YOLO26 + ByteTrack + Full Metrics)")

    # ===================== CONFIG =====================
    model_name = job.get("model", {}).get("name", "yolo26m.pt")
    conf = job.get("model", {}).get("conf", 0.35)
    iou = job.get("model", {}).get("iou", 0.45)
    input_path = job.get("input", {}).get("source") or job.get("input", {}).get("path") or 0

    params = job.get("params", {})
    device = params.get("device", "cuda")
    pixels_per_meter = params.get("pixels_per_meter", 0.05)
    resize_to = params.get("resize_to", None)   # e.g. 1280

    logger.info(f"Input: {input_path} | Device: {device.upper()} | Resize: {resize_to or 'Original'}")

    # ===================== LOAD MODEL =====================
    model_path = Path(model_name)

    # Create models folder if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Final path where model should be
    model_path = models_dir / model_name

    if not model_path.exists():
        logger.info(f"Downloading {model_name}... (first time only)")
        temp_model = YOLO(model_name).save(str(model_path))

    logger.info(f"Using model: {model_path}")
    model = YOLO(str(model_path)).to(device)

    # ===================== SETUP =====================
    byte_tracker = sv.ByteTrack(
        track_activation_threshold=0.35,
        lost_track_buffer=30,
        minimum_matching_threshold=0.7,
    )
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open input: {input_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow("Percept OS - Realtime", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Percept OS - Realtime", 1280, 720)

    out_video = None
    if job.get("output", {}).get("save_video", True):
        out_path = paths.artifacts_dir / "output.mp4"
        out_video = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), orig_fps, (orig_width, orig_height))

    # ===================== METRICS VARIABLES =====================
    inference_times = []
    confidences = []
    detections_per_frame = []
    id_switches = 0
    prev_ids = set()
    peak_gpu_memory = 0

    frame_count = 0
    detections_count = 0
    unique_objects = 0
    track_history = {}          # track_id → last centroid
    total_speed_sum = 0.0
    speed_count = 0

    logger.info(f"Starting stream @ {orig_width}x{orig_height}")

    prev_time = time.time()
    while True:
        # ===================== READ FRAME =====================
        ret, frame = cap.read()
        if not ret:
            break

        # ===================== RESIZE (if needed) =====================
        if resize_to is not None and frame.shape[1] > resize_to:
            scale = resize_to / frame.shape[1]
            new_height = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (resize_to, new_height), interpolation=cv2.INTER_AREA)

        # ===================== INFERENCE + TIMING =====================
        t0 = time.perf_counter()
        results = model(frame, conf=conf, iou=iou, verbose=False)[0]
        inference_ms = (time.perf_counter() - t0) * 1000
        inference_times.append(inference_ms)

        # ===================== DETECTION + TRACKING =====================
        detections = sv.Detections.from_ultralytics(results)
        detections = byte_tracker.update_with_detections(detections)

        # ===================== METRICS COLLECTION =====================
        frame_count += 1
        detections_count += len(detections)
        detections_per_frame.append(len(detections))

        if len(detections) > 0:
            confidences.extend(detections.confidence.tolist())

        # ID switches
        current_ids = set(detections.tracker_id) if detections.tracker_id is not None else set()
        id_switches += len(current_ids - prev_ids)
        prev_ids = current_ids

        # GPU memory
        if torch.cuda.is_available():
            peak_gpu_memory = max(peak_gpu_memory, torch.cuda.max_memory_allocated() // (1024 * 1024))

        # FPS calculation
        current_time = time.time()
        fps_now = 1 / (current_time - prev_time)
        prev_time = current_time

        # ===================== SPEED ESTIMATION =====================
        if detections.tracker_id is not None:
            for i, tid in enumerate(detections.tracker_id):
                x1, y1, x2, y2 = detections.xyxy[i]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                if tid in track_history:
                    prev_cx, prev_cy = track_history[tid]
                    pixel_dist = np.hypot(cx - prev_cx, cy - prev_cy)
                    speed_mps = pixel_dist * fps_now * pixels_per_meter
                    speed_kmh = speed_mps * 3.6
                    if speed_kmh > 1.0:
                        total_speed_sum += speed_kmh
                        speed_count += 1

                track_history[tid] = (cx, cy)

        avg_speed = round(total_speed_sum / speed_count, 1) if speed_count > 0 else 0.0
        unique_objects = len(set(list(track_history.keys()) + list(current_ids)))

        # ===================== ANNOTATION + DISPLAY =====================
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        labels = [f"{model.names[int(c)]} {detections.confidence[i]:.2f}" for i, c in enumerate(detections.class_id)]
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # Black background + text
        #cv2.rectangle(annotated_frame, (10, 25), (300, 70), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"FPS: {fps_now:.1f}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        #cv2.rectangle(annotated_frame, (10, 75), (300, 115), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"Unique: {unique_objects}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        #cv2.rectangle(annotated_frame, (10, 120), (300, 160), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"Avg Speed: {avg_speed} km/h", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        if out_video:
            out_video.write(annotated_frame)

        cv2.imshow("Percept OS - Realtime", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if out_video:
        out_video.release()
    cv2.destroyAllWindows()

    # ===================== FINAL METRICS =====================
    total_time_ms = timer.ms()
    avg_fps = round(frame_count * 1000 / total_time_ms, 2) if total_time_ms > 0 else 0.0
    avg_inference_ms = round(sum(inference_times) / len(inference_times), 2) if inference_times else 0.0
    
    avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
    min_conf = round(min(confidences), 3) if confidences else 0.0
    max_conf = round(max(confidences), 3) if confidences else 0.0
    avg_dets_per_frame = round(sum(detections_per_frame) / len(detections_per_frame), 2) if detections_per_frame else 0.0

    logger.blank()
    logger.info("=== REALTIME METRICS END ===")
    logger.info(f"Frames: {frame_count} | Unique objects: {unique_objects}")
    logger.info(f"Avg FPS: {avg_fps} | Avg inference: {avg_inference_ms} ms")
    logger.info(f"Avg detections/frame: {avg_dets_per_frame} | Total detections: {detections_count}")
    logger.info(f"Avg conf: {avg_conf} (min {min_conf} – max {max_conf})")
    logger.info(f"ID switches: {id_switches} | Peak GPU memory: {peak_gpu_memory} MB")
    logger.info(f"Avg speed: {avg_speed} km/h")
    logger.ok(f"Realtime finished → {unique_objects} objects tracked @ {avg_fps} FPS")

    return {
        "frames_processed": int(frame_count),
        "unique_objects": int(unique_objects),
        "avg_fps": float(avg_fps),
        "avg_inference_ms": float(avg_inference_ms),
        "avg_detections_per_frame": float(avg_dets_per_frame),
        "total_detections": int(detections_count),
        "avg_confidence": float(avg_conf),
        "min_confidence": float(min_conf),
        "max_confidence": float(max_conf),
        "id_switches": int(id_switches),
        "peak_gpu_memory_mb": int(peak_gpu_memory),
        "avg_speed_kmh": float(avg_speed),
        "total_time_ms": float(round(total_time_ms, 2)),
        "output_video": str(paths.artifacts_dir / "output.mp4") if out_video else None
    }