from pathlib import Path
import time
import torch
import numpy as np

import cv2
import supervision as sv
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image



def run(job: dict, ctx: dict) -> dict:
    """
    Aerial Pipeline - High-resolution image detection using SAHI + YOLO26
    """
    logger.info("Starting AERIAL pipeline (SAHI + YOLO26)")

    logger =    ["logger"]
    paths = ctx["paths"]
    timer = ctx["timer"]

    # ===================== CONFIG =====================
    model_name = job.get("model", {}).get("name", "yolo26m.pt")
    conf = job.get("model", {}).get("conf", 0.35)

    # SAHI settings from JSON (configurable)
    params = job.get("params", {})
    slice_size = params.get("sahi_slice_size", 512)
    overlap = params.get("sahi_overlap", 0.10)
    device = params.get("device", "cpu")

    logger.info(f"Model: {model_name} | Device: {device.upper()}")
    logger.info(f"SAHI: {slice_size}x{slice_size} tiles | Overlap: {int(overlap*100)}% | Conf: {conf}")

    # ===================== MODEL LOADING =====================
    # Create models folder if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Final path where model should be
    model_path = models_dir / model_name

    # Download to our models/ folder if missing
    if not model_path.exists():
        logger.info(f"Downloading {model_name} → models/ folder (first time only)")
        temp_model = YOLO(model_name)           # downloads from Ultralytics
        temp_model.save(str(model_path))

    logger.info(f"Using model: {model_path}")       

    # SAHI wrapper for YOLO
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=str(model_path),
        confidence_threshold=conf,
        device=device,
    )

    # ===================== SOURCE PREPARATION / LOAD ALL IMAGES (BATCH) =====================
    source = job.get("input", {}).get("path")
    if not source:
        raise ValueError("No source provided in job.input")

    source_path = Path(source)
    # If it's a folder → batch mode (this is the key part)
    image_paths = [source_path] if source_path.is_file() else \
                list(source_path.glob("*.jpg")) + list(source_path.glob("*.jpeg")) + \
                list(source_path.glob("*.png")) + list(source_path.glob("*.avif"))

    if not image_paths:
        raise ValueError(f"No valid images found in {source}")

    logger.info(f"Found {len(image_paths)} image(s) to process")

    # ===================== PREPARE ANNOTATORS =====================
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # ===================== METRICS INITIALIZATION =====================
    total_detections = 0
    all_confidences = []
    detections_per_image = []
    total_sahi_tiles = 0
    class_counter = {}          # For detections per class
    inference_times = []

    # ===================== PROCESS EVERY IMAGE/ MAIN LOOP =====================
    for img_path in image_paths:
        logger.info(f"Processing: {img_path.name}")

        # Read image
        image = read_image(str(img_path))

        # ===================== SAHI INFERENCE =====================
        t0 = time.perf_counter()
        result = get_sliced_prediction(
            image=image,
            detection_model=detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            postprocess_class_agnostic=True,
        )
        inference_ms = (time.perf_counter() - t0) * 1000
        inference_times.append(inference_ms)

        total_sahi_tiles += len(result.object_prediction_list)  # rough tile count

        # ===================== SAFE DETECTION HANDLING =====================
        object_preds = result.object_prediction_list
        # Safe handling for zero detections
        if len(object_preds) == 0:
            detections = sv.Detections.empty()
            logger.warn(f"No objects detected in {img_path.name}")
        else:
            xyxy = np.array([box.bbox.to_xyxy() for box in object_preds])
            confidence = np.array([box.score.value for box in object_preds])
            class_id = np.array([box.category.id for box in object_preds])

            detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id,
            )

        # ===================== METRICS COLLECTION =====================
        num_dets = len(object_preds)
        total_detections += num_dets
        detections_per_image.append(num_dets)

        if num_dets > 0:
            all_confidences.extend([box.score.value for box in object_preds])

            # Count per class
            for box in object_preds:
                cls_name = box.category.name
                class_counter[cls_name] = class_counter.get(cls_name, 0) + 1

        # ===================== ANNOTATION & SAVE =====================
        # Save annotated image
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
        labels = [f"{box.category.name} {box.score.value:.2f}" for box in object_preds]
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

        out_img = paths.artifacts_dir / f"annotated_{img_path.stem}.jpg"
        cv2.imwrite(str(out_img), annotated_frame)

    # ===================== FINAL METRICS CALCULATION =====================
    total_time_ms = timer.ms()
    num_images = len(image_paths)
    avg_dets_per_image = round(total_detections / num_images, 2) if num_images > 0 else 0
    avg_conf = round(sum(all_confidences) / len(all_confidences), 3) if all_confidences else 0.0
    min_conf = round(min(all_confidences), 3) if all_confidences else 0.0
    max_conf = round(max(all_confidences), 3) if all_confidences else 0.0
    avg_time_per_image = round(total_time_ms / num_images, 2) if num_images > 0 else 0
    effective_fps = round(num_images * 1000 / total_time_ms, 2) if total_time_ms > 0 else 0
    avg_tiles_per_image = round(total_sahi_tiles / num_images, 1) if num_images > 0 else 0
    peak_gpu_memory = torch.cuda.max_memory_allocated() // (1024 * 1024) if torch.cuda.is_available() else 0

    # ===================== FINAL LOGGING =====================
    logger.blank()
    logger.info("=== AERIAL METRICS END ===")
    logger.info(f"Total images: {num_images}")
    logger.info(f"Total detections: {total_detections} | Avg per image: {avg_dets_per_image}")
    logger.info(f"Avg confidence: {avg_conf} (min {min_conf} – max {max_conf})")
    logger.info(f"Total SAHI tiles: {total_sahi_tiles} | Avg tiles per image: {avg_tiles_per_image}")
    logger.info(f"Batch process time: {round(total_time_ms, 2)} ms | Avg single inference time: {avg_time_per_image} ms")
    logger.info(f"Effective FPS: {effective_fps}")
    logger.info(f"Peak GPU memory: {peak_gpu_memory} MB")
    logger.info(f"Classes detected: {len(class_counter)}")
    logger.info(f"Detections per class: {dict(sorted(class_counter.items(), key=lambda x: x[1], reverse=True))}")
    logger.info(f"Output saved in: {paths.artifacts_dir}")

    logger.ok(f"Aerial pipeline finished — {total_detections} objects detected across {num_images} images")

    # ===================== RETURN ALL METRICS =====================
    return {
        "total_images": num_images,
        "total_detections": total_detections,
        "avg_detections_per_image": avg_dets_per_image,
        "avg_confidence": avg_conf,
        "min_confidence": min_conf,
        "max_confidence": max_conf,
        "total_sahi_tiles": total_sahi_tiles,
        "avg_tiles_per_image": avg_tiles_per_image,
        "total_time_ms": round(total_time_ms, 2),
        "avg_time_per_image_ms": avg_time_per_image,
        "effective_fps": effective_fps,
        "peak_gpu_memory_mb": peak_gpu_memory,
        "detections_per_class": dict(class_counter),
        "output_dir": str(paths.artifacts_dir),
    }