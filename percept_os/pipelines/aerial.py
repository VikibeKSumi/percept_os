from pathlib import Path
import cv2
import numpy as np
import supervision as sv
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image
import time

def run(job: dict, ctx: dict) -> dict:
    logger = ctx["logger"]
    paths = ctx["paths"]
    timer = ctx["timer"]

    logger.info("Starting AERIAL pipeline (SAHI + YOLO11)")

    # model and path
    model_name = job.get("model", {}).get("name", "yolo26m.pt")
    conf = job.get("model", {}).get("conf", 0.35)
    input_path = job.get("input", {}).get("path")

    # setting device and SAHI parameters
    params = job.get("params", {})
    device = params.get("device", "cuda")
    slice_size = params.get("sahi_slice_size", 512)
    overlap = params.get("sahi_overlap", 0.1)

    if not input_path:
        raise ValueError("No path provided in job.input")

    
    if not Path(model_name).exists():
        logger.info(f"Downloading {model_name}... (first time only, saved in root folder)")

    logger.info(f"Using model: {model_name}")

    # YOLO(model).info(verbose=True)  # <- uncomment this line to see full table
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",  # SAHI only accepts "yolov8" as 'model_type'
        model_path=model_name,
        confidence_threshold=conf,
        device=device,
    )

    source_path = Path(input_path)
    image_paths = [source_path] if source_path.is_file() else list(source_path.glob("*.jpg")) + \
                  list(source_path.glob("*.jpeg")) + list(source_path.glob("*.png"))

    if not image_paths:
        logger.warn(f"No images found for path: {source_path}")

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    results = []
    total_conf = 0.0
    processed_images = 0
    skipped_unreadable_images = 0

    for img_path in image_paths:
        logger.info(f"Processing {img_path.name}")
        try:
            image = read_image(str(img_path))
        except Exception as e:
            skipped_unreadable_images += 1
            logger.warn(f"Skipping unreadable image '{img_path.name}': {e}")
            continue

        processed_images += 1
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
        sahi_ms = (time.perf_counter() - t0) * 1000

        object_preds = result.object_prediction_list

        if len(object_preds) == 0:
            detections = sv.Detections.empty()
            logger.warn("No objects detected in this image")
        else:
            xyxy = np.array([box.bbox.to_xyxy() for box in object_preds])
            confidence = np.array([box.score.value for box in object_preds])
            class_id = np.array([box.category.id for box in object_preds])

            detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id,
            )

        # ==================Annotate========================
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=[f"{box.category.name} {box.score.value:.2f}" for box in object_preds],
        )

        out_img = paths.artifacts_dir / f"annotated_{img_path.stem}.jpg"
        cv2.imwrite(str(out_img), annotated_frame)

        detections_list = [{
            "class": box.category.name,
            "confidence": float(box.score.value),
            "bbox": [int(x) for x in box.bbox.to_xyxy()],
        } for box in object_preds]

        results.append({
            "image": img_path.name,
            "detections": detections_list,
            "count": len(detections_list),
        })

        total_conf += sum(box.score.value for box in object_preds)

    # ==============Metrics======================
    total_time_ms = timer.ms()
    num_images = len(image_paths)
    total_detections = sum(r["count"] for r in results)
    avg_conf = round(total_conf / total_detections, 3) if total_detections > 0 else 0.0
    avg_per_processed_image = round(total_detections / processed_images, 2) if processed_images > 0 else 0

    metrics_summary = {
        "total_images": num_images,
        "processed_images": processed_images,
        "skipped_unreadable_images": skipped_unreadable_images,
        "total_detections": total_detections,
        "avg_detections_per_image": avg_per_processed_image,
        "overall_avg_confidence": avg_conf,
        "total_time_ms": round(total_time_ms, 2),
        "output_dir": str(paths.artifacts_dir),
    }

    logger.info("=== METRICS END ===")
    logger.info(f"Processed imgs : {processed_images}")
    logger.info(f"Skipped imgs   : {skipped_unreadable_images}")
    logger.info(f"Total objects  : {total_detections}")
    logger.info(f"Avg conf       : {avg_conf}")
    logger.info(f"SAHI inference: {sahi_ms:.1f} ms")
    logger.ok(f"Aerial pipeline finished - {total_detections} objects detected")
    
    return metrics_summary
