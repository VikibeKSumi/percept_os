# Percept_OS

**A unified real-time & aerial computer vision runtime for edge robotics and drone applications.**

Percept_OS is a lightweight, modular system that brings together two powerful vision pipelines in one framework:

- **Aerial Pipeline** → High-resolution image analysis (drone, satellite, top-down) using **SAHI + YOLO26-OBB**
- **Realtime Pipeline** → Live video / webcam / RTSP detection & tracking using **YOLO26 + ByteTrack**

Built with simplicity and performance in mind, it is perfect for learning systems architecture, pipeline design, and running computer vision on resource-constrained hardware (laptop + GTX 1650 Ti).

---

### ✨ Features

- Modular and easy-to-understand pipeline architecture
- Full SAHI support for high-resolution aerial images
- Real-time object detection + robust tracking (ByteTrack)
- YOLO26 OBB (Oriented Bounding Box) ready
- Rich, professional metrics for both pipelines
- Everything configurable via simple JSON files
- Batch processing support for aerial images
- GPU optimized with clean logging and artifact management

---

### 🛠 Tech Stack

- **Core**: Ultralytics YOLO26, Supervision, SAHI
- **Vision**: OpenCV, PyTorch
- **Others**: Rich (beautiful logging), Docker-ready, Git

---

### 📈 Performance Metrics

Percept_OS automatically tracks rich, professional metrics for both pipelines. All results are saved in the `runs/` folder for every execution.

#### Aerial Pipeline
- **Performance**: Total images, Time per image, Effective FPS
- **Detection Quality**: Total detections, Avg detections per image, Confidence (avg/min/max)
- **SAHI Analysis**: Total tiles used, Avg tiles per image
- **Scene Understanding**: Detections per class, Number of unique classes
- **Resource Usage**: Peak GPU memory

#### Realtime Pipeline
- **Performance**: Average FPS, Inference time per frame, Total frames
- **Detection Quality**: Average / Min / Max confidence, Detections per frame
- **Tracking Quality**: Unique objects tracked, ID switches
- **Movement**: Average object speed (km/h)
- **Resource Usage**: Peak GPU memory

---

### ⚙️ Configuration (Easy Tuning)

All important parameters are controlled from JSON files.

#### Aerial default (`jobs/aerial_demo.json`)
```json
    "model": {
    "name": "yolo26l-obb.pt",
    "conf": 0.45
    },
    "params": {
    "device": "cuda",
    "sahi_slice_size": 512,
    "sahi_overlap": 0.15
    }
```
#### Realtime default (`jobs/edge_demo.json`)
```JSON
    "model": {
    "name": "yolo26m.pt",
    "conf": 0.50,
    "iou": 0.45
    },
    "params": {
    "device": "cuda",
    "resize_to": 1280,
    "pixels_per_meter": 0.05
    }
```

### 🚀 How to Run

#### 1. Clone the Repository
```Bash
    git clone https://github.com/yourusername/percept_os.git
    cd percept_os
```

#### 2. Create Virtual Environment and activate it (Recommended)
```Bash
    python -m venv venv
    venv\Scripts\activate
```

#### 3. Install Dependencies
```Bash
    pip install -e .
```

#### 4. Add Your Own Image or Video

For Aerial (Images):
Edit jobs/aerial_demo.json and put your image path:
```JSON
    "input": {
    "path": "samples/--HERE--.jpg" // ← your image file
    }
```

For Realtime (Video / Webcam):
Edit jobs/edge_demo.json and put your video path:
```JSON
    "input": {
    "path": "samples/--HERE--.mp4"     // ← your video file; use '0' for webcam
    }
```

#### 5. Install dependencies
```bash
    pip install -e .
```

#### 6. Run Realtime (webcam or video)
```Bash
    python -m percept_os.run jobs/edge_demo.json
```

#### 7. Run Aerial (single image or folder)
```Bash
    python -m percept_os.run jobs/aerial_demo.json
```


### 📁 Project Structure
```text
textpercept_os/
├── run.py                    # Main entry point
├── pipelines/
│   ├── aerial.py             # Aerial + SAHI pipeline
│   └── realtime.py           # Realtime + tracking pipeline
├── core/
│   ├── logger.py
│   └── utils.py
├── jobs/                     # ← Put your demo JSONs here
├── samples/                  # Test images & videos
├── runs/                     # All results + metrics saved here
└── models/                   # Auto-downloaded YOLO models
```

### 🛣 Future Roadmap
- Model fine-tuning guide (for better people/animal detection in aerial)
- Instance & semantic segmentation support
- Multi-camera / multi-stream realtime
- TensorRT + FP16 optimization for higher FPS
- Web dashboard for live monitoring