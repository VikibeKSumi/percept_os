# Percept_OS

A unified computer vision system for **Aerial Image Analysis** and **Realtime video processing** . Built with Python, Ultralytics YOLO, OpenCV, and professional practices--clean, modular, and configurable.


---

### Features

**Realtime Pipeline (edge_road)**
- Live webcam / video / RTSP processing
- YOLO26 + ByteTrack for stable object tracking
- On-screen display with black background (FPS, Unique Objects, Avg Speed)
- Full 12-metric tracking system
- Input resize support for better performance
- Speed estimation (km/h)

**Aerial Pipeline (aerial_space)**
- High-resolution image processing with SAHI slicing
- Fully configurable slice size & overlap via JSON
- Support for multiple YOLO26 models (s/m/l/x)
- Safe zero-detection handling

**General**
- Job-based configuration using JSON files
- Automatic run folders with logs, artifacts & metrics
- Rich colored logging
- Clean separation between pipelines

---

### How to Run

Sample image and video provided in sample folder.

# 0. Use your own image or video data.
In jobs/ folder
- replace with your image in aerial_demo.json
    ```
        "input": {
        "path": "sample\\ --YOUR IMAGE--"
        }
    ```
- replace with your video in edge_demo.json
    ```
        "input": {
        "path": "sample\\ --YOUR VIDEO--"
        }
    ```
  
# 1. Clone the repository (first time only)
```bash
git clone https://github.com/yourusername/percept_os.git
cd percept_os
```

# 2. Create virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate
```

# 3. Install the project (this installs all dependencies from pyproject.toml)
```bash
pip install -e .
```

# 4. Run the Pipeline
```bash
# Realtime
python -m percept_os.run jobs/edge_demo.json
# Aerial
python -m percept_os.run jobs/aerial_demo.json
```