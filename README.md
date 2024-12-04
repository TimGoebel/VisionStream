# Streamlit Video Analyzer

This repository hosts a real-time video streaming and analysis app using Streamlit and YOLO. The app provides options for live video streams from various sources, and integrates YOLO for real-time object detection, pose estimation, and segmentation.

## Table of Contents


- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Scripts Overview](#scripts-overview)

## Installation
1. Clone the repository:

    ```bash
  git clone https://github.com/username/Streamlit-Video-Analyzer.git
  cd Streamlit-Video-Analyzer
  ```


2. Install the required Python packages:

  ```bash
  pip install -r requirements.txt
  ```
## Usage
To start the app, ensure that you have the correct model files and video sources. You can run the app as follows:

### Running the Pipeline

Execute the following command to start the pipeline:
```bash
streamlit run main.py
```

### Arguments

- `source`: Select a video source from `webcam`, `video files`, or `RTSP stream`.
- `model_type` : Choose the model type for video analysis, including `object detection`, `pose estimation`, or `segmentation`.

## Directory Structure
The project’s directory structure is as follows:
```
Streamlit-Video-Analyzer/
│
├── camera/
│   ├── capture_video.py
│   ├── capture_video_d.py
│   └── ...
├── models/
│   ├── yolov8.pt
│   ├── pose_estimationv8.pt
│   ├── Segmentationv8.pt
│
├── class/
│   ├── labels.txt
│   └── ...
├── main.py
├── read_json.py
├── prompt.json
├── requirements.txt
└── README.md
```

## Scripts Overview
- **`main.py`: The main script that starts the Streamlit app and loads the interface.
- **`capture_video.py`: Contains the function to capture video from standard sources.
- **`capture_video_d.py`: Contains the function to capture video from a specific camera type (e.g., Depth Camera).
- **`draw.py`: draws the inferencing.
- **`read_json.py`: reads in json and over writes two locations
- **`prompt.json`: is the prompt
- **`models/': Directory for storing pre-trained YOLO models.
- **`classes/': Directory for storing text files of classes.

## Scripts Overview

- **`Video Source`: Choose between webcam, RTSP stream, and video files.
- **`image`: saves image to a direstor
- **`Model Type`: Select the type of analysis model (Object Detection, Pose Estimation, or Segmentation).
- **`Resolution`: Customize resolution for the video stream.
- **`Confidence Threshold`: Set a confidence threshold for the model predictions.
- **`Classes`: Select the class text file
