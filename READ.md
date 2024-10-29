# Streamlit Video Analyzer

This repository hosts a real-time video streaming and analysis app using Streamlit and YOLO. The app provides options for live video streams from various sources, and integrates YOLO for real-time object detection, pose estimation, and segmentation.

## Table of Contents


- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Scripts Overview](#scripts-overview)
- [Link to dataset](#Link-to-dataset)

## Installation
1. Clone the repository:

    ```bash
  git clone https://github.com/username/Streamlit-Video-Analyzer.git
  cd Streamlit-Video-Analyzer
  ```

## Setup

### Prerequisites

- Python 3.8+
- Install required packages using the provided `requirements.txt`

```bash
pip install -r requirements.txt

Installation
Clone the repository:

git clone https://github.com/yourusername/video-streaming-app.git
cd video-streaming-app


Set up the necessary models and video sources. Follow instructions in the Configuration section below.

Configuration
YOLO Models: Pre-trained YOLO models can be downloaded from Ultralytics YOLO repository.
RTSP Stream URL: Ensure the RTSP stream URL is accessible for your network.

Usage
Run the application with the following command:
streamlit run app.py


Available Settings
Video Source: Select from options like d435, web, video, and RTSP.
Model Source: Choose between none, object_detection, pose_estimation, or segmentation.
Resolution: Adjust the width and height of the video stream.
Confidence Threshold: Set the threshold level for detection confidence.
Capturing Images
To grab an image during video streaming, press the "Grab Image" button. The image will be saved temporarily, and a preview will be shown in the app.

Directory Structure
project-root/
│
├── camera/
│   ├── capture_video.py
│   ├── capture_video_d.py
│   └── ...
├── models/
│   ├── yolov5s.pt (sample model)
│   └── ...
├── app.py
└── requirements.txt



Acknowledgments
Streamlit for building an amazing interactive web app framework.
Ultralytics YOLO for their powerful object detection models.
OpenCV for video processing and image capture utilities.

