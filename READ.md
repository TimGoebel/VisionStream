# Video Stream with Continuous Input using Streamlit

This project provides a real-time video streaming application built with Streamlit, utilizing YOLO for object detection, pose estimation, and segmentation models. The app supports various video sources, including webcams, RTSP streams, and uploaded videos, with an intuitive sidebar for settings adjustments.

## Features

- **Real-time Video Streaming**: Capture video from various sources such as webcam, RTSP streams, or video files.
- **Dynamic Model Loading**: Choose and load pre-trained models for object detection, pose estimation, or segmentation.
- **Adjustable Settings**: Configure video source, model type, resolution, and confidence threshold directly in the UI.
- **Image Capture**: Option to grab and save frames during the video stream.
- **Text Input**: Interactive text inputs for automated notes and prompt entries during streaming.

## Demo

<img src="path/to/demo.gif" alt="Demo GIF" width="700"/>

## Setup

### Prerequisites

- Python 3.8+
- Install required packages using the provided `requirements.txt`

```bash
pip install -r requirements.txt

git clone https://github.com/yourusername/video-streaming-app.git
cd video-streaming-app

Set up the necessary models and video sources. Follow instructions in the Configuration section below.

Configuration
YOLO Models: Pre-trained YOLO models can be downloaded from Ultralytics YOLO repository.
RTSP Stream URL: Ensure the RTSP stream URL is accessible for your network.

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

