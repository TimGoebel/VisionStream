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
