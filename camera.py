# camera.py
import cv2
import os
import streamlit as st
import concurrent.futures
from ultralytics import YOLO
from drawing import process_objects1
from read_json import generate_report
from datetime import datetime

# Constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
TEXT_COLOR = (255, 255, 255)
FONT_THICKNESS = 2

JOINT_NAMES = ["noise", "lf_eye", "rt_eye", "lf_ear", "rt_ear", "lf_shoulder",
               "rt_shoulder", "lf_elbow", "rt_elbow", "lt_wrist", "rt_wrist",
               "lf_hip", "rt_hip", "lf_knee", "rt_knee", "lf_ankle", "rt_ankle"]

def object_detection_worker(args):
    model, classnames, color_image, confid_threshold, model_type = args
    results = model(color_image, conf=confid_threshold)
    detected_objects = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_objects1, info, color_image, classnames, JOINT_NAMES, confid_threshold, FONT, 
                                   FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, model_type) for info in results]
        for future in concurrent.futures.as_completed(futures):
            detected_objects.extend(future.result())
    
    return detected_objects, color_image

def capture_video(source, width, height, model_type, model=None, classnames=None, confid_threshold=0.5, save_path=None):
    cap = cv2.VideoCapture(source)
    stframe = st.empty()  # Placeholder for video frame
    detected_objects_placeholder = st.empty()  # Single placeholder for detected objects
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to retrieve frame.")
            break

        if model is not None and classnames is not None:
            # Object detection
            args = (model, classnames, frame, confid_threshold, model_type)
            future = executor.submit(object_detection_worker, args)
            detected_objects, frame = future.result()

            # Update detected objects in the placeholderst.session_state["job_function"]
            if detected_objects:  # Check if there are any detected objects
                detection_info = detected_objects[0]
                positions = [det[-1] for det in detection_info]
                position_string = ', '.join(positions)  # Join list into a comma-separated string
                detected_objects_placeholder.write(f"<b style='color: red;'>Detected objects: {position_string}</b>", unsafe_allow_html=True)
            else:
                detected_objects_placeholder.write("Detected objects: No objects detected")

        # Display the video frame
        frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")

        # Save the frame if the grab image flag is set
        if st.session_state.get("grab_image_flag", False):
            if save_path:  # Check if save_path is provided
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_path = os.path.join(save_path, f"grabbed_image_{timestamp}.png")
                cv2.imwrite(img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                st.success(f"Image saved at: {img_path}")  # Notify user
            st.session_state.grab_image_flag = False

        # Save the frame if the grab image flag is set
        if st.session_state.get("prompt_flag", False):
            job_function = st.session_state.get("job_function", "Not Selected")
            report = generate_report(job_function, position_string)
            st.write(f"<b style='color: red;'>Generate a detection report based on the following detected objects: {report}</b>", unsafe_allow_html=True)
            st.session_state.prompt_flag = False

    # Release resources
    cap.release()
    executor.shutdown()
