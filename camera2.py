import pyrealsense2 as rs
import cv2
import numpy as np
from drawing import process_objects
from ultralytics import YOLO
import concurrent.futures
import streamlit as st
from datetime import datetime  # Import datetime


# Replace these with appropriate values or constants from your original code
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
TEXT_COLOR = (255, 255, 255)
FONT_THICKNESS = 2

JOINT_NAMES = ["noise","lf_eye","rt_eye","lf_ear","rt_ear","lf_shoulder",
            "rt_shoulder","lf_elbow","rt_elbow","lt_wrist","rt_wrist",
            "lf_hip","rt_hip","lf_knee","rt_knee","lf_ankle","rt_ankle"]

def object_detection_worker(args):
    model, classnames, color_image, depth_image, CONFID_THRESHOLD, model_type = args
    # Perform pose estimation using YOLOv8 Pose model
    results = model(color_image,conf=CONFID_THRESHOLD)
    detected_objects = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_objects, info, color_image, depth_image, classnames,JOINT_NAMES,CONFID_THRESHOLD,FONT, 
                        FONT_SCALE,TEXT_COLOR, FONT_THICKNESS,model_type) for info in results]
            for future in concurrent.futures.as_completed(futures):
                detected_objects.extend(future.result())
    
    return detected_objects, color_image, depth_image

def capture_video_d(source, width, height, model_type, model=None, classnames=None, CONFID_THRESHOLD=0.5,save_path=None):
    pipeline = rs.pipeline()
    stframe = st.empty()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 9)
    colorizer.set_option(rs.option.visual_preset, 0)
    colorizer.set_option(rs.option.histogram_equalization_enabled, 1)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    depth_image = np.zeros((height, width, 3), dtype=np.uint8)
    depth_image_normalized = np.zeros((height, width), dtype=np.uint8)
    depth_image_8bit = np.zeros((height, width, 3), dtype=np.uint8)
    depth_colormap = np.zeros((height, width, 3), dtype=np.uint8)
    combine = np.zeros((height, width*2, 3), dtype=np.uint8)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            hdr_merge = rs.hdr_merge()
            depth_frame_aligned = hdr_merge.process(depth_frame)

            threshold_filter = rs.threshold_filter()
            threshold_filter.set_option(rs.option.min_distance, 0)
            threshold_filter.set_option(rs.option.max_distance, 16)
            depth_frame_aligned = threshold_filter.process(depth_frame_aligned)

            disparity_transformer = rs.disparity_transform(True)
            depth_frame_aligned = disparity_transformer.process(depth_frame_aligned)

            spatial = rs.spatial_filter()
            spatial.set_option(rs.option.filter_magnitude, 2)
            spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
            spatial.set_option(rs.option.filter_smooth_delta, 20)
            spatial.set_option(rs.option.holes_fill, 0)
            depth_frame_aligned = spatial.process(depth_frame_aligned)

            temporal_filter = rs.temporal_filter()
            temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
            temporal_filter.set_option(rs.option.filter_smooth_delta, 20)
            depth_frame_aligned = temporal_filter.process(depth_frame_aligned)

            hole_filling = rs.hole_filling_filter(1)
            depth_frame_aligned = hole_filling.process(depth_frame_aligned)

            disparity_transformer = rs.disparity_transform(True)
            disparity_frame = disparity_transformer.process(depth_frame_aligned)

            disparity_to_depth = rs.disparity_transform(False)
            depth_frame_aligned = disparity_to_depth.process(disparity_frame)

            color_image = np.asanyarray(color_frame.get_data())

            depth_colormap[:] = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            depth_image_normalized = cv2.normalize(depth_colormap[:, :, 0], None, 0, 255, norm_type=cv2.NORM_MINMAX)
            depth_image_8bit = depth_image_normalized.astype(np.uint8)
            depth_image[:] = cv2.applyColorMap(depth_image_8bit, cv2.COLORMAP_JET)

            if model is not None and classnames is not None:
                args = (model, classnames,color_image, depth_image,CONFID_THRESHOLD,model_type)
                future = executor.submit(object_detection_worker, args)
                detected_objects, color_image, depth_image = future.result()

            combine[:, :width, :] = color_image
            combine[:, width:, :] = depth_image

            # combined_image = np.hstack((color_image, depth_image))
            # cv2.imshow('RGB and Depth', combine)
            
            stframe.image(combine, channels="BGR")

            # Save the frame if the button was pressed
            if st.session_state.grab_image_flag:
                if save_path:  # Check if save_path is provided
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_path = os.path.join(save_path, "grabbed_image.png")
                    img_path = os.path.join(save_path, f"grabbed_image_{timestamp}.png")
                    cv2.imwrite(img_path, cv2.cvtColor(combine, cv2.COLOR_RGB2BGR))
                    st.success(f"Image saved at: {img_path}")  # Notify user
                st.session_state.grab_image_flag = False

            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

