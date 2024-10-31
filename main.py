# main.py
import streamlit as st
import tempfile
import os
from camera import capture_video
from camera2 import capture_video_d
from ultralytics import YOLO
import cv2

class VideoSettings:
    def __init__(self):
        self.options_video = ['d435', 'web', 'video', 'rstp']
        self.type_model = ['none', 'Object_detection', 'pose_estimation', 'segmentation']
        self.source_mapping = {'d435': 0, 'web': 0, 'video': '', 'rstp': ''}
        self.width = 640
        self.height = 480
        self.confidence_threshold = 0.5
        self.model_type = 'none'
        self.selected_source = 'web'
        self.rtsp_url = ''
        self.save_path = ""  # Initialize save_path here
        self.save_path_options = list(filter(lambda x: os.path.isdir(x), [os.path.join(os.getcwd(), d) for d in os.listdir(os.getcwd())]))
    
    def display_sidebar(self):
        st.sidebar.header("Settings")
        self.selected_source = st.sidebar.selectbox("Video Source", self.options_video)
        # st.sidebar.write(f"Source Selected: {self.selected_source}")
        
        self.model_type = st.sidebar.selectbox("Model Source", self.type_model)
        # st.sidebar.write(f"Model Selected: {self.model_type}")

        self.width = st.sidebar.number_input("Resolution Width", min_value=100, max_value=4096, value=self.width)
        self.height = st.sidebar.number_input("Resolution Height", min_value=100, max_value=4096, value=self.height)

        self.confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=self.confidence_threshold, step=0.01)
        # st.sidebar.write(f"Confidence Threshold: {self.confidence_threshold}")
        self.save_path = st.sidebar.selectbox("Select Image Save Path", self.save_path_options)


    def get_video_source(self):
        if self.selected_source == 'video':
            uploaded_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    self.source_mapping['video'] = tmp_file.name
                st.sidebar.write(f"Selected video file: {uploaded_file.name}")
        elif self.selected_source == 'rstp':
            self.rtsp_url = st.sidebar.text_input("Enter RTSP Stream URL", placeholder="rtsp://your.rtsp.url")
            st.sidebar.write(f"RTSP URL: {self.rtsp_url}")
            self.source_mapping['rstp'] = self.rtsp_url
        
        return self.source_mapping[self.selected_source]


class ModelLoader:
    def __init__(self):
        self.model = None
        self.classnames = None

    def load_model_and_labels(self, model_type):
        if model_type != 'none':
            uploaded_model = st.sidebar.file_uploader("Choose a model file", type=["pt"])
            uploaded_label = st.sidebar.file_uploader("Choose a label file", type=["txt"])
            if uploaded_model:
                self._load_model(uploaded_model)
            if uploaded_label:
                self._load_labels(uploaded_label)

    def _load_model(self, uploaded_model):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_model_file:
            tmp_model_file.write(uploaded_model.read())
            model_path = tmp_model_file.name
        st.sidebar.write(f"Selected model file: {uploaded_model.name}")
        self.model = YOLO(model_path)

    def _load_labels(self, uploaded_label):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_label_file:
            tmp_label_file.write(uploaded_label.read())
            label_path = tmp_label_file.name
        st.sidebar.write(f"Selected label file: {uploaded_label.name}")
        with open(label_path, 'r') as f:
            self.classnames = f.read().splitlines()


class App:
    def __init__(self):
        self.video_settings = VideoSettings()
        self.model_loader = ModelLoader()
        # self.video_placeholder = st.empty()

        # Initialize session state for grabbing image and streaming
        if "grab_image_flag" not in st.session_state:
            st.session_state.grab_image_flag = False
        # if "img_path" not in st.session_state:
        #     st.session_state.img_path = None
        if "is_streaming" not in st.session_state:
            st.session_state.is_streaming = False

    def run(self):
        st.title("Video Stream with Continuous Input")
        self.add_custom_css()
        self.video_settings.display_sidebar()
        self.model_loader.load_model_and_labels(self.video_settings.model_type)
        self.start_streaming_controls()

    def add_custom_css(self):
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(to bottom, #00f0ff, #000080);
        }
        </style>
        """, unsafe_allow_html=True)

    def start_streaming_controls(self):
        col1, col2 = st.columns(2)
        if col1.button("Start Video"):
            st.session_state.is_streaming = True
        if col2.button("Grab Image"):
            st.session_state.grab_image_flag = True

        if st.session_state.is_streaming:
            source = self.video_settings.get_video_source()
            self.stream_video(source)
        

    def stream_video(self, source):
        # Capture video frame based on selected source
        if self.video_settings.selected_source == 'd435':
            frame = capture_video_d(source, self.video_settings.width, self.video_settings.height, 
                                    self.video_settings.model_type, self.model_loader.model, 
                                    self.model_loader.classnames, self.video_settings.confidence_threshold)
            
        else:
            frame = capture_video(source, self.video_settings.width, self.video_settings.height, 
                                  self.video_settings.model_type, self.model_loader.model, 
                                  self.model_loader.classnames, self.video_settings.confidence_threshold,
                                  self.video_settings.save_path)
 

if __name__ == "__main__":
    app = App()
    app.run()