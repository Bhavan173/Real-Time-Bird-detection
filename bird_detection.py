from ultralytics import YOLO
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av


st.title("Real-Time Bird Detection App")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

@st.fragment(run_every=1)
def detection_status():
    if ctx.video_processor:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "🐦 Birds Detected",
                ctx.video_processor.count
            )

        with col2:
            st.metric(
                "🎯 Confidence",
                f"{ctx.video_processor.lastconf:.2f}"
            )

        with col3:
            st.metric(
                "🔍 Detection",
                "Yes" if ctx.video_processor.detected else "No"
            )

class YOLOVideoTransformer(VideoTransformerBase):

    def __init__(self):
        self.detected = False
        self.count = 0
        self.lastconf = 0.0

    def recv(self, frame:av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, conf=0.6,imgsz=512,verbose=False)

        self.detected = len(results[0].boxes)>0
        if self.detected:
            self.count = len(results[0].boxes)
            self.lastconf = float(results[0].boxes.conf.max())
        else:
            self.count = 0
            self.lastconf = 0.0

        annotated_frame = results[0].plot()
        annotated_frame = av.VideoFrame.from_ndarray(annotated_frame,format="bgr24")
        annotated_frame.pts = frame.pts
        annotated_frame.time_base = frame.time_base
        return annotated_frame
    
ctx = webrtc_streamer(
    key="yolo_stream",
    video_processor_factory=YOLOVideoTransformer,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
    media_stream_constraints={
        "video": True,
        "audio": False,
    }
)

detection_status()