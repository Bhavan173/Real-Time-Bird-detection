from ultralytics import YOLO
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("Real-Time Bird Detection App")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

class YOLOVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, conf=0.6)
        annotated_frame = results[0].plot()
        return annotated_frame
    
webrtc_streamer(
    key="yolo-stream",
    video_processor_factory=YOLOVideoTransformer
)