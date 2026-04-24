import streamlit as st
from detect import run_detection

st.set_page_config(page_title="Feeding Behavior Detection", layout="wide")

st.title("🐄 Feeding Behavior Detection (YOLOv8)")
st.write("Upload a video to detect feeding-related objects.")

uploaded_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:
    st.video(uploaded_file)

    st.info("Running detection... please wait ⏳")

    output_video_path = run_detection(uploaded_file)

    if output_video_path:
        st.success("Detection completed ✅")

        st.video(output_video_path)

        with open(output_video_path, "rb") as f:
            st.download_button(
                label="Download Result",
                data=f,
                file_name="output.mp4"
            )
    else:
        st.error("No output generated.")
