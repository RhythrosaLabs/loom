import streamlit as st
from lumaai import LumaAI
import requests
import time
import base64
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import io
from moviepy.editor import (
    VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip, vfx
)
import os
import sys
import numpy as np
import traceback

# Redirect stderr to stdout
sys.stderr = sys.stdout

# Initialize session state
if 'generated_content' not in st.session_state:
    st.session_state.generated_content = []

def main():
    st.set_page_config(page_title="AI Video Suite", layout="wide")
    st.title("All-in-One AI Video Solution")

    # Sidebar for API Keys
    st.sidebar.header("API Keys")
    luma_api_key = st.sidebar.text_input("Enter your Luma AI API Key", type="password")
    stability_api_key = st.sidebar.text_input("Enter your Stability AI API Key", type="password")

    if not luma_api_key and not stability_api_key:
        st.warning("Please enter at least one API Key to proceed.")
        return

    # Initialize clients
    luma_client = LumaAI(auth_token=luma_api_key) if luma_api_key else None

    st.header("Content Generation")

    # Mode selection
    mode = st.selectbox("Select Mode", ["Text-to-Video (Luma AI)", "Text-to-Video (Stability AI)", "Image-to-Video (Stability AI)"])

    # Input fields
    if mode == "Text-to-Video (Luma AI)":
        if not luma_api_key:
            st.error("Luma AI API Key is required for this mode.")
            return
        prompt = st.text_area("Prompt", "A teddy bear in sunglasses playing electric guitar and dancing")
        aspect_ratio = st.selectbox("Aspect Ratio", ["9:16", "16:9", "1:1", "3:4", "4:3"])
        loop = st.checkbox("Loop Video", value=False)

        # Camera Motions
        st.subheader("Camera Motion")
        try:
            supported_camera_motions = luma_client.generations.camera_motion.list()
            camera_motion = st.selectbox("Select Camera Motion", ["None"] + supported_camera_motions)
            if camera_motion != "None":
                prompt = f"{prompt}, {camera_motion}"
        except Exception as e:
            st.error(f"Could not fetch camera motions: {e}")
            camera_motion = None

        # Keyframes
        st.subheader("Keyframes")
        keyframe_option = st.selectbox(
            "Keyframe Options",
            ["None", "Start Image", "End Image", "Start and End Image", "Start Generation", "End Generation", "Start and End Generation"]
        )
        keyframes = {}

        if keyframe_option in ["Start Image", "Start and End Image"]:
            start_image_url = st.text_input("Start Image URL")
            if start_image_url:
                keyframes["frame0"] = {
                    "type": "image",
                    "url": start_image_url
                }

        if keyframe_option in ["End Image", "Start and End Image"]:
            end_image_url = st.text_input("End Image URL")
            if end_image_url:
                keyframes["frame1"] = {
                    "type": "image",
                    "url": end_image_url
                }

        if keyframe_option in ["Start Generation", "Start and End Generation"]:
            start_generation_id = st.text_input("Start Generation ID")
            if start_generation_id:
                keyframes["frame0"] = {
                    "type": "generation",
                    "id": start_generation_id
                }

        if keyframe_option in ["End Generation", "Start and End Generation"]:
            end_generation_id = st.text_input("End Generation ID")
            if end_generation_id:
                keyframes["frame1"] = {
                    "type": "generation",
                    "id": end_generation_id
                }

        # Advanced Options
        st.header("Advanced Options")

        # Callback URL
        callback_url = st.text_input("Callback URL (Optional)")

        # Video Editing Options
        st.subheader("Video Editing Tools")
        apply_filters = st.checkbox("Apply Filters", value=False)
        if apply_filters:
            brightness = st.slider("Brightness", 0.5, 2.0, 1.0)
            contrast = st.slider("Contrast", 0.5, 2.0, 1.0)
            saturation = st.slider("Saturation", 0.5, 2.0, 1.0)

        add_text_overlay = st.checkbox("Add Text Overlay", value=False)
        if add_text_overlay:
            overlay_text = st.text_input("Overlay Text", "Sample Text")
            font_size = st.slider("Font Size", 10, 100, 40)
            font_color = st.color_picker("Font Color", "#FFFFFF")
            text_position = st.selectbox("Text Position", ["Top", "Center", "Bottom"])

        # Generate button
        if st.button("Generate Video"):
            with st.spinner("Generating video... this may take a few minutes."):
                try:
                    # Prepare generation parameters
                    generation_params = {
                        "prompt": prompt,
                        "aspect_ratio": aspect_ratio,
                        "loop": loop,
                    }

                    if keyframes:
                        generation_params["keyframes"] = keyframes

                    if callback_url:
                        generation_params["callback_url"] = callback_url

                    generation = luma_client.generations.create(**generation_params)
                    completed = False
                    while not completed:
                        generation = luma_client.generations.get(id=generation.id)
                        if generation.state == "completed":
                            completed = True
                        elif generation.state == "failed":
                            st.error(f"Generation failed: {generation.failure_reason}")
                            return
                        else:
                            time.sleep(5)

                    video_url = generation.assets.video

                    # Download video
                    response = requests.get(video_url)
                    video_path = f"{generation.id}.mp4"
                    with open(video_path, "wb") as f:
                        f.write(response.content)

                    # Apply video editing tools
                    video_clip = VideoFileClip(video_path)

                    if apply_filters:
                        video_clip = video_clip.fx(vfx.colorx, brightness)
                        video_clip = video_clip.fx(vfx.lum_contrast, contrast=contrast*100)
                        # Saturation adjustment would require custom implementation

                    if add_text_overlay:
                        txt_clip = TextClip(
                            overlay_text, fontsize=font_size, color=font_color.replace("#", ""), font="Arial"
                        )
                        txt_clip = txt_clip.set_position(text_position.lower()).set_duration(video_clip.duration)
                        video_clip = CompositeVideoClip([video_clip, txt_clip])

                    # Save edited video
                    edited_video_path = f"{generation.id}_edited.mp4"
                    video_clip.write_videofile(edited_video_path, codec="libx264", audio_codec="aac")

                    st.video(edited_video_path)
                    st.markdown(f"[Download Video]({edited_video_path})")

                    # Clean up
                    video_clip.close()
                    os.remove(video_path)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.error(traceback.format_exc())

    elif mode == "Text-to-Video (Stability AI)":
        if not stability_api_key:
            st.error("Stability AI API Key is required for this mode.")
            return
        prompt = st.text_area("Enter a text prompt for video generation", height=100)
        num_segments = st.slider("Number of video segments to generate", 1, 10, 3)
        cfg_scale = st.slider("CFG Scale", 0.0, 10.0, 7.0)
        seed = st.number_input("Seed (0 for random)", min_value=0, max_value=4294967294, value=0)
        crossfade_duration = st.slider("Crossfade Duration (seconds)", 0.0, 2.0, 0.5, 0.1)

        # Generate button
        if st.button("Generate Video"):
            with st.spinner("Generating video..."):
                try:
                    # Implement video generation using Stability AI
                    # Placeholder for actual implementation
                    st.info("This feature is under development.")

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.error(traceback.format_exc())

    elif mode == "Image-to-Video (Stability AI)":
        if not stability_api_key:
            st.error("Stability AI API Key is required for this mode.")
            return
        image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        cfg_scale = st.slider("CFG Scale", 0.0, 10.0, 7.0)
        seed = st.number_input("Seed (0 for random)", min_value=0, max_value=4294967294, value=0)

        # Generate button
        if st.button("Generate Video"):
            if image_file is None:
                st.error("Please upload an image.")
                return
            with st.spinner("Generating video..."):
                try:
                    # Implement video generation using Stability AI
                    # Placeholder for actual implementation
                    st.info("This feature is under development.")

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.error(traceback.format_exc())

    # Footer with style
    st.markdown("""
    <style>
    .reportview-container {
        background: #1a1a1a;
        color: white;
    }
    .sidebar .sidebar-content {
        background: #333333;
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
