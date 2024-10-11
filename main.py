import streamlit as st
from lumaai import LumaAI
import replicate
import requests
import time
import base64
from PIL import Image
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
if 'generations' not in st.session_state:
    st.session_state.generations = []  # List to store generation metadata

def main():
    st.set_page_config(page_title="AI Video Suite", layout="wide")
    st.title("All-in-One AI Video Solution")

    # Sidebar for API Keys
    st.sidebar.header("API Keys")
    luma_api_key = st.sidebar.text_input("Enter your Luma AI API Key", type="password")
    stability_api_key = st.sidebar.text_input("Enter your Stability AI API Key", type="password")
    replicate_api_key = st.sidebar.text_input("Enter your Replicate API Key", type="password")

    # Set Replicate API token
    if replicate_api_key:
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_key

    if not luma_api_key and not stability_api_key and not replicate_api_key:
        st.warning("Please enter at least one API Key to proceed.")
        return

    # Initialize clients
    luma_client = LumaAI(auth_token=luma_api_key) if luma_api_key else None

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Generator", "History", "Edit"])

    # -------------------------------------------
    # Generator Tab
    # -------------------------------------------
    with tab1:
        st.header("Content Generation")

        # Mode selection
        mode = st.selectbox("Select Mode", ["Text-to-Video (Luma AI)", "Text-to-Video (Stability AI)", "Image-to-Video (Stability AI)", "Image Generation (Replicate AI)"])

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

                        st.session_state.generations.append({
                            "id": generation.id,
                            "type": "video",
                            "path": video_path,
                            "source": "Luma AI",
                            "prompt": prompt,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        })

                        st.video(video_path)
                        st.success("Video generated and saved to history.")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.error(traceback.format_exc())

        elif mode == "Text-to-Video (Stability AI)":
            st.info("This feature is under development.")

        elif mode == "Image-to-Video (Stability AI)":
            st.info("This feature is under development.")

        elif mode == "Image Generation (Replicate AI)":
            if not replicate_api_key:
                st.error("Replicate API Key is required for this mode.")
                return
            prompt = st.text_area("Enter a prompt for image generation", "A serene landscape with mountains and a river")
            aspect_ratio = st.selectbox("Aspect Ratio", ["1:1", "16:9", "9:16"])
            output_format = st.selectbox("Output Format", ["jpg", "png", "webp"])
            output_quality = st.slider("Output Quality", 1, 100, 80)
            safety_tolerance = st.slider("Safety Tolerance", 0, 5, 2)
            prompt_upsampling = st.checkbox("Prompt Upsampling", value=True)

            if st.button("Generate Image"):
                with st.spinner("Generating image..."):
                    try:
                        output = replicate.run(
                            "black-forest-labs/flux-1.1-pro",
                            input={
                                "prompt": prompt,
                                "aspect_ratio": aspect_ratio,
                                "output_format": output_format,
                                "output_quality": output_quality,
                                "safety_tolerance": safety_tolerance,
                                "prompt_upsampling": prompt_upsampling
                            }
                        )
                        image_url = output[0]
                        image_response = requests.get(image_url)
                        image = Image.open(io.BytesIO(image_response.content))

                        image_path = f"replicate_image_{len(st.session_state.generations)+1}.{output_format}"
                        image.save(image_path)

                        st.session_state.generations.append({
                            "id": f"replicate_{len(st.session_state.generations)+1}",
                            "type": "image",
                            "path": image_path,
                            "source": "Replicate AI",
                            "prompt": prompt,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        })

                        st.image(image)
                        st.success("Image generated and saved to history.")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.error(traceback.format_exc())

    # -------------------------------------------
    # History Tab
    # -------------------------------------------
    with tab2:
        st.header("Generation History")
        if st.session_state.generations:
            for gen in st.session_state.generations[::-1]:  # Display newest first
                st.subheader(f"ID: {gen['id']} | Type: {gen['type']} | Source: {gen['source']} | Time: {gen['timestamp']}")
                st.write(f"Prompt: {gen['prompt']}")
                if gen['type'] == "video":
                    st.video(gen['path'])
                elif gen['type'] == "image":
                    st.image(gen['path'])
                st.markdown("---")
        else:
            st.info("No generations yet. Generate content in the Generator tab.")

    # -------------------------------------------
    # Edit Tab
    # -------------------------------------------
    with tab3:
        st.header("Edit Generations")
        if not st.session_state.generations:
            st.info("No generations available to edit.")
            return

        # Select generation to edit
        gen_options = [f"{gen['id']} ({gen['type']}) - {gen['timestamp']}" for gen in st.session_state.generations]
        selected_gen = st.selectbox("Select a generation to edit", gen_options)
        selected_gen_index = gen_options.index(selected_gen)
        gen_to_edit = st.session_state.generations[selected_gen_index]

        if gen_to_edit['type'] == "video":
            st.video(gen_to_edit['path'])
        elif gen_to_edit['type'] == "image":
            st.image(gen_to_edit['path'])

        # Video Editing Tools
        if gen_to_edit['type'] == "video":
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

            # Apply Edits
            if st.button("Apply Edits"):
                with st.spinner("Applying edits..."):
                    try:
                        video_clip = VideoFileClip(gen_to_edit['path'])

                        if apply_filters:
                            video_clip = video_clip.fx(vfx.colorx, brightness)
                            video_clip = video_clip.fx(vfx.lum_contrast, contrast=contrast*100)
                            # Note: Adjusting saturation requires custom implementation or external libraries

                        if add_text_overlay:
                            txt_clip = TextClip(
                                overlay_text, fontsize=font_size, color=font_color.replace("#", ""), font="Arial"
                            )
                            txt_clip = txt_clip.set_position(text_position.lower()).set_duration(video_clip.duration)
                            video_clip = CompositeVideoClip([video_clip, txt_clip])

                        # Save edited video
                        edited_video_path = f"{gen_to_edit['id']}_edited.mp4"
                        video_clip.write_videofile(edited_video_path, codec="libx264", audio_codec="aac")

                        # Update generation history
                        st.session_state.generations.append({
                            "id": f"{gen_to_edit['id']}_edited",
                            "type": "video",
                            "path": edited_video_path,
                            "source": gen_to_edit['source'],
                            "prompt": gen_to_edit['prompt'],
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        })

                        st.video(edited_video_path)
                        st.success("Edits applied and new video saved to history.")

                        # Clean up
                        video_clip.close()

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.error(traceback.format_exc())

        elif gen_to_edit['type'] == "image":
            st.info("Image editing features are under development.")

    # -------------------------------------------
    # Footer with style
    # -------------------------------------------
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
