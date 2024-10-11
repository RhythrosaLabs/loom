import streamlit as st
from lumaai import LumaAI
import replicate
import requests
import time
import base64
from PIL import Image
import io
from moviepy.editor import (
    VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip, vfx, ImageClip
)
import os
import sys
import numpy as np
import traceback
import zipfile

# Redirect stderr to stdout to avoid issues with logging in some environments
sys.stderr = sys.stdout

# Initialize session state
if 'generations' not in st.session_state:
    st.session_state.generations = []  # List to store generation metadata
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'generated_videos' not in st.session_state:
    st.session_state.generated_videos = []
if 'final_video' not in st.session_state:
    st.session_state.final_video = None

def resize_image(image):
    width, height = image.size
    if (width, height) in [(1024, 576), (576, 1024), (768, 768)]:
        return image
    else:
        st.warning("Resizing image to 768x768 (default)")
        return image.resize((768, 768))

def generate_image_from_text(api_key, prompt):
    url = "https://api.stability.ai/v1beta/generation/stable-diffusion-v1-6/text-to-image"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "text_prompts": [{"text": prompt}],
        "cfg_scale": 7,
        "height": 768,
        "width": 768,
        "samples": 1,
        "steps": 30,
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        image_data = response.json()['artifacts'][0]['base64']
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        return image
    except requests.exceptions.RequestException as e:
        st.error(f"Error generating image: {str(e)}")
        return None

def start_video_generation(api_key, image, cfg_scale=1.8, motion_bucket_id=127, seed=0):
    url = "https://api.stability.ai/v2beta/image-to-video"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    files = {
        "image": ("image.png", img_byte_arr, "image/png")
    }
    data = {
        "seed": str(seed),
        "cfg_scale": str(cfg_scale),
        "motion_bucket_id": str(motion_bucket_id)
    }
    try:
        response = requests.post(url, headers=headers, files=files, data=data)
        response.raise_for_status()
        return response.json().get('id')
    except requests.exceptions.RequestException as e:
        st.error(f"Error starting video generation: {str(e)}")
        return None

def poll_for_video(api_key, generation_id):
    url = f"https://api.stability.ai/v2beta/image-to-video/result/{generation_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "video/*"
    }
    max_attempts = 60
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 202:
                st.write(f"Video generation in progress... Polling attempt {attempt + 1}/{max_attempts}")
                time.sleep(10)
            elif response.status_code == 200:
                return response.content
            else:
                response.raise_for_status()
        except requests.exceptions.RequestException as e:
            st.error(f"Error polling for video: {str(e)}")
            return None
    st.error("Video generation timed out. Please try again.")
    return None

def validate_video_clip(video_path):
    if not os.path.exists(video_path):
        st.error(f"Video file not found: {video_path}")
        return False
    try:
        clip = VideoFileClip(video_path)
        if clip is None:
            st.error(f"Failed to load video clip: {video_path}")
            return False
        duration = clip.duration
        clip.close()
        st.write(f"Validated video clip: {video_path}, Duration: {duration} seconds")
        return duration > 0
    except Exception as e:
        st.error(f"Invalid video segment: {video_path}, Error: {str(e)}")
        return False

def get_last_frame_image(video_path):
    if not os.path.exists(video_path):
        st.error(f"Video file not found: {video_path}")
        return None
    try:
        video_clip = VideoFileClip(video_path)
        if video_clip is None:
            st.error(f"Failed to load video clip: {video_path}")
            return None
        if video_clip.duration <= 0:
            st.error(f"Invalid video duration for {video_path}")
            video_clip.close()
            return None
        last_frame = video_clip.get_frame(video_clip.duration - 0.001)
        last_frame_image = Image.fromarray(np.uint8(last_frame)).convert('RGB')
        video_clip.close()
        return last_frame_image
    except Exception as e:
        st.error(f"Error extracting last frame from {video_path}: {str(e)}")
        return None

def concatenate_videos(video_clips, crossfade_duration=0):
    valid_clips = []
    for clip_path in video_clips:
        st.write(f"Attempting to load clip: {clip_path}")
        if validate_video_clip(clip_path):
            try:
                clip = VideoFileClip(clip_path)
                if clip is not None and clip.duration > 0:
                    valid_clips.append(clip)
                    st.write(f"Successfully loaded clip: {clip_path}, Duration: {clip.duration} seconds")
                else:
                    st.warning(f"Skipping invalid clip: {clip_path}")
            except Exception as e:
                st.warning(f"Error loading clip {clip_path}: {str(e)}")
        else:
            st.warning(f"Validation failed for clip: {clip_path}")

    if not valid_clips:
        st.error("No valid video segments found. Unable to concatenate.")
        return None, None

    try:
        st.write(f"Attempting to concatenate {len(valid_clips)} valid clips")
        
        # Trim the last frame from all clips except the last one
        trimmed_clips = []
        for i, clip in enumerate(valid_clips):
            if i < len(valid_clips) - 1:
                # Subtract a small duration (e.g., 1/30 second) to remove approximately one frame
                trimmed_clip = clip.subclip(0, clip.duration - 1/30)
                trimmed_clips.append(trimmed_clip)
            else:
                trimmed_clips.append(clip)

        if crossfade_duration > 0:
            st.write(f"Applying crossfade of {crossfade_duration} seconds")
            # Apply crossfade transition
            final_clips = []
            for i, clip in enumerate(trimmed_clips):
                if i == 0:
                    final_clips.append(clip)
                else:
                    # Create a crossfade transition
                    fade_out = trimmed_clips[i-1].fx(vfx.fadeout, duration=crossfade_duration)
                    fade_in = clip.fx(vfx.fadein, duration=crossfade_duration)
                    transition = CompositeVideoClip([fade_out, fade_in])
                    transition = transition.set_duration(crossfade_duration)
                    
                    # Add the transition and the full clip
                    final_clips.append(transition)
                    final_clips.append(clip)
            
            final_video = concatenate_videoclips(final_clips)
        else:
            final_video = concatenate_videoclips(trimmed_clips)

        st.write(f"Concatenation successful. Final video duration: {final_video.duration} seconds")
        return final_video, valid_clips
    except Exception as e:
        st.error(f"Error concatenating videos: {str(e)}")
        for clip in valid_clips:
            clip.close()
        return None, None

def generate_multiple_images(api_key, prompt, num_images):
    images = []
    for i in range(num_images):
        st.write(f"Generating image {i+1}/{num_images}...")
        image = generate_image_from_text(api_key, prompt)
        if image:
            images.append(image)
        else:
            st.error(f"Failed to generate image {i+1}")
    return images

def create_zip_file(images, videos, output_path="generated_content.zip"):
    if not images and not videos:
        st.error("No images or videos to create a zip file.")
        return None

    try:
        with zipfile.ZipFile(output_path, 'w') as zipf:
            for i, img in enumerate(images):
                img_path = f"image_{i+1}.png"
                img.save(img_path)
                zipf.write(img_path)
                os.remove(img_path)
            
            for video in videos:
                if os.path.exists(video):
                    zipf.write(video)
                else:
                    st.warning(f"Video file not found: {video}")
        
        return output_path
    except Exception as e:
        st.error(f"Error creating zip file: {str(e)}")
        return None

def display_images_in_grid(images, columns=3):
    """Display images in a grid layout with captions."""
    for i in range(0, len(images), columns):
        cols = st.columns(columns)
        for j in range(columns):
            if i + j < len(images):
                with cols[j]:
                    st.image(images[i + j], use_column_width=True, caption=f"Image {i + j + 1}")
                    st.markdown(f"<p style='text-align: center;'>Image {i + j + 1}</p>", unsafe_allow_html=True)

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
    tab1, tab2, tab3 = st.tabs(["Generator", "Images", "Videos"])

    # -------------------------------------------
    # Generator Tab
    # -------------------------------------------
    with tab1:
        st.header("Content Generation")

        # Mode selection
        mode = st.selectbox("Select Mode", [
            "Luma",
            "Text-to-Video (Stability AI)",
            "Image-to-Video (Stability AI)",
            "Image Generation (Replicate AI)",
            "Snapshot Mode (Stability AI)"
        ])

        if mode == "Luma":
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
                start_image_file = st.file_uploader("Upload Start Image", type=["png", "jpg", "jpeg"])
                if start_image_file:
                    # Here you need to upload the image to a hosting service to get a URL
                    st.warning("Please upload the image to a hosting service and provide the URL.")
                    start_image_url = st.text_input("Start Image URL after uploading")
                    if start_image_url:
                        keyframes["frame0"] = {
                            "type": "image",
                            "url": start_image_url
                        }

            if keyframe_option in ["End Image", "Start and End Image"]:
                end_image_file = st.file_uploader("Upload End Image", type=["png", "jpg", "jpeg"])
                if end_image_file:
                    # Here you need to upload the image to a hosting service to get a URL
                    st.warning("Please upload the image to a hosting service and provide the URL.")
                    end_image_url = st.text_input("End Image URL after uploading")
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

                        st.session_state.generated_videos.append(video_path)
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

        # The rest of your code remains unchanged...

    # -------------------------------------------
    # Images Tab
    # -------------------------------------------
    with tab2:
        st.subheader("Generated Images")
        if st.session_state.generated_images:
            display_images_in_grid(st.session_state.generated_images)
        else:
            st.write("No images generated yet. Use the Generator tab to create images.")

    # -------------------------------------------
    # Videos Tab
    # -------------------------------------------
    with tab3:
        st.subheader("Generated Videos")
        if st.session_state.generated_videos:
            for i, video_path in enumerate(st.session_state.generated_videos):
                if os.path.exists(video_path):
                    st.video(video_path)
                    st.write(f"Video Segment {i+1}")
                    with open(video_path, "rb") as f:
                        st.download_button(f"Download Video Segment {i+1}", f, file_name=f"video_segment_{i+1}.mp4")
                else:
                    st.error(f"Video file not found: {video_path}")
            
            if st.session_state.final_video and os.path.exists(st.session_state.final_video):
                st.subheader("Final Longform Video")
                st.video(st.session_state.final_video)
                with open(st.session_state.final_video, "rb") as f:
                    st.download_button("Download Longform Video", f, file_name="longform_video.mp4")
        else:
            st.write("No videos generated yet. Use the Generator tab to create videos.")

        # Add download all button
        if st.session_state.generated_images or st.session_state.generated_videos:
            zip_path = create_zip_file(st.session_state.generated_images, st.session_state.generated_videos)
            with open(zip_path, "rb") as f:
                st.download_button("Download All Content (ZIP)", f, file_name="generated_content.zip")
            os.remove(zip_path)

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
