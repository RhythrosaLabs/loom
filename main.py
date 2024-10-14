import streamlit as st
from lumaai import LumaAI
import runwayml
import replicate
import requests
import time
import base64
from PIL import Image
import io
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip, vfx, ImageClip
import os
import sys
import numpy as np
import traceback
import zipfile

# Redirect stderr to stdout to capture all logs in Streamlit
sys.stderr = sys.stdout

# -----------------------------
# Initialize Session State
# -----------------------------
if 'generations' not in st.session_state:
    st.session_state.generations = []  # List to store generation metadata
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'generated_videos' not in st.session_state:
    st.session_state.generated_videos = []
if 'final_video' not in st.session_state:
    st.session_state.final_video = None

# -----------------------------
# Helper Functions
# -----------------------------

def resize_image(image, target_size):
    return image.resize(target_size)

def generate_image_from_text_stability(api_key, prompt):
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
        st.error(f"Error generating image with Stable Diffusion: {str(e)}")
        return None

def generate_image_from_text_flux(prompt, aspect_ratio, output_format, output_quality, safety_tolerance, prompt_upsampling):
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
        # Access the URL directly from the FileOutput object
        image_url = output.url
        image_response = requests.get(image_url)
        image = Image.open(io.BytesIO(image_response.content))
        return image
    except Exception as e:
        st.error(f"Error generating image with Flux: {e}")
        st.error(traceback.format_exc())
        return None

def generate_image_from_text_dalle(api_key, prompt, size, quality):
    url = "https://api.openai.com/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "dall-e-3",
        "prompt": prompt,
        "n": 1,
        "size": size,
        "response_format": "url",
        "quality": quality  # "standard" or "hd"
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        image_url = response_data['data'][0]['url']
        # Optionally, you can access the revised prompt
        revised_prompt = response_data['data'][0].get('revised_prompt', '')
        if revised_prompt:
            st.write(f"**Revised Prompt:** {revised_prompt}")
        # Download image
        image_response = requests.get(image_url)
        image = Image.open(io.BytesIO(image_response.content))
        return image
    except Exception as e:
        st.error(f"Error generating image with DALL·E: {e}")
        st.error(traceback.format_exc())
        return None

def start_video_generation_stability(api_key, image, cfg_scale=1.8, motion_bucket_id=127, seed=0):
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
        st.error(f"Error starting video generation with Stability AI: {str(e)}")
        return None

def poll_for_video_stability(api_key, generation_id):
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
            st.error(f"Error polling for video with Stability AI: {str(e)}")
            return None
    st.error("Video generation timed out with Stability AI. Please try again.")
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

def create_video_from_images(images, fps, output_path):
    clips = [ImageClip(np.array(img)).set_duration(1/fps) for img in images]
    video = concatenate_videoclips(clips, method="compose")
    video.write_videofile(output_path, fps=fps, codec="libx264")
    return output_path

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

def generate_video_runwayml(runway_api_key, prompt_image_url, prompt_text):
    client = runwayml.RunwayML(api_key=runway_api_key)
    try:
        response = client.image_to_video.create(
            model="gen3a_turbo",
            prompt_image=prompt_image_url,
            prompt_text=prompt_text,
        )
        generation_id = response.id
        st.write(f"RunwayML Video Generation ID: {generation_id}")
        # Poll until completion
        while True:
            generation = client.image_to_video.get(id=generation_id)
            if generation.state == "completed":
                st.success("RunwayML Video Generation Completed.")
                video_url = generation.assets.video
                # Download video
                video_response = requests.get(video_url)
                video_path = f"runwayml_video_{generation_id}.mp4"
                with open(video_path, "wb") as f:
                    f.write(video_response.content)
                st.write(f"✅ Saved RunwayML video to {video_path}")
                st.session_state.generated_videos.append(video_path)
                st.session_state.final_video = video_path
                st.video(video_path)
                break
            elif generation.state == "failed":
                st.error(f"RunwayML Video Generation Failed: {generation.failure_reason}")
                break
            else:
                st.write("⌛ RunwayML Video Generation in progress... Waiting for completion.")
                time.sleep(10)
    except runwayml.APIConnectionError as e:
        st.error("RunwayML API Connection Error.")
        st.error(e.__cause__)  # an underlying Exception, likely raised within httpx.
    except runwayml.RateLimitError as e:
        st.error("RunwayML Rate Limit Exceeded. Please wait and try again.")
    except runwayml.APIStatusError as e:
        st.error(f"RunwayML API returned an error: {e.status_code}")
        st.error(e.response)
    except Exception as e:
        st.error(f"An unexpected error occurred with RunwayML: {e}")
        st.error(traceback.format_exc())

# -----------------------------
# Main Application Function
# -----------------------------
def main():
    # -------------------------
    # Streamlit Page Configuration
    # -------------------------
    st.set_page_config(page_title="AI Video Suite", layout="wide", page_icon="🎬")
    
    # -------------------------
    # Custom CSS for Enhanced UI
    # -------------------------
    st.markdown("""
    <style>
    /* Background and Text Color */
    .reportview-container {
        background-color: #1a1a1a;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #333333;
    }
    
    /* Header Styling */
    h1 {
        color: #FFD700;
        text-align: center;
        font-family: 'Helvetica', sans-serif;
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        height: 3em;
        width: 15em;
        border-radius:10px;
        border: 1px solid #4CAF50;
        font-size:20px;
    }
    
    .stButton>button:hover {
        background-color: #45a049;
    }
    
    /* Success and Error Message Styling */
    .css-1aumxhk.edgvbvh3 {
        background-color: #1a1a1a;
    }
    
    /* Tooltip Styling */
    div.tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted black;
    }
    
    div.tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px 0;
        position: absolute;
        z-index: 1;
        bottom: 125%; 
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    div.tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Grid Layout for Images */
    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        grid-gap: 10px;
    }
    
    /* Footer Styling */
    footer {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

    # -------------------------
    # Application Header
    # -------------------------
    st.title("🎬 AI Video Suite")

    # -------------------------
    # Sidebar Navigation with Tabs
    # -------------------------
    sidebar_tabs = st.sidebar.tabs(["🔑 API Keys", "ℹ️ About"])

    # API Keys Tab
    with sidebar_tabs[0]:
        st.header("🔑 API Keys")
        # Use unique keys for each input field to prevent duplicate element IDs
        st.text_input("Enter your Luma AI API Key", type="password", key="luma_api_key")
        st.text_input("Enter your Stability AI API Key", type="password", key="stability_api_key")
        st.text_input("Enter your Replicate API Key", type="password", key="replicate_api_key")
        st.text_input("Enter your OpenAI API Key (for DALL·E)", type="password", key="openai_api_key")
        st.text_input("Enter your RunwayML API Key", type="password", key="runway_api_key")

    # About Tab
    with sidebar_tabs[1]:
        st.header("ℹ️ About")
        st.markdown("""
        ### **AI Video Suite**

        **AI Video Suite** is an all-in-one platform that leverages multiple AI image and video services to generate and concatenate content. Whether you're a content creator, marketer, or simply an AI enthusiast, this application provides a seamless experience to create and manage AI-generated media.

        #### **Features**

        - **Snapshot Mode:** Generate a series of images using DALL·E, Stable Diffusion, or Flux and compile them into a cohesive video.
        - **Text-to-Video:** Luma, Stable Diffusion, and Runway
        - **Image-to-Video:** Luma, Stable Diffusion, and Runway
        - **Image Generation:** Flux, Stable Diffusion, DALLE3


        #### **Supported AI Services**

        - **DALL·E 3:** text-to-image.
        - **Stable Diffusion:** text-to-image, text-to-video, image-to-video.
        - **Flux (Replicate AI):** text-to-image.
        - **RunwayML:** text-to-video and image-to-video
        - **Luma AI:** text-to-video and image-to-video

        #### **How to Use**

        1. **API Keys:**
           - Navigate to the **API Keys** tab in the sidebar.
           - Enter your respective API keys for **Luma AI**, **Stability AI**, **Replicate AI**, **OpenAI**, and **RunwayML**.
           - Ensure that your API keys are valid and have the necessary permissions.

        2. **Generate Content:**
           - Go to the **Generator** tab.
           - Select your desired mode (e.g., **Snapshot Mode**, **RunwayML Image-to-Video**).
           - Input your prompts and adjust settings as needed.
           - Click on the **Generate** button to initiate the creation process.

        3. **View and Download:**
           - Generated images will appear in the **Images** tab.
           - Generated videos will be available in the **Videos** tab, where you can view and download them individually or as a ZIP file.

        - **API Usage and Costs:**
          Be mindful of the usage limits and potential costs associated with each API. Monitor your usage to avoid unexpected charges.

        - **Security:**
          Never share or expose your API keys publicly. Ensure they are kept secure to prevent unauthorized access.

        - **Performance:**
          Generating a large number of images or complex videos may consume significant resources and time. Adjust your settings accordingly.

        #### **Support**
        - If you encounter any issues or have questions, please feel free to get in touch.
        - [View and fork the code](https://github.com/RhythrosaLabs/loom/)

        #### **Credits**

        - **Daniel Sheils:** [LinkedIn](http://linkedin.com/in/danielsheils/) | [Portfolio](https://danielsheils.myportfolio.com) | [Rhythrosa Labs](https://rhythrosalabs.com) | [brAInstormer](https://brainstormer.streamlit.app) | [Game Maker](https://game-maker2.streamlit.app)
        


        
        """)

    # -------------------------
    # Retrieve API Keys from Session State
    # -------------------------
    luma_api_key = st.session_state.get('luma_api_key', '')
    stability_api_key = st.session_state.get('stability_api_key', '')
    replicate_api_key = st.session_state.get('replicate_api_key', '')
    openai_api_key = st.session_state.get('openai_api_key', '')
    runway_api_key = st.session_state.get('runway_api_key', '')
    
    # -------------------------
    # Set Replicate API Token
    # -------------------------
    if replicate_api_key:
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_key

    # -------------------------
    # Prompt User to Enter at Least One API Key
    # -------------------------
    if not any([luma_api_key, stability_api_key, replicate_api_key, openai_api_key, runway_api_key]):
        st.warning("🔑 Please enter at least one API Key in the **API Keys** tab to proceed.")
        st.stop()

    # -------------------------
    # Initialize Luma AI Client
    # -------------------------
    if luma_api_key:
        try:
            luma_client = LumaAI(auth_token=luma_api_key)
        except Exception as e:
            st.error(f"❌ Error initializing Luma AI client: {e}")
            luma_client = None
    else:
        luma_client = None

    # -------------------------
    # Main Tabs: Generator, Images, Videos
    # -------------------------
    tab1, tab2, tab3 = st.tabs(["🎨 Generator", "🖼️ Images", "📽️ Videos"])

    # -------------------------
    # Generator Tab
    # -------------------------
    with tab1:
        st.header("🎨 Content Generation")

        # Mode Selection
        mode = st.selectbox("Select Generation Mode", [
            "Snapshot Mode",
            "Text-to-Video (Stability AI)",
            "Image-to-Video (Stability AI)",
            "Image Generation (Replicate AI)",
            "RunwayML Image-to-Video",
            "Luma Integration"
        ])

        # ---------------------
        # Snapshot Mode
        # ---------------------
        if mode == "Snapshot Mode":
            st.subheader("📸 Snapshot Mode")
            snapshot_generator = st.selectbox("Select Image Generator", ["DALL·E", "Stable Diffusion", "Flux"], key="snapshot_generator")
            prompt = st.text_area("Enter a text prompt for Snapshot Mode", height=100, key="snapshot_prompt")
            num_images = st.slider("Number of images to generate", 2, 300, 10, key="snapshot_num_images")
            fps = st.slider("Frames per second", 1, 60, 24, key="snapshot_fps")
            if snapshot_generator in ["Flux", "DALL·E"]:
                aspect_ratio = st.selectbox("Aspect Ratio", ["1:1", "16:9", "9:16"], key="snapshot_aspect_ratio")
            else:
                aspect_ratio = "1:1"

            # Check for required API keys
            if snapshot_generator == "Stable Diffusion" and not stability_api_key:
                st.error("🚫 Stability AI API Key is required for Stable Diffusion.")
                st.stop()
            if snapshot_generator == "Flux" and not replicate_api_key:
                st.error("🚫 Replicate API Key is required for Flux.")
                st.stop()
            if snapshot_generator == "DALL·E" and not openai_api_key:
                st.error("🚫 OpenAI API Key is required for DALL·E.")
                st.stop()

            if st.button("✨ Generate Video"):
                if not prompt:
                    st.error("❗ Please enter a text prompt.")
                    st.stop()

                try:
                    st.success(f"🔄 Generating {num_images} images using {snapshot_generator}...")
                    images = []
                    for i in range(num_images):
                        st.write(f"Generating image {i+1}/{num_images}...")
                        if snapshot_generator == "Stable Diffusion":
                            image = generate_image_from_text_stability(stability_api_key, prompt)
                        elif snapshot_generator == "Flux":
                            image = generate_image_from_text_flux(
                                prompt,
                                aspect_ratio=aspect_ratio,
                                output_format="png",
                                output_quality=80,
                                safety_tolerance=2,
                                prompt_upsampling=True
                            )
                        elif snapshot_generator == "DALL·E":
                            if aspect_ratio == "1:1":
                                size = "1024x1024"
                            elif aspect_ratio == "16:9":
                                size = "1792x1024"
                            elif aspect_ratio == "9:16":
                                size = "1024x1792"
                            quality = "standard"  # or "hd"
                            image = generate_image_from_text_dalle(openai_api_key, prompt, size, quality)
                        else:
                            st.error(f"🚫 Unsupported generator: {snapshot_generator}")
                            continue
                        if image:
                            images.append(image)
                            st.session_state.generated_images.append(image)
                        else:
                            st.error(f"❌ Failed to generate image {i+1}")

                    if images:
                        st.success("✅ All images generated successfully!")
                        st.write("🎞️ Creating video from generated images...")
                        video_path = "snapshot_mode_video.mp4"
                        create_video_from_images(images, fps, video_path)
                        st.session_state.generated_videos.append(video_path)
                        st.session_state.final_video = video_path
                        st.success(f"🎬 Snapshot Mode video created: {video_path}")
                        st.video(video_path)
                    else:
                        st.error("❌ Failed to generate images for Snapshot Mode.")

                except Exception as e:
                    st.error(f"❗ An unexpected error occurred: {str(e)}")
                    st.write("🛠️ Error details:", str(e))
                    st.write("📜 Traceback:", traceback.format_exc())

        # ---------------------
        # Text-to-Video (Stability AI)
        # ---------------------
        elif mode == "Text-to-Video (Stability AI)":
            st.subheader("📜 Text-to-Video (Stability AI)")
            prompt = st.text_area("Enter a text prompt for video generation", height=100, key="stability_video_prompt")
            cfg_scale = st.slider("CFG Scale (Controls adherence to prompt)", 0.0, 10.0, 1.8, key="stability_cfg_scale")
            motion_bucket_id = st.slider("Motion Bucket ID (1-255)", 1, 255, 127, key="stability_motion_bucket")
            seed = st.number_input("Seed (0 for random)", min_value=0, max_value=4294967294, value=0, key="stability_seed")
            num_segments = st.slider("Number of video segments to generate", 1, 60, 5, key="stability_num_segments")
            crossfade_duration = st.slider("Crossfade Duration (seconds)", 0.0, 2.0, 0.0, 0.01, key="stability_crossfade")

            if st.button("🎥 Generate Video with Stability AI"):
                if not prompt:
                    st.error("❗ Please enter a text prompt.")
                    st.stop()

                try:
                    st.success("🔄 Generating initial image from text prompt...")
                    image = generate_image_from_text_stability(stability_api_key, prompt)
                    if image is None:
                        st.error("❌ Failed to generate the initial image.")
                        st.stop()
                    image = resize_image(image, (768, 768))
                    st.session_state.generated_images.append(image)
                    
                    video_clips = []
                    current_image = image

                    for i in range(num_segments):
                        st.write(f"🎞️ Generating video segment {i+1}/{num_segments}...")
                        generation_id = start_video_generation_stability(stability_api_key, current_image, cfg_scale, motion_bucket_id, seed)

                        if generation_id:
                            video_content = poll_for_video_stability(stability_api_key, generation_id)

                            if video_content:
                                video_path = f"video_segment_{i+1}.mp4"
                                with open(video_path, "wb") as f:
                                    f.write(video_content)
                                st.write(f"✅ Saved video segment to {video_path}")
                                video_clips.append(video_path)
                                st.session_state.generated_videos.append(video_path)

                                last_frame_image = get_last_frame_image(video_path)
                                if last_frame_image:
                                    current_image = last_frame_image
                                    st.session_state.generated_images.append(current_image)
                                else:
                                    st.warning(f"⚠️ Could not extract last frame from segment {i+1}. Using previous image.")
                            else:
                                st.error(f"❌ Failed to retrieve video content for segment {i+1}.")
                        else:
                            st.error(f"❌ Failed to start video generation for segment {i+1}.")

                    if video_clips:
                        st.success("🔗 Concatenating video segments into one longform video...")
                        final_video, valid_clips = concatenate_videos(video_clips, crossfade_duration=crossfade_duration)
                        if final_video:
                            try:
                                final_video_path = "longform_video.mp4"
                                final_video.write_videofile(final_video_path, codec="libx264", audio_codec="aac")
                                st.session_state.final_video = final_video_path
                                st.success(f"🎬 Longform video created: {final_video_path}")
                                st.video(final_video_path)
                            except Exception as e:
                                st.error(f"❌ Error writing final video: {str(e)}")
                                st.write("📜 Traceback:", traceback.format_exc())
                            finally:
                                if final_video:
                                    final_video.close()
                                if valid_clips:
                                    for clip in valid_clips:
                                        clip.close()
                            
                            # Clean up individual video segments
                            for video_file in video_clips:
                                if os.path.exists(video_file):
                                    os.remove(video_file)
                                    st.write(f"🗑️ Removed temporary file: {video_file}")
                                else:
                                    st.warning(f"⚠️ Could not find file to remove: {video_file}")
                        else:
                            st.error("❌ Failed to create the final video.")
                        
                        # Final Video Display
                        if st.session_state.final_video and os.path.exists(st.session_state.final_video):
                            st.write(f"### 🎞️ Final Video: {st.session_state.final_video}")
                            st.video(st.session_state.final_video)
                    else:
                        st.error("❌ No video segments were successfully generated.")

                except Exception as e:
                    st.error(f"❗ An unexpected error occurred: {str(e)}")
                    st.write("🛠️ Error details:", str(e))
                    st.write("📜 Traceback:", traceback.format_exc())

        # ---------------------
        # Image-to-Video (Stability AI)
        # ---------------------
        elif mode == "Image-to-Video (Stability AI)":
            st.subheader("🖼️ Image-to-Video (Stability AI)")
            image_file = st.file_uploader("📂 Upload an image", type=["png", "jpg", "jpeg"], key="stability_image_upload")
            cfg_scale = st.slider("CFG Scale (Controls adherence to prompt)", 0.0, 10.0, 1.8, key="stability_image_cfg_scale")
            motion_bucket_id = st.slider("Motion Bucket ID (1-255)", 1, 255, 127, key="stability_image_motion_bucket")
            seed = st.number_input("Seed (0 for random)", min_value=0, max_value=4294967294, value=0, key="stability_image_seed")

            if st.button("🎥 Generate Video from Image"):
                if not image_file:
                    st.error("❗ Please upload an image.")
                    st.stop()
                try:
                    image = Image.open(image_file)
                    image = resize_image(image, (768, 768))
                    st.session_state.generated_images.append(image)

                    st.success("🔄 Starting video generation from uploaded image...")
                    generation_id = start_video_generation_stability(stability_api_key, image, cfg_scale, motion_bucket_id, seed)

                    if generation_id:
                        video_content = poll_for_video_stability(stability_api_key, generation_id)

                        if video_content:
                            video_path = "image_to_video.mp4"
                            with open(video_path, "wb") as f:
                                f.write(video_content)
                            st.success(f"✅ Image-to-Video created: {video_path}")
                            st.session_state.generated_videos.append(video_path)
                            st.session_state.final_video = video_path
                            st.video(video_path)
                        else:
                            st.error("❌ Failed to retrieve video content.")
                    else:
                        st.error("❌ Failed to start video generation.")

                except Exception as e:
                    st.error(f"❗ An unexpected error occurred: {e}")
                    st.error(traceback.format_exc())

        # ---------------------
        # Image Generation (Replicate AI)
        # ---------------------
        elif mode == "Image Generation (Replicate AI)":
            st.subheader("🖼️ Image Generation (Replicate AI)")
            prompt = st.text_area("Enter a prompt for image generation", "A serene landscape with mountains and a river", height=100, key="replicate_prompt")
            aspect_ratio = st.selectbox("Aspect Ratio", ["1:1", "16:9", "9:16"], key="replicate_aspect_ratio")
            output_format = st.selectbox("Output Format", ["jpg", "png", "webp"], key="replicate_output_format")
            output_quality = st.slider("Output Quality", 1, 100, 80, key="replicate_output_quality")
            safety_tolerance = st.slider("Safety Tolerance", 0, 5, 2, key="replicate_safety_tolerance")
            prompt_upsampling = st.checkbox("Prompt Upsampling", value=True, key="replicate_prompt_upsampling")

            if st.button("✨ Generate Image with Replicate AI"):
                if not prompt:
                    st.error("❗ Please enter a prompt.")
                    st.stop()

                with st.spinner("🔄 Generating image..."):
                    try:
                        image = generate_image_from_text_flux(
                            prompt,
                            aspect_ratio=aspect_ratio,
                            output_format=output_format,
                            output_quality=output_quality,
                            safety_tolerance=safety_tolerance,
                            prompt_upsampling=prompt_upsampling
                        )
                        if image:
                            image_path = f"replicate_image_{len(st.session_state.generations)+1}.{output_format}"
                            image.save(image_path)
                            st.session_state.generated_images.append(image)
                            st.session_state.generations.append({
                                "id": f"replicate_{len(st.session_state.generations)+1}",
                                "type": "image",
                                "path": image_path,
                                "source": "Replicate AI",
                                "prompt": prompt,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            })

                            st.image(image, caption=f"Image {len(st.session_state.generated_images)}", use_column_width=True)
                            st.success("✅ Image generated and saved to history.")
                        else:
                            st.error("❌ Failed to generate image.")

                    except Exception as e:
                        st.error(f"❗ An error occurred: {e}")
                        st.error(traceback.format_exc())

        # ---------------------
        # RunwayML Image-to-Video
        # ---------------------
        elif mode == "RunwayML Image-to-Video":
            st.subheader("🎥 RunwayML Image-to-Video")
            prompt_image_url = st.text_input("📌 Enter the URL of the prompt image", key="runway_prompt_image_url")
            prompt_text = st.text_area("📝 Enter the text prompt for the video", "A futuristic cityscape at sunset", height=100, key="runway_prompt_text")

            if st.button("✨ Generate Video with RunwayML"):
                if not prompt_image_url:
                    st.error("❗ Please enter the URL of the prompt image.")
                    st.stop()
                if not prompt_text:
                    st.error("❗ Please enter a text prompt.")
                    st.stop()
                try:
                    st.success("🔄 Initiating RunwayML video generation...")
                    generate_video_runwayml(runway_api_key, prompt_image_url, prompt_text)
                except Exception as e:
                    st.error(f"❗ An unexpected error occurred with RunwayML: {e}")
                    st.error(traceback.format_exc())

        # ---------------------
        # Luma Integration
        # ---------------------
        elif mode == "Luma Integration":
            st.subheader("🎞️ Luma Integration")
            prompt = st.text_area("📝 Enter your prompt", "A teddy bear in sunglasses playing electric guitar and dancing", height=100, key="luma_prompt")
            aspect_ratio = st.selectbox("Aspect Ratio", ["9:16", "16:9", "1:1", "3:4", "4:3"], key="luma_aspect_ratio")
            loop = st.checkbox("🔁 Loop Video", value=False, key="luma_loop")

            # Camera Motions
            st.markdown("### 🎥 Camera Motion")
            try:
                supported_camera_motions = luma_client.generations.camera_motion.list()
                camera_motion = st.selectbox("Select Camera Motion", ["None"] + supported_camera_motions, key="luma_camera_motion")
                if camera_motion != "None":
                    prompt = f"{prompt}, {camera_motion}"
            except Exception as e:
                st.error(f"🚫 Could not fetch camera motions: {e}")
                camera_motion = None

            # Keyframes
            st.markdown("### 🎞️ Keyframes")
            keyframe_option = st.selectbox(
                "Select Keyframe Options",
                ["None", "Start Image", "End Image", "Start and End Image", "Start Generation", "End Generation", "Start and End Generation"],
                key="luma_keyframe_option"
            )
            keyframes = {}

            if keyframe_option in ["Start Image", "Start and End Image"]:
                start_image_url = st.text_input("📌 Start Image URL", key="luma_start_image_url")
                if start_image_url:
                    keyframes["frame0"] = {
                        "type": "image",
                        "url": start_image_url
                    }

            if keyframe_option in ["End Image", "Start and End Image"]:
                end_image_url = st.text_input("📌 End Image URL", key="luma_end_image_url")
                if end_image_url:
                    keyframes["frame1"] = {
                        "type": "image",
                        "url": end_image_url
                    }

            if keyframe_option in ["Start Generation", "Start and End Generation"]:
                start_generation_id = st.text_input("🔑 Start Generation ID", key="luma_start_generation_id")
                if start_generation_id:
                    keyframes["frame0"] = {
                        "type": "generation",
                        "id": start_generation_id
                    }

            if keyframe_option in ["End Generation", "Start and End Generation"]:
                end_generation_id = st.text_input("🔑 End Generation ID", key="luma_end_generation_id")
                if end_generation_id:
                    keyframes["frame1"] = {
                        "type": "generation",
                        "id": end_generation_id
                    }

            # Generate Button
            if st.button("✨ Generate Video with Luma AI"):
                if not prompt:
                    st.error("❗ Please enter a prompt.")
                    st.stop()

                try:
                    with st.spinner("🔄 Generating video with Luma AI..."):
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
                                st.error(f"❌ Generation failed: {generation.failure_reason}")
                                st.stop()
                            else:
                                st.write("⌛ Video generation in progress... Waiting for completion.")
                                time.sleep(5)

                        video_url = generation.assets.video

                        # Download video
                        response = requests.get(video_url)
                        video_path = f"{generation.id}.mp4"
                        with open(video_path, "wb") as f:
                            f.write(response.content)

                        st.session_state.generated_videos.append(video_path)
                        st.session_state.final_video = video_path
                        st.success(f"✅ Video generated and saved to {video_path}")
                        st.video(video_path)

                except Exception as e:
                    st.error(f"❗ An error occurred: {e}")
                    st.error(traceback.format_exc())

    # -------------------------
    # Images Tab
    # -------------------------
    with tab2:
        st.header("🖼️ Generated Images")
        if st.session_state.generated_images:
            st.write(f"### Total Images: {len(st.session_state.generated_images)}")
            # Display images in a responsive grid
            num_columns = 3
            for i in range(0, len(st.session_state.generated_images), num_columns):
                cols = st.columns(num_columns)
                for j in range(num_columns):
                    idx = i + j
                    if idx < len(st.session_state.generated_images):
                        with cols[j]:
                            st.image(st.session_state.generated_images[idx], use_column_width=True, caption=f"Image {idx + 1}")
        else:
            st.info("🎨 No images generated yet. Use the **Generator** tab to create images.")

    # -------------------------
    # Videos Tab
    # -------------------------
    with tab3:
        st.header("📽️ Generated Videos")
        if st.session_state.generated_videos:
            st.write(f"### Total Videos: {len(st.session_state.generated_videos)}")
            for i, video_path in enumerate(st.session_state.generated_videos):
                if os.path.exists(video_path):
                    st.write(f"#### Video {i+1}")
                    st.video(video_path)
                    with open(video_path, "rb") as f:
                        st.download_button(
                            label=f"📥 Download Video {i+1}",
                            data=f,
                            file_name=os.path.basename(video_path),
                            mime="video/mp4"
                        )
                else:
                    st.error(f"❌ Video file not found: {video_path}")
            
            # Final Video Display
            if st.session_state.final_video and os.path.exists(st.session_state.final_video):
                st.write(f"### 🎞️ Final Video: {st.session_state.final_video}")
                st.video(st.session_state.final_video)
                with open(st.session_state.final_video, "rb") as f:
                    st.download_button(
                        label="📥 Download Final Video",
                        data=f,
                        file_name=os.path.basename(st.session_state.final_video),
                        mime="video/mp4"
                    )
        else:
            st.info("📽️ No videos generated yet. Use the **Generator** tab to create videos.")
        
        # ---------------------
        # Download All Content as ZIP
        # ---------------------
        if st.session_state.generated_images or st.session_state.generated_videos:
            with st.expander("📦 Download All Content (ZIP)"):
                zip_path = create_zip_file(st.session_state.generated_images, st.session_state.generated_videos)
                if zip_path:
                    with open(zip_path, "rb") as f:
                        st.download_button(
                            label="📥 Download ZIP",
                            data=f,
                            file_name="generated_content.zip",
                            mime="application/zip"
                        )
                    # Optionally, remove the ZIP after download
                    os.remove(zip_path)
        else:
            st.info("📦 No content available for ZIP download.")

    # -------------------------
    # Footer Styling (Optional)
    # -------------------------
    st.markdown("""
    <style>
    /* Footer Styling */
    footer {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# Run the Application
# -----------------------------
if __name__ == "__main__":
    main()
