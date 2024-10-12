import streamlit as st
from lumaai import LumaAI
import replicate
import requests
import time
import base64
from PIL import Image
import io
from moviepy.editor import (
    VideoFileClip, concatenate_videoclips, CompositeVideoClip, vfx, ImageClip
)
import os
import sys
import numpy as np
import traceback
import zipfile
import openai  # For DALL·E integration

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
    if (width, height) in [(1024, 576), (576, 1024), (768, 768), (1024, 1024)]:
        return image
    else:
        st.warning("Resizing image to 768x768 (default)")
        return image.resize((768, 768))

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
        image_url = output[0]
        image_response = requests.get(image_url)
        image = Image.open(io.BytesIO(image_response.content))
        return image
    except Exception as e:
        st.error(f"Error generating image with Flux: {e}")
        st.error(traceback.format_exc())
        return None

def generate_image_from_text_dalle(api_key, prompt, size):
    openai.api_key = api_key
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size=size,
            response_format="url"  # Get the image URL
        )
        image_url = response['data'][0]['url']
        image_response = requests.get(image_url)
        image = Image.open(io.BytesIO(image_response.content))
        return image
    except Exception as e:
        st.error(f"Error generating image with DALL·E: {e}")
        st.error(traceback.format_exc())
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

def main():
    st.set_page_config(page_title="AI Video Suite", layout="wide")
    st.title("All-in-One AI Video Solution")

    # Sidebar for API Keys
    st.sidebar.header("API Keys")
    luma_api_key = st.sidebar.text_input("Enter your Luma AI API Key", type="password")
    stability_api_key = st.sidebar.text_input("Enter your Stability AI API Key", type="password")
    replicate_api_key = st.sidebar.text_input("Enter your Replicate API Key", type="password")
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key (for DALL·E)", type="password")

    # Set Replicate API token
    if replicate_api_key:
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_key

    if not luma_api_key and not stability_api_key and not replicate_api_key and not openai_api_key:
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
            "Snapshot Mode",
            "Text-to-Video (Stability AI)",
            "Image-to-Video (Stability AI)",
            "Image Generation (Replicate AI)",
            "Luma"
        ])

        if mode == "Snapshot Mode":
            st.subheader("Snapshot Mode")
            snapshot_generator = st.selectbox("Select Image Generator", ["DALL·E", "Stable Diffusion", "Flux"])
            prompt = st.text_area("Enter a text prompt for Snapshot Mode", height=100)
            num_images = st.slider("Number of images to generate", 2, 300, 10)
            fps = st.slider("Frames per second", 1, 60, 24)
            if snapshot_generator in ["Flux", "DALL·E"]:
                aspect_ratio = st.selectbox("Aspect Ratio", ["1:1", "16:9", "9:16"])
            else:
                aspect_ratio = "1:1"

            # Check for required API keys
            if snapshot_generator == "Stable Diffusion" and not stability_api_key:
                st.error("Stability AI API Key is required for Stable Diffusion.")
                return
            if snapshot_generator == "Flux" and not replicate_api_key:
                st.error("Replicate API Key is required for Flux.")
                return
            if snapshot_generator == "DALL·E" and not openai_api_key:
                st.error("OpenAI API Key is required for DALL·E.")
                return

            if st.button("Generate Video"):
                if not prompt:
                    st.error("Please enter a text prompt.")
                    return

                try:
                    st.write(f"Generating {num_images} images using {snapshot_generator}...")
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
                            size = "1024x1024" if aspect_ratio == "1:1" else ("1024x576" if aspect_ratio == "16:9" else "576x1024")
                            image = generate_image_from_text_dalle(openai_api_key, prompt, size)
                        else:
                            st.error(f"Unsupported generator: {snapshot_generator}")
                            return
                        if image:
                            images.append(image)
                        else:
                            st.error(f"Failed to generate image {i+1}")
                    if images:
                        st.session_state.generated_images.extend(images)
                        st.write("Creating video from generated images...")
                        video_path = "snapshot_mode_video.mp4"
                        create_video_from_images(images, fps, video_path)
                        st.session_state.generated_videos.append(video_path)
                        st.session_state.final_video = video_path
                        st.success(f"Snapshot Mode video created: {video_path}")
                        st.video(video_path)
                    else:
                        st.error("Failed to generate images for Snapshot Mode.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
                    st.write("Error details:", str(e))
                    st.write("Traceback:", traceback.format_exc())

        elif mode == "Text-to-Video (Stability AI)":
            if not stability_api_key:
                st.error("Stability AI API Key is required for this mode.")
                return
            prompt = st.text_area("Enter a text prompt for video generation", height=100)
            cfg_scale = st.slider("CFG Scale (Stick to original image)", 0.0, 10.0, 1.8)
            motion_bucket_id = st.slider("Motion Bucket ID (Less motion to more motion)", 1, 255, 127)
            seed = st.number_input("Seed (0 for random)", min_value=0, max_value=4294967294, value=0)
            num_segments = st.slider("Number of video segments to generate", 1, 60, 5)
            crossfade_duration = st.slider("Crossfade Duration (seconds)", 0.0, 2.0, 0.0, 0.01)

            if st.button("Generate Video"):
                if not prompt:
                    st.error("Please enter a text prompt.")
                    return
                # Call the combined function for Text-to-Video
                try:
                    st.write("Generating image from text prompt...")
                    image = generate_image_from_text_stability(stability_api_key, prompt)
                    if image is None:
                        return
                    image = resize_image(image)
                    st.session_state.generated_images.append(image)
                    
                    video_clips = []
                    current_image = image

                    for i in range(num_segments):
                        st.write(f"Generating video segment {i+1}/{num_segments}...")
                        generation_id = start_video_generation(stability_api_key, current_image, cfg_scale, motion_bucket_id, seed)

                        if generation_id:
                            video_content = poll_for_video(stability_api_key, generation_id)

                            if video_content:
                                video_path = f"video_segment_{i+1}.mp4"
                                with open(video_path, "wb") as f:
                                    f.write(video_content)
                                st.write(f"Saved video segment to {video_path}")
                                video_clips.append(video_path)
                                st.session_state.generated_videos.append(video_path)

                                last_frame_image = get_last_frame_image(video_path)
                                if last_frame_image:
                                    current_image = last_frame_image
                                    st.session_state.generated_images.append(current_image)
                                else:
                                    st.warning(f"Could not extract last frame from segment {i+1}. Using previous image.")
                            else:
                                st.error(f"Failed to retrieve video content for segment {i+1}.")
                        else:
                            st.error(f"Failed to start video generation for segment {i+1}.")

                    if video_clips:
                        st.write("Concatenating video segments into one longform video...")
                        final_video, valid_clips = concatenate_videos(video_clips, crossfade_duration=crossfade_duration)
                        if final_video:
                            try:
                                final_video_path = "longform_video.mp4"
                                final_video.write_videofile(final_video_path, codec="libx264", audio_codec="aac")
                                st.session_state.final_video = final_video_path
                                st.success(f"Longform video created: {final_video_path}")
                                st.video(final_video_path)
                            except Exception as e:
                                st.error(f"Error writing final video: {str(e)}")
                                st.write("Traceback:", traceback.format_exc())
                            finally:
                                if final_video:
                                    final_video.close()
                                if valid_clips:
                                    for clip in valid_clips:
                                        clip.close()
                        else:
                            st.error("Failed to create the final video.")
                        
                        # Clean up individual video segments
                        for video_file in video_clips:
                            if os.path.exists(video_file):
                                os.remove(video_file)
                                st.write(f"Removed temporary file: {video_file}")
                            else:
                                st.warning(f"Could not find file to remove: {video_file}")
                    else:
                        st.error("No video segments were successfully generated.")

                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
                    st.write("Error details:", str(e))
                    st.write("Traceback:", traceback.format_exc())

        elif mode == "Image-to-Video (Stability AI)":
            if not stability_api_key:
                st.error("Stability AI API Key is required for this mode.")
                return
            image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
            cfg_scale = st.slider("CFG Scale (Stick to original image)", 0.0, 10.0, 1.8)
            motion_bucket_id = st.slider("Motion Bucket ID (Less motion to more motion)", 1, 255, 127)
            seed = st.number_input("Seed (0 for random)", min_value=0, max_value=4294967294, value=0)

            if st.button("Generate Video"):
                if not image_file:
                    st.error("Please upload an image.")
                    return
                image = Image.open(image_file)
                image = resize_image(image)
                st.session_state.generated_images.append(image)

                st.write("Generating video from uploaded image...")
                generation_id = start_video_generation(stability_api_key, image, cfg_scale, motion_bucket_id, seed)

                if generation_id:
                    video_content = poll_for_video(stability_api_key, generation_id)

                    if video_content:
                        video_path = "image_to_video.mp4"
                        with open(video_path, "wb") as f:
                            f.write(video_content)
                        st.write(f"Saved video to {video_path}")
                        st.session_state.generated_videos.append(video_path)
                        st.session_state.final_video = video_path
                        st.success(f"Image-to-Video created: {video_path}")
                        st.video(video_path)
                    else:
                        st.error("Failed to retrieve video content.")
                else:
                    st.error("Failed to start video generation.")

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

                            st.image(image)
                            st.success("Image generated and saved to history.")
                        else:
                            st.error("Failed to generate image.")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.error(traceback.format_exc())

        elif mode == "Luma":
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
                    st.write(f"Video {i+1}")
                    with open(video_path, "rb") as f:
                        st.download_button(f"Download Video {i+1}", f, file_name=f"video_{i+1}.mp4")
                else:
                    st.error(f"Video file not found: {video_path}")
            
            if st.session_state.final_video and os.path.exists(st.session_state.final_video):
                st.subheader("Final Video")
                st.video(st.session_state.final_video)
                with open(st.session_state.final_video, "rb") as f:
                    st.download_button("Download Final Video", f, file_name="final_video.mp4")
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
