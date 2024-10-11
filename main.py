import streamlit as st
from lumaai import LumaAI
import replicate
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
if 'generations' not in st.session_state:
    st.session_state.generations = []  # List to store generation metadata
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # Chat history with AI Assistant
if 'automation_tasks' not in st.session_state:
    st.session_state.automation_tasks = []  # List to store automation tasks

# Functions for Stability AI
def resize_image(image):
    width, height = image.size
    if (width, height) in [(1024, 576), (576, 1024), (768, 768)]:
        return image
    else:
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

def main():
    st.set_page_config(page_title="AI Video Suite", layout="wide")
    st.title("🚀 All-in-One AI Video Solution")

    # Sidebar with tabs: Settings, Chat, About
    with st.sidebar:
        sidebar_tabs = st.tabs(["Settings", "Chat", "About"])

        # Settings Tab
        with sidebar_tabs[0]:
            st.header("Settings")
            # API Keys
            st.subheader("API Keys")
            luma_api_key = st.text_input("Luma AI API Key", type="password")
            stability_api_key = st.text_input("Stability AI API Key", type="password")
            replicate_api_key = st.text_input("Replicate API Key", type="password")
            openai_api_key = st.text_input("OpenAI API Key", type="password")

            # Set Replicate API token
            if replicate_api_key:
                os.environ["REPLICATE_API_TOKEN"] = replicate_api_key

            # Set OpenAI API key
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key

            # Initialize clients
            luma_client = LumaAI(auth_token=luma_api_key) if luma_api_key else None

        # Chat Tab
        with sidebar_tabs[1]:
            st.header("Chat with AI Assistant")
            if not openai_api_key:
                st.error("OpenAI API Key is required for this feature.")
            else:
                # Display chat history
                for chat in st.session_state.chat_history:
                    if chat['role'] == "user":
                        st.markdown(f"**You:** {chat['content']}")
                    else:
                        st.markdown(f"**Assistant:** {chat['content']}")

                user_input = st.text_input("You:", key="chat_input")
                if st.button("Send", key="chat_send"):
                    if user_input:
                        # Prepare the messages
                        messages = [{"role": "system", "content": "You are an AI assistant that helps users with their generated content."}]
                        # Add chat history
                        for chat in st.session_state.chat_history:
                            messages.append(chat)
                        # Add user input
                        messages.append({"role": "user", "content": user_input})

                        with st.spinner("Assistant is typing..."):
                            try:
                                response = requests.post(
                                    "https://api.openai.com/v1/chat/completions",
                                    headers={
                                        "Content-Type": "application/json",
                                        "Authorization": f"Bearer {openai_api_key}"
                                    },
                                    json={
                                        "model": "gpt-3.5-turbo",
                                        "messages": messages
                                    }
                                )
                                response.raise_for_status()
                                assistant_reply = response.json()['choices'][0]['message']['content']

                                # Update chat history
                                st.session_state.chat_history.append({"role": "user", "content": user_input})
                                st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})

                                # Display assistant reply
                                st.markdown(f"**Assistant:** {assistant_reply}")

                            except Exception as e:
                                st.error(f"An error occurred: {e}")
                                st.error(traceback.format_exc())
                    else:
                        st.warning("Please enter a message.")

        # About Tab
        with sidebar_tabs[2]:
            st.header("About")
            st.info("""
            **All-in-One AI Video Solution**

            This application allows you to generate and edit AI-powered videos and images using various models like Luma AI, Stability AI, and Replicate AI. You can also interact with an AI assistant for guidance and automate tasks.

            **Features:**
            - Generate videos from text prompts (Luma AI, Stability AI)
            - Generate images from text prompts (DALL·E 3, Replicate AI)
            - Edit generated content with filters and text overlays
            - Chat with an AI assistant
            - Automate content generation tasks
            """)

    # Main content with tabs: Generate, Edit, Automate, History
    main_tabs = st.tabs(["Generate", "Edit", "Automate", "History"])

    # Generate Tab
    with main_tabs[0]:
        # Content for the Generate tab
        st.header("🎨 Content Generation")

        # Mode selection with icons
        mode = st.selectbox("Select Mode", [
            "🖼️ Text-to-Image (DALL·E 3)",
            "🎥 Text-to-Video (Luma AI)",
            "🖌️ Image Generation (Replicate AI)",
            "🎥 Text-to-Video (Stability AI)",
            "🎥 Image-to-Video (Stability AI)"
        ])

        if mode == "🎥 Text-to-Video (Stability AI)":
            if not stability_api_key:
                st.error("Stability AI API Key is required for this mode.")
                return

            prompt = st.text_area("Enter a text prompt for video generation", height=100)

            with st.expander("Settings", expanded=False):
                cfg_scale = st.slider("CFG Scale", 0.0, 10.0, 7.0)
                motion_bucket_id = st.slider("Motion Bucket ID", 1, 255, 127)
                seed = st.number_input("Seed (0 for random)", min_value=0, max_value=4294967294, value=0)

            if st.button("🚀 Generate Video"):
                if not prompt:
                    st.error("Please enter a text prompt.")
                    return
                with st.spinner("Generating video... this may take a few minutes."):
                    try:
                        # Generate initial image from text
                        image = generate_image_from_text(stability_api_key, prompt)
                        if image is None:
                            st.error("Failed to generate initial image.")
                            return

                        # Start video generation
                        generation_id = start_video_generation(stability_api_key, image, cfg_scale, motion_bucket_id, seed)
                        if generation_id is None:
                            st.error("Failed to start video generation.")
                            return

                        # Poll for video completion
                        video_content = poll_for_video(stability_api_key, generation_id)
                        if video_content is None:
                            st.error("Failed to retrieve video content.")
                            return

                        # Save video
                        video_path = f"stability_text_to_video_{len(st.session_state.generations)+1}.mp4"
                        with open(video_path, "wb") as f:
                            f.write(video_content)

                        # Save to generations
                        st.session_state.generations.append({
                            "id": f"stability_t2v_{len(st.session_state.generations)+1}",
                            "type": "video",
                            "path": video_path,
                            "source": "Stability AI",
                            "prompt": prompt,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        })

                        st.video(video_path)
                        st.success("Video generated and saved to history.")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.error(traceback.format_exc())

        elif mode == "🎥 Image-to-Video (Stability AI)":
            if not stability_api_key:
                st.error("Stability AI API Key is required for this mode.")
                return

            image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

            with st.expander("Settings", expanded=False):
                cfg_scale = st.slider("CFG Scale", 0.0, 10.0, 7.0)
                motion_bucket_id = st.slider("Motion Bucket ID", 1, 255, 127)
                seed = st.number_input("Seed (0 for random)", min_value=0, max_value=4294967294, value=0)

            if st.button("🚀 Generate Video"):
                if image_file is None:
                    st.error("Please upload an image.")
                    return
                with st.spinner("Generating video... this may take a few minutes."):
                    try:
                        # Load image
                        image = Image.open(image_file)
                        image = resize_image(image)

                        # Start video generation
                        generation_id = start_video_generation(stability_api_key, image, cfg_scale, motion_bucket_id, seed)
                        if generation_id is None:
                            st.error("Failed to start video generation.")
                            return

                        # Poll for video completion
                        video_content = poll_for_video(stability_api_key, generation_id)
                        if video_content is None:
                            st.error("Failed to retrieve video content.")
                            return

                        # Save video
                        video_path = f"stability_image_to_video_{len(st.session_state.generations)+1}.mp4"
                        with open(video_path, "wb") as f:
                            f.write(video_content)

                        # Save to generations
                        st.session_state.generations.append({
                            "id": f"stability_i2v_{len(st.session_state.generations)+1}",
                            "type": "video",
                            "path": video_path,
                            "source": "Stability AI",
                            "prompt": "Image-to-Video",
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        })

                        st.video(video_path)
                        st.success("Video generated and saved to history.")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.error(traceback.format_exc())

        elif mode == "🖼️ Text-to-Image (DALL·E 3)":
            if not openai_api_key:
                st.error("OpenAI API Key is required for this mode.")
                return
            prompt = st.text_area("Enter a prompt for image generation", "A surreal landscape with floating islands")

            num_images = st.slider("Number of images to generate", 1, 5, 1)

            if st.button("🖼️ Generate Image(s)"):
                with st.spinner("Generating image(s)..."):
                    try:
                        response = requests.post(
                            "https://api.openai.com/v1/images/generations",
                            headers={
                                "Content-Type": "application/json",
                                "Authorization": f"Bearer {openai_api_key}"
                            },
                            json={
                                "prompt": prompt,
                                "n": num_images,
                                "size": "1024x1024"
                            }
                        )
                        response.raise_for_status()
                        data = response.json()
                        for i, img_data in enumerate(data['data']):
                            image_url = img_data['url']
                            image_response = requests.get(image_url)
                            image = Image.open(io.BytesIO(image_response.content))

                            image_path = f"dalle_image_{len(st.session_state.generations)+1}_{i+1}.png"
                            image.save(image_path)

                            st.session_state.generations.append({
                                "id": f"dalle_{len(st.session_state.generations)+1}_{i+1}",
                                "type": "image",
                                "path": image_path,
                                "source": "DALL·E 3",
                                "prompt": prompt,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            })

                            st.image(image)
                        st.success("Image(s) generated and saved to history.")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.error(traceback.format_exc())

        elif mode == "🖌️ Image Generation (Replicate AI)":
            if not replicate_api_key:
                st.error("Replicate API Key is required for this mode.")
                return
            prompt = st.text_area("Enter a prompt for image generation", "A serene landscape with mountains and a river")
            aspect_ratio = st.selectbox("Aspect Ratio", ["1:1", "16:9", "9:16"])
            output_format = st.selectbox("Output Format", ["jpg", "png", "webp"])
            output_quality = st.slider("Output Quality", 1, 100, 80)
            safety_tolerance = st.slider("Safety Tolerance", 0, 5, 2)
            prompt_upsampling = st.checkbox("Prompt Upsampling", value=True)

            if st.button("🖌️ Generate Image"):
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

        elif mode == "🎥 Text-to-Video (Luma AI)":
            if not luma_api_key:
                st.error("Luma AI API Key is required for this mode.")
                return
            prompt = st.text_area("Prompt", "A futuristic cityscape at sunset")
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
            if st.button("🚀 Generate Video"):
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

    # Edit Tab
    with main_tabs[1]:
        st.header("✏️ Edit Generations")
        if not st.session_state.generations:
            st.info("No generations available to edit.")
        else:
            # Select generation to edit
            gen_options = [f"{gen['id']} ({gen['type']}) - {gen['timestamp']}" for gen in st.session_state.generations]
            selected_gen = st.selectbox("Select a generation to edit", gen_options)
            selected_gen_index = gen_options.index(selected_gen)
            gen_to_edit = st.session_state.generations[selected_gen_index]

            if gen_to_edit['type'] == "video":
                st.video(gen_to_edit['path'])
            elif gen_to_edit['type'] == "image":
                st.image(gen_to_edit['path'])

            # Video and Image Editing Tools
            # [Include editing tools as per previous code]

    # Automate Tab
    with main_tabs[2]:
        st.header("🤖 AI-Powered Automation")
        # [Implement automation functionality as per previous code]

    # History Tab
    with main_tabs[3]:
        st.header("📜 Generation History")
        if st.session_state.generations:
            for gen in st.session_state.generations[::-1]:
                st.subheader(f"ID: {gen['id']} | Type: {gen['type']} | Source: {gen['source']} | Time: {gen['timestamp']}")
                st.write(f"Prompt: {gen['prompt']}")
                if gen['type'] == "video":
                    st.video(gen['path'])
                    with open(gen['path'], "rb") as f:
                        st.download_button("Download Video", f, file_name=os.path.basename(gen['path']))
                elif gen['type'] == "image":
                    st.image(gen['path'])
                    with open(gen['path'], "rb") as f:
                        st.download_button("Download Image", f, file_name=os.path.basename(gen['path']))
                st.markdown("---")
        else:
            st.info("No generations yet. Generate content in the Generate tab.")

    # Apply custom styles
    st.markdown("""
    <style>
    .reportview-container {
        background-color: #0E1117;
    }
    .css-1v3fvcr {
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #1DB954;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
