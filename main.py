import streamlit as st
from lumaai import LumaAI
import replicate
import openai
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
import threading
import json

# Redirect stderr to stdout
sys.stderr = sys.stdout

# Initialize session state
if 'generations' not in st.session_state:
    st.session_state.generations = []  # List to store generation metadata
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # Chat history with ChatGPT
if 'automation_tasks' not in st.session_state:
    st.session_state.automation_tasks = []  # List to store automation tasks
if 'users' not in st.session_state:
    st.session_state.users = {}  # Dictionary to store user sessions

# Load API keys from a config file for multi-user support (optional)
def load_api_keys():
    if os.path.exists("api_keys.json"):
        with open("api_keys.json", "r") as f:
            return json.load(f)
    else:
        return {}

def save_api_keys(api_keys):
    with open("api_keys.json", "w") as f:
        json.dump(api_keys, f)

def main():
    st.set_page_config(page_title="AI Video Suite", layout="wide")
    st.title("🚀 All-in-One AI Video Solution")

    # User Authentication (Simple Implementation)
    st.sidebar.header("User Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if username and password:
        # For simplicity, passwords are not hashed here
        if username not in st.session_state.users:
            st.session_state.users[username] = {'password': password, 'api_keys': {}}
        elif st.session_state.users[username]['password'] != password:
            st.error("Incorrect password")
            return
    else:
        st.warning("Please enter your username and password to proceed.")
        return

    # Sidebar for API Keys
    st.sidebar.header("API Keys")
    api_keys = st.session_state.users[username]['api_keys']
    luma_api_key = st.sidebar.text_input("Luma AI API Key", value=api_keys.get('luma_api_key', ''), type="password")
    stability_api_key = st.sidebar.text_input("Stability AI API Key", value=api_keys.get('stability_api_key', ''), type="password")
    replicate_api_key = st.sidebar.text_input("Replicate API Key", value=api_keys.get('replicate_api_key', ''), type="password")
    openai_api_key = st.sidebar.text_input("OpenAI API Key", value=api_keys.get('openai_api_key', ''), type="password")
    midjourney_api_key = st.sidebar.text_input("Midjourney API Key", value=api_keys.get('midjourney_api_key', ''), type="password")

    # Save API keys
    st.session_state.users[username]['api_keys'] = {
        'luma_api_key': luma_api_key,
        'stability_api_key': stability_api_key,
        'replicate_api_key': replicate_api_key,
        'openai_api_key': openai_api_key,
        'midjourney_api_key': midjourney_api_key
    }

    # Optionally, save to a config file
    # api_keys_data = load_api_keys()
    # api_keys_data[username] = st.session_state.users[username]['api_keys']
    # save_api_keys(api_keys_data)

    # Set Replicate API token
    if replicate_api_key:
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_key

    # Set OpenAI API key
    if openai_api_key:
        openai.api_key = openai_api_key

    # Initialize clients
    luma_client = LumaAI(auth_token=luma_api_key) if luma_api_key else None

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Generator", "History", "Edit", "Chat", "Automation", "Analytics"
    ])

    # -------------------------------------------
    # Generator Tab
    # -------------------------------------------
    with tab1:
        st.header("🎨 Content Generation")

        # Mode selection with icons
        mode = st.selectbox("Select Mode", [
            "🖼️ Text-to-Image (DALL·E 3)",
            "🎥 Text-to-Video (Luma AI)",
            "🖌️ Image Generation (Replicate AI)",
            "🎨 Text-to-Image (Midjourney)",
            "🎥 Image-to-Video (Stability AI)"
        ])

        if mode == "🎥 Text-to-Video (Luma AI)":
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
                            "user": username,
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
                        response = openai.Image.create(
                            prompt=prompt,
                            n=num_images,
                            size="1024x1024"
                        )
                        for i, img_data in enumerate(response['data']):
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
                                "user": username,
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
                            "user": username,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        })

                        st.image(image)
                        st.success("Image generated and saved to history.")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.error(traceback.format_exc())

        elif mode == "🎨 Text-to-Image (Midjourney)":
            if not midjourney_api_key:
                st.error("Midjourney API Key is required for this mode.")
                return
            prompt = st.text_area("Enter a prompt for image generation", "An abstract painting of a futuristic city")
            # Midjourney API integration would go here
            st.info("Midjourney integration is under development.")

        elif mode == "🎥 Image-to-Video (Stability AI)":
            if not stability_api_key:
                st.error("Stability AI API Key is required for this mode.")
                return
            st.info("This feature is under development.")

    # -------------------------------------------
    # History Tab
    # -------------------------------------------
    with tab2:
        st.header("📜 Generation History")
        user_gens = [gen for gen in st.session_state.generations if gen['user'] == username]
        if user_gens:
            for gen in user_gens[::-1]:  # Display newest first
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
            st.info("No generations yet. Generate content in the Generator tab.")

    # -------------------------------------------
    # Edit Tab
    # -------------------------------------------
    with tab3:
        st.header("✏️ Edit Generations")
        user_gens = [gen for gen in st.session_state.generations if gen['user'] == username]
        if not user_gens:
            st.info("No generations available to edit.")
            return

        # Select generation to edit
        gen_options = [f"{gen['id']} ({gen['type']}) - {gen['timestamp']}" for gen in user_gens]
        selected_gen = st.selectbox("Select a generation to edit", gen_options)
        selected_gen_index = gen_options.index(selected_gen)
        gen_to_edit = user_gens[selected_gen_index]

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
            if st.button("💾 Apply Edits"):
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
                            "user": username,
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
            st.subheader("Image Editing Tools")
            apply_filters = st.checkbox("Apply Filters", value=False)
            if apply_filters:
                brightness = st.slider("Brightness", 0.0, 2.0, 1.0)
                contrast = st.slider("Contrast", 0.0, 2.0, 1.0)
                saturation = st.slider("Saturation", 0.0, 2.0, 1.0)

            add_text_overlay = st.checkbox("Add Text Overlay", value=False)
            if add_text_overlay:
                overlay_text = st.text_input("Overlay Text", "Sample Text")
                font_size = st.slider("Font Size", 10, 100, 40)
                font_color = st.color_picker("Font Color", "#FFFFFF")
                text_position = st.selectbox("Text Position", ["Top", "Center", "Bottom"])

            # Apply Edits
            if st.button("💾 Apply Edits"):
                with st.spinner("Applying edits..."):
                    try:
                        image = Image.open(gen_to_edit['path'])

                        if apply_filters:
                            enhancer = ImageEnhance.Brightness(image)
                            image = enhancer.enhance(brightness)
                            enhancer = ImageEnhance.Contrast(image)
                            image = enhancer.enhance(contrast)
                            enhancer = ImageEnhance.Color(image)
                            image = enhancer.enhance(saturation)

                        if add_text_overlay:
                            draw = ImageDraw.Draw(image)
                            font = ImageFont.truetype("arial.ttf", font_size)
                            text_width, text_height = draw.textsize(overlay_text, font=font)
                            width, height = image.size

                            if text_position == "Top":
                                position = ((width - text_width) / 2, 10)
                            elif text_position == "Center":
                                position = ((width - text_width) / 2, (height - text_height) / 2)
                            else:  # Bottom
                                position = ((width - text_width) / 2, height - text_height - 10)

                            draw.text(position, overlay_text, fill=font_color, font=font)

                        # Save edited image
                        edited_image_path = f"{gen_to_edit['id']}_edited.png"
                        image.save(edited_image_path)

                        # Update generation history
                        st.session_state.generations.append({
                            "id": f"{gen_to_edit['id']}_edited",
                            "type": "image",
                            "path": edited_image_path,
                            "source": gen_to_edit['source'],
                            "prompt": gen_to_edit['prompt'],
                            "user": username,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        })

                        st.image(image)
                        st.success("Edits applied and new image saved to history.")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.error(traceback.format_exc())

    # -------------------------------------------
    # Chat Tab
    # -------------------------------------------
    with tab4:
        st.header("💬 Chat with AI Assistant")
        if not openai_api_key:
            st.error("OpenAI API Key is required for this feature.")
            return

        # Display chat history
        for chat in st.session_state.chat_history:
            if chat['role'] == "user":
                st.markdown(f"**{username}:** {chat['content']}")
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
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=messages
                        )
                        assistant_reply = response.choices[0].message['content']

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

    # -------------------------------------------
    # Automation Tab
    # -------------------------------------------
    with tab5:
        st.header("🤖 AI-Powered Automation")
        st.info("Set up automated workflows for content generation and editing.")

        automation_task = st.text_input("Define an automation task (e.g., 'Generate a daily image of a sunrise')")

        if st.button("Add Automation Task"):
            if automation_task:
                st.session_state.automation_tasks.append({
                    "task": automation_task,
                    "user": username,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                st.success("Automation task added.")
            else:
                st.warning("Please enter a task.")

        st.subheader("Your Automation Tasks")
        user_tasks = [task for task in st.session_state.automation_tasks if task['user'] == username]
        if user_tasks:
            for task in user_tasks:
                st.write(f"- {task['task']} (Added on {task['timestamp']})")
        else:
            st.info("No automation tasks added yet.")

    # -------------------------------------------
    # Analytics Tab
    # -------------------------------------------
    with tab6:
        st.header("📊 Analytics Dashboard")
        st.info("Monitor your content generation statistics and API usage.")

        user_gens = [gen for gen in st.session_state.generations if gen['user'] == username]
        total_generations = len(user_gens)
        image_generations = len([gen for gen in user_gens if gen['type'] == 'image'])
        video_generations = len([gen for gen in user_gens if gen['type'] == 'video'])

        st.subheader("Generation Statistics")
        st.write(f"- Total Generations: {total_generations}")
        st.write(f"- Images Generated: {image_generations}")
        st.write(f"- Videos Generated: {video_generations}")

        # Additional analytics can be added here

    # -------------------------------------------
    # Footer with style
    # -------------------------------------------
    st.markdown("""
    <style>
    .css-1d391kg {
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
