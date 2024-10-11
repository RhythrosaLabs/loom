import streamlit as st
from lumaai import LumaAI
import replicate
import requests
import time
import base64
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import io
import os
import sys
import numpy as np
import traceback
from datetime import datetime, timedelta
import threading

# Redirect stderr to stdout
sys.stderr = sys.stdout

# Initialize session state
if 'generations' not in st.session_state:
    st.session_state.generations = []  # List to store generation metadata
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # Chat history with AI Assistant
if 'automation_tasks' not in st.session_state:
    st.session_state.automation_tasks = []  # List to store automation tasks
if 'assistant' not in st.session_state:
    st.session_state.assistant = None  # Assistant instance

# Function to analyze images using Replicate's image captioning model (BLIP)
def analyze_image(image):
    try:
        model = replicate.models.get("salesforce/blip")
        version = model.versions.get("c2e454e0c809c550b9d7386079468c9fc9ec4b2d2bcd68e048333209d289bc69")
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        output = version.predict(image=img_byte_arr)
        caption = output
        return caption
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return "An image was generated."

# Automation task runner
def run_automation_tasks():
    while True:
        now = datetime.now()
        for task in st.session_state.automation_tasks:
            task_time = task['time']
            if now >= task_time and not task.get('completed', False):
                st.write(f"Running automation task: {task['description']}")
                # Implement task execution based on task['action']
                if task['action'] == "Generate Text-to-Image":
                    # Call the text-to-image function
                    prompt = task['prompt']
                    generate_text_to_image(prompt)
                elif task['action'] == "Generate Text-to-Video":
                    prompt = task['prompt']
                    generate_text_to_video(prompt)
                task['completed'] = True
        time.sleep(60)

def generate_text_to_image(prompt):
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API Key is required for this feature.")
        return
    try:
        response = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_api_key}"
            },
            json={
                "prompt": prompt,
                "n": 1,
                "size": "1024x1024"
            }
        )
        response.raise_for_status()
        data = response.json()
        img_data = data['data'][0]
        image_url = img_data['url']
        image_response = requests.get(image_url)
        image = Image.open(io.BytesIO(image_response.content))

        image_path = f"automated_image_{len(st.session_state.generations)+1}.png"
        image.save(image_path)

        # Analyze image
        analysis = analyze_image(image)

        st.session_state.generations.append({
            "id": f"automated_{len(st.session_state.generations)+1}",
            "type": "image",
            "path": image_path,
            "source": "Automation",
            "prompt": prompt,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "analysis": analysis
        })

        st.image(image)
        st.success("Automated image generated and saved to history.")

    except Exception as e:
        st.error(f"An error occurred during automation: {e}")
        st.error(traceback.format_exc())

def generate_text_to_video(prompt):
    # Placeholder for text-to-video generation in automation
    st.info(f"Automated video generation is not implemented yet for prompt: {prompt}")

# Start automation thread
if 'automation_thread' not in st.session_state:
    st.session_state.automation_thread = threading.Thread(target=run_automation_tasks, daemon=True)
    st.session_state.automation_thread.start()

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

                        # Add context from generated content
                        for gen in st.session_state.generations[-5:]:  # Include last 5 generations
                            messages.append({"role": "system", "content": f"Generated content: {gen['analysis']}"})

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

            This application allows you to generate and analyze AI-powered images and videos using various models like Luma AI, Stability AI, and Replicate AI. You can also interact with an AI assistant for guidance and automate tasks.

            **Features:**
            - Generate videos from text prompts (Luma AI, Stability AI)
            - Generate images from text prompts (DALL·E 3, Replicate AI)
            - Analyze generated content automatically
            - Chat with an AI assistant
            - Automate content generation tasks
            """)

    # Main content with tabs: Generate, Automate, History
    main_tabs = st.tabs(["Generate", "Automate", "History"])

    # Generate Tab
    with main_tabs[0]:
        # Content for the Generate tab
        st.header("🎨 Content Generation")

        # Mode selection with icons
        mode = st.selectbox("Select Mode", [
            "🖼️ Text-to-Image (DALL·E 3)",
            "🖌️ Image Generation (Replicate AI)"
        ])

        if mode == "🖼️ Text-to-Image (DALL·E 3)":
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

                            # Analyze image
                            analysis = analyze_image(image)

                            st.session_state.generations.append({
                                "id": f"dalle_{len(st.session_state.generations)+1}_{i+1}",
                                "type": "image",
                                "path": image_path,
                                "source": "DALL·E 3",
                                "prompt": prompt,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "analysis": analysis
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

                        # Analyze image
                        analysis = analyze_image(image)

                        st.session_state.generations.append({
                            "id": f"replicate_{len(st.session_state.generations)+1}",
                            "type": "image",
                            "path": image_path,
                            "source": "Replicate AI",
                            "prompt": prompt,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "analysis": analysis
                        })

                        st.image(image)
                        st.success("Image generated and saved to history.")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.error(traceback.format_exc())

    # Automate Tab
    with main_tabs[1]:
        st.header("🤖 AI-Powered Automation")
        st.info("Set up automated tasks for content generation.")
        task_description = st.text_input("Task Description")
        task_datetime = st.datetime_input("Scheduled Time", datetime.now() + timedelta(minutes=1))
        task_action = st.selectbox("Action", ["Generate Text-to-Image"])
        task_prompt = st.text_area("Prompt")

        if st.button("Add Automation Task"):
            if task_description and task_datetime and task_action and task_prompt:
                st.session_state.automation_tasks.append({
                    "description": task_description,
                    "time": task_datetime,
                    "action": task_action,
                    "prompt": task_prompt,
                    "completed": False
                })
                st.success("Automation task added.")
            else:
                st.error("Please fill in all the fields.")

        st.subheader("Scheduled Tasks")
        if st.session_state.automation_tasks:
            for task in st.session_state.automation_tasks:
                st.write(f"Task: {task['description']}, Time: {task['time']}, Action: {task['action']}, Completed: {task['completed']}")
        else:
            st.write("No scheduled tasks.")

    # History Tab
    with main_tabs[2]:
        st.header("📜 Generation History")
        if st.session_state.generations:
            for gen in st.session_state.generations[::-1]:
                st.subheader(f"ID: {gen['id']} | Type: {gen['type']} | Source: {gen['source']} | Time: {gen['timestamp']}")
                st.write(f"Prompt: {gen['prompt']}")
                st.write(f"Analysis: {gen['analysis']}")
                if gen['type'] == "image":
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
