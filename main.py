import streamlit as st
from lumaai import LumaAI
import replicate
import requests
import time
import base64
from PIL import Image
import io
import os
import sys
import numpy as np
import traceback
from datetime import datetime, timedelta

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

def main():
    st.set_page_config(page_title="AI Content Suite", layout="wide")
    st.title("🚀 All-in-One AI Content Solution")

    # Sidebar with tabs: Settings, Chat, About
    with st.sidebar:
        sidebar_tabs = st.tabs(["Settings", "Chat", "About"])

        # Settings Tab
        with sidebar_tabs[0]:
            st.header("Settings")
            # API Keys
            st.subheader("API Keys")
            luma_api_key = st.text_input("Luma AI API Key", type="password")
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
            **All-in-One AI Content Solution**

            This application allows you to generate and analyze AI-powered images using various models like Luma AI, Replicate AI, and OpenAI's DALL·E 3. You can also interact with an AI assistant for guidance and automate tasks.

            **Features:**
            - Generate images from text prompts (DALL·E 3, Replicate AI)
            - Analyze generated content automatically
            - Chat with an AI assistant
            - Automate content generation workflows
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
        st.header("🤖 Workflow Automation")
        st.info("Set up workflows to automate your content creation process.")

        st.subheader("Create a Workflow")
        workflow_name = st.text_input("Workflow Name")
        selected_mode = st.selectbox("Select Generation Mode", ["🖼️ Text-to-Image (DALL·E 3)", "🖌️ Image Generation (Replicate AI)"])
        workflow_prompt = st.text_area("Prompt")

        if st.button("Save Workflow"):
            if workflow_name and selected_mode and workflow_prompt:
                st.session_state.automation_tasks.append({
                    "name": workflow_name,
                    "mode": selected_mode,
                    "prompt": workflow_prompt
                })
                st.success("Workflow saved.")
            else:
                st.error("Please fill in all the fields.")

        st.subheader("Your Workflows")
        if st.session_state.automation_tasks:
            for idx, task in enumerate(st.session_state.automation_tasks):
                st.write(f"**Workflow {idx + 1}: {task['name']}**")
                st.write(f"Mode: {task['mode']}")
                st.write(f"Prompt: {task['prompt']}")
                if st.button(f"Run Workflow {idx + 1}", key=f"run_{idx}"):
                    # Execute the workflow
                    if task['mode'] == "🖼️ Text-to-Image (DALL·E 3)":
                        prompt = task['prompt']
                        # Reuse the generate_text_to_image function
                        generate_text_to_image(prompt)
                    elif task['mode'] == "🖌️ Image Generation (Replicate AI)":
                        prompt = task['prompt']
                        # Reuse the generate_image_replicate function
                        generate_image_replicate(prompt)
        else:
            st.write("No workflows saved.")

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

# Function to generate image using DALL·E 3 (Reused in workflows)
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

        image_path = f"workflow_image_{len(st.session_state.generations)+1}.png"
        image.save(image_path)

        # Analyze image
        analysis = analyze_image(image)

        st.session_state.generations.append({
            "id": f"workflow_{len(st.session_state.generations)+1}",
            "type": "image",
            "path": image_path,
            "source": "Workflow Automation",
            "prompt": prompt,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "analysis": analysis
        })

        st.image(image)
        st.success("Image generated and saved to history.")

    except Exception as e:
        st.error(f"An error occurred during workflow execution: {e}")
        st.error(traceback.format_exc())

# Function to generate image using Replicate AI (Reused in workflows)
def generate_image_replicate(prompt):
    replicate_api_key = os.environ.get("REPLICATE_API_TOKEN")
    if not replicate_api_key:
        st.error("Replicate API Key is required for this feature.")
        return
    try:
        output = replicate.run(
            "black-forest-labs/flux-1.1-pro",
            input={
                "prompt": prompt,
                "aspect_ratio": "1:1",
                "output_format": "png",
                "output_quality": 80,
                "safety_tolerance": 2,
                "prompt_upsampling": True
            }
        )
        image_url = output[0]
        image_response = requests.get(image_url)
        image = Image.open(io.BytesIO(image_response.content))

        image_path = f"workflow_image_{len(st.session_state.generations)+1}.png"
        image.save(image_path)

        # Analyze image
        analysis = analyze_image(image)

        st.session_state.generations.append({
            "id": f"workflow_{len(st.session_state.generations)+1}",
            "type": "image",
            "path": image_path,
            "source": "Workflow Automation",
            "prompt": prompt,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "analysis": analysis
        })

        st.image(image)
        st.success("Image generated and saved to history.")

    except Exception as e:
        st.error(f"An error occurred during workflow execution: {e}")
        st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
