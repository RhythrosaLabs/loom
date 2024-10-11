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
import traceback
from datetime import datetime

# Redirect stderr to stdout for better error visibility
sys.stderr = sys.stdout

# Initialize session state
if 'generations' not in st.session_state:
    st.session_state.generations = []  # List to store generation metadata
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # Chat history with AI Assistant

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

# Function to analyze video frames (first frame) using Replicate's BLIP model
def analyze_video_frame(video_path):
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            caption = analyze_image(image)
            return caption
        else:
            return "A video was generated."
    except Exception as e:
        st.error(f"Error analyzing video frame: {str(e)}")
        return "A video was generated."

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
            stability_api_key = st.text_input("Stability AI API Key", type="password")
            replicate_api_key = st.text_input("Replicate API Key", type="password")
            openai_api_key = st.text_input("OpenAI API Key", type="password")

            # Set Replicate API token
            if replicate_api_key:
                os.environ["REPLICATE_API_TOKEN"] = replicate_api_key

            # Set OpenAI API key
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key

            # Initialize Luma AI client
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
                            messages.append({"role": "system", "content": f"Generated content analysis: {gen['analysis']}"})

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

            This application allows you to generate and analyze AI-powered images and videos using various models like Luma AI, Stability AI, Replicate AI, and OpenAI's DALL·E 3. You can also interact with an AI assistant for guidance.

            **Features:**
            - Generate images from text prompts (DALL·E 3, Replicate AI)
            - Generate videos from text prompts (Luma AI, Stability AI)
            - Analyze generated content automatically
            - Chat with an AI assistant
            """)

    # Main content with tabs: Generate, History
    main_tabs = st.tabs(["Generate", "History"])

    # Generate Tab
    with main_tabs[0]:
        st.header("🎨 Content Generation")

        # Mode selection with icons
        mode = st.selectbox("Select Mode", [
            "🖼️ Text-to-Image (DALL·E 3)",
            "🖌️ Image Generation (Replicate AI)",
            "🎥 Text-to-Video (Luma AI)",
            "🎥 Text-to-Video (Stability AI)",
            "🎥 Image-to-Video (Stability AI)"
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
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "analysis": analysis
                            })

                            st.image(image, caption=f"DALL·E 3 Image: {prompt}")
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
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "analysis": analysis
                        })

                        st.image(image, caption=f"Replicate AI Image: {prompt}")
                        st.success("Image generated and saved to history.")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.error(traceback.format_exc())

        elif mode == "🎥 Text-to-Video (Luma AI)":
            if not luma_api_key:
                st.error("Luma AI API Key is required for this mode.")
                return
            prompt = st.text_area("Enter a prompt for video generation", "A futuristic cityscape at sunset")
            aspect_ratio = st.selectbox("Aspect Ratio", ["16:9", "9:16", "1:1", "3:4", "4:3"])
            loop = st.checkbox("Loop Video", value=False)

            if st.button("🎥 Generate Video"):
                with st.spinner("Generating video... this may take a few minutes."):
                    try:
                        # Prepare generation parameters
                        generation_params = {
                            "prompt": prompt,
                            "aspect_ratio": aspect_ratio,
                            "loop": loop,
                        }

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

                        # Analyze video frame
                        analysis = analyze_video_frame(video_path)

                        st.session_state.generations.append({
                            "id": generation.id,
                            "type": "video",
                            "path": video_path,
                            "source": "Luma AI",
                            "prompt": prompt,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "analysis": analysis
                        })

                        st.video(video_path)
                        st.success("Video generated and saved to history.")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.error(traceback.format_exc())

        elif mode == "🎥 Text-to-Video (Stability AI)":
            if not stability_api_key:
                st.error("Stability AI API Key is required for this mode.")
                return

            prompt = st.text_area("Enter a text prompt for video generation", height=100)

            if st.button("🎥 Generate Video"):
                with st.spinner("Generating video..."):
                    try:
                        # Stability AI Text-to-Video implementation
                        # Placeholder since actual API implementation may vary
                        st.info("Stability AI Text-to-Video is under development.")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.error(traceback.format_exc())

        elif mode == "🎥 Image-to-Video (Stability AI)":
            if not stability_api_key:
                st.error("Stability AI API Key is required for this mode.")
                return

            image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

            if st.button("🎥 Generate Video"):
                if image_file is None:
                    st.error("Please upload an image.")
                    return
                with st.spinner("Generating video..."):
                    try:
                        # Stability AI Image-to-Video implementation
                        # Placeholder since actual API implementation may vary
                        st.info("Stability AI Image-to-Video is under development.")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.error(traceback.format_exc())

    # History Tab
    with main_tabs[1]:
        st.header("📜 Generation History")
        if st.session_state.generations:
            for gen in st.session_state.generations[::-1]:
                st.subheader(f"ID: {gen['id']} | Type: {gen['type']} | Source: {gen['source']} | Time: {gen['timestamp']}")
                st.write(f"**Prompt:** {gen['prompt']}")
                st.write(f"**Analysis:** {gen['analysis']}")
                if gen['type'] == "image":
                    st.image(gen['path'])
                    with open(gen['path'], "rb") as f:
                        st.download_button("Download Image", f, file_name=os.path.basename(gen['path']))
                elif gen['type'] == "video":
                    st.video(gen['path'])
                    with open(gen['path'], "rb") as f:
                        st.download_button("Download Video", f, file_name=os.path.basename(gen['path']))
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
