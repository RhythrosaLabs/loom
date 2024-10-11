import streamlit as st
from lumaai import LumaAI
import requests
import time

def main():
    st.title("Luma AI Dream Machine")

    # Sidebar for API Key
    st.sidebar.header("API Authentication")
    api_key = st.sidebar.text_input("Enter your Luma AI API Key", type="password")

    if api_key:
        client = LumaAI(auth_token=api_key)

        st.header("Generation Parameters")

        # Input fields
        prompt = st.text_input("Prompt", "A teddy bear in sunglasses playing electric guitar and dancing")

        aspect_ratio = st.selectbox("Aspect Ratio", ["9:16", "16:9", "1:1", "3:4", "4:3"])

        loop = st.checkbox("Loop Video", value=False)

        # Keyframes
        st.subheader("Keyframes")
        keyframe_option = st.selectbox("Keyframe Options", ["None", "Start Image", "End Image", "Start and End Image"])
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

        # Submit button
        if st.button("Generate Video"):
            with st.spinner("Generating video... this may take a few minutes."):
                try:
                    generation = client.generations.create(
                        prompt=prompt,
                        aspect_ratio=aspect_ratio,
                        loop=loop,
                        keyframes=keyframes if keyframes else None
                    )
                    completed = False
                    while not completed:
                        generation = client.generations.get(id=generation.id)
                        if generation.state == "completed":
                            completed = True
                        elif generation.state == "failed":
                            st.error(f"Generation failed: {generation.failure_reason}")
                            return
                        else:
                            time.sleep(5)

                    video_url = generation.assets.video

                    # Display video
                    st.video(video_url)

                    # Option to download video
                    st.markdown(f"[Download Video]({video_url})")

                except Exception as e:
                    st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter your Luma AI API Key to proceed.")

if __name__ == "__main__":
    main()
