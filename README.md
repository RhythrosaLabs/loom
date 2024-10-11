All-in-One AI Video Solution

This Streamlit app is a powerful, customizable AI video solution that integrates both Luma AI and Stability AI APIs. It allows you to generate videos from text prompts or images, apply advanced video editing tools like filters and text overlays, and provides a sleek, modern interface.

Features

API Integration: Supports both Luma AI and Stability AI APIs.
Text-to-Video: Generate videos from text prompts.
Image-to-Video: Create videos starting from an image.
Camera Motion Control: Select camera motions to enhance your video.
Aspect Ratio Options: Choose from various aspect ratios.
Looping: Option to loop the generated video.
Keyframes: Use images or previous generations as start/end frames.
Video Editing Tools:
Filters: Adjust brightness, contrast, and saturation.
Text Overlays: Add customizable text to your videos.
Cropping: Planned for future implementation.
Modern UI: A sleek and intuitive interface.
Setup Instructions

Prerequisites
Python 3.7 or higher.
Luma AI API Key: Obtain from Luma AI Dream Machine API Keys.
Stability AI API Key: Obtain from Stability AI API Keys.
Installation
Clone the repository or download the files
bash
Copy code
git clone https://github.com/yourusername/ai-video-suite.git
cd ai-video-suite
Create a virtual environment (optional but recommended)
bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required packages
bash
Copy code
pip install -r requirements.txt
Running the App
Run the Streamlit app
bash
Copy code
streamlit run app.py
Open your web browser
Navigate to http://localhost:8501 if it doesn't open automatically.
Usage
Enter Your API Keys
Go to the sidebar and input your Luma AI and/or Stability AI API Keys.
Select Mode
Choose between "Text-to-Video (Luma AI)", "Text-to-Video (Stability AI)", or "Image-to-Video (Stability AI)".
Set Generation Parameters
Prompt: Enter the text prompt (if applicable).
Aspect Ratio: Select the desired aspect ratio.
Loop Video: Choose to loop the video.
Camera Motion: Select a camera motion to include in your prompt.
Keyframes: Optionally set start/end frames using images or generation IDs.
Advanced Options
Video Editing Tools:
Adjust brightness, contrast, and saturation.
Add text overlays with customizable font size, color, and position.
Generate Video
Click "Generate Video" and wait for the process to complete.
View and Download
The generated video will be displayed.
Click the download link to save the video.
Notes
API Keys: Ensure you have valid API keys for the services you wish to use.
Video Editing: Some features like cropping are planned for future implementation.
Stability AI Modes: The Stability AI integration is under development.
Troubleshooting
Errors: Check if your API keys and inputs are correct.
Dependencies: Ensure all packages in requirements.txt are installed.
Logs: Refer to the console or terminal for error messages.
License
This project is licensed under the MIT License.

Acknowledgments
Luma AI for their Dream Machine API.
Stability AI for their text-to-image and image-to-video APIs.
Streamlit for providing an easy-to-use web app framework.
