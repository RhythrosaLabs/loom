Luma AI Dream Machine Streamlit App

This Streamlit app allows you to generate videos using the Luma AI Dream Machine API. You can input your API key, set various parameters, and generate videos based on text prompts and/or images.

Features

API Key Input: Securely input your Luma AI API Key.
Text Prompt: Provide a text prompt for video generation.
Aspect Ratio: Select from available aspect ratios.
Loop Option: Choose to loop the generated video.
Keyframes: Use start and/or end images as keyframes.
Video Display: View the generated video within the app.
Download Link: Download the generated video directly.
Setup Instructions

Prerequisites
Python 3.7 or higher installed.
A Luma AI API Key. Obtain one from Luma AI Dream Machine API Keys.
Installation
Clone the repository or download the files
bash
Copy code
git clone https://github.com/yourusername/lumaai-streamlit-app.git
cd lumaai-streamlit-app
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
The app should automatically open. If not, navigate to http://localhost:8501.
Usage
Enter Your API Key
Go to the sidebar and input your Luma AI API Key.
Set Generation Parameters
Prompt: Enter the text prompt.
Aspect Ratio: Choose an aspect ratio.
Loop Video: Check if you want the video to loop.
Keyframes
Keyframe Options: Select keyframe usage.
Start Image URL: Provide if using a start image.
End Image URL: Provide if using an end image.
Generate Video
Click "Generate Video" and wait for the process to complete.
View and Download
The generated video will be displayed.
Click the download link to save the video.
Notes
Image URLs: Provide accessible image URLs for keyframes.
API Limits: Be aware of any rate limits with your API key.
Troubleshooting
Errors: Check if your API key and inputs are correct.
Connection: Ensure you have internet access.
Logs: Refer to the console or terminal for error messages.
License
This project is licensed under the MIT License.
