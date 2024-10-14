# AI Video Suite

AI Video Suite is a Streamlit-based application designed to generate and process videos and images using multiple AI services such as Luma AI, Replicate, Stable Diffusion, DALL·E, and RunwayML. This app allows users to create custom AI-generated media, process it, and download the results seamlessly.

## Features

- **Text-to-Image**: Generate images from text prompts using DALL·E, Stable Diffusion, or Flux.
- **Text-to-Video**: Generate videos from text prompts using Luma AI, Stable Diffusion, or RunwayML.
- **Image-to-Video**: Generate videos from an uploaded image using Luma AI or Stable Diffusion.
- **Video Concatenation**: Automatically merge generated video clips into a single video.
- **Downloadable Media**: Download generated images, videos, or a ZIP file of all generated media.

## How to Use

1. **API Keys**: 
   - Navigate to the **API Keys** tab on the sidebar.
   - Input your API keys for Luma AI, Stability AI, Replicate AI, OpenAI (DALL·E), and RunwayML.
   - Make sure your API keys are valid to access the respective AI services.

2. **Generate Content**:
   - Use the **Generator** tab to create images or videos based on text or uploaded images.
   - Select the AI service, provide the necessary prompts or images, and click **Generate**.

3. **Download**:
   - View generated content in the **Images** and **Videos** tabs.
   - Download individual files or all content as a ZIP file.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/ai-video-suite.git
    cd ai-video-suite
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:
    ```bash
    streamlit run app.py
    ```

## API Services

- **Luma AI**: For generating videos from text or images.
- **Stable Diffusion**: For generating images and videos from text prompts.
- **DALL·E**: Text-to-image generation using OpenAI's DALL·E.
- **Replicate AI**: Image generation using Flux models.
- **RunwayML**: For generating videos from text or images.

## Credits

- **Daniel Sheils**: [LinkedIn](http://linkedin.com/in/danielsheils/) | [Portfolio](https://danielsheils.myportfolio.com) | [Rhythrosa Labs](https://rhythrosalabs.com)
