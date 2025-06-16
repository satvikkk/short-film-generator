# VideoGen Agent

A tool that generates short films from text prompts. Enter a prompt like "A lonely robot on Mars finds a plant" and get a 1-2 minute short film with visuals, background music, voiceover narration, and transitions.

## Features

- **Script Generation**: Transforms your prompt into a detailed script with scenes and narration using Phi-4-mini-instruct
- **Scene-by-Scene Video Generation**: Creates high-quality video clips for each scene using CogVideoX
- **Voiceover & Dialogue Generation**: Converts narration and character dialogue to audio
- **Background Music & Sound Effects**: Adds mood-appropriate audio based on scene context
- **Automatic Video Editing**: Assembles all components into a final video with transitions

## Requirements

- Python 3.8+ with pip
- [FFmpeg](https://ffmpeg.org/download.html) installed and available in PATH
- Enough disk space for temporary files (~2GB)
- For optimal performance: NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- Apple Silicon Macs can use MPS acceleration for video generation

## Setup

1. Clone this repository
   ```
   git clone https://github.com/yourusername/videogen-agent.git
   cd videogen-agent
   ```

2. Run the setup script:
   ```
   ./run.sh
   ```

   Or manually:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. (Optional) Create a `.env` file from the example:
   ```
   cp .env.example .env
   ```
   Then edit `.env` to add your Hugging Face token if you have one.

## Usage

### Web Interface

1. Start the application:
   ```
   python app.py
   ```

2. Open `http://localhost:5000` in your browser
3. Enter your prompt and click "Generate Video"
4. Wait for the processing to complete (may take 5-20 minutes depending on hardware)
5. Download or play your generated video

### Resource Usage

This application uses several AI models:
- **Text generation**: Phi-4-mini-instruct (4B parameter LLM, much more efficient than larger models)
- **Video generation**: CogVideoX-2b (state-of-the-art text-to-video model)
- **Text-to-speech**: Simulated audio (synthetic waveform generation)
- **Music generation**: Simulated audio with mood parameters
- **Video editing**: FFmpeg via MoviePy

## Hardware Requirements

Different hardware configurations will affect performance:

| Hardware | Script Generation | Video Generation | Total Time |
|----------|------------------|------------------|------------|
| NVIDIA GPU (8GB+ VRAM) | ~10 seconds | ~2-5 mins per scene | ~10-15 mins |
| NVIDIA GPU (4GB VRAM) | ~20 seconds | ~5-10 mins per scene | ~15-30 mins |
| Apple Silicon (M1/M2/M3) | ~30 seconds | ~10-15 mins per scene | ~30-45 mins |
| CPU only | ~1 min | ~30+ mins per scene | ~2+ hours |

## Project Structure

```
videogen-agent/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── run.sh                  # Setup and run script
├── modules/                # Core functionality modules
│   ├── __init__.py         # Module initialization
│   ├── script_generator.py # Text-to-script generation with Phi-4-mini-instruct
│   ├── video_generator.py  # Scene-by-scene video generation with CogVideoX
│   ├── audio_generator.py  # Voiceover and music generation
│   └── video_editor.py     # Final video assembly
├── templates/              # Web interface templates
│   └── index.html          # Main web interface
├── outputs/                # Generated content (gitignored)
│   ├── scripts/            # Generated scripts in JSON format
│   ├── videos/             # Generated scene videos
│   ├── audio/              # Generated audio files
│   └── final/              # Final rendered videos
```

## Limitations

While this project uses state-of-the-art open models, there are still some limitations:

- Memory requirements for video generation (8GB+ VRAM recommended)
- Video generation can take significant time even on powerful hardware
- First-time model downloads may be large (several GB)
- Audio quality is basic compared to commercial text-to-speech

## Future Improvements

- Integration with more advanced audio models
- Support for more video styles and aspect ratios
- User customization of generation parameters
- Background task queue for handling multiple generation requests
- More advanced video transitions and effects

## License

This project is released under the MIT License. See the LICENSE file for details. 