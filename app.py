import os
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import logging
from threading import Thread

from modules.script_generator import ScriptGenerator
from modules.video_generator import VideoGenerator
from modules.audio_generator import AudioGenerator
from modules.video_editor import VideoEditor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload

# Create required directories if they don't exist
os.makedirs('outputs', exist_ok=True)
os.makedirs('outputs/scripts', exist_ok=True)
os.makedirs('outputs/videos', exist_ok=True)
os.makedirs('outputs/audio', exist_ok=True)
os.makedirs('outputs/final', exist_ok=True)

# Initialize modules
script_generator = ScriptGenerator()
video_generator = VideoGenerator()
audio_generator = AudioGenerator()
video_editor = VideoEditor()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global job storage
jobs = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_video():
    prompt = request.form.get('prompt', '')
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    # Generate a unique job ID
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'status': 'processing',
        'progress': 0,
        'message': 'Starting video generation...',
        'output_file': None
    }
    
    # Start generation process in background
    thread = Thread(target=process_video, args=(prompt, job_id))
    thread.daemon = True
    thread.start()
    
    return jsonify({'job_id': job_id})

@app.route('/status/<job_id>', methods=['GET'])
def job_status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Invalid job ID'}), 404
    
    return jsonify(jobs[job_id])

@app.route('/download/<job_id>', methods=['GET'])
def download_video(job_id):
    if job_id not in jobs or jobs[job_id]['output_file'] is None:
        return jsonify({'error': 'Video not found'}), 404
    
    filename = jobs[job_id]['output_file']
    return send_from_directory('outputs/final', filename, as_attachment=True)

def process_video(prompt, job_id):
    try:
        # 1. Generate script
        update_job_status(job_id, 10, 'Generating script...')
        script_data = script_generator.generate(prompt)
        script_path = f"outputs/scripts/{job_id}.json"
        with open(script_path, 'w') as f:
            import json
            json.dump(script_data, f)
        
        # 2. Generate video for each scene
        update_job_status(job_id, 20, 'Generating video scenes...')
        scene_video_paths = []
        
        # Track successful and failed scenes
        failed_scenes = []
        
        for i, scene in enumerate(script_data['scenes']):
            try:
                update_job_status(job_id, 20 + (i+1)*30//len(script_data['scenes']), 
                                f'Generating scene {i+1} of {len(script_data["scenes"])}...')
                video_path = video_generator.generate_scene(scene['description'], f"{job_id}_scene_{i}")
                scene_video_paths.append(video_path)
            except Exception as e:
                logger.error(f"Error generating scene {i}: {str(e)}")
                failed_scenes.append(i)
                # Create a fallback scene if CogVideoX fails
                fallback_text = f"Scene {i+1}: {scene['description'][:50]}..."
                fallback_path = create_fallback_scene(fallback_text, f"{job_id}_scene_{i}_fallback")
                scene_video_paths.append(fallback_path)
        
        # 3. Generate audio (narration and dialogue)
        update_job_status(job_id, 50, 'Generating audio...')
        narration_path = audio_generator.generate_narration(script_data['narration'], f"{job_id}_narration")
        
        # 4. Generate background music
        update_job_status(job_id, 70, 'Generating background music...')
        music_paths = []
        for i, scene in enumerate(script_data['scenes']):
            music_path = audio_generator.generate_music(scene['mood'], f"{job_id}_music_{i}")
            music_paths.append(music_path)
        
        # 5. Edit and combine all elements
        update_job_status(job_id, 80, 'Editing final video...')
        output_filename = f"{job_id}_final.mp4"
        video_editor.create_final_video(
            scene_video_paths,
            narration_path,
            music_paths,
            script_data,
            output_filename
        )
        
        update_job_status(job_id, 100, 'Video generation complete!', output_filename)
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        update_job_status(job_id, -1, f"Error: {str(e)}")

def create_fallback_scene(text, scene_id):
    """Create a simple fallback scene when video generation fails."""
    import cv2
    import numpy as np
    
    # Create output directory if it doesn't exist
    os.makedirs("outputs/videos", exist_ok=True)
    
    # Path for the output video
    output_path = f"outputs/videos/{scene_id}.mp4"
    
    # Generate a sequence of simple frames
    frames = []
    for i in range(5):  # 5 frames
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Add a gradient background
        for y in range(720):
            for x in range(1280):
                frame[y, x, 0] = int(255 * (x / 1280))  # Blue
                frame[y, x, 1] = int(255 * (y / 720))   # Green
        
        # Add a simple animation
        radius = 100 + i * 20
        cv2.circle(frame, (640, 360), radius, (0, 0, 255), -1)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Video Generation Error", (400, 300), font, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, text, (300, 360), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame {i+1}", (600, 420), font, 0.7, (255, 255, 255), 2)
        frames.append(frame)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, 1, (1280, 720))
    
    # Add each frame to the video
    for frame in frames:
        video_writer.write(frame)
    
    # Release the video writer
    video_writer.release()
    
    return output_path

def update_job_status(job_id, progress, message, output_file=None):
    if job_id in jobs:
        jobs[job_id]['progress'] = progress
        jobs[job_id]['message'] = message
        if progress == 100 and output_file:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['output_file'] = output_file
        elif progress == -1:
            jobs[job_id]['status'] = 'error'

if __name__ == '__main__':
    app.run(debug=True) 