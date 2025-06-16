import os
import torch
import logging
import numpy as np
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
import cv2
import gc

logger = logging.getLogger(__name__)

class VideoGenerator:
    def __init__(self):
        """Initialize the video generator with CogVideoX model from Hugging Face."""
        logger.info("Initializing VideoGenerator...")
        try:
            # Check for CUDA or MPS availability
            if torch.cuda.is_available():
                device = "cuda"
                dtype = torch.float16
                logger.info("Using CUDA for video generation with float16")
            elif torch.backends.mps.is_available():
                device = "mps"
                dtype = torch.float16  # Try float16 on MPS too
                # Set MPS memory limit
                os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.5"  # Reduce to 50% of available memory
                logger.info("Using MPS (Apple Silicon) for video generation with float16")
            else:
                device = "cpu"
                dtype = torch.float16
                logger.warning("Neither CUDA nor MPS available. Running on CPU will be very slow.")
            
            # Load CogVideoX model with memory optimizations
            self.pipeline = CogVideoXPipeline.from_pretrained(
                "THUDM/CogVideoX-2b",
                torch_dtype=dtype,
                use_safetensors=True,  # More memory efficient
                low_cpu_mem_usage=True,
            )
            
            # Move to device after loading to control memory usage
            self.pipeline = self.pipeline.to(device)
            
            # Enable memory efficient attention if on CUDA
            if device == "cuda":
                self.pipeline.enable_xformers_memory_efficient_attention()
                self.pipeline.transformer.to(memory_format=torch.channels_last)
                torch.cuda.empty_cache()
                try:
                    self.pipeline.transformer = torch.compile(
                        self.pipeline.transformer, mode="reduce-overhead", fullgraph=True
                    )
                    logger.info("Successfully compiled transformer for optimization")
                except Exception as e:
                    logger.warning(f"Could not compile transformer: {e}")
            
            # Set model to evaluation mode to save memory
            self.pipeline.transformer.eval()
            
            self.device = device
            self.dtype = dtype
                
            logger.info(f"VideoGenerator initialized successfully on {device}.")
        except Exception as e:
            logger.error(f"Error initializing VideoGenerator: {e}")
            # Try fallback to CPU
            try:
                logger.info("Attempting fallback to CPU with minimal memory usage...")
                self.pipeline = CogVideoXPipeline.from_pretrained(
                    "THUDM/CogVideoX-2b",
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    low_cpu_mem_usage=True,
                ).to("cpu")
                self.device = "cpu"
                self.dtype = torch.float16
                logger.info("Fallback to CPU successful.")
            except Exception as e2:
                logger.error(f"Fallback failed: {e2}")
                raise
    
    def generate_scene(self, description, scene_id):
        """
        Generate a video clip for a scene description using CogVideoX.
        
        Args:
            description (str): The scene description.
            scene_id (str): Unique identifier for the scene.
            
        Returns:
            str: Path to the generated video file.
        """
        logger.info(f"Generating video for scene: {description[:50]}...")
        
        # Create output directory if it doesn't exist
        os.makedirs("outputs/videos", exist_ok=True)
        
        # Path for the output video
        output_path = f"outputs/videos/{scene_id}.mp4"
        
        try:
            # Free up memory before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Simplify the prompt to reduce complexity
            simplified_prompt = self._simplify_prompt(description)
            
            # Generate video with CogVideoX using memory-optimized settings
            video_frames = self.pipeline(
                prompt=simplified_prompt,
                guidance_scale=3.5,  # Further reduced from 4.5 to save memory
                num_inference_steps=20,  # Further reduced from 30 to save time and memory
                height=192,  # Further reduced resolution
                width=192,  # Further reduced resolution
            ).frames[0]
            
            # Export video to file
            temp_path = f"outputs/videos/{scene_id}_temp.mp4"
            export_to_video(video_frames, temp_path, fps=8)
            
            # Optionally post-process the video (e.g., resize, add effects)
            self._post_process_video(temp_path, output_path)
            
            # Clean up temp file if needed
            if os.path.exists(temp_path) and temp_path != output_path:
                os.remove(temp_path)
            
            # Free memory after generation
            del video_frames
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            logger.info(f"Video generated at {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating video: {e}")
            # Create a fallback video
            return self._create_fallback_video(scene_id, description)
    
    def _simplify_prompt(self, description):
        """Simplify the prompt to reduce memory usage while maintaining quality."""
        # Truncate description if too long
        if len(description) > 100:  # Further reduced from 200
            description = description[:100]
        
        # Add just one enhancement instead of multiple
        enhancement = np.random.choice([
            "cinematic quality",
            "detailed",
            "professional lighting"
        ])
        
        # Combine the original description with the enhancement
        simplified_prompt = f"{description}, {enhancement}"
        logger.info(f"Simplified prompt: {simplified_prompt}")
        
        return simplified_prompt
    
    def _post_process_video(self, input_path, output_path):
        """Apply post-processing to the generated video."""
        # Load the video
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Optionally upscale to 480p if the input is smaller
        target_width = max(width, 640)
        target_height = max(height, 480)
        
        # Create video writer for the processed video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
        
        # Process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize if needed
            if width != target_width or height != target_height:
                frame = cv2.resize(frame, (target_width, target_height))
            
            # Write the processed frame
            out.write(frame)
            
        # Release resources
        cap.release()
        out.release()
    
    def _create_fallback_video(self, scene_id, description):
        """Create a simple fallback video when generation fails."""
        output_path = f"outputs/videos/{scene_id}_fallback.mp4"
        
        # Create a simple video with text
        width, height = 640, 480
        fps = 8
        duration = 5  # seconds
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Generate frames
        for i in range(fps * duration):
            # Create a gradient background
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    frame[y, x, 0] = int(255 * (x / width))  # Blue
                    frame[y, x, 1] = int(255 * (y / height))  # Green
            
            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, "Scene Description:", (50, 100), font, 1, (255, 255, 255), 2)
            
            # Wrap text
            words = description.split()
            line = ""
            y_pos = 150
            for word in words:
                test_line = line + word + " "
                text_size = cv2.getTextSize(test_line, font, 0.7, 1)[0]
                if text_size[0] > width - 100:
                    cv2.putText(frame, line, (50, y_pos), font, 0.7, (255, 255, 255), 1)
                    y_pos += 30
                    line = word + " "
                else:
                    line = test_line
            
            # Add the last line
            if line:
                cv2.putText(frame, line, (50, y_pos), font, 0.7, (255, 255, 255), 1)
            
            # Add frame counter
            cv2.putText(frame, f"Frame {i+1}/{fps * duration}", (width - 200, height - 30), 
                       font, 0.6, (255, 255, 255), 1)
            
            # Write frame
            out.write(frame)
        
        # Release resources
        out.release()
        
        return output_path 