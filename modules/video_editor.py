import os
import logging
import numpy as np
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips, concatenate_audioclips, CompositeAudioClip, TextClip, CompositeVideoClip
import moviepy.video.fx as vfx
import moviepy.audio.fx as afx

logger = logging.getLogger(__name__)

class VideoEditor:
    def __init__(self):
        """Initialize the video editor."""
        logger.info("Initializing VideoEditor...")
        try:
            # Initialize output directory
            os.makedirs("outputs/final", exist_ok=True)
            
            # Define transition settings
            self.transition_duration = 0.5  # seconds
            self.fade_duration = 0.8  # seconds
            self.text_duration = 2.0  # seconds for title screen
            
            logger.info("VideoEditor initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing VideoEditor: {e}")
            raise
    
    def create_final_video(self, scene_video_paths, narration_path, music_paths, script_data, output_filename):
        """
        Combine all video and audio elements into a final video.
        
        Args:
            scene_video_paths (list): Paths to individual scene videos.
            narration_path (str): Path to narration audio.
            music_paths (list): Paths to background music clips.
            script_data (dict): Script data with titles and other metadata.
            output_filename (str): Name for the output video file.
            
        Returns:
            str: Path to the final video.
        """
        logger.info("Assembling final video...")
        
        try:
            # Load all video clips
            video_clips = []
            audio_clips = []
            current_time = 0
            total_duration = 0
            
            # Create title screen
            title_text = script_data.get("title", "Generated Short Film")
            title_clip = self._create_title_clip(title_text, self.text_duration)
            video_clips.append(title_clip)
            total_duration += title_clip.duration
            
            # Load narration audio
            narration_audio = AudioFileClip(narration_path)
            
            # Process each scene with transitions
            for i, video_path in enumerate(scene_video_paths):
                # Load video clip
                scene_clip = VideoFileClip(video_path)
                
                # Apply fade in/out
                scene_clip = scene_clip.fx(vfx.fadein, self.fade_duration/2)
                scene_clip = scene_clip.fx(vfx.fadeout, self.fade_duration/2)
                
                # Add scene to video clips
                video_clips.append(scene_clip)
                total_duration += scene_clip.duration
                
                # Load background music for this scene if available
                if i < len(music_paths):
                    music_clip = AudioFileClip(music_paths[i])
                    
                    # Loop music to match scene duration if needed
                    if music_clip.duration < scene_clip.duration:
                        repeats = int(np.ceil(scene_clip.duration / music_clip.duration))
                        music_clip = concatenate_audioclips([music_clip] * repeats)
                    
                    # Trim music to match scene length
                    music_clip = music_clip.subclip(0, scene_clip.duration)
                    
                    # Adjust volume
                    music_clip = music_clip.fx(afx.volumex, 0.3)  # Lower volume for background music
                    
                    # Set start time
                    music_clip = music_clip.set_start(current_time)
                    audio_clips.append(music_clip)
                
                # Update current time
                current_time += scene_clip.duration
            
            # Set narration to span the entire video
            if narration_audio.duration > total_duration:
                narration_audio = narration_audio.subclip(0, total_duration)
            narration_audio = narration_audio.fx(afx.volumex, 0.8)  # Ensure narration is clearer than music
            audio_clips.append(narration_audio)
            
            # Concatenate all video clips with crossfade transitions
            final_video = concatenate_videoclips(video_clips, method="chain")
            
            # Combine all audio clips
            final_audio = CompositeAudioClip(audio_clips)
            
            # Set audio to video
            final_video = final_video.set_audio(final_audio)
            
            # Add ending credits
            credits_text = f"Generated from prompt: \"{script_data.get('prompt', '')}\""
            credits_clip = self._create_credits_clip(credits_text, self.text_duration)
            
            # Add credits to the end of the video
            final_video = concatenate_videoclips([final_video, credits_clip], method="chain")
            
            # Write final video to file
            output_path = f"outputs/final/{output_filename}"
            final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=24, threads=4)
            
            # Close all clips to free resources
            self._close_clips([final_video, final_audio] + video_clips + audio_clips)
            
            logger.info(f"Final video created at {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating final video: {e}")
            # Create a simple fallback video if creation fails
            return self._create_fallback_final_video(output_filename)
    
    def _create_title_clip(self, title_text, duration):
        """Create a title screen clip with the given text."""
        try:
            # Create text clip with MoviePy v2 syntax
            txt_clip = TextClip(
                font="Arial",
                text=title_text,
                font_size=70,
                color='white',
                size=(1280, 720),
                method='caption',
                align='center',
                bg_color='black'
            ).with_duration(duration)
            
            # Add fade in/out
            txt_clip = txt_clip.fx(vfx.fadein, self.fade_duration/2)
            txt_clip = txt_clip.fx(vfx.fadeout, self.fade_duration/2)
            
            return txt_clip
        except Exception as e:
            logger.error(f"Error creating title clip: {e}")
            # Return a simple black screen as fallback
            return self._create_blank_clip(duration)
    
    def _create_credits_clip(self, text, duration):
        """Create an ending credits clip."""
        try:
            # Create text clip with MoviePy v2 syntax
            txt_clip = TextClip(
                font="Arial",
                text=text,
                font_size=40,
                color='white',
                size=(1280, 720),
                method='caption',
                align='center',
                bg_color='black'
            ).with_duration(duration)
            
            # Add fade in/out
            txt_clip = txt_clip.fx(vfx.fadein, self.fade_duration/2)
            txt_clip = txt_clip.fx(vfx.fadeout, self.fade_duration/2)
            
            return txt_clip
        except Exception as e:
            logger.error(f"Error creating credits clip: {e}")
            # Return a simple black screen as fallback
            return self._create_blank_clip(duration)
    
    def _create_blank_clip(self, duration):
        """Create a blank black clip for use as a fallback."""
        clip = TextClip(
            font="Arial",
            text=" ",
            font_size=1,
            color='white',
            size=(1280, 720),
            bg_color='black'
        ).with_duration(duration)
        return clip
    
    def _close_clips(self, clips):
        """Close all clips to free resources."""
        for clip in clips:
            try:
                if hasattr(clip, 'close'):
                    clip.close()
            except Exception as e:
                logger.warning(f"Error closing clip: {e}")
    
    def _create_fallback_final_video(self, output_filename):
        """Create a simple fallback video if the main process fails."""
        output_path = f"outputs/final/{output_filename}"
        
        try:
            # Create a simple text clip with MoviePy v2 syntax
            error_text = "Video generation encountered an error.\nPlease try again."
            error_clip = TextClip(
                font="Arial",
                text=error_text,
                font_size=70,
                color='white',
                size=(1280, 720),
                method='caption',
                align='center',
                bg_color='black'
            ).with_duration(5)  # 5 second video
            
            # Write fallback video to file
            error_clip.write_videofile(output_path, codec="libx264", fps=24)
            
            # Close clip
            error_clip.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating fallback video: {e}")
            return None 