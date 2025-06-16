import os
import torch
import logging
import numpy as np
from scipy.io import wavfile
import requests
from pydub import AudioSegment
import random
import json

logger = logging.getLogger(__name__)

class AudioGenerator:
    def __init__(self):
        """Initialize the audio generator."""
        logger.info("Initializing AudioGenerator...")
        try:
            # Initialize directories
            os.makedirs("outputs/audio", exist_ok=True)
            
            # Check for CUDA or MPS availability
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("Using CUDA for audio generation")
            elif torch.backends.mps.is_available():
                device = "mps"
                logger.info("Using MPS (Apple Silicon) for audio generation")
            else:
                device = "cpu"
                logger.warning("Neither CUDA nor MPS available. Running on CPU will be slow.")
            
            self.device = device
            
            # We'll use simple TTS for narration and music generation
            # In a production system, we would use models like:
            # - Facebook's MMS or Bark for TTS
            # - AudioCraft for music generation
            # But for this prototype, we'll use simpler approaches
            
            # Dictionary of mood-based music selections (could be expanded)
            self.mood_samples = {
                "happy": ["happy_1", "cheerful_1", "upbeat_1"],
                "sad": ["sad_1", "melancholy_1", "emotional_1"],
                "tense": ["suspense_1", "thriller_1", "dark_1"],
                "mysterious": ["mystery_1", "eerie_1", "strange_1"],
                "exciting": ["action_1", "adventure_1", "epic_1"],
                "calm": ["peaceful_1", "ambient_1", "relaxing_1"],
                "romantic": ["romance_1", "love_1", "heartfelt_1"],
                "scary": ["horror_1", "frightening_1", "spooky_1"],
                "suspenseful": ["tension_1", "anticipation_1", "buildup_1"],
                "hopeful": ["inspirational_1", "uplifting_1", "motivational_1"]
            }
            
            logger.info(f"AudioGenerator initialized successfully on {device}.")
        except Exception as e:
            logger.error(f"Error initializing AudioGenerator: {e}")
            raise
    
    def generate_narration(self, narration_text, output_id):
        """
        Generate audio for narration text.
        
        Args:
            narration_text (str): The narration text to convert to speech.
            output_id (str): Identifier for the output file.
            
        Returns:
            str: Path to the generated audio file.
        """
        logger.info(f"Generating narration audio for: {narration_text[:50]}...")
        
        output_path = f"outputs/audio/{output_id}_narration.wav"
        
        try:
            # In a production environment, we would use a proper TTS model
            # For this prototype, we'll create a simulated TTS audio file
            self._create_simulated_speech(narration_text, output_path)
            
            logger.info(f"Narration audio generated at {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating narration: {e}")
            # Create fallback audio
            return self._create_fallback_audio(output_id, "narration")
    
    def generate_music(self, mood, output_id):
        """
        Generate background music based on mood.
        
        Args:
            mood (str): The mood of the scene (e.g., happy, sad, tense).
            output_id (str): Identifier for the output file.
            
        Returns:
            str: Path to the generated audio file.
        """
        logger.info(f"Generating {mood} music...")
        
        output_path = f"outputs/audio/{output_id}_music.wav"
        
        try:
            # In a real scenario, we'd use music generation models or a library of samples
            # For this prototype, we'll create a simulated music track
            self._create_simulated_music(mood, output_path)
            
            logger.info(f"Music audio generated at {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating music: {e}")
            # Create fallback audio
            return self._create_fallback_audio(output_id, "music")
    
    def _create_simulated_speech(self, text, output_path):
        """
        Create simulated speech audio.
        In a production environment, this would call a TTS model.
        """
        # Generate a simple audio waveform based on text
        sample_rate = 24000  # CD quality
        duration = len(text) * 0.07  # Rough speech rate
        
        # Generate a basic carrier wave
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = np.zeros_like(t)
        
        # Map each character to frequencies to create a speech-like sound
        for i, char in enumerate(text):
            if char == " ":
                continue  # Skip spaces
                
            # Map character to a frequency between 200-500Hz
            freq = 200 + (ord(char) % 300)
            
            # Calculate start and end times for this character
            char_start = i * 0.07
            char_end = (i + 1) * 0.07
            
            # Create a mask for this character's time slice
            mask = (t >= char_start) & (t < char_end)
            
            # Add a sine wave for this character
            audio[mask] += 0.5 * np.sin(2 * np.pi * freq * (t[mask] - char_start))
        
        # Normalize audio
        audio = 0.5 * audio / np.max(np.abs(audio))
        
        # Add some noise to make it sound more natural
        noise = np.random.normal(0, 0.01, audio.shape)
        audio = audio + noise
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Save as WAV
        wavfile.write(output_path, sample_rate, audio_int16)
    
    def _create_simulated_music(self, mood, output_path):
        """
        Create simulated music based on mood.
        In a production environment, this would use a music generation model or library.
        """
        # Set music parameters based on mood
        if mood.lower() in self.mood_samples:
            mood_key = mood.lower()
        else:
            mood_key = random.choice(list(self.mood_samples.keys()))
            
        # Generate mood-specific parameters
        mood_params = {
            "happy": {"base_freq": 440, "tempo": 0.3, "harmonics": [1, 3, 5]},
            "sad": {"base_freq": 392, "tempo": 0.6, "harmonics": [1, 2, 6]},
            "tense": {"base_freq": 466, "tempo": 0.25, "harmonics": [1, 2, 7]},
            "mysterious": {"base_freq": 415, "tempo": 0.5, "harmonics": [1, 4, 7]},
            "exciting": {"base_freq": 523, "tempo": 0.2, "harmonics": [1, 3, 6]},
            "calm": {"base_freq": 349, "tempo": 0.8, "harmonics": [1, 5, 8]},
            "romantic": {"base_freq": 392, "tempo": 0.5, "harmonics": [1, 3, 5]},
            "scary": {"base_freq": 370, "tempo": 0.4, "harmonics": [1, 7, 13]},
            "suspenseful": {"base_freq": 415, "tempo": 0.35, "harmonics": [1, 3, 9]},
            "hopeful": {"base_freq": 523, "tempo": 0.4, "harmonics": [1, 5, 8]}
        }
        
        params = mood_params.get(mood_key, mood_params["calm"])  # Default to calm
        
        # Generate a 10-second sample
        sample_rate = 44100  # CD quality
        duration = 10  # 10 seconds of music
        
        # Generate time array
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Start with silence
        audio = np.zeros_like(t)
        
        # Generate a melody
        for i in range(20):  # 20 notes
            # Note start and duration
            note_start = i * params["tempo"]
            note_duration = params["tempo"] * 0.8  # Note takes up 80% of its time slot
            
            # Stop if we've gone beyond the duration
            if note_start >= duration:
                break
                
            # Select a note from a scale (major or minor depending on mood)
            if mood_key in ["happy", "exciting", "hopeful"]:
                scale = [0, 2, 4, 5, 7, 9, 11]  # Major scale
            else:
                scale = [0, 2, 3, 5, 7, 8, 10]  # Minor scale
                
            # Choose a note from the scale
            note_offset = scale[i % len(scale)]
            
            # Calculate frequency (adjust octave every 7 notes)
            octave_adjust = 2 ** (i // len(scale))
            note_freq = params["base_freq"] * octave_adjust * 2 ** (note_offset / 12)
            
            # Create a mask for this note's time slice
            mask = (t >= note_start) & (t < note_start + note_duration)
            
            # Add a sine wave for the fundamental frequency
            envelope = np.sin(np.pi * (t[mask] - note_start) / note_duration) ** 0.5
            
            # Add harmonics
            for harmonic in params["harmonics"]:
                # The higher the harmonic, the lower its amplitude
                amplitude = 1.0 / harmonic
                audio[mask] += amplitude * envelope * np.sin(2 * np.pi * note_freq * harmonic * (t[mask] - note_start))
        
        # Normalize audio
        audio = 0.5 * audio / np.max(np.abs(audio))
        
        # Add some reverb effect
        reverb_delay = int(sample_rate * 0.1)  # 100ms reverb delay
        reverb = np.zeros_like(audio)
        reverb[reverb_delay:] = 0.6 * audio[:-reverb_delay]
        audio = audio + reverb[:len(audio)]
        
        # Normalize again after effects
        audio = 0.9 * audio / np.max(np.abs(audio))
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Save as WAV
        wavfile.write(output_path, sample_rate, audio_int16)
    
    def _create_fallback_audio(self, output_id, audio_type):
        """Create fallback audio when generation fails."""
        output_path = f"outputs/audio/{output_id}_{audio_type}_fallback.wav"
        
        # Generate a simple sine wave
        sample_rate = 44100
        duration = 5  # 5 seconds
        
        # Different frequencies for different audio types
        if audio_type == "narration":
            freq = 440  # A4
        else:  # music
            freq = 523  # C5
            
        # Generate time array
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Generate sine wave
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        
        # Add a simple envelope
        envelope = np.ones_like(t)
        attack = int(0.1 * sample_rate)
        release = int(0.3 * sample_rate)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(1, 0, release)
        audio = audio * envelope
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Save as WAV
        wavfile.write(output_path, sample_rate, audio_int16)
        
        return output_path 