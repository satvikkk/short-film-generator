import os
import torch
import json
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gc

logger = logging.getLogger(__name__)

class ScriptGenerator:
    def __init__(self):
        """Initialize the script generator with Phi-4-mini-instruct."""
        logger.info("Initializing ScriptGenerator with Phi-4-mini-instruct...")
        try:
            # Use Phi-4-mini-instruct
            model_path = "microsoft/Phi-4-mini-instruct"
            
            # Set random seed for reproducibility
            torch.random.manual_seed(0)
            
            # Determine which device we're using
            if torch.cuda.is_available():
                device = "cuda"
                torch_dtype = torch.float16
                # Set CUDA memory optimization
                torch.cuda.empty_cache()
                logger.info("Using CUDA for script generation with float16")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                torch_dtype = torch.float16
                # Set environment variable for MPS memory limit
                os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.5"  # Use 50% of available memory
                logger.info("Using MPS (Apple Silicon) for script generation with float16")
            else:
                device = "cpu"
                torch_dtype = torch.float16
                logger.warning("Neither CUDA nor MPS available. Running on CPU will be very slow.")
            
            # Load model and tokenizer with memory optimizations but without bitsandbytes
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            
            # Move model to device after loading
            self.model = self.model.to(device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Create pipeline with memory optimization
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device,
            )
            
            # Generation arguments optimized for memory
            self.generation_args = {
                "max_new_tokens": 800,  # Reduced from 1000
                "return_full_text": False,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
                "use_cache": True,  # Enable KV caching
                "num_beams": 1,  # Disable beam search to save memory
            }
                
            logger.info(f"ScriptGenerator initialized successfully with Phi-4-mini-instruct on {device}.")
        except Exception as e:
            logger.error(f"Error initializing ScriptGenerator with Phi-4: {e}")
            # Try fallback to CPU with even more aggressive memory optimization
            try:
                logger.info("Attempting fallback to CPU with minimal memory usage...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="cpu",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
                logger.info("Fallback to CPU successful.")
            except Exception as e2:
                logger.error(f"Fallback failed: {e2}")
                raise

    def generate(self, prompt):
        """
        Generate a script from a prompt using Phi-4-mini-instruct.
        
        Args:
            prompt (str): The user's prompt describing the short film.
            
        Returns:
            dict: Script data including scenes, narration, and other metadata.
        """
        logger.info(f"Generating script for prompt: {prompt}")
        
        # Free up memory before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # On MPS, manually trigger garbage collection
            gc.collect()
        
        # Create a simpler prompt to reduce memory usage
        simplified_prompt = f"""Write a short film script about: "{prompt}". Include title, 3 scenes with descriptions and moods, and narration."""
        
        try:
            # Generate the script text with Phi-4 using the pipeline
            result = self.pipe(
                simplified_prompt,
                **self.generation_args
            )
            
            script_text = result[0]['generated_text']
            
            # Remove the prompt from the output
            if simplified_prompt in script_text:
                script_text = script_text.replace(simplified_prompt, "").strip()
            
            # Parse the generated text into structured script data
            script_data = self._parse_script(script_text, prompt)
            
            logger.info(f"Script generated successfully with {len(script_data['scenes'])} scenes.")
            return script_data
            
        except Exception as e:
            logger.error(f"Error generating script with Phi-4: {e}")
            # If generation fails, create a fallback script
            return self._create_fallback_script(prompt)
    
    def _parse_script(self, script_text, original_prompt):
        """
        Parse the generated script text into structured data.
        
        In case of difficult parsing, this creates a reasonable structure.
        """
        # Start with a base structure
        script_data = {
            "title": f"Film: {original_prompt[:30]}...",
            "prompt": original_prompt,
            "scenes": [],
            "narration": ""
        }
        
        # Try to extract a title
        title_match = False
        lines = script_text.split("\n")
        for i, line in enumerate(lines):
            line = line.strip()
            if "TITLE:" in line.upper() or "Title:" in line:
                if i+1 < len(lines) and lines[i+1].strip():
                    script_data["title"] = lines[i+1].strip()
                    title_match = True
                    break
            elif not title_match and (len(line) < 60 and len(line) > 5):
                # If no explicit title marker but looks like a title
                script_data["title"] = line
                break
                
        # Extract scenes
        scene_blocks = []
        current_scene = {"description": [], "mood": "neutral"}
        in_scene_section = False
        in_narration_section = False
        narration_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            if "SCENES:" in line.upper() or "Scenes:" in line:
                in_scene_section = True
                in_narration_section = False
                continue
                
            if "NARRATION:" in line.upper() or "Narration:" in line:
                if current_scene["description"]:
                    scene_blocks.append(current_scene)
                    current_scene = {"description": [], "mood": "neutral"}
                in_scene_section = False
                in_narration_section = True
                continue
            
            # Process scenes
            if in_scene_section:
                # Check for scene headers
                if line.lower().startswith("scene ") and ":" in line:
                    if current_scene["description"]:
                        scene_blocks.append(current_scene)
                        current_scene = {"description": [], "mood": "neutral"}
                    current_scene["description"].append(line)
                # Check for mood indicators
                elif line.lower().startswith("mood:"):
                    mood = line.lower().replace("mood:", "").strip()
                    current_scene["mood"] = mood
                else:
                    current_scene["description"].append(line)
            
            # Process narration
            elif in_narration_section:
                narration_lines.append(line)
        
        # Add the last scene if it exists
        if current_scene["description"]:
            scene_blocks.append(current_scene)
        
        # Process each scene block
        for i, scene in enumerate(scene_blocks):
            description = " ".join(scene["description"])
            mood = scene["mood"]
            
            # Extract mood if not explicitly stated
            if mood == "neutral":
                mood_keywords = ["happy", "sad", "tense", "exciting", "mysterious", 
                               "calm", "romantic", "scary", "suspenseful", "hopeful"]
                for keyword in mood_keywords:
                    if keyword in description.lower():
                        mood = keyword
                        break
            
            # Create a scene entry
            scene_entry = {
                "id": i, 
                "description": description,
                "mood": mood
            }
            
            script_data["scenes"].append(scene_entry)
        
        # If no proper scenes were extracted, create default scenes
        if not script_data["scenes"]:
            script_data = self._create_fallback_script(original_prompt)
            
        # Set narration
        if narration_lines:
            script_data["narration"] = " ".join(narration_lines)
        else:
            # Create a default narration based on the original prompt
            script_data["narration"] = f"In a world where {original_prompt}, our story unfolds."
            
        return script_data
    
    def _create_fallback_script(self, prompt):
        """Create a detailed script if generation or parsing fails."""
        words = prompt.split()
        mid_point = len(words) // 2
        
        # Create more detailed scene descriptions
        opening_scene = f"Opening scene: A wide establishing shot reveals {' '.join(words[:mid_point])}. The lighting is soft and diffused, creating a mysterious atmosphere. The camera slowly pans across the scene, showing intricate details and textures."
        
        middle_scene = f"Middle scene: A medium close-up shot focuses on {' '.join(words[1:3])}. The color palette shifts to more intense hues, with dramatic lighting creating strong shadows. Various objects in the foreground and background add depth to the composition."
        
        final_scene = f"Final scene: The camera pulls back to reveal {' '.join(words[-2:])} in a new context. Golden hour lighting bathes everything in a warm glow. The scene is carefully composed with balanced elements and a clear focal point."
        
        return {
            "title": f"The Journey of {' '.join(words[:3])}",
            "prompt": prompt,
            "scenes": [
                {"id": 0, "description": opening_scene, "mood": "mysterious"},
                {"id": 1, "description": middle_scene, "mood": "tense"},
                {"id": 2, "description": final_scene, "mood": "hopeful"}
            ],
            "narration": f"In a world where {prompt}, our story unfolds. We witness the journey through challenges and discovery, leading to an unexpected conclusion that changes everything."
        } 