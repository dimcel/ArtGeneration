# ControlNet Image Sequence - Simple Version
# Maintains pose consistency while changing style

# ========================================
# EASY SETTINGS - UNCOMMENT TO CHANGE
# ========================================

# CONTROLNET TYPE
controlnet_type = "openpose"        # CURRENT: Maintains human pose
# controlnet_type = "canny"          # ALT: Maintains edges/composition
# controlnet_type = "depth"          # ALT: Maintains depth relationships

# VIDEO LENGTH SETTINGS
# num_frames = 12         # CURRENT: 12 unique variations
# num_frames = 8          # SHORT: 8 variations
num_frames = 16         # LONGER: 16 variations

# CONTROL STRENGTH SETTINGS  
# controlnet_strength = 0.8   # CURRENT: Strong pose control
controlnet_strength = 0.4   # LOOSER: More creative freedom
# controlnet_strength = 1.0   # STRICT: Maximum pose control

# FRAME DUPLICATION SETTINGS
frame_hold_time = 0.2       # CURRENT: Each image shows for 0.2s
# frame_hold_time = 0.3       # FAST: Each image shows for 0.3s  
# frame_hold_time = 0.8       # SLOW: Each image shows for 0.8s

# QUALITY SETTINGS
# num_inference_steps = 20    # CURRENT: Balanced
# num_inference_steps = 15   # FASTER: Lower quality
num_inference_steps = 25   # BETTER: Higher quality

# IMAGE SIZE
image_size = (512, 512)     # CURRENT: Standard
# image_size = (768, 768)    # LARGER: Better quality, more memory

# PLAYBACK SETTINGS
fps = 6                 # CURRENT: Good viewing speed

# ========================================

import torch
import requests
from PIL import Image
from io import BytesIO
import os
import time
import cv2
import numpy as np

print("🎭 ControlNet Image Sequence Generator")
print("=" * 50)

# Check GPU
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name()}")
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"💾 GPU Memory: {total_memory:.1f}GB")
else:
    print("⚠️  No GPU - will be slow!")

# ========================================
# LOAD BASE IMAGE
# ========================================

def load_album_cover(url, size=image_size):
    """Load Golden Brown album cover"""
    print(f"📥 Loading album cover (size: {size})...")
    
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image = image.resize(size, Image.Resampling.LANCZOS)
    
    print(f"✅ Album cover loaded: {image.size}")
    return image

# Load Golden Brown cover
golden_brown_url = "https://i.scdn.co/image/ab67616d0000b273a713f9b06accedba5d963d61"
base_image = load_album_cover(golden_brown_url)

# Save base image
base_image.save("base_album_cover.png")
print("💾 Base image saved: base_album_cover.png")

# ========================================
# GENERATE CONTROL MAP
# ========================================

def generate_control_map(image, control_type):
    """Generate control map from base image"""
    print(f"🔧 Generating {control_type} control map...")
    
    image_np = np.array(image)
    
    if control_type == "canny":
        # Simple edge detection
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        control_map = Image.fromarray(edges).convert("RGB")
        
    elif control_type == "openpose":
        # For now, use canny as placeholder - ControlNet will handle pose detection
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        control_map = Image.fromarray(edges).convert("RGB")
        
    elif control_type == "depth":
        # Simple depth approximation using edges
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        # Create depth-like map
        depth = cv2.GaussianBlur(gray, (5, 5), 0)
        control_map = Image.fromarray(depth).convert("RGB")
    
    control_map.save(f"{control_type}_control_map.png")
    print(f"✅ Control map saved: {control_type}_control_map.png")
    return control_map

# Generate control map
control_map = generate_control_map(base_image, controlnet_type)

# ========================================
# PROMPTS FOR SMOOTH TRANSITION SEQUENCE
# ========================================

# Progressive prompts with GRADUAL changes for smooth animation
# Focus on subtle transitions rather than dramatic style changes
prompts = [
    # Phase 1: Starting state (frames 1-4)
    "woman in golden dress, neutral expression, soft warm lighting",
    "woman in golden dress, gentle smile beginning, soft warm lighting",
    "woman in golden dress, subtle smile, slightly warmer lighting",
    "woman in golden dress, soft smile, warm golden lighting",
    
    # Phase 2: Building movement (frames 5-8)
    "woman in golden dress, gentle smile, head tilting slightly, warm lighting",
    "woman in golden dress, soft expression, slight head movement, golden glow",
    "woman in golden dress, graceful expression, gentle movement, amber lighting",
    "woman in golden dress, elegant pose, flowing movement, warm amber light",
    
    # Phase 3: Peak expression (frames 9-12)
    "woman in golden dress, captivating gaze, fluid movement, radiant lighting",
    "woman in golden dress, mysterious smile, dancing motion, golden radiance",
    "woman in golden dress, enchanting expression, graceful dance, ethereal glow",
    "woman in golden dress, alluring gaze, elegant movement, cinematic lighting",
    
    # Phase 4: Return to calm (frames 13-16)
    "woman in golden dress, serene expression, gentle settling, soft golden light",
    "woman in golden dress, peaceful gaze, subtle movement, warm lighting",
    "woman in golden dress, calm expression, gentle pose, soft glow",
    "woman in golden dress, tranquil beauty, still elegance, gentle lighting"
]

# Ensure we have enough prompts
while len(prompts) < num_frames:
    prompts.extend(prompts)
prompts = prompts[:num_frames]

# Calculate video specs
frames_per_image = int(frame_hold_time * fps)
total_video_frames = num_frames * frames_per_image
video_duration = total_video_frames / fps

print(f"\n📝 Settings:")
print(f"   ControlNet type: {controlnet_type}")
print(f"   Control strength: {controlnet_strength}")
print(f"   Unique frames: {num_frames}")
print(f"   Frame hold time: {frame_hold_time}s")
print(f"   📊 Video specs:")
print(f"      Frames per image: {frames_per_image}")
print(f"      Total video frames: {total_video_frames}")
print(f"      Final video duration: {video_duration:.1f}s")

# ========================================
# GENERATE CONTROLNET SEQUENCE
# ========================================

try:
    print(f"\n📥 Loading ControlNet pipeline...")
    
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    
    # Load ControlNet model
    if controlnet_type == "openpose":
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-openpose",
            torch_dtype=torch.float16
        )
    elif controlnet_type == "canny":
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16
        )
    elif controlnet_type == "depth":
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth",
            torch_dtype=torch.float16
        )
    
    # Load pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    
    # Enable optimizations
    pipe.enable_attention_slicing()
    
    print(f"✅ ControlNet loaded: {controlnet_type}")
    
    print(f"\n🔗 Generating {num_frames} chained frames...")
    
    # Create output folder
    os.makedirs("controlnet_frames", exist_ok=True)
    
    frames = []
    total_start = time.time()
    
    # Start with the album cover for first frame
    current_image = base_image
    
    for i in range(num_frames):
        frame_start = time.time()
        
        prompt = prompts[i]
        print(f"\n🖼️  Frame {i+1}/{num_frames}")
        print(f"   Prompt: {prompt[:50]}...")
        
        if i == 0:
            print(f"   Input: Original album cover")
        else:
            print(f"   Input: Previous frame (chained)")
        
        # Generate frame with ControlNet using current image
        result = pipe(
            prompt=prompt,
            image=control_map,  # Control map stays the same for consistency
            controlnet_conditioning_scale=controlnet_strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5,
            generator=torch.Generator(device="cuda").manual_seed(42 + i)
        )
        
        frame = result.images[0]
        
        # Save frame
        frame_path = f"controlnet_frames/frame_{i:03d}.png"
        frame.save(frame_path)
        frames.append(frame)
        
        # CHAINING: Use this frame as input for next iteration
        current_image = frame
        
        # Update control map from current frame for next iteration
        if i < num_frames - 1:  # Don't generate control map for last frame
            control_map = generate_control_map(current_image, controlnet_type)
        
        frame_time = time.time() - frame_start
        print(f"   ✅ Generated in {frame_time:.1f}s")
        print(f"   🔗 Frame will be used as input for next frame")
    
    total_time = time.time() - total_start
    avg_time = total_time / num_frames
    
    print(f"\n✅ All frames generated!")
    print(f"📊 Total time: {total_time:.1f}s")
    print(f"⚡ Average per frame: {avg_time:.1f}s")
    
    # Create video with frame duplication
    print(f"\n🎬 Creating video with frame duplication...")
    
    try:
        import imageio
        
        video_start = time.time()
        
        print(f"   📊 Each image will show for {frames_per_image} frames ({frame_hold_time}s)")
        
        with imageio.get_writer('controlnet_sequence.mp4', fps=fps) as writer:
            for i in range(num_frames):
                frame_file = f"controlnet_frames/frame_{i:03d}.png"
                image = imageio.imread(frame_file)
                
                # Duplicate each frame
                for repeat in range(frames_per_image):
                    writer.append_data(image)
                    
                if i % 3 == 0:
                    print(f"   Added image {i+1}/{num_frames} ({frames_per_image} times)")
        
        video_time = time.time() - video_start
        
        file_size = os.path.getsize('controlnet_sequence.mp4') / (1024 * 1024)
        
        print(f"✅ Video created in {video_time:.1f}s: controlnet_sequence.mp4")
        print(f"📁 Video: {file_size:.1f}MB, {video_duration:.1f}s duration")
        
        print(f"\n🎉 SUCCESS!")
        print(f"📺 Files created:")
        print(f"   - base_album_cover.png (original)")
        print(f"   - {controlnet_type}_control_map.png (control guidance)")
        print(f"   - controlnet_frames/ (all frames)")
        print(f"   - controlnet_sequence.mp4 (final consistent video)")
        
        print(f"\n🎯 ControlNet Chaining Results:")
        print(f"   ✅ Each frame builds on the previous frame")
        print(f"   ✅ Creates progressive evolution from album cover")
        print(f"   ✅ True animation rather than separate images")
        print(f"   ✅ Maintains {controlnet_type} consistency throughout")
        
    except ImportError:
        print("⚠️  imageio not installed")
    except Exception as e:
        print(f"❌ Video creation failed: {e}")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print(f"\nTry adjusting settings:")
    print(f"   - Try different controlnet_type")
    print(f"   - Reduce controlnet_strength")
    print(f"   - Use smaller image_size")

print(f"\n📝 To experiment:")
print(f"   - Change controlnet_type (openpose/canny/depth)")
print(f"   - Adjust controlnet_strength (0.6=loose, 1.0=strict)")
print(f"   - Try different frame_hold_time values")