# Image-to-Image Sequence with Better Prompts
# Back to original approach but with progressive prompts

# ========================================
# EASY SETTINGS - UNCOMMENT TO CHANGE
# ========================================

# MODEL SETTINGS
base_model = "runwayml/stable-diffusion-v1-5"
# base_model = "stabilityai/stable-diffusion-2-1"
# base_model = "CompVis/stable-diffusion-v1-4"

# VIDEO LENGTH SETTINGS
num_frames = 16         # CURRENT: 16 unique variations
# num_frames = 12         # SHORTER: 12 variations
# num_frames = 20         # LONGER: 20 variations

# CHANGE INTENSITY SETTINGS
strength = 0.5          # CURRENT: Balanced change
# strength = 0.3          # SUBTLE: Small changes
# strength = 0.7          # DRAMATIC: Big changes

# FRAME DUPLICATION SETTINGS
frame_hold_time = 0.2   # CURRENT: Each image shows for 0.2s
# frame_hold_time = 0.3   # SLOWER: Each image shows for 0.3s  
# frame_hold_time = 0.1   # FASTER: Each image shows for 0.1s

# QUALITY SETTINGS
num_inference_steps = 25    # CURRENT: Higher quality
# num_inference_steps = 20   # BALANCED: Standard quality
# num_inference_steps = 15   # FASTER: Lower quality

guidance_scale = 7.5    # CURRENT: Standard prompt following
# guidance_scale = 5.0    # LOOSE: More creative
# guidance_scale = 10.0   # STRICT: Follow prompts closely

# IMAGE SIZE SETTINGS
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

print("🎨 Image-to-Image with Better Progressive Prompts")
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
# PROGRESSIVE PROMPTS FOR SMOOTH TRANSITIONS
# ========================================

# These are the better prompts from the ControlNet version
# Focus on gradual changes for smooth animation
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

# Ensure we have enough prompts for the frames
while len(prompts) < num_frames:
    prompts.extend(prompts)  # Repeat if needed

prompts = prompts[:num_frames]  # Trim to exact number

# Calculate video specs with duplication
frames_per_image = int(frame_hold_time * fps)
total_video_frames = num_frames * frames_per_image
video_duration = total_video_frames / fps

print(f"\n📝 Settings:")
print(f"   Model: {base_model}")
print(f"   Unique frames: {num_frames}")
print(f"   Frame hold time: {frame_hold_time}s")
print(f"   Strength: {strength} (change intensity)")
print(f"   Image size: {image_size}")
print(f"   📊 Video specs:")
print(f"      Frames per image: {frames_per_image}")
print(f"      Total video frames: {total_video_frames}")
print(f"      Final video duration: {video_duration:.1f}s")

# ========================================
# GENERATE SEQUENCE
# ========================================

try:
    print(f"\n📥 Loading Stable Diffusion img2img...")
    
    from diffusers import StableDiffusionImg2ImgPipeline
    
    # Load pipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    
    # Enable optimizations
    pipe.enable_attention_slicing()
    
    print(f"✅ Model loaded: {base_model}")
    
    print(f"\n🎨 Generating {num_frames} frames with progressive prompts...")
    
    # Create output folder
    os.makedirs("img2img_frames", exist_ok=True)
    
    frames = []
    total_start = time.time()
    
    for i in range(num_frames):
        frame_start = time.time()
        
        prompt = prompts[i]
        print(f"\n🖼️  Frame {i+1}/{num_frames}")
        print(f"   Prompt: {prompt[:50]}...")
        
        # Generate frame using original img2img approach
        result = pipe(
            prompt=prompt,
            image=base_image,  # Always use original album cover as base
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(42 + i)
        )
        
        frame = result.images[0]
        
        # Save frame
        frame_path = f"img2img_frames/frame_{i:03d}.png"
        frame.save(frame_path)
        frames.append(frame)
        
        frame_time = time.time() - frame_start
        print(f"   ✅ Generated in {frame_time:.1f}s")
    
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
        
        with imageio.get_writer('img2img_progressive.mp4', fps=fps) as writer:
            for i in range(num_frames):
                frame_file = f"img2img_frames/frame_{i:03d}.png"
                image = imageio.imread(frame_file)
                
                # Duplicate each frame multiple times
                for repeat in range(frames_per_image):
                    writer.append_data(image)
                    
                if i % 4 == 0:  # Progress update
                    print(f"   Added image {i+1}/{num_frames} ({frames_per_image} times)")
        
        video_time = time.time() - video_start
        
        file_size = os.path.getsize('img2img_progressive.mp4') / (1024 * 1024)
        
        print(f"✅ Video created in {video_time:.1f}s: img2img_progressive.mp4")
        print(f"📁 Video: {file_size:.1f}MB, {video_duration:.1f}s duration")
        
        print(f"\n🎉 SUCCESS!")
        print(f"📺 Files created:")
        print(f"   - base_album_cover.png (original)")
        print(f"   - img2img_frames/ (all frames)")
        print(f"   - img2img_progressive.mp4 (final video)")
        
        print(f"\n🎯 Image-to-Image with Progressive Prompts:")
        print(f"   ✅ Uses actual Golden Brown album cover as base")
        print(f"   ✅ Progressive prompts for smoother conceptual flow")
        print(f"   ✅ Original img2img style (closer to target)")
        print(f"   ✅ {video_duration:.1f}s video with gradual transitions")
        
    except ImportError:
        print("⚠️  imageio not installed")
        print("💡 Frames saved in img2img_frames/ folder")
    except Exception as e:
        print(f"❌ Video creation failed: {e}")
        print("💡 Frames should be good in img2img_frames/ folder")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print(f"\nTry adjusting settings:")
    print(f"   - Reduce num_frames")
    print(f"   - Use smaller image_size")
    print(f"   - Try different base_model")

print(f"\n📝 To experiment:")
print(f"   - Change strength (0.3=subtle, 0.7=dramatic)")
print(f"   - Try different frame_hold_time values")
print(f"   - Adjust num_frames for longer/shorter videos")