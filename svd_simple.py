# Simple SVD Test - Animate Golden Brown Cover
# Takes an image and generates video directly

import torch
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import requests
from io import BytesIO
import os
import time

print("🎬 Stable Video Diffusion (SVD) Test")
print("=" * 40)

# Check GPU
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name()}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1e9:.1f}GB")
else:
    print("⚠️  No GPU - will be very slow!")

# ========================================
# SVD MODEL OPTIONS
# ========================================

# DEFAULT MODEL (14 frames, faster)
svd_model = "stabilityai/stable-video-diffusion-img2vid"

# #################### OTHER SVD MODELS ####################
# # Option 1: Extended version (25 frames, longer video)
# svd_model = "stabilityai/stable-video-diffusion-img2vid-xt"
# #################### END SVD MODELS ####################

print(f"\n🎯 Using SVD model: {svd_model}")

# ========================================
# IMAGE INPUT
# ========================================

def load_image_from_url(url, size=(512, 512)):
    """Download and prepare image for SVD"""
    print(f"📥 Downloading image from URL...")
    
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    
    # Resize to optimal size for SVD
    image = image.resize(size, Image.Resampling.LANCZOS)
    
    print(f"✅ Image loaded: {image.size}")
    return image

def load_local_image(path, size=(512, 512)):
    """Load local image file"""
    print(f"📁 Loading image from: {path}")
    
    image = Image.open(path).convert("RGB")
    image = image.resize(size, Image.Resampling.LANCZOS)
    
    print(f"✅ Image loaded: {image.size}")
    return image

# Choose your input method:
# Option 1: Golden Brown cover from Spotify
golden_brown_url = "https://i.scdn.co/image/ab67616d0000b273a713f9b06accedba5d963d61"

# Option 2: Local file (uncomment to use)
# input_image = load_local_image("your_image.jpg")

# Load the Golden Brown cover
input_image = load_image_from_url(golden_brown_url)

# Save input image for reference
input_image.save("input_image.png")
print("💾 Input image saved as: input_image.png")

# ========================================
# SVD GENERATION SETTINGS
# ========================================

# Frame settings
num_frames = 14          # Number of frames (14 for base model, 25 for XT)
fps = 7                  # Frames per second for output video

# MOTION CONTROL SETTINGS - Experiment with these!
# motion_bucket_id = 127        # MOTION INTENSITY: 0=static, 127=default, 255=max motion
# noise_aug_strength = 0.02     # VARIATION AMOUNT: 0.0=minimal change, 1.0=dramatic change
decode_chunk_size = 2         # Memory optimization (reduce if GPU issues)

# #################### MOTION EXPERIMENTS ####################
# # Minimal motion (very subtle)
# motion_bucket_id = 50
# noise_aug_strength = 0.01
# 
# # Medium motion (more noticeable)
# motion_bucket_id = 150
# noise_aug_strength = 0.05
# 
# # Maximum motion (dramatic changes)
# motion_bucket_id = 255
# noise_aug_strength = 0.2
# 
# # Crazy experimental (may look chaotic)
motion_bucket_id = 255
noise_aug_strength = 0.5
# #################### END MOTION EXPERIMENTS ####################

print(f"\n📝 Generation settings:")
print(f"   Frames: {num_frames}")
print(f"   Video duration: {num_frames/fps:.1f} seconds")
print(f"   FPS: {fps}")
print(f"   🎭 Motion bucket ID: {motion_bucket_id} (0=static, 255=max)")
print(f"   🌊 Noise aug strength: {noise_aug_strength} (0.0=minimal, 1.0=dramatic)")
print(f"   💾 Decode chunk size: {decode_chunk_size}")

if motion_bucket_id > 200:
    print(f"   ⚠️  HIGH MOTION: Expect dramatic/chaotic movement!")
elif motion_bucket_id < 80:
    print(f"   😴 LOW MOTION: Expect very subtle changes")
else:
    print(f"   ✅ BALANCED MOTION: Should have decent movement")

# ========================================
# GENERATE VIDEO
# ========================================

try:
    print("\n📥 Loading SVD model...")
    load_start = time.time()
    
    # Load SVD pipeline
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        svd_model,
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe = pipe.to("cuda")
    
    load_time = time.time() - load_start
    print(f"✅ SVD model loaded in {load_time:.1f}s")
    
    print(f"\n🎬 Generating video from image...")
    generation_start = time.time()
    
    # Generate video with motion controls
    generator = torch.manual_seed(42)  # For reproducible results
    
    frames = pipe(
        image=input_image,
        motion_bucket_id=motion_bucket_id,      # Control motion intensity
        noise_aug_strength=noise_aug_strength,  # Control variation amount
        decode_chunk_size=decode_chunk_size,
        num_frames=num_frames,
        generator=generator
    ).frames[0]  # Get the first (and only) video
    
    generation_time = time.time() - generation_start
    
    print(f"✅ Generated {len(frames)} frames in {generation_time:.1f}s")
    print(f"⚡ Speed: {generation_time/len(frames):.2f}s per frame")
    print(f"📊 Total time: {load_time + generation_time:.1f}s")
    
    # Save individual frames
    print(f"\n💾 Saving frames...")
    os.makedirs("svd_frames", exist_ok=True)
    
    for i, frame in enumerate(frames):
        frame_path = f"svd_frames/frame_{i:03d}.png"
        frame.save(frame_path)
        if i % 5 == 0:
            print(f"   Saved frame {i+1}/{len(frames)}")
    
    # Create video
    print(f"\n🎬 Creating video...")
    
    try:
        import imageio
        
        video_start = time.time()
        
        # Read frames and create video
        frame_files = [f"svd_frames/frame_{i:03d}.png" for i in range(len(frames))]
        
        with imageio.get_writer('svd_output.mp4', fps=fps) as writer:
            for frame_file in frame_files:
                image = imageio.imread(frame_file)
                writer.append_data(image)
        
        video_time = time.time() - video_start
        
        # Video info
        file_size = os.path.getsize('svd_output.mp4') / (1024 * 1024)
        duration = len(frames) / fps
        
        print(f"✅ Video created in {video_time:.1f}s: svd_output.mp4")
        print(f"📁 Video: {file_size:.1f}MB, {duration:.1f}s duration at {fps}fps")
        print(f"🎭 Motion settings: bucket_id={motion_bucket_id}, noise_strength={noise_aug_strength}")
        
        print(f"\n🎉 SUCCESS!")
        print(f"📺 Files created:")
        print(f"   - input_image.png (original)")
        print(f"   - svd_frames/ (individual frames)")
        print(f"   - svd_output.mp4 (final video)")
        print(f"\n💡 To experiment: Uncomment different motion settings above!")
        
    except ImportError:
        print("⚠️  imageio not installed, install with: !pip install imageio[ffmpeg]")
        print("💡 Individual frames saved in 'svd_frames' folder")
    except Exception as e:
        print(f"❌ Video creation failed: {e}")
        print("💡 Individual frames should be good in 'svd_frames' folder")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nTroubleshooting:")
    print("- Try the XT model for different frame count")
    print("- Reduce decode_chunk_size to 1 if memory issues")
    print("- Make sure image is not too large")

print(f"\n📊 COMPARISON NOTES:")
print(f"   SVD vs AnimateDiff:")
print(f"   - Input: Image vs Text prompt")
print(f"   - Speed: Should be faster")
print(f"   - Control: Less vs More")
print(f"   - Quality: Specialized for video")

print(f"\n📝 TODO - Still need to test:")
print("- [ ] Image-to-Image sequence method")