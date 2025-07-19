# I2VGen-XL ONLY - Memory Optimized with Easy Settings
# Clean test of just I2VGen-XL with all optimizations

# ========================================
# EASY SETTINGS - UNCOMMENT TO CHANGE
# ========================================

# VIDEO LENGTH SETTINGS
# num_frames = 8          # CURRENT: Short (1.3s at 6fps) - MEMORY SAFE
num_frames = 16         # LONGER: Medium (2.7s at 6fps) - Try this!
# num_frames = 24         # LONGEST: Long (4s at 6fps) - Might cause OOM
# num_frames = 32         # VERY LONG: (5.3s at 6fps) - High OOM risk

# QUALITY SETTINGS  
# num_inference_steps = 10   # FASTEST: Lower quality, very fast
num_inference_steps = 20   # CURRENT: Balanced quality/speed
# num_inference_steps = 30   # HIGHER: Better quality, slower
# num_inference_steps = 50   # BEST: Highest quality, slowest

# PROMPT FOLLOWING SETTINGS
# guidance_scale = 5.0    # LOOSE: More creative, less prompt following
guidance_scale = 7.0    # CURRENT: Balanced
# guidance_scale = 9.0    # STRICT: Follows prompt more closely
# guidance_scale = 12.0   # VERY STRICT: Maximum prompt adherence

# IMAGE SIZE SETTINGS
# image_size = (256, 256)  # SMALLEST: Saves memory, lower quality
image_size = (512, 512)  # CURRENT: Good balance
# image_size = (768, 768)  # LARGER: Better quality, more memory
# image_size = (1024, 576) # WIDESCREEN: Cinematic, high memory usage

# PLAYBACK SPEED SETTINGS
# fps = 4              # VERY SLOW: Easy to see details
fps = 6              # CURRENT: Good viewing speed  
# fps = 8              # FASTER: More natural motion
# fps = 12             # FAST: Smooth motion

# PROMPT VARIATIONS - UNCOMMENT TO CHANGE STYLE
text_prompt = "elegant woman in golden dress dancing slowly, mysterious and seductive, golden brown lighting, cinematic quality"
# text_prompt = "mysterious woman in flowing golden fabric, ethereal movement, warm amber lighting, dreamy atmosphere"
# text_prompt = "woman in shimmering gold dancing gracefully, hypnotic motion, 1980s aesthetic, soft golden glow"
# text_prompt = "enchanting female figure in golden attire, fluid dance movements, seductive lighting, film noir style"

# ========================================
# RECOMMENDED COMBINATIONS:
# ========================================
# FOR LONGER VIDEOS (medium memory usage):
#   num_frames = 16, num_inference_steps = 20, guidance_scale = 7.0
#
# FOR BEST QUALITY (high memory usage):  
#   num_frames = 12, num_inference_steps = 30, guidance_scale = 9.0
#
# FOR FAST TESTING (low memory usage):
#   num_frames = 8, num_inference_steps = 10, guidance_scale = 5.0
#
# FOR MAXIMUM LENGTH (highest memory risk):
#   num_frames = 24, num_inference_steps = 15, guidance_scale = 7.0
# ========================================

import torch
import gc
import os
from PIL import Image
import requests
from io import BytesIO
import time

# ========================================
# MEMORY OPTIMIZATION SETUP
# ========================================

def clear_cuda_memory():
    """Clear CUDA memory thoroughly"""
    torch.cuda.empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def setup_memory_optimizations():
    """Setup memory optimizations"""
    print("🔧 Setting up memory optimizations...")
    
    # PyTorch CUDA memory settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Clear any existing CUDA memory
    clear_cuda_memory()
    
    # Check available memory
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated_memory = torch.cuda.memory_allocated() / 1e9
        free_memory = total_memory - allocated_memory
        
        print(f"💾 Total GPU Memory: {total_memory:.1f}GB")
        print(f"💾 Allocated Memory: {allocated_memory:.1f}GB") 
        print(f"💾 Free Memory: {free_memory:.1f}GB")
        
        return free_memory

print("🎬 I2VGen-XL Memory Optimized Test")
print("=" * 50)

# Check GPU
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name()}")
    free_memory = setup_memory_optimizations()
else:
    print("❌ No GPU available!")
    exit()

# ========================================
# INPUT SETUP
# ========================================

def load_image_optimized(url, size=(512, 512)):
    """Load image with memory-conscious sizing"""
    print(f"📥 Loading image (size: {size})...")
    
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image = image.resize(size, Image.Resampling.LANCZOS)
    
    print(f"✅ Image loaded: {image.size}")
    return image

# Load Golden Brown cover
golden_brown_url = "https://i.scdn.co/image/ab67616d0000b273a713f9b06accedba5d963d61"
input_image = load_image_optimized(golden_brown_url, image_size)

# Save input for reference
input_image.save("input_image.png")
print("💾 Input image saved as: input_image.png")

# ========================================
# I2VGEN-XL OPTIMIZED GENERATION  
# ========================================

print(f"\n📝 Current Settings:")
print(f"   Video length: {num_frames} frames = {num_frames/fps:.1f}s at {fps}fps")
print(f"   Image size: {image_size}")
print(f"   Quality: {num_inference_steps} inference steps")
print(f"   Prompt following: {guidance_scale} guidance scale")
print(f"   Prompt: {text_prompt[:50]}...")

print(f"\n💾 Estimated memory usage:")
if num_frames <= 8:
    print(f"   🟢 LOW - Should work on most GPUs")
elif num_frames <= 16:
    print(f"   🟡 MEDIUM - Requires 12+ GB GPU memory")
elif num_frames <= 24:
    print(f"   🟠 HIGH - Requires 15+ GB GPU memory")
else:
    print(f"   🔴 VERY HIGH - May cause OOM errors")

try:
    print("\n📥 Loading I2VGen-XL with ALL optimizations...")
    load_start = time.time()
    
    from diffusers import I2VGenXLPipeline
    
    # Clear memory before loading
    clear_cuda_memory()
    
    # Load pipeline with optimizations
    pipe = I2VGenXLPipeline.from_pretrained(
        "ali-vilab/i2vgen-xl",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    
    # Apply ALL memory optimizations
    print("🔧 Applying memory optimizations...")
    
    # 1. CPU Offloading - moves parts to CPU when not in use
    pipe.enable_model_cpu_offload()
    print("   ✅ CPU offloading enabled")
    
    # 2. VAE Slicing - processes video in smaller chunks
    pipe.enable_vae_slicing()
    print("   ✅ VAE slicing enabled")
    
    # 3. Attention Slicing - reduces attention memory usage
    pipe.enable_attention_slicing()
    print("   ✅ Attention slicing enabled")
    
    # Don't use .to("cuda") when using cpu_offload - it handles GPU placement
    
    load_time = time.time() - load_start
    print(f"✅ I2VGen-XL loaded in {load_time:.1f}s with optimizations")
    
    # Check memory after loading
    print(f"\n💾 Memory after loading:")
    allocated = torch.cuda.memory_allocated() / 1e9
    print(f"   Allocated: {allocated:.1f}GB")
    
    print(f"\n🎬 Generating video...")
    generation_start = time.time()
    
    # Clear memory before generation
    clear_cuda_memory()
    
    # Generate with conservative settings
    generator = torch.manual_seed(42)
    
    result = pipe(
        prompt=text_prompt,
        image=input_image,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    )
    
    frames = result.frames[0]
    generation_time = time.time() - generation_start
    
    print(f"✅ Generated {len(frames)} frames in {generation_time:.1f}s")
    print(f"⚡ Speed: {generation_time/len(frames):.2f}s per frame")
    print(f"📊 Total time: {load_time + generation_time:.1f}s")
    
    # Save frames
    print(f"\n💾 Saving frames...")
    os.makedirs("i2vgen_frames", exist_ok=True)
    
    for i, frame in enumerate(frames):
        frame_path = f"i2vgen_frames/frame_{i:03d}.png"
        frame.save(frame_path)
        if i % 3 == 0:
            print(f"   Saved frame {i+1}/{len(frames)}")
    
    # Clear frames from memory before video creation
    del frames
    clear_cuda_memory()
    
    # Create video
    print(f"\n🎬 Creating video...")
    
    try:
        import imageio
        
        fps = 6  # Slower for easier viewing
        video_start = time.time()
        
        frame_files = [f"i2vgen_frames/frame_{i:03d}.png" for i in range(num_frames)]
        
        with imageio.get_writer('i2vgen_output.mp4', fps=fps) as writer:
            for frame_file in frame_files:
                image = imageio.imread(frame_file)
                writer.append_data(image)
        
        video_time = time.time() - video_start
        
        file_size = os.path.getsize('i2vgen_output.mp4') / (1024 * 1024)
        duration = num_frames / fps
        
        print(f"✅ Video created in {video_time:.1f}s: i2vgen_output.mp4")
        print(f"📁 Video: {file_size:.1f}MB, {duration:.1f}s duration at {fps}fps")
        
        print(f"\n🎉 SUCCESS!")
        print(f"📺 Files created:")
        print(f"   - input_image.png (original album cover)")
        print(f"   - i2vgen_frames/ (individual frames)")
        print(f"   - i2vgen_output.mp4 (final video)")
        
        print(f"\n🎯 I2VGen-XL Results:")
        print(f"   ✅ Uses both image + text input")
        print(f"   ✅ Memory optimizations successful")
        print(f"   ✅ Golden Brown cover animated with text guidance")
        
    except ImportError:
        print("⚠️  imageio not installed, install with: !pip install imageio[ffmpeg]")
        print("💡 Frames are saved in 'i2vgen_frames' folder")
    except Exception as e:
        print(f"❌ Video creation failed: {e}")
        print("💡 Frames should still be good in 'i2vgen_frames' folder")

except torch.cuda.OutOfMemoryError as e:
    print(f"\n💥 CUDA OUT OF MEMORY!")
    print(f"Error: {e}")
    print(f"\n💡 Memory optimization suggestions:")
    print(f"   - Reduce num_frames to 4 or 6")
    print(f"   - Reduce image size to (256, 256)")
    print(f"   - Reduce num_inference_steps to 10")
    print(f"   - Restart notebook to clear all memory")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print(f"\nTroubleshooting:")
    print(f"   - Check if I2VGen-XL model exists")
    print(f"   - Try restarting notebook") 
    print(f"   - Install missing dependencies")

finally:
    # Clean up memory
    clear_cuda_memory()
    print(f"\n🧹 Memory cleaned up")