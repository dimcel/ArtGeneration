# AnimateDiff Long Video Generator - 15 Second Video (5 segments)
# Generate multiple short videos and stitch them together

import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from PIL import Image
import os
import time

print("🎬 AnimateDiff Long Video Generator")
print("Target: 15-second video from 5 segments")
print("=" * 50)

# Check GPU
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name()}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1e9:.1f}GB")
else:
    print("⚠️  No GPU - will be very slow!")

# ========================================
# MODEL OPTIONS - Comment/uncomment to test different models
# ========================================

# DEFAULT MODEL (Medium size, good quality)
base_model = "runwayml/stable-diffusion-v1-5"

# #################### SMALLER/FASTER MODELS ####################
# # Option 1: Smaller SD model (faster)
# base_model = "CompVis/stable-diffusion-v1-4" 
# 
# # Option 2: Tiny model (very fast but lower quality)
# base_model = "nota-ai/bk-sdm-small"
# #################### END SMALLER MODELS ####################

# #################### LARGER/BETTER MODELS ####################
# # Option 1: Better SD model (slower but higher quality)
# base_model = "stabilityai/stable-diffusion-2-1"
# 
# # Option 2: Specialized anime/art model
# base_model = "andite/anything-v4.0"
# 
# # Option 3: Realistic model (photorealistic results)
# base_model = "SG161222/Realistic_Vision_V2.0"
# #################### END LARGER MODELS ####################

# ========================================
# MOTION ADAPTER OPTIONS - Comment/uncomment to test different motion styles
# ========================================

# DEFAULT MOTION ADAPTER (General motion, stable)
motion_adapter = "guoyww/animatediff-motion-adapter-v1-5-2"

# #################### OTHER MOTION ADAPTERS ####################
# # Option 1: Improved motion, less artifacts
# motion_adapter = "guoyww/animatediff-motion-adapter-v1-5-3"
# 
# # Option 2: For SD 2.x models (use with stabilityai/stable-diffusion-2-1)
# motion_adapter = "guoyww/animatediff-motion-adapter-sd-v2"
# 
# # Option 3: Experimental motion (may have different movement style)
# motion_adapter = "wangfuyun/AnimateLCM-SVD-xt"
# #################### END MOTION ADAPTERS ####################

print(f"\n🎯 Using model: {base_model}")
print(f"🎭 Motion adapter: {motion_adapter}")

# ========================================
# PROMPT SETTINGS FOR 5 SEGMENTS
# ========================================

# 5 different prompts for 5 video segments (3 seconds each = 15 total)
segment_prompts = [
    # Segment 1: Young woman farmer
    "young woman farmer working in a green field, sunny day, detailed face, realistic, beautiful countryside",
    
    # Segment 2: Farmer with animals
    "woman farmer feeding chickens in farmyard, golden hour lighting, detailed portrait, rural setting",
    
    # Segment 3: Transition to elderly
    "middle-aged woman sitting on porch, sunset, peaceful expression, detailed face, warm lighting",
    
    # Segment 4: Elderly indoor
    "elderly woman knitting in cozy living room, soft lamp light, detailed hands, serene face",
    
    # Segment 5: Final scene
    "wise old grandmother reading book by window, glasses, natural light, peaceful, detailed portrait"
]

# Generation settings per segment
frames_per_segment = 18  # 18 frames per segment (safe limit)
inference_steps = 25     # Good quality
guidance_scale = 12.0    # Strong prompt following
fps = 6                  # 18 frames ÷ 6 fps = 3 seconds per segment

print(f"\n📝 Video Plan:")
print(f"   Segments: {len(segment_prompts)}")
print(f"   Frames per segment: {frames_per_segment}")
print(f"   Duration per segment: {frames_per_segment/fps:.1f}s")
print(f"   Total duration: {len(segment_prompts) * frames_per_segment/fps:.1f}s")
print(f"   Inference steps: {inference_steps}")
print(f"   Guidance scale: {guidance_scale}")

print(f"\n🎨 Segment prompts:")
for i, prompt in enumerate(segment_prompts, 1):
    print(f"   Segment {i} ({(i-1)*3:.1f}-{i*3:.1f}s): {prompt[:60]}...")

# ========================================
# VIDEO GENERATION FUNCTION
# ========================================

def generate_segment(pipe, prompt, segment_num, frames=18):
    """Generate one video segment"""
    print(f"\n🎬 Generating segment {segment_num}/5...")
    print(f"   Prompt: {prompt}")
    
    start_time = time.time()
    
    # Generate frames
    result = pipe(
        prompt=prompt,
        num_frames=frames,
        num_inference_steps=inference_steps,
        guidance_scale=guidance_scale
    )
    
    generation_time = time.time() - start_time
    frames = result.frames[0]
    
    print(f"   ✅ Generated {len(frames)} frames in {generation_time:.1f}s")
    print(f"   ⚡ Speed: {generation_time/len(frames):.2f}s per frame")
    
    # Save frames to segment folder
    segment_folder = f"segments/segment_{segment_num:02d}"
    os.makedirs(segment_folder, exist_ok=True)
    
    for i, frame in enumerate(frames):
        frame_path = f"{segment_folder}/frame_{i:03d}.png"
        frame.save(frame_path)
    
    print(f"   💾 Saved {len(frames)} frames to {segment_folder}/")
    
    return frames, generation_time

def stitch_segments_to_video():
    """Stitch all segments into final video"""
    try:
        import imageio
        
        print(f"\n🔗 Stitching {len(segment_prompts)} segments into final video...")
        
        # Collect all frame files in order
        all_frame_files = []
        for segment_num in range(1, len(segment_prompts) + 1):
            segment_folder = f"segments/segment_{segment_num:02d}"
            
            # Get frames from this segment
            segment_frames = []
            for i in range(frames_per_segment):
                frame_file = f"{segment_folder}/frame_{i:03d}.png"
                if os.path.exists(frame_file):
                    segment_frames.append(frame_file)
            
            all_frame_files.extend(segment_frames)
            print(f"   Added {len(segment_frames)} frames from segment {segment_num}")
        
        print(f"   Total frames to stitch: {len(all_frame_files)}")
        
        # Create final video
        stitch_start = time.time()
        with imageio.get_writer('long_video_final.mp4', fps=fps) as writer:
            for i, frame_file in enumerate(all_frame_files):
                image = imageio.imread(frame_file)
                writer.append_data(image)
                
                if i % 20 == 0:  # Progress every 20 frames
                    print(f"   Stitching progress: {i+1}/{len(all_frame_files)} frames")
        
        stitch_time = time.time() - stitch_start
        
        # Final video info
        file_size = os.path.getsize('long_video_final.mp4') / (1024 * 1024)
        duration = len(all_frame_files) / fps
        
        print(f"\n🎉 FINAL VIDEO COMPLETE!")
        print(f"   File: long_video_final.mp4")
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   Size: {file_size:.1f} MB")
        print(f"   Frames: {len(all_frame_files)}")
        print(f"   Stitching time: {stitch_time:.1f}s")
        
        return True
        
    except ImportError:
        print("❌ imageio not available - install with: !pip install imageio[ffmpeg]")
        return False
    except Exception as e:
        print(f"❌ Stitching failed: {e}")
        return False

# ========================================
# MAIN GENERATION LOOP
# ========================================

try:
    # Create output directory
    os.makedirs("segments", exist_ok=True)
    
    print("\n📥 Loading models...")
    load_start = time.time()
    
    # Load motion adapter
    adapter = MotionAdapter.from_pretrained(motion_adapter)
    
    # Load pipeline  
    pipe = AnimateDiffPipeline.from_pretrained(
        base_model,
        motion_adapter=adapter,
        torch_dtype=torch.float16
    ).to("cuda")
    
    # Setup scheduler
    scheduler = EulerDiscreteScheduler.from_pretrained(
        base_model,
        subfolder="scheduler"
    )
    pipe.scheduler = scheduler
    
    load_time = time.time() - load_start
    print(f"✅ Models loaded in {load_time:.1f}s")
    
    # Generate all segments
    total_generation_time = 0
    total_frames = 0
    
    for i, prompt in enumerate(segment_prompts, 1):
        frames, gen_time = generate_segment(pipe, prompt, i, frames_per_segment)
        total_generation_time += gen_time
        total_frames += len(frames)
    
    print(f"\n📊 GENERATION SUMMARY:")
    print(f"   Total segments: {len(segment_prompts)}")
    print(f"   Total frames: {total_frames}")
    print(f"   Total generation time: {total_generation_time:.1f}s")
    print(f"   Average per frame: {total_generation_time/total_frames:.2f}s")
    print(f"   Model loading + generation: {load_time + total_generation_time:.1f}s")
    
    # Stitch all segments together
    if stitch_segments_to_video():
        print(f"\n🎯 SUCCESS! Check 'long_video_final.mp4' for your 15-second video!")
    else:
        print(f"\n💡 Individual segments saved in 'segments/' folder")
        print("You can manually stitch them together later")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nTroubleshooting:")
    print("- Try reducing frames_per_segment to 16 or 12")
    print("- Try a different model from the options above")
    print("- Check GPU memory")

print("\n📝 Files generated:")
print("- segments/segment_01/ through segments/segment_05/ (individual frames)")
print("- long_video_final.mp4 (final stitched video)")
print("\n📺 Expected result: 15-second video showing progression from young farmer to elderly woman")