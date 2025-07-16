import librosa
import numpy as np
import tomllib
from pathlib import Path
from typing import Dict, Any, Optional

def load_config(config_path: str = "config.toml") -> Dict[str, Any]:
    """Load configuration from TOML file"""
    try:
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {config_path} not found")
    except Exception as e:
        raise ValueError(f"Error loading config: {e}")

def validate_audio_file(audio_path: str, config: Dict[str, Any]) -> bool:
    """Validate audio file before processing"""
    path = Path(audio_path)
    
    # Check if file exists
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Check file extension
    supported_formats = config["audio"]["supported_formats"]
    if path.suffix.lower() not in supported_formats:
        raise ValueError(f"Unsupported format: {path.suffix}. Supported: {supported_formats}")
    
    return True

def extract_tempo(y: np.ndarray, sr: int, config: Dict[str, Any]) -> float:
    """Extract tempo from audio"""
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return round(tempo.item(), config["output"]["float_precision"])

def extract_energy(y: np.ndarray, config: Dict[str, Any]) -> float:
    """Extract energy (RMS) from audio"""
    frame_length = config["processing"]["energy_frame_length"]
    hop_length = config["processing"]["energy_hop_length"]
    
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    energy = np.mean(rms)
    return round(float(energy), config["output"]["float_precision"])

def extract_spectral_brightness(y: np.ndarray, sr: int, config: Dict[str, Any]) -> float:
    """Extract spectral brightness (mean spectral centroid)"""
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    brightness = np.mean(spectral_centroid)
    return round(float(brightness), config["output"]["float_precision"])

def extract_key(y: np.ndarray, sr: int, config: Dict[str, Any]) -> int:
    """Extract musical key from audio"""
    if config["processing"]["key_method"] == "chroma":
        # More sophisticated key detection
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        # Smooth over time for more stable detection
        window_frames = int(config["processing"]["key_smooth_window"] * sr / 512)
        if window_frames > 1:
            # Simple moving average
            kernel = np.ones(window_frames) / window_frames
            chroma_smooth = np.array([np.convolve(chroma[i], kernel, mode='same') for i in range(12)])
        else:
            chroma_smooth = chroma
            
        key = np.argmax(np.mean(chroma_smooth, axis=1))
    else:
        # Simple method - just strongest frequency
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key = np.argmax(np.mean(chroma, axis=1))
    
    return int(key)

def extract_duration(y: np.ndarray, sr: int, config: Dict[str, Any]) -> float:
    """Extract duration from audio"""
    duration = len(y) / sr
    return round(float(duration), config["output"]["float_precision"])

def check_duration_limits(duration: float, config: Dict[str, Any]) -> None:
    """Check if duration is within acceptable limits"""
    max_duration = config["validation"]["max_duration_minutes"] * 60
    min_duration = config["validation"]["min_duration_seconds"]
    
    if duration > max_duration:
        raise ValueError(f"Audio too long: {duration:.1f}s (max: {max_duration}s)")
    if duration < min_duration:
        raise ValueError(f"Audio too short: {duration:.1f}s (min: {min_duration}s)")

def process_audio_file(audio_path: str, config_path: str = "config.toml") -> Dict[str, Any]:
    """
    Main function to extract audio features from a music file
    
    Args:
        audio_path: Path to audio file
        config_path: Path to configuration file
        
    Returns:
        Dictionary with extracted features
    """
    # Load configuration
    config = load_config(config_path)
    
    # Validate input file
    validate_audio_file(audio_path, config)
    
    try:
        # Load audio
        sr = config["audio"]["sample_rate"]
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Check duration limits
        duration = extract_duration(y, sr, config)
        if config["validation"]["check_file_integrity"]:
            check_duration_limits(duration, config)
        
        # Extract features based on config
        features = {"audio_path": audio_path}
        
        if config["features"]["extract_duration"]:
            features["duration"] = duration
            
        if config["features"]["extract_tempo"]:
            features["tempo"] = extract_tempo(y, sr, config)
            
        if config["features"]["extract_energy"]:
            features["energy"] = extract_energy(y, config)
            
        if config["features"]["extract_key"]:
            features["key"] = extract_key(y, sr, config)
            
        if config["features"]["extract_spectral_brightness"]:
            features["spectral_brightness"] = extract_spectral_brightness(y, sr, config)
            
        # Note: valence is disabled by default in config - needs better implementation
        if config["features"]["extract_valence"]:
            # Placeholder - implement proper valence calculation later
            features["valence"] = 0.5
        
        # Add debug info if requested
        if config["output"]["include_debug_info"]:
            features["debug"] = {
                "sample_rate": sr,
                "samples": len(y),
                "config_used": config_path
            }
        
        return features
        
    except Exception as e:
        raise RuntimeError(f"Error processing audio file {audio_path}: {e}")

if __name__ == "__main__":
    try:
        features = process_audio_file("song.mp3")
        
        print("Extracted features:")
        for key, value in features.items():
            if key != "debug":
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. A config.toml file in the same directory")
        print("2. An audio file to test with")
        print("3. librosa installed (pip install librosa)")