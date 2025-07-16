import librosa
import numpy as np
import tomllib
from pathlib import Path
from typing import Dict, Any, List, Tuple
from scipy import signal

def load_config(config_path: str = "config.toml") -> Dict[str, Any]:
    """Load configuration from TOML file"""
    try:
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {config_path} not found")
    except Exception as e:
        raise ValueError(f"Error loading config: {e}")

def extract_tempo_advanced(y: np.ndarray, sr: int, config: Dict[str, Any]) -> float:
    """
    Advanced tempo detection using multiple methods and consensus
    """
    tempos = []
    
    # Method 1: Standard beat tracking
    try:
        tempo1, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempos.append(tempo1.item())
    except:
        pass
    
    # Method 2: Onset-based tempo detection
    try:
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        if len(onset_times) > 1:
            # Calculate intervals between onsets
            intervals = np.diff(onset_times)
            # Filter out very short/long intervals (outliers)
            intervals = intervals[(intervals > 0.2) & (intervals < 2.0)]
            if len(intervals) > 0:
                avg_interval = np.median(intervals)  # Use median to avoid outliers
                tempo2 = 60.0 / avg_interval
                tempos.append(tempo2)
    except:
        pass
    
    # Method 3: Autocorrelation-based tempo
    try:
        # Get onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        # Autocorrelation
        autocorr = np.correlate(onset_env, onset_env, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation
        peaks, _ = signal.find_peaks(autocorr, height=np.max(autocorr)*0.3)
        if len(peaks) > 0:
            # Convert lag to tempo
            lag_frames = peaks[0]
            lag_time = lag_frames * (512 / sr)  # Default hop_length is 512
            tempo3 = 60.0 / lag_time
            if 40 <= tempo3 <= 300:  # Reasonable tempo range
                tempos.append(tempo3)
    except:
        pass
    
    # Method 4: Fourier-based tempo detection
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo_freqs = np.fft.rfftfreq(len(onset_env), d=512/sr)
        tempo_fft = np.abs(np.fft.rfft(onset_env))
        
        # Look for peaks in tempo range (40-300 BPM)
        tempo_bpm = tempo_freqs * 60
        valid_range = (tempo_bpm >= 40) & (tempo_bpm <= 300)
        if np.any(valid_range):
            peak_idx = np.argmax(tempo_fft[valid_range])
            tempo4 = tempo_bpm[valid_range][peak_idx]
            tempos.append(tempo4)
    except:
        pass
    
    # Consensus: clean up the tempos and find the best one
    if not tempos:
        return 120.0  # Default fallback
    
    # Remove outliers (more than 50% different from median)
    tempos = np.array(tempos)
    median_tempo = np.median(tempos)
    cleaned_tempos = tempos[np.abs(tempos - median_tempo) < median_tempo * 0.5]
    
    if len(cleaned_tempos) == 0:
        final_tempo = median_tempo
    else:
        final_tempo = np.mean(cleaned_tempos)
    
    # Snap to common musical tempos (helps with slight inaccuracies)
    common_tempos = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 200]
    closest_common = min(common_tempos, key=lambda x: abs(x - final_tempo))
    
    # Only snap if we're within 5 BPM of a common tempo
    if abs(final_tempo - closest_common) <= 5:
        final_tempo = closest_common
    
    return round(final_tempo, config["output"]["float_precision"])

def get_krumhansl_schmuckler_profiles():
    """
    Return the Krumhansl-Schmuckler key profiles
    These are experimentally derived weights for how each note
    sounds in major and minor keys
    """
    # Major key profile (how much each note "belongs" in a major key)
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    
    # Minor key profile
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    # Normalize profiles
    major = major / np.sum(major)
    minor = minor / np.sum(minor)
    
    return major, minor

def extract_key_advanced(y: np.ndarray, sr: int, config: Dict[str, Any]) -> Tuple[int, str]:
    """
    Advanced key detection using Krumhansl-Schmuckler algorithm
    Returns: (key_number, mode) where mode is 'major' or 'minor'
    """
    try:
        # Extract chroma features
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, fmin=librosa.note_to_hz('C1'))
        
        # Average chroma over time (gives us the overall harmonic content)
        avg_chroma = np.mean(chroma, axis=1)
        
        # Normalize to get relative strengths
        avg_chroma = avg_chroma / np.sum(avg_chroma)
        
        # Get the Krumhansl-Schmuckler profiles
        major_profile, minor_profile = get_krumhansl_schmuckler_profiles()
        
        # Test all 24 possible keys (12 major + 12 minor)
        correlations = []
        
        for tonic in range(12):  # For each possible root note
            # Rotate the profiles to test different keys
            major_rotated = np.roll(major_profile, tonic)
            minor_rotated = np.roll(minor_profile, tonic)
            
            # Calculate correlation between our chroma and each key profile
            major_corr = np.corrcoef(avg_chroma, major_rotated)[0, 1]
            minor_corr = np.corrcoef(avg_chroma, minor_rotated)[0, 1]
            
            correlations.append((tonic, 'major', major_corr))
            correlations.append((tonic, 'minor', minor_corr))
        
        # Find the key with highest correlation
        best_key = max(correlations, key=lambda x: x[2] if not np.isnan(x[2]) else -1)
        
        return int(best_key[0]), best_key[1]
        
    except Exception as e:
        print(f"Key detection failed: {e}, falling back to simple method")
        # Fallback to simple method
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key = np.argmax(np.mean(chroma, axis=1))
        return int(key), 'unknown'

def extract_energy_advanced(y: np.ndarray, config: Dict[str, Any]) -> Dict[str, float]:
    """
    Advanced energy analysis with multiple metrics
    """
    # RMS Energy (what we had before)
    rms = librosa.feature.rms(y=y)[0]
    rms_energy = float(np.mean(rms))
    
    # Peak energy
    peak_energy = float(np.max(np.abs(y)))
    
    # Zero crossing rate (indicates percussive content)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    avg_zcr = float(np.mean(zcr))
    
    # Dynamic range (difference between loud and quiet parts)
    dynamic_range = float(np.max(rms) - np.min(rms))
    
    precision = config["output"]["float_precision"]
    
    return {
        "energy": round(rms_energy, precision),
        "peak_energy": round(peak_energy, precision),
        "dynamic_range": round(dynamic_range, precision),
        "percussive_content": round(avg_zcr, precision)
    }

def extract_spectral_features_advanced(y: np.ndarray, sr: int, config: Dict[str, Any]) -> Dict[str, float]:
    """
    Advanced spectral analysis
    """
    precision = config["output"]["float_precision"]
    
    # Spectral centroid (brightness)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    brightness = float(np.mean(spectral_centroid))
    
    # Spectral rolloff (where 85% of energy is below this frequency)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    avg_rolloff = float(np.mean(rolloff))
    
    # Spectral bandwidth (spread of frequencies)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    avg_bandwidth = float(np.mean(bandwidth))
    
    # Spectral contrast (difference between peaks and valleys in spectrum)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    avg_contrast = float(np.mean(contrast))
    
    return {
        "spectral_brightness": round(brightness, precision),
        "spectral_rolloff": round(avg_rolloff, precision),
        "spectral_bandwidth": round(avg_bandwidth, precision),
        "spectral_contrast": round(avg_contrast, precision)
    }

def extract_rhythmic_features_advanced(y: np.ndarray, sr: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Advanced rhythmic analysis
    """
    precision = config["output"]["float_precision"]
    
    # Beat tracking with confidence
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    
    # Beat consistency (how regular are the beats?)
    if len(beat_times) > 1:
        beat_intervals = np.diff(beat_times)
        beat_consistency = 1.0 - (np.std(beat_intervals) / np.mean(beat_intervals))
        beat_consistency = max(0.0, min(1.0, beat_consistency))  # Clamp to 0-1
    else:
        beat_consistency = 0.0
    
    # Onset density (how many note starts per second)
    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    onset_density = len(onsets) / (len(y) / sr)
    
    return {
        "beat_consistency": round(float(beat_consistency), precision),
        "onset_density": round(float(onset_density), precision),
        "num_beats": len(beat_times)
    }

def process_audio_file_advanced(audio_path: str, config_path: str = "config.toml") -> Dict[str, Any]:
    """
    Advanced audio processing with sophisticated feature extraction
    """
    # Load configuration
    config = load_config(config_path)
    
    try:
        # Load audio
        sr = config["audio"]["sample_rate"]
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Basic info
        duration = len(y) / sr
        features = {
            "audio_path": audio_path,
            "duration": round(duration, config["output"]["float_precision"])
        }
        
        print(f"🎵 Processing: {Path(audio_path).name}")
        print(f"   Duration: {duration:.1f}s")
        
        # Advanced tempo detection
        if config["features"]["extract_tempo"]:
            print("   🥁 Analyzing tempo...")
            features["tempo"] = extract_tempo_advanced(y, sr, config)
        
        # Advanced key detection
        if config["features"]["extract_key"]:
            print("   🎹 Detecting key...")
            key_num, key_mode = extract_key_advanced(y, sr, config)
            features["key"] = key_num
            features["key_mode"] = key_mode
            
            # Convert to note name for readability
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            features["key_name"] = f"{note_names[key_num]} {key_mode}"
        
        # Advanced energy analysis
        if config["features"]["extract_energy"]:
            print("   ⚡ Analyzing energy...")
            energy_features = extract_energy_advanced(y, config)
            features.update(energy_features)
        
        # Advanced spectral features
        if config["features"]["extract_spectral_brightness"]:
            print("   🌈 Analyzing spectrum...")
            spectral_features = extract_spectral_features_advanced(y, sr, config)
            features.update(spectral_features)
        
        # Advanced rhythmic features
        print("   🎶 Analyzing rhythm...")
        rhythmic_features = extract_rhythmic_features_advanced(y, sr, config)
        features.update(rhythmic_features)
        
        print("   ✅ Analysis complete!")
        
        return features
        
    except Exception as e:
        raise RuntimeError(f"Error processing audio file {audio_path}: {e}")

# Example usage and comparison
if __name__ == "__main__":
    try:
        print("🚀 Advanced Music Preprocessor")
        print("=" * 40)
        
        # Process a song with advanced methods
        features = process_audio_file_advanced("song.mp3")
        
        print("\n📊 Extracted Features:")
        print("-" * 30)
        
        # Group features for better readability
        basic_info = ["audio_path", "duration"]
        tempo_info = ["tempo", "beat_consistency", "onset_density", "num_beats"]
        key_info = ["key", "key_mode", "key_name"]
        energy_info = ["energy", "peak_energy", "dynamic_range", "percussive_content"]
        spectral_info = ["spectral_brightness", "spectral_rolloff", "spectral_bandwidth", "spectral_contrast"]
        
        sections = [
            ("📁 Basic Info", basic_info),
            ("🥁 Tempo & Rhythm", tempo_info),
            ("🎹 Musical Key", key_info),
            ("⚡ Energy Analysis", energy_info),
            ("🌈 Spectral Analysis", spectral_info)
        ]
        
        for section_name, feature_list in sections:
            print(f"\n{section_name}:")
            for feature in feature_list:
                if feature in features:
                    print(f"  {feature}: {features[feature]}")
                    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. A config.toml file in the same directory")
        print("2. An audio file to test with")
        print("3. librosa and scipy installed")
        print("   pip install librosa scipy")