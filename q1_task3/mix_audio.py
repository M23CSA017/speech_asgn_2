import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json
import torchaudio
import torchaudio.transforms as T
import torch

# Set constants
TARGET_SR = 16000  # Target sample rate (16 kHz)
MIX_OUTPUT_DIR = "voxmix_data"  # Output directory for mixed data

def load_audio(path, target_sr=TARGET_SR):
    """Load audio and resample to target sample rate"""
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        resample = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resample(waveform)
    return waveform

def align_audio(wave1, wave2):
    """Crop or pad waveforms to equal length"""
    min_len = min(wave1.shape[-1], wave2.shape[-1])
    wave1 = wave1[..., :min_len]
    wave2 = wave2[..., :min_len]
    return wave1, wave2

def mix_signals(wave1, wave2):
    """Mix audio signals by adding them"""
    mixture = wave1 + wave2
    mixture = torch.clamp(mixture, -1.0, 1.0)  # Avoid clipping
    return mixture

def save_audio(waveform, path, sample_rate=TARGET_SR):
    """Save audio to specified path"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchaudio.save(path, waveform, sample_rate)

def process_metadata(metadata_file, output_dir, split):
    """Process metadata and create mixed data"""
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    mix_dir = os.path.join(output_dir, split, "mix")
    s1_dir = os.path.join(output_dir, split, "s1")
    s2_dir = os.path.join(output_dir, split, "s2")

    os.makedirs(mix_dir, exist_ok=True)
    os.makedirs(s1_dir, exist_ok=True)
    os.makedirs(s2_dir, exist_ok=True)

    for mix_id, data in metadata.items():
        utt1_path = data["utt1_path"]
        utt2_path = data["utt2_path"]

        # Load and mix audio
        try:
            w1 = load_audio(utt1_path)
            w2 = load_audio(utt2_path)
        except Exception as e:
            print(f"Error loading audio: {e}")
            continue

        w1, w2 = align_audio(w1, w2)
        mix = mix_signals(w1, w2)

        # Save mixture and sources
        mix_path = os.path.join(mix_dir, f"{mix_id}.wav")
        s1_path = os.path.join(s1_dir, f"{mix_id}.wav")
        s2_path = os.path.join(s2_dir, f"{mix_id}.wav")

        save_audio(mix, mix_path)
        save_audio(w1, s1_path)
        save_audio(w2, s2_path)

        print(f" Mixed: {mix_id}")

# Run for both train and test sets
process_metadata("train_voxmix_metadata.json", MIX_OUTPUT_DIR, "train")
process_metadata("test_voxmix_metadata.json", MIX_OUTPUT_DIR, "test")

print("VoxCeleb2 mixture creation complete!")
