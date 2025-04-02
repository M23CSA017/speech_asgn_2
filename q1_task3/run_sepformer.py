import os
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from speechbrain.inference import SepformerSeparation

# Set paths
MIXED_DATA_DIR = "voxmix_data"
OUTPUT_DIR = "voxmix_data/enhanced"

# Load SepFormer pre-trained model
model = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-whamr",
    savedir="sepformer_model"
)

# Define resampling function
resample_to_16k = T.Resample(orig_freq=8000, new_freq=16000)

def separate_and_save(mix_path, enh1_path, enh2_path):
    """Separate mixed audio and save enhanced outputs"""
    # Apply SepFormer to separate audio
    est_sources = model.separate_file(mix_path)

    # Resample to 16 kHz before saving
    est1_16k = resample_to_16k(est_sources[:, :, 0])
    est2_16k = resample_to_16k(est_sources[:, :, 1])

    # Save properly resampled sources
    torchaudio.save(enh1_path, est1_16k, 16000)
    torchaudio.save(enh2_path, est2_16k, 16000)

def process_split(split="train", max_files=None):

    """Process only first `max_files` in the split for testing"""
    mix_dir = os.path.join(MIXED_DATA_DIR, split, "mix")
    enh1_dir = os.path.join(OUTPUT_DIR, split, "enh1")
    enh2_dir = os.path.join(OUTPUT_DIR, split, "enh2")

    os.makedirs(enh1_dir, exist_ok=True)
    os.makedirs(enh2_dir, exist_ok=True)

    mix_files = [f for f in os.listdir(mix_dir) if f.endswith(".wav")]
    # Process all files if max_files is not specified
    if max_files is not None:
        mix_files = sorted(mix_files)[:max_files]
    else:
        mix_files = sorted(mix_files)


    for file in tqdm(mix_files, desc=f"Processing {split}"):
        mix_path = os.path.join(mix_dir, file)
        enh1_path = os.path.join(enh1_dir, file)
        enh2_path = os.path.join(enh2_dir, file)

        try:
            separate_and_save(mix_path, enh1_path, enh2_path)
        except Exception as e:
            print(f"Error processing {file}: {e}")


# Run for both train and test splits
process_split("train")
process_split("test")

print("SepFormer separation complete and correctly resampled to 16kHz!")
    