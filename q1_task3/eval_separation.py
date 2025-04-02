
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import torchaudio
from pesq import pesq
from mir_eval.separation import bss_eval_sources
from collections import defaultdict

# Define paths
DATA_DIR = "voxmix_data"
ENH_DIR = os.path.join(DATA_DIR, "enhanced", "test")
S1_DIR = os.path.join(DATA_DIR, "test", "s1")
S2_DIR = os.path.join(DATA_DIR, "test", "s2")

def compute_bss_metrics(estimate, reference):
    """Compute SDR, SIR, and SAR with permutation handling"""
    sdr, sir, sar, _ = bss_eval_sources(reference, estimate, compute_permutation=True)
    return sdr, sir, sar

def compute_pesq(enh_path, ref_path, sr=16000):
    """Compute PESQ with length matching"""
    enh_wave, _ = torchaudio.load(enh_path)
    ref_wave, _ = torchaudio.load(ref_path)
    
    enh_wave = enh_wave.squeeze(0).numpy()
    ref_wave = ref_wave.squeeze(0).numpy()
    
    min_len = min(len(enh_wave), len(ref_wave))
    return pesq(sr, ref_wave[:min_len], enh_wave[:min_len], "wb")

def evaluate_file(mix_id, enh1_path, enh2_path):
    """Evaluate metrics for a single mixture"""
    s1_path = os.path.join(S1_DIR, f"{mix_id}.wav")
    s2_path = os.path.join(S2_DIR, f"{mix_id}.wav")

    # Load and prepare audio
    s1_wave, _ = torchaudio.load(s1_path)
    s2_wave, _ = torchaudio.load(s2_path)
    enh1_wave, _ = torchaudio.load(enh1_path)
    enh2_wave, _ = torchaudio.load(enh2_path)

    # Stack and match lengths
    reference = np.stack([s1_wave.squeeze().numpy(), s2_wave.squeeze().numpy()])
    estimate = np.stack([enh1_wave.squeeze().numpy(), enh2_wave.squeeze().numpy()])
    min_len = min(reference.shape[1], estimate.shape[1])
    reference, estimate = reference[:, :min_len], estimate[:, :min_len]

    # Compute metrics
    sdr, sir, sar = compute_bss_metrics(estimate, reference)
    pesq1 = compute_pesq(enh1_path, s1_path)
    pesq2 = compute_pesq(enh2_path, s2_path)

    return {
        "sdr": sdr, "sir": sir, "sar": sar,
        "pesq": [pesq1, pesq2]
    }

def evaluate_all(num_files=None):
    """Evaluate all files with statistical summary"""
    enh1_dir = os.path.join(ENH_DIR, "enh1")
    enh2_dir = os.path.join(ENH_DIR, "enh2")
    files = sorted([f for f in os.listdir(enh1_dir) if f.endswith(".wav")])[:num_files]
    
    metrics = defaultdict(list)
    results = []

    for file in files:
        try:
            mix_id = file.replace(".wav", "")
            res = evaluate_file(mix_id, 
                                os.path.join(enh1_dir, file),
                                os.path.join(enh2_dir, file))
            
            # Store individual results
            results.append({"mix_id": mix_id, **res})
            
            # Aggregate for statistics
            for metric in ['sdr', 'sir', 'sar', 'pesq']:
                metrics[metric].extend(res[metric])
                
        except Exception as e:
            print(f"Error processing {file}: {e}")

    # Calculate statistics
    stats = {
        metric: {
            'mean': np.nanmean(values),
            'std': np.nanstd(values),
            'min': np.nanmin(values),
            'max': np.nanmax(values)
        }
        for metric, values in metrics.items()
    }
    
    return results, stats

# Run evaluation
results, stats = evaluate_all(num_files=None)

# Print detailed results
print("\n Individual Results:")
for res in results:
    print(f"\nFile: {res['mix_id']}")
    for i in range(2):
        print(f"  Speaker {i+1}:")
        print(f"    SDR: {res['sdr'][i]:.2f} dB, SIR: {res['sir'][i]:.2f} dB, SAR: {res['sar'][i]:.2f} dB")
        print(f"    PESQ: {res['pesq'][i]:.2f}")

# Print statistical summary in your requested format
print("\n4.3 Summary:")
print("Metric\tMean\tStd Dev\tMin\tMax")

for metric in ['sdr', 'sir', 'sar', 'pesq']:
    values = stats[metric]
    print(f"{metric.upper()}\t{values['mean']:.2f}\t{values['std']:.2f}\t{values['min']:.2f}\t{values['max']:.2f}")
