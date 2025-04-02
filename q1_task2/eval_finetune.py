import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from torch import nn
from torch.nn import functional as F
from peft import LoraConfig, PeftModel



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/wav2vec2-large-xlsr-53"
checkpoint_path ="/home/m23csa017/speech_assn_2/task1A/outputs_final/checkpoints/best_model.pt"
trial_file = "/home/m23csa017/speech_assn_2/task1A/veri_test2.txt"
wav_root = "/DATA2/PMCAll/speech/vox1/vox1_test_wav/wav"
sample_rate = 16000
max_length = 6
embedding_dim = 256


LORA_R = 8
LORA_ALPHA = 8
LORA_DROPOUT = 0.05



class SpeakerEncoder(nn.Module):
    def __init__(self, hidden_size=1024, embedding_size=256):
        super().__init__()
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, embedding_size)
        )
        for layer in [self.attention_pool[0], self.attention_pool[2]]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features):
        weights = self.attention_pool(features)
        embeddings = torch.sum(weights * features, dim=1)
        return self.projection(embeddings)

class SpeakerVerificationModel(nn.Module):
    def __init__(self, num_classes=100):  
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
        self.speaker_encoder = SpeakerEncoder(
            hidden_size=self.wav2vec2.config.hidden_size,
            embedding_size=embedding_dim
        )
        
        self.arcface = nn.Linear(embedding_dim, num_classes) if num_classes else None
    
    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        frame_features = outputs.last_hidden_state
        embeddings = self.speaker_encoder(frame_features)
        return F.normalize(embeddings, p=2, dim=1)

def apply_lora_to_model(model):
    
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["k_proj", "v_proj", "q_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    model.wav2vec2.encoder = PeftModel(model.wav2vec2.encoder, config)
    return model



feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

def load_audio(path):
    
    try:
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:  # convert to mono
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        return waveform.squeeze(0)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def get_embedding(model, audio_path):
    waveform = load_audio(audio_path)
    if waveform is None:
        return None
    
    
    inputs = feature_extractor(
        waveform.numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )
    
    with torch.no_grad():
        emb = model(
            inputs.input_values.to(device),
            attention_mask=inputs.attention_mask.to(device)
        )
    return emb.squeeze(0).cpu().numpy()



def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    return brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

def compute_tar_at_far(labels, scores, far_point=0.01):
    fpr, tpr, _ = roc_curve(labels, scores)
    return tpr[np.argmin(np.abs(fpr - far_point))]

def speaker_identification_accuracy(embeddings_dict):
    
    spk_embs = {}
    for path, emb in embeddings_dict.items():
        spk = path.split('/')[-2]  
        spk_embs.setdefault(spk, []).append(emb)
    
    centroids = {spk: np.mean(embs, axis=0) for spk, embs in spk_embs.items()}
    
    correct = 0
    total = 0
    for path, emb in embeddings_dict.items():
        gt_spk = path.split('/')[-2]
        best_score = -1
        best_spk = None
        for spk, centroid in centroids.items():
            score = np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid))
            if score > best_score:
                best_score = score
                best_spk = spk
        if best_spk == gt_spk:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0



def main():
    
    print("Loading model...")
    model = SpeakerVerificationModel(num_classes=100) 
    model = apply_lora_to_model(model)
    model.to(device)
    
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    
    print("Loading trial pairs")
    trials = []
    with open(trial_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                trials.append((
                    os.path.join(wav_root, parts[1]),
                    os.path.join(wav_root, parts[2]),
                    int(parts[0])
                ))
    print(f"Loaded {len(trials)} trial pairs")
    
    
    print("Computing embeddings")
    unique_files = set(p1 for p1, p2, _ in trials).union(set(p2 for p1, p2, _ in trials))
    embeddings_cache = {}
    for file_path in tqdm(unique_files, desc="Processing audio files"):
        emb = get_embedding(model, file_path)
        if emb is not None:
            embeddings_cache[file_path] = emb
    
    
    print("Evaluating pairs")
    scores, labels = [], []
    for p1, p2, label in tqdm(trials, desc="Evaluating trials"):
        if p1 in embeddings_cache and p2 in embeddings_cache:
            emb1 = embeddings_cache[p1]
            emb2 = embeddings_cache[p2]
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            scores.append(similarity)
            labels.append(label)
    
    
    if len(scores) > 0:
        eer = compute_eer(labels, scores)
        tar = compute_tar_at_far(labels, scores)
        id_acc = speaker_identification_accuracy(embeddings_cache)
        
        print("Final Results:")
        print(f"Processed {len(scores)}/{len(trials)} pairs")
        print(f"EER: {eer*100:.2f}%")
        print(f"TAR@1%FAR: {tar*100:.2f}%")
        print(f"Speaker ID Accuracy: {id_acc*100:.2f}%")
        
        
        results = pd.DataFrame({
            "enroll_path": [t[0] for t in trials[:len(scores)]],
            "test_path": [t[1] for t in trials[:len(scores)]],
            "label": labels,
            "score": scores
        })
        results.to_csv("verification_results.csv", index=False)
        print("Results saved to verification_results.csv")
    else:
        print("No valid pairs processed")

if __name__ == "__main__":
    main()


