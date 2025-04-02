
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

# ============================
# CONFIGURATION
# ============================
DATA_DIR = "voxmix_data"
ENH_DIR = os.path.join(DATA_DIR, "enhanced", "test")
S1_DIR = os.path.join(DATA_DIR, "test", "s1")
S2_DIR = os.path.join(DATA_DIR, "test", "s2")
CHECKPOINT_PATH = "/home/m23csa017/speech_assn_2/task1A/outputs_final/checkpoints/best_model.pt"
SAMPLE_RATE = 16000
EMBEDDING_DIM = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_CSV = "identification_results.csv"

# ============================
# MODEL ARCHITECTURE (Matching finetune.py)
# ============================
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
        
    def forward(self, features):
        weights = self.attention_pool(features)
        embeddings = torch.sum(weights * features, dim=1)
        return self.projection(embeddings)

class SpeakerVerificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        self.speaker_encoder = SpeakerEncoder(
            hidden_size=self.wav2vec2.config.hidden_size,
            embedding_size=EMBEDDING_DIM
        )
    
    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask
        )
        frame_features = outputs.last_hidden_state
        return F.normalize(self.speaker_encoder(frame_features), p=2, dim=1)

# ============================
# UTILITY FUNCTIONS
# ============================
def load_audio(file_path, target_sr=SAMPLE_RATE):
    """Load and preprocess audio file."""
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        if waveform.shape[0] > 1:  # Convert stereo to mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)
        return waveform.squeeze(0)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

def preprocess_audio(waveform):
    """Preprocess audio using the same feature extractor as training"""
    inputs = feature_extractor(
        waveform.numpy(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )
    return inputs.input_values.to(DEVICE), inputs.attention_mask.to(DEVICE)

# ============================
# MODEL LOADING (Handles both regular and LoRA checkpoints)
# ============================
def load_model():
    print("Loading model...")
    model = SpeakerVerificationModel().to(DEVICE)
    
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model_state_dict = checkpoint['model_state_dict']
        
        # Handle DataParallel if used during training
        model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
        
        # Check if this is a LoRA checkpoint
        has_lora = any('lora' in k.lower() for k in model_state_dict.keys())
        
        if has_lora:
            print("Loading LoRA-enhanced model (strict=False)")
            model.load_state_dict(model_state_dict, strict=False)
        else:
            print("Loading regular checkpoint")
            model.load_state_dict(model_state_dict)
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise
    
    model.eval()
    print("Model loaded successfully!")
    return model

# ============================
# EMBEDDING EXTRACTION
# ============================
def get_embedding(audio_path, model):
    waveform = load_audio(audio_path)
    if waveform is None:
        return None
    
    input_values, attention_mask = preprocess_audio(waveform)
    with torch.no_grad():
        if isinstance(model, SpeakerVerificationModel):
            embedding = model(input_values, attention_mask=attention_mask)
        else:  # For pre-trained model
            outputs = model(input_values, attention_mask=attention_mask)
            embedding = outputs.last_hidden_state.mean(dim=1)
            embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.squeeze(0).cpu()

# ============================
# EVALUATION FUNCTION
# ============================
def evaluate_sid(model, model_type="fine-tuned", num_files=None):
    """Evaluate SID model and return detailed results."""
    enh1_dir = os.path.join(ENH_DIR, "enh1")
    enh2_dir = os.path.join(ENH_DIR, "enh2")
    files = sorted([f for f in os.listdir(enh1_dir) if f.endswith(".wav")])

    if num_files is not None:
        files = files[:num_files]

    results = []
    correct_predictions = 0
    
    for file in tqdm(files, desc=f"Evaluating {model_type} model"):
        # Get paths
        s1_path = os.path.join(S1_DIR, file)
        s2_path = os.path.join(S2_DIR, file)
        enh1_path = os.path.join(enh1_dir, file)
        enh2_path = os.path.join(enh2_dir, file)
        
        # Get embeddings
        s1_embed = get_embedding(s1_path, model)
        s2_embed = get_embedding(s2_path, model)
        enh1_embed = get_embedding(enh1_path, model)
        enh2_embed = get_embedding(enh2_path, model)
        
        if None in [s1_embed, s2_embed, enh1_embed, enh2_embed]:
            continue
        
        # Calculate similarities
        sim1_s1 = F.cosine_similarity(enh1_embed.unsqueeze(0), s1_embed.unsqueeze(0)).item()
        sim1_s2 = F.cosine_similarity(enh1_embed.unsqueeze(0), s2_embed.unsqueeze(0)).item()
        sim2_s1 = F.cosine_similarity(enh2_embed.unsqueeze(0), s1_embed.unsqueeze(0)).item()
        sim2_s2 = F.cosine_similarity(enh2_embed.unsqueeze(0), s2_embed.unsqueeze(0)).item()
        
        # Determine correctness
        correct_order = (sim1_s1 > sim1_s2 and sim2_s2 > sim2_s1)
        swapped_order = (sim1_s2 > sim1_s1 and sim2_s1 > sim2_s2)
        # is_correct = correct_order or swapped_order

        combined_similarity = (sim1_s1 + sim2_s2) - (sim1_s2 + sim2_s1)
        is_correct = combined_similarity > 0
        
        # Record results
        results.append({
            'file': file,
            'sim1_s1': sim1_s1,
            'sim1_s2': sim1_s2,
            'sim2_s1': sim2_s1,
            'sim2_s2': sim2_s2,
            'correct_order': correct_order,
            'swapped_order': swapped_order,
            'is_correct': is_correct,
            'model_type': model_type
        })
        
        if is_correct:
            correct_predictions += 1
    
    # Save results for analysis
    pd.DataFrame(results).to_csv(RESULTS_CSV, mode='a', header=not os.path.exists(RESULTS_CSV))
    
    accuracy = correct_predictions / len(files) if len(files) > 0 else 0
    print(f"\n{model_type} Model Results:")
    print(f"Files Processed: {len(files)}")
    print(f"Rank-1 Accuracy: {accuracy:.2%} ({correct_predictions}/{len(files)})")
    
    return accuracy, results

# ============================
# MAIN EXECUTION
# ============================
if __name__ == "__main__":
    try:
        # Clear previous results
        if os.path.exists(RESULTS_CSV):
            os.remove(RESULTS_CSV)

        # Load fine-tuned model (handles both regular and LoRA checkpoints)
        print("\n🔍 Evaluating Fine-Tuned Model...")
        fine_tuned_model = load_model()
        ft_accuracy, ft_results = evaluate_sid(fine_tuned_model, "fine-tuned")
        
        # Load pre-trained model
        print("\n🔍 Evaluating Pre-trained Model...")
        pre_trained_model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-large-xlsr-53"
        ).to(DEVICE)
        pre_trained_model.eval()
        pt_accuracy, pt_results = evaluate_sid(pre_trained_model, "pre-trained")

        # Comparative analysis
        improvement = ft_accuracy - pt_accuracy
        print(f"\nFine-Tuned Improvement: {improvement:.2%}")
        print(f" Results saved to {RESULTS_CSV}")

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        raise