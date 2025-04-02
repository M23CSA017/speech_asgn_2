import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
from peft import LoraConfig, PeftModel
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve



seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")



DATA_ROOT = "/DATA2/PMCAll/speech/vox2"
VOX2_AAC_DIR = os.path.join(DATA_ROOT, "aac")
VOX2_TXT_DIR = os.path.join(DATA_ROOT, "txt")
OUTPUT_DIR = "./outputs_final"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
writer = SummaryWriter(os.path.join(OUTPUT_DIR, "tensorboard_logs"))


# Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 3e-5
NUM_EPOCHS = 10
MAX_AUDIO_LENGTH = 6  
MIN_AUDIO_LENGTH = 3  
SAMPLE_RATE = 16000
EMBEDDING_DIM = 256
MARGIN = 0.5
SCALE = 30.0
LORA_R = 8
LORA_ALPHA = 8
LORA_DROPOUT = 0.05


def get_speaker_ids(audio_dir):
    
    speaker_paths = glob.glob(os.path.join(audio_dir, "id*"))
    return sorted([os.path.basename(path) for path in speaker_paths])

def split_speakers(all_speaker_ids):
    
    random.shuffle(all_speaker_ids)
    train_speaker_ids = all_speaker_ids[:100]
    val_speaker_ids = all_speaker_ids[100:118]
    return train_speaker_ids, val_speaker_ids

all_speaker_ids = get_speaker_ids(VOX2_AAC_DIR)
train_speaker_ids, val_speaker_ids = split_speakers(all_speaker_ids)
print(f"Found {len(all_speaker_ids)} speakers.")
print(f"Training with {len(train_speaker_ids)} speakers: {train_speaker_ids[:3]}...")
print(f"Validation with {len(val_speaker_ids)} speakers: {val_speaker_ids[:3]}...")

# Creating label mappings
id_to_label = {speaker_id: idx for idx, speaker_id in enumerate(train_speaker_ids)}
val_id_to_label = {speaker_id: idx for idx, speaker_id in enumerate(val_speaker_ids)}



def load_audio(file_path):
    
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)
        return waveform.squeeze(0)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None



class VoxCeleb2Dataset(Dataset):
    def __init__(self, audio_dir, speaker_ids, feature_extractor, is_val=False, max_samples_per_speaker=None):
        self.audio_dir = audio_dir
        self.speaker_ids = speaker_ids
        self.feature_extractor = feature_extractor
        self.is_val = is_val
        self.samples = []
        
        speaker_counts = {s: 0 for s in speaker_ids}
        for speaker_id in tqdm(speaker_ids, desc="Building dataset"):
            speaker_path = os.path.join(audio_dir, speaker_id)
            if not os.path.exists(speaker_path):
                continue
                
            for root, _, files in os.walk(speaker_path):
                for file in files:
                    if file.endswith(".m4a"):
                        if max_samples_per_speaker and speaker_counts[speaker_id] >= max_samples_per_speaker:
                            continue
                            
                        audio_file = os.path.join(root, file)
                        label = val_id_to_label.get(speaker_id, -1) if is_val else id_to_label[speaker_id]
                        self.samples.append({
                            "audio_path": audio_file, 
                            "label": label,
                            "speaker_id": speaker_id
                        })
                        speaker_counts[speaker_id] += 1

        print(f"Dataset built with {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        waveform = load_audio(sample["audio_path"])
        if waveform is None:
            random_idx = random.randint(0, len(self)-1)
            return self[random_idx]
            
        return {
            "waveform": waveform,
            "label": sample["label"],
            "speaker_id": sample["speaker_id"],
            "audio_path": sample["audio_path"]
        }

def get_collate_fn(feature_extractor):
    def collate_fn(batch):
        waveforms = [item["waveform"].numpy() for item in batch if item["waveform"] is not None]
        labels = torch.tensor([item["label"] for item in batch if item["waveform"] is not None])
        speaker_ids = [item["speaker_id"] for item in batch if item["waveform"] is not None]
        audio_paths = [item["audio_path"] for item in batch if item["waveform"] is not None]

        inputs = feature_extractor(
            waveforms,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        )

        return {
            "input_values": inputs.input_values,
            "attention_mask": inputs.attention_mask,
            "labels": labels,
            "speaker_ids": speaker_ids,
            "audio_paths": audio_paths
        }
    return collate_fn



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

class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_size, num_classes, s=SCALE, m=MARGIN):
        super(ArcFaceLoss, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weights = F.normalize(self.weight, p=2, dim=1)
        cos_theta = F.linear(embeddings, weights)
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        theta_m = theta + self.m * one_hot
        cos_theta_m = torch.cos(theta_m)
        logits = self.s * (one_hot * cos_theta_m + (1 - one_hot) * cos_theta)
        return F.cross_entropy(logits, labels)

class SpeakerVerificationModel(nn.Module):
    def __init__(self, num_classes, pretrained_model_name="facebook/wav2vec2-large-xlsr-53"):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
            
        self.speaker_encoder = SpeakerEncoder(
            hidden_size=self.wav2vec2.config.hidden_size,
            embedding_size=EMBEDDING_DIM
        )
        
        self.arcface = ArcFaceLoss(EMBEDDING_DIM, num_classes)
    
    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask
        )
        
        frame_features = outputs.last_hidden_state
        embeddings = self.speaker_encoder(frame_features)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        if labels is not None:
            loss = self.arcface(embeddings, labels)
            return embeddings, loss
            
        return embeddings

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
    model.wav2vec2.encoder.print_trainable_parameters()
    return model



def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def compute_tar_at_far(labels, scores, far_point=0.01):
    fpr, tpr, _ = roc_curve(labels, scores)
    return tpr[np.argmin(np.abs(fpr - far_point))]

def batched_cosine_similarity(embeddings, batch_size=1000):
    N = embeddings.size(0)
    similarity_matrix = torch.zeros((N, N), device=embeddings.device)
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        for j in range(0, N, batch_size):
            end_j = min(j + batch_size, N)
            similarity_matrix[i:end_i, j:end_j] = F.cosine_similarity(
                embeddings[i:end_i].unsqueeze(1),
                embeddings[j:end_j].unsqueeze(0),
                dim=2
            )
    return similarity_matrix

def speaker_identification_accuracy(embeddings, labels):
    similarity_matrix = batched_cosine_similarity(embeddings)
    predicted_labels = []
    for i in range(len(labels)):
        similarity_matrix[i, i] = -1
        pred_idx = similarity_matrix[i].argmax().item()
        predicted_labels.append(labels[pred_idx].item())
    return accuracy_score(labels.cpu().numpy(), np.array(predicted_labels)) * 100

def validate_identification(model, val_loader):
    model.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            embeddings = model(input_values)
            all_embeddings.append(embeddings)
            all_labels.append(labels)
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return speaker_identification_accuracy(all_embeddings, all_labels)



def train_epoch(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch in progress:
        inputs = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            _, loss = model(inputs, attention_mask=attention_mask, labels=labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        
        total_loss += loss.item()
        progress.set_postfix({'loss': loss.item()})
    
    return total_loss / len(train_loader)



def main():
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    
    train_dataset = VoxCeleb2Dataset(
        VOX2_AAC_DIR, train_speaker_ids, feature_extractor, 
        is_val=False, max_samples_per_speaker=100
    )
    val_dataset = VoxCeleb2Dataset(
        VOX2_AAC_DIR, val_speaker_ids, feature_extractor,
        is_val=True, max_samples_per_speaker=20
    )
    
    collate_fn = get_collate_fn(feature_extractor)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=4, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True
    )
    
    model = SpeakerVerificationModel(num_classes=len(train_speaker_ids))
    model = apply_lora_to_model(model)
    model.to(device)
    
    optimizer = optim.AdamW([
        {'params': model.wav2vec2.parameters(), 'lr': 1e-5},
        {'params': model.speaker_encoder.parameters(), 'lr': 3e-5},
        {'params': model.arcface.parameters(), 'lr': 5e-5}
    ])
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    best_val_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        avg_loss = train_epoch(model, train_loader, optimizer, epoch)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        
        val_acc = validate_identification(model, val_loader)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            print(f"New best model saved with val accuracy: {best_val_acc:.2f}%")
        
        scheduler.step()
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_acc': best_val_acc,
    }, os.path.join(CHECKPOINT_DIR, "final_model.pt"))
    
    writer.close()
    print("Training completed successfully!")

if __name__ == "__main__":
    main()