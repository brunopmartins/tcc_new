# Implementation Guide: Building a Kinship Classification System

## 1. Quick Start

### 1.1 Environment Setup

```bash
# Create conda environment
conda create -n kinship python=3.8
conda activate kinship

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
pip install numpy pandas scikit-learn matplotlib tqdm
pip install opencv-python pillow albumentations

# Install face recognition models
pip install facenet-pytorch  # FaceNet
pip install insightface onnxruntime-gpu  # ArcFace

# Optional: Transformers for ViT
pip install transformers timm

# Optional: Experiment tracking
pip install wandb tensorboard
```

### 1.2 Repository Setup

```bash
# Clone key repositories
git clone https://github.com/visionjo/FIW_KRT.git        # FIW Toolbox
git clone https://github.com/garynlfd/KFC.git            # Fairness-aware
git clone https://github.com/zhenyuzhouiu/FaCoR.git      # FaCoRNet

# Project structure
mkdir kinship_project
cd kinship_project
mkdir -p data models logs configs
```

---

## 2. Data Preparation

### 2.1 FIW Dataset Download

1. Visit: https://web.northeastern.edu/smilelab/fiw/
2. Request access and agree to terms
3. Download the dataset archive
4. Extract to `data/FIW/`

### 2.2 Dataset Structure

```
data/
├── FIW/
│   ├── FIDs/
│   │   ├── FID0001/
│   │   │   ├── MID1/
│   │   │   │   ├── P00001_face0.jpg
│   │   │   │   └── ...
│   │   │   ├── MID2/
│   │   │   └── F0001.csv
│   │   ├── FID0002/
│   │   └── ...
│   ├── FIW_PIDs.csv
│   ├── FIW_FIDs.csv
│   └── FIW_RIDs.csv
├── train_pairs.csv
├── val_pairs.csv
└── test_pairs.csv
```

### 2.3 Data Loading

```python
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class FIWDataset(Dataset):
    """
    FIW Dataset for kinship verification.
    """
    def __init__(self, root_dir, pairs_csv, transform=None):
        """
        Args:
            root_dir: Path to FIW/FIDs/
            pairs_csv: CSV with columns [img1_path, img2_path, label, relation]
            transform: Image transforms
        """
        self.root_dir = root_dir
        self.pairs = pd.read_csv(pairs_csv)
        self.transform = transform or self._default_transform()
    
    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((112, 112)),  # ArcFace input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        
        img1_path = os.path.join(self.root_dir, row['img1_path'])
        img2_path = os.path.join(self.root_dir, row['img2_path'])
        
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        label = row['label']  # 1 = kin, 0 = non-kin
        relation = row.get('relation', -1)  # FS=0, FD=1, MS=2, MD=3, etc.
        
        return img1, img2, label, relation


def create_dataloaders(data_dir, batch_size=32, num_workers=4):
    """Create train, val, test dataloaders."""
    
    train_dataset = FIWDataset(
        root_dir=os.path.join(data_dir, 'FIW/FIDs'),
        pairs_csv=os.path.join(data_dir, 'train_pairs.csv'),
        transform=get_train_transform()
    )
    
    val_dataset = FIWDataset(
        root_dir=os.path.join(data_dir, 'FIW/FIDs'),
        pairs_csv=os.path.join(data_dir, 'val_pairs.csv'),
        transform=get_val_transform()
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomCrop((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
```

### 2.4 Pair Generation

```python
import random
from collections import defaultdict

def generate_pairs(fiw_root, output_csv, split='train', neg_ratio=1.0):
    """
    Generate positive and negative pairs from FIW.
    
    Args:
        fiw_root: Path to FIW/FIDs/
        output_csv: Output path for pairs CSV
        split: 'train', 'val', or 'test'
        neg_ratio: Ratio of negative to positive pairs
    """
    # Load family relationships
    families = load_families(fiw_root)
    
    # Split families
    all_fids = list(families.keys())
    random.shuffle(all_fids)
    n = len(all_fids)
    
    if split == 'train':
        fids = all_fids[:int(0.7*n)]
    elif split == 'val':
        fids = all_fids[int(0.7*n):int(0.85*n)]
    else:
        fids = all_fids[int(0.85*n):]
    
    pairs = []
    
    # Generate positive pairs (kin)
    for fid in fids:
        family = families[fid]
        for mid1, mid2 in family['kin_pairs']:
            # Get image paths for both members
            img1_paths = get_member_images(fiw_root, fid, mid1)
            img2_paths = get_member_images(fiw_root, fid, mid2)
            
            for img1 in img1_paths:
                for img2 in img2_paths:
                    pairs.append({
                        'img1_path': img1,
                        'img2_path': img2,
                        'label': 1,
                        'relation': family['relations'].get((mid1, mid2), -1)
                    })
    
    # Generate negative pairs (non-kin)
    positive_count = len(pairs)
    negative_count = int(positive_count * neg_ratio)
    
    all_images = []
    for fid in fids:
        for img in get_all_family_images(fiw_root, fid):
            all_images.append((fid, img))
    
    neg_pairs = 0
    while neg_pairs < negative_count:
        # Sample two images from different families
        (fid1, img1), (fid2, img2) = random.sample(all_images, 2)
        if fid1 != fid2:
            pairs.append({
                'img1_path': img1,
                'img2_path': img2,
                'label': 0,
                'relation': -1
            })
            neg_pairs += 1
    
    # Save to CSV
    df = pd.DataFrame(pairs)
    df.to_csv(output_csv, index=False)
    print(f"Generated {len(pairs)} pairs ({positive_count} pos, {neg_pairs} neg)")
```

---

## 3. Model Implementation

### 3.1 Baseline: Siamese with ArcFace

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceEncoder(nn.Module):
    """
    ArcFace backbone encoder.
    """
    def __init__(self, pretrained_path=None):
        super().__init__()
        from insightface.recognition.arcface_torch import get_model
        self.backbone = get_model('r100', dropout=0.0, fp16=False)
        
        if pretrained_path:
            self.backbone.load_state_dict(torch.load(pretrained_path))
    
    def forward(self, x):
        return self.backbone(x)


class SiameseKinshipNet(nn.Module):
    """
    Siamese network for kinship verification.
    """
    def __init__(self, encoder, embedding_dim=512, freeze_encoder=False):
        super().__init__()
        self.encoder = encoder
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Comparison head
        self.head = nn.Sequential(
            nn.Linear(embedding_dim * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, img1, img2):
        # Extract embeddings
        emb1 = self.encoder(img1)
        emb2 = self.encoder(img2)
        
        # Normalize
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)
        
        # Feature interactions
        diff = emb1 - emb2
        prod = emb1 * emb2
        sq_diff = (emb1 - emb2) ** 2
        
        # Concatenate
        combined = torch.cat([diff, prod, sq_diff, emb1 + emb2], dim=1)
        
        # Predict
        logits = self.head(combined)
        return logits.squeeze(-1), emb1, emb2
    
    def get_embeddings(self, img):
        emb = self.encoder(img)
        return F.normalize(emb, p=2, dim=1)
```

### 3.2 Contrastive Learning Model

```python
class ContrastiveKinshipNet(nn.Module):
    """
    Kinship network with contrastive learning.
    """
    def __init__(self, encoder, embedding_dim=512, proj_dim=256):
        super().__init__()
        self.encoder = encoder
        
        # Projection head for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, proj_dim)
        )
        
        # Classification head (optional)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, img1, img2, return_embeddings=False):
        # Encode
        feat1 = self.encoder(img1)
        feat2 = self.encoder(img2)
        
        # Project
        z1 = self.projector(feat1)
        z2 = self.projector(feat2)
        
        # Normalize for contrastive loss
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        if return_embeddings:
            return z1, z2
        
        # Classification
        combined = torch.cat([z1, z2, z1 - z2, z1 * z2], dim=1)
        logits = self.classifier(combined)
        
        return logits.squeeze(-1), z1, z2
```

---

## 4. Training Pipeline

### 4.1 Training Script

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb

def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for img1, img2, labels, _ in pbar:
        img1 = img1.to(device)
        img2 = img2.to(device)
        labels = labels.float().to(device)
        
        optimizer.zero_grad()
        
        if scaler:  # Mixed precision
            with autocast():
                logits, emb1, emb2 = model(img1, img2)
                loss = criterion(logits, labels, emb1, emb2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, emb1, emb2 = model(img1, img2)
            loss = criterion(logits, labels, emb1, emb2)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
    
    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for img1, img2, labels, _ in tqdm(dataloader, desc='Validating'):
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.float().to(device)
            
            logits, emb1, emb2 = model(img1, img2)
            loss = criterion(logits, labels, emb1, emb2)
            
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute AUC
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(all_labels, all_preds)
    
    return total_loss / len(dataloader), correct / total, auc


def train(config):
    """Main training function."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    train_loader, val_loader = create_dataloaders(
        config['data_dir'],
        batch_size=config['batch_size']
    )
    
    # Model
    encoder = ArcFaceEncoder(config.get('pretrained_path'))
    model = SiameseKinshipNet(encoder, freeze_encoder=config.get('freeze_encoder', False))
    model = model.to(device)
    
    # Loss
    criterion = CombinedLoss(
        bce_weight=config.get('bce_weight', 1.0),
        contrastive_weight=config.get('contrastive_weight', 0.5),
        temperature=config.get('temperature', 0.08)
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=1e-6
    )
    
    # Mixed precision
    scaler = GradScaler() if config.get('fp16', True) else None
    
    # Training loop
    best_auc = 0
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_acc, val_auc = validate(
            model, val_loader, criterion, device
        )
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), f"models/best_model.pth")
            print(f"Saved best model with AUC: {val_auc:.4f}")
        
        # Log to wandb
        if config.get('use_wandb', False):
            wandb.log({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_auc': val_auc,
                'lr': optimizer.param_groups[0]['lr']
            })
    
    return model, best_auc


class CombinedLoss(nn.Module):
    """Combined BCE and contrastive loss."""
    def __init__(self, bce_weight=1.0, contrastive_weight=0.5, temperature=0.08):
        super().__init__()
        self.bce_weight = bce_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, labels, emb1, emb2):
        # BCE loss
        bce_loss = self.bce(logits, labels)
        
        # Contrastive loss (for positive pairs only)
        if self.contrastive_weight > 0:
            pos_mask = labels == 1
            if pos_mask.sum() > 0:
                similarity = F.cosine_similarity(emb1[pos_mask], emb2[pos_mask])
                contrastive_loss = (1 - similarity).mean()
            else:
                contrastive_loss = 0
        else:
            contrastive_loss = 0
        
        return self.bce_weight * bce_loss + self.contrastive_weight * contrastive_loss
```

### 4.2 Configuration

```python
# config.py
config = {
    # Data
    'data_dir': 'data/',
    'batch_size': 32,
    
    # Model
    'pretrained_path': 'models/arcface_r100.pth',
    'freeze_encoder': False,
    
    # Training
    'epochs': 50,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'fp16': True,
    
    # Loss
    'bce_weight': 1.0,
    'contrastive_weight': 0.5,
    'temperature': 0.08,
    
    # Logging
    'use_wandb': True,
    'project_name': 'kinship-verification',
}

if __name__ == '__main__':
    train(config)
```

---

## 5. Evaluation

### 5.1 Evaluation Script

```python
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

def evaluate_model(model, test_loader, device):
    """Comprehensive model evaluation."""
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    all_relations = []
    
    with torch.no_grad():
        for img1, img2, labels, relations in tqdm(test_loader):
            img1 = img1.to(device)
            img2 = img2.to(device)
            
            logits, _, _ = model(img1, img2)
            probs = torch.sigmoid(logits)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend((probs > 0.5).float().cpu().numpy())
            all_labels.extend(labels.numpy())
            all_relations.extend(relations.numpy())
    
    # Convert to numpy
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_relations = np.array(all_relations)
    
    # Overall metrics
    results = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs),
        'f1': f1_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
    }
    
    # Per-relation metrics
    relation_names = {0: 'FS', 1: 'FD', 2: 'MS', 3: 'MD', 4: 'Siblings'}
    results['per_relation'] = {}
    
    for rel_id, rel_name in relation_names.items():
        mask = all_relations == rel_id
        if mask.sum() > 0:
            results['per_relation'][rel_name] = {
                'accuracy': accuracy_score(all_labels[mask], all_preds[mask]),
                'count': mask.sum()
            }
    
    return results


def print_results(results):
    """Pretty print evaluation results."""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {results['accuracy']*100:.2f}%")
    print(f"  AUC:       {results['auc']:.4f}")
    print(f"  F1 Score:  {results['f1']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    
    print(f"\nPer-Relation Accuracy:")
    for rel, metrics in results['per_relation'].items():
        print(f"  {rel}: {metrics['accuracy']*100:.2f}% (n={metrics['count']})")
```

---

## 6. Using Existing Repositories

### 6.1 FIW_KRT (Official Toolbox)

```bash
cd FIW_KRT

# Install
pip install -r requirements.txt

# Download FIW dataset (follow instructions)
# ...

# Run baseline
python src/models/train_model.py --config configs/baseline.yaml
```

### 6.2 KFC (Fairness-Aware)

```bash
cd KFC

# Install
pip install -r requirements.txt

# Download datasets (Google Drive link in README)
# Place in same directory as train.py

# Train
python train.py --batch_size 25 \
                --sample ./data_files \
                --save_path ./log_files \
                --epochs 100 --beta 0.08 \
                --log_path log_files/experiment.txt \
                --gpu 0

# Find optimal threshold
python find.py --sample ./data_files \
               --save_path ./log_files \
               --batch_size 50 \
               --gpu 0

# Test
python test.py --sample ./sample0 \
               --save_path ./log_files \
               --threshold <from_find.py> \
               --batch_size 50 \
               --gpu 0
```

### 6.3 FaCoR (Cross-Attention)

```bash
cd FaCoR

# Setup environment
conda env create -f environment.yml
conda activate facor

# Download data (Google Drive link)
# Extract to ./

# Run training
sh run.sh
```

---

## 7. Inference

### 7.1 Single Pair Inference

```python
def predict_kinship(model, img1_path, img2_path, device, transform):
    """Predict kinship for a single pair."""
    model.eval()
    
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    
    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits, _, _ = model(img1, img2)
        prob = torch.sigmoid(logits).item()
    
    return {
        'probability': prob,
        'prediction': 'Kin' if prob > 0.5 else 'Non-kin',
        'confidence': max(prob, 1-prob)
    }
```

### 7.2 Batch Inference

```python
def batch_predict(model, pairs_csv, data_dir, device, batch_size=32):
    """Batch inference on pairs."""
    dataset = FIWDataset(data_dir, pairs_csv)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    results = []
    model.eval()
    
    with torch.no_grad():
        for img1, img2, _, _ in tqdm(loader):
            img1 = img1.to(device)
            img2 = img2.to(device)
            
            logits, _, _ = model(img1, img2)
            probs = torch.sigmoid(logits)
            
            results.extend(probs.cpu().numpy().tolist())
    
    return results
```

---

## 8. Common Issues and Solutions

### 8.1 Out of Memory

```python
# Solutions:
# 1. Reduce batch size
config['batch_size'] = 16

# 2. Use gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. Use mixed precision (fp16)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# 4. Freeze encoder layers
for param in model.encoder.parameters():
    param.requires_grad = False
```

### 8.2 Slow Training

```python
# 1. Use multiple workers
DataLoader(..., num_workers=8, pin_memory=True)

# 2. Enable cudnn benchmarking
torch.backends.cudnn.benchmark = True

# 3. Precompute embeddings (if encoder is frozen)
# Save embeddings to disk, train only head
```

### 8.3 Poor Convergence

```python
# 1. Learning rate warmup
from transformers import get_linear_schedule_with_warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=1000, num_training_steps=total_steps
)

# 2. Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. Label smoothing
criterion = nn.BCEWithLogitsLoss(label_smoothing=0.1)
```

---

## 9. Deployment Checklist

- [ ] Model exported to ONNX/TorchScript
- [ ] Inference pipeline tested
- [ ] Threshold tuned on validation set
- [ ] Fairness metrics computed
- [ ] Error cases documented
- [ ] API endpoints created (if needed)
- [ ] Logging and monitoring setup
- [ ] Privacy/consent mechanisms in place

---

*This implementation guide provides a complete path from data preparation to deployment for kinship classification systems.*
