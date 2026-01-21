"""
Shared loss functions for kinship classification models.
Includes: BCE, Contrastive, Triplet, SupCon, Fair Contrastive, Relation-Guided.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ContrastiveLoss(nn.Module):
    """
    Standard contrastive loss for Siamese networks.
    
    L = (1-Y) * 0.5 * D^2 + Y * 0.5 * max(0, margin - D)^2
    
    where D is the distance between embeddings.
    """
    
    def __init__(self, margin: float = 1.0, distance: str = "euclidean"):
        super().__init__()
        self.margin = margin
        self.distance = distance
    
    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            emb1: First embeddings [B, D]
            emb2: Second embeddings [B, D]
            labels: Binary labels (1=similar, 0=dissimilar) [B]
        """
        if self.distance == "euclidean":
            distances = F.pairwise_distance(emb1, emb2)
        else:  # cosine
            distances = 1 - F.cosine_similarity(emb1, emb2)
        
        # Contrastive loss
        loss_positive = labels * torch.pow(distances, 2)
        loss_negative = (1 - labels) * torch.pow(
            torch.clamp(self.margin - distances, min=0.0), 2
        )
        
        loss = 0.5 * (loss_positive + loss_negative)
        return loss.mean()


class CosineContrastiveLoss(nn.Module):
    """
    InfoNCE-style contrastive loss using cosine similarity.
    Similar to what FaCoR uses.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss.
        
        Args:
            emb1: First embeddings [B, D]
            emb2: Second embeddings [B, D]
            labels: Optional labels (not used in standard InfoNCE)
        """
        batch_size = emb1.size(0)
        
        # Normalize embeddings
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)
        
        # Concatenate for symmetric loss
        x1x2 = torch.cat([emb1, emb2], dim=0)  # [2B, D]
        x2x1 = torch.cat([emb2, emb1], dim=0)  # [2B, D]
        
        # Compute similarity matrix
        sim_matrix = torch.mm(x1x2, x1x2.t()) / self.temperature  # [2B, 2B]
        
        # Positive pairs are (i, i+B) and (i+B, i)
        pos_sim = torch.sum(x1x2 * x2x1, dim=1) / self.temperature  # [2B]
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=emb1.device).bool()
        sim_matrix.masked_fill_(mask, float('-inf'))
        
        # Numerator: positive pair similarity
        numerator = torch.exp(pos_sim)
        
        # Denominator: sum of all similarities
        denominator = torch.exp(sim_matrix).sum(dim=1)
        
        # Loss
        loss = -torch.log(numerator / denominator)
        return loss.mean()


class TripletLoss(nn.Module):
    """
    Triplet loss with online hard mining.
    
    L = max(0, d(a, p) - d(a, n) + margin)
    """
    
    def __init__(self, margin: float = 0.3, mining: str = "hard"):
        super().__init__()
        self.margin = margin
        self.mining = mining
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: All embeddings [B, D]
            labels: Class/identity labels [B]
        """
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)
        
        # Get masks for positive and negative pairs
        labels = labels.unsqueeze(1)
        pos_mask = (labels == labels.t()).float()
        neg_mask = (labels != labels.t()).float()
        
        # Remove self-comparisons
        pos_mask.fill_diagonal_(0)
        
        if self.mining == "hard":
            # Hardest positive: max distance among positives
            pos_distances = distances * pos_mask
            pos_distances[pos_mask == 0] = float('-inf')
            hardest_pos = pos_distances.max(dim=1)[0]
            
            # Hardest negative: min distance among negatives
            neg_distances = distances * neg_mask
            neg_distances[neg_mask == 0] = float('inf')
            hardest_neg = neg_distances.min(dim=1)[0]
            
            # Triplet loss
            loss = F.relu(hardest_pos - hardest_neg + self.margin)
        else:
            # All triplets
            loss = F.relu(
                distances.unsqueeze(2) - distances.unsqueeze(1) + self.margin
            )
            # Mask valid triplets
            triplet_mask = pos_mask.unsqueeze(2) * neg_mask.unsqueeze(1)
            loss = (loss * triplet_mask).sum() / (triplet_mask.sum() + 1e-8)
            return loss
        
        return loss.mean()


class FairContrastiveLoss(nn.Module):
    """
    Fair contrastive loss from KFC (BMVC 2023).
    Incorporates demographic-specific margins to reduce bias.
    """
    
    def __init__(
        self,
        temperature: float = 0.08,
        demographic_margins: Optional[dict] = None,
    ):
        super().__init__()
        self.temperature = temperature
        
        # Default margins for different demographics (from KFC paper)
        self.demographic_margins = demographic_margins or {
            "AA": 0.0,  # African American
            "A": 0.0,   # Asian
            "C": 0.0,   # Caucasian
            "I": 0.0,   # Indian
        }
    
    def compute_bias_margin(
        self,
        demographics: torch.Tensor,
        bias_map: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample debiasing margin."""
        return torch.sum(bias_map, dim=1) / bias_map.size(1)
    
    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        labels: torch.Tensor,
        demographics: Optional[torch.Tensor] = None,
        bias_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            emb1: First embeddings [B, D]
            emb2: Second embeddings [B, D]
            labels: Kinship labels [B]
            demographics: Demographic group indices [B]
            bias_map: Learned bias margins [B, num_groups]
        """
        batch_size = emb1.size(0)
        
        # Normalize
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)
        
        # Concatenate
        x1x2 = torch.cat([emb1, emb2], dim=0)
        x2x1 = torch.cat([emb2, emb1], dim=0)
        
        # Similarity matrix
        sim_matrix = torch.mm(x1x2, x1x2.t()) / self.temperature
        
        # Positive similarities with debiasing
        pos_sim = torch.sum(x1x2 * x2x1, dim=1)
        
        if bias_map is not None:
            debias_margin = self.compute_bias_margin(demographics, bias_map)
            debias_margin = torch.cat([debias_margin, debias_margin], dim=0)
            pos_sim = (pos_sim - debias_margin) / self.temperature
        else:
            pos_sim = pos_sim / self.temperature
        
        # Mask self-similarity
        mask = torch.eye(2 * batch_size, device=emb1.device).bool()
        sim_matrix.masked_fill_(mask, float('-inf'))
        
        # Loss
        numerator = torch.exp(pos_sim)
        denominator = torch.exp(sim_matrix).sum(dim=1)
        
        loss = -torch.log(numerator / denominator)
        return loss.mean()


class RelationGuidedContrastiveLoss(nn.Module):
    """
    Relation-guided contrastive loss inspired by FaCoR.
    Uses attention maps to dynamically adjust temperature.
    """
    
    def __init__(self, base_temperature: float = 0.07, alpha: float = 0.5):
        super().__init__()
        self.base_temperature = base_temperature
        self.alpha = alpha
    
    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        attention_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            emb1: First embeddings [B, D]
            emb2: Second embeddings [B, D]
            attention_map: Attention weights from cross-attention [B, H, W] or [B]
        """
        batch_size = emb1.size(0)
        
        # Normalize
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)
        
        # Dynamic temperature based on attention
        if attention_map is not None:
            if attention_map.dim() > 2:
                attention_map = attention_map.mean(dim=[1, 2])  # Global average
            temperature = self.base_temperature * (1 + self.alpha * attention_map)
            temperature = temperature.unsqueeze(1)
        else:
            temperature = self.base_temperature
        
        # Concatenate
        x1x2 = torch.cat([emb1, emb2], dim=0)
        x2x1 = torch.cat([emb2, emb1], dim=0)
        
        # Similarity with dynamic temperature
        if isinstance(temperature, torch.Tensor):
            temp_cat = torch.cat([temperature, temperature], dim=0)
            sim_matrix = torch.mm(x1x2, x1x2.t()) / temp_cat
            pos_sim = torch.sum(x1x2 * x2x1, dim=1) / temp_cat.squeeze()
        else:
            sim_matrix = torch.mm(x1x2, x1x2.t()) / temperature
            pos_sim = torch.sum(x1x2 * x2x1, dim=1) / temperature
        
        # Mask self-similarity
        mask = torch.eye(2 * batch_size, device=emb1.device).bool()
        sim_matrix.masked_fill_(mask, float('-inf'))
        
        # Loss
        numerator = torch.exp(pos_sim)
        denominator = torch.exp(sim_matrix).sum(dim=1)
        
        loss = -torch.log(numerator / denominator)
        return loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss with multiple objectives."""
    
    def __init__(
        self,
        losses: dict,
        weights: Optional[dict] = None,
    ):
        """
        Args:
            losses: Dict of loss name -> loss module
            weights: Dict of loss name -> weight
        """
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = weights or {name: 1.0 for name in losses}
    
    def forward(self, **kwargs) -> Tuple[torch.Tensor, dict]:
        """Compute combined loss."""
        total_loss = 0
        loss_dict = {}
        
        for name, loss_fn in self.losses.items():
            loss = loss_fn(**kwargs)
            loss_dict[name] = loss.item()
            total_loss = total_loss + self.weights.get(name, 1.0) * loss
        
        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict


def get_loss(loss_type: str, **kwargs) -> nn.Module:
    """Factory function to get loss by name."""
    losses = {
        "bce": nn.BCEWithLogitsLoss(),
        "contrastive": ContrastiveLoss(**kwargs),
        "cosine_contrastive": CosineContrastiveLoss(**kwargs),
        "triplet": TripletLoss(**kwargs),
        "fair_contrastive": FairContrastiveLoss(**kwargs),
        "relation_guided": RelationGuidedContrastiveLoss(**kwargs),
    }
    return losses.get(loss_type, ContrastiveLoss(**kwargs))
