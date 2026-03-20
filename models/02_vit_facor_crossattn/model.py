"""
Vision Transformer + FaCoR Cross-Attention Model for Kinship Verification.

Key Innovation:
- Replace CNN backbone with Vision Transformer for global feature extraction
- Apply FaCoR-style cross-attention between ViT patch tokens
- Combine local patch interactions with global CLS token features

Architecture:
    Input Pair → ViT Backbone → Patch Tokens
                      ↓
              Cross-Attention Module
                      ↓
              Channel Attention
                      ↓
           Contrastive Embeddings
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import timm


class CrossAttentionModule(nn.Module):
    """
    FaCoR-style cross-attention between two sets of features.
    Allows each face to attend to relevant regions of the other face.
    """
    
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections for cross-attention
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # FFN after attention
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Cross-attention between two feature sets.
        
        Args:
            x1: Features from first image [B, N, D]
            x2: Features from second image [B, N, D]
        
        Returns:
            out1: Attended features for first image [B, N, D]
            out2: Attended features for second image [B, N, D]
            attn_map: Cross-attention weights [B, H, N, N]
        """
        B, N, D = x1.shape
        
        # x1 attends to x2
        q1 = self.q_proj(x1).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k2 = self.k_proj(x2).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v2 = self.v_proj(x2).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention: x1 queries, x2 keys/values
        attn = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out1 = (attn @ v2).transpose(1, 2).reshape(B, N, D)
        out1 = self.out_proj(out1)
        out1 = self.norm1(x1 + out1)
        out1 = self.norm2(out1 + self.ffn(out1))
        
        # x2 attends to x1 (symmetric)
        q2 = self.q_proj(x2).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k1 = self.k_proj(x1).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v1 = self.v_proj(x1).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn2 = F.softmax(attn2, dim=-1)
        attn2 = self.dropout(attn2)
        
        out2 = (attn2 @ v1).transpose(1, 2).reshape(B, N, D)
        out2 = self.out_proj(out2)
        out2 = self.norm1(x2 + out2)
        out2 = self.norm2(out2 + self.ffn(out2))
        
        # Return average attention map for visualization
        attn_map = (attn + attn2.transpose(-2, -1)) / 2
        
        return out1, out2, attn_map


class ChannelAttention(nn.Module):
    """
    Channel attention layer (squeeze-and-excitation style).
    Reweights feature channels based on global information.
    """
    
    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, D] or [B, N, D]
        """
        if x.dim() == 3:
            # Global average pooling over sequence
            avg = x.mean(dim=1)  # [B, D]
        else:
            avg = x
        
        weights = self.fc(avg)  # [B, D]
        
        if x.dim() == 3:
            weights = weights.unsqueeze(1)  # [B, 1, D]
        
        return x * weights


class ViTFaCoRModel(nn.Module):
    """
    Vision Transformer with FaCoR-style Cross-Attention for Kinship Verification.
    
    Key features:
    - ViT backbone for global feature extraction
    - Cross-attention between patch tokens of two faces
    - Channel attention for feature refinement
    - Contrastive learning compatible output
    """
    
    def __init__(
        self,
        vit_model: str = "vit_base_patch16_224",
        pretrained: bool = True,
        embedding_dim: int = 512,
        num_cross_attn_layers: int = 2,
        cross_attn_heads: int = 8,
        channel_reduction: int = 16,
        dropout: float = 0.1,
        freeze_vit: bool = False,
    ):
        super().__init__()
        
        # ViT backbone
        self.vit = timm.create_model(
            vit_model,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
        )
        vit_dim = self.vit.embed_dim  # Usually 768 for base
        
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionModule(
                dim=vit_dim,
                num_heads=cross_attn_heads,
                dropout=dropout,
            )
            for _ in range(num_cross_attn_layers)
        ])
        
        # Channel attention
        self.channel_attn = ChannelAttention(vit_dim, reduction=channel_reduction)
        
        # Projection to embedding space
        self.projection = nn.Sequential(
            nn.Linear(vit_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )
        
        self.vit_dim = vit_dim
        self.embedding_dim = embedding_dim
    
    def extract_patch_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract ViT patch tokens and CLS token.
        
        Args:
            x: Input image [B, 3, H, W]
        
        Returns:
            patch_tokens: Patch embeddings [B, N-1, D]
            cls_token: CLS token embedding [B, D]
        """
        # Get all tokens from ViT
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        
        # Pass through transformer blocks
        for block in self.vit.blocks:
            x = block(x)
        
        x = self.vit.norm(x)
        
        # Split CLS and patch tokens
        cls_token = x[:, 0]  # [B, D]
        patch_tokens = x[:, 1:]  # [B, N-1, D]
        
        return patch_tokens, cls_token
    
    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with cross-attention between two faces.
        
        Args:
            img1: First face image [B, 3, H, W]
            img2: Second face image [B, 3, H, W]
        
        Returns:
            emb1: Embedding for first face [B, D]
            emb2: Embedding for second face [B, D]
            attn_map: Cross-attention map [B, H, N, N]
        """
        # Extract patch tokens
        patches1, cls1 = self.extract_patch_tokens(img1)
        patches2, cls2 = self.extract_patch_tokens(img2)
        
        # Apply cross-attention layers
        attn_maps = []
        for cross_attn in self.cross_attn_layers:
            patches1, patches2, attn = cross_attn(patches1, patches2)
            attn_maps.append(attn)
        
        # Average attention maps
        attn_map = torch.stack(attn_maps, dim=0).mean(dim=0)
        
        # Global average pooling over patches
        feat1 = patches1.mean(dim=1)  # [B, D]
        feat2 = patches2.mean(dim=1)  # [B, D]
        
        # Channel attention
        feat1 = self.channel_attn(feat1)
        feat2 = self.channel_attn(feat2)
        
        # Combine with CLS token
        feat1 = feat1 + cls1
        feat2 = feat2 + cls2
        
        # Project to embedding space
        emb1 = self.projection(feat1)
        emb2 = self.projection(feat2)
        
        # L2 normalize
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)
        
        return emb1, emb2, attn_map
    
    def get_similarity(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> torch.Tensor:
        """Get cosine similarity between two faces."""
        emb1, emb2, _ = self.forward(img1, img2)
        return F.cosine_similarity(emb1, emb2, dim=1)


class ViTFaCoRClassifier(nn.Module):
    """
    ViT-FaCoR with classification head for direct kinship prediction.
    Wraps the base model with a binary classifier.
    """
    
    def __init__(self, base_model: ViTFaCoRModel):
        super().__init__()
        self.base_model = base_model
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(base_model.embedding_dim * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
    
    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with classification output.
        
        Returns:
            logits: Kinship prediction logits [B, 1]
            emb1, emb2: Embeddings [B, D]
            attn_map: Attention map
        """
        emb1, emb2, attn_map = self.base_model(img1, img2)
        
        # Combine embeddings for classification
        diff = emb1 - emb2
        product = emb1 * emb2
        combined = torch.cat([emb1, emb2, diff, product], dim=1)
        
        logits = self.classifier(combined)
        
        return logits, emb1, emb2, attn_map


def build_vit_facor_model(
    *,
    vit_model: str = "vit_base_patch16_224",
    pretrained: bool = True,
    embedding_dim: int = 512,
    num_cross_attn_layers: int = 2,
    cross_attn_heads: int = 8,
    channel_reduction: int = 16,
    dropout: float = 0.1,
    freeze_vit: bool = False,
    use_classifier_head: bool = False,
):
    """Create either the embedding model or the BCE classifier wrapper."""
    base_model = ViTFaCoRModel(
        vit_model=vit_model,
        pretrained=pretrained,
        embedding_dim=embedding_dim,
        num_cross_attn_layers=num_cross_attn_layers,
        cross_attn_heads=cross_attn_heads,
        channel_reduction=channel_reduction,
        dropout=dropout,
        freeze_vit=freeze_vit,
    )
    if use_classifier_head:
        return ViTFaCoRClassifier(base_model)
    return base_model


def parse_model_outputs(outputs):
    """Normalize model outputs into embeddings, attention, and scalar scores."""
    if not isinstance(outputs, tuple):
        raise ValueError("ViT-FaCoR outputs are expected to be a tuple.")

    if len(outputs) >= 4:
        logits, emb1, emb2, attn_map = outputs[:4]
        if logits.ndim > 1 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        scores = torch.sigmoid(logits)
        return {
            "logits": logits,
            "emb1": emb1,
            "emb2": emb2,
            "attn_map": attn_map,
            "scores": scores,
        }

    if len(outputs) >= 3:
        emb1, emb2, attn_map = outputs[:3]
        scores = (F.cosine_similarity(emb1, emb2, dim=1) + 1) / 2
        return {
            "emb1": emb1,
            "emb2": emb2,
            "attn_map": attn_map,
            "scores": scores,
        }

    if len(outputs) >= 2:
        emb1, emb2 = outputs[:2]
        scores = (F.cosine_similarity(emb1, emb2, dim=1) + 1) / 2
        return {
            "emb1": emb1,
            "emb2": emb2,
            "attn_map": None,
            "scores": scores,
        }

    raise ValueError("Unsupported ViT-FaCoR output format.")


def create_model(config=None) -> ViTFaCoRModel:
    """Factory function to create model from config."""
    if config is None:
        return ViTFaCoRModel()
    
    return build_vit_facor_model(
        vit_model=getattr(config, "vit_model", "vit_base_patch16_224"),
        pretrained=getattr(config, "pretrained", True),
        embedding_dim=getattr(config, "embedding_dim", 512),
        num_cross_attn_layers=getattr(config, "cross_attn_layers", 2),
        cross_attn_heads=getattr(config, "cross_attn_heads", 8),
        channel_reduction=getattr(config, "channel_reduction", 16),
        dropout=getattr(config, "dropout", 0.1),
        freeze_vit=getattr(config, "freeze_vit", False),
        use_classifier_head=getattr(config, "use_classifier_head", False),
    )


if __name__ == "__main__":
    # Test model
    model = ViTFaCoRModel()
    
    # Dummy input
    img1 = torch.randn(4, 3, 224, 224)
    img2 = torch.randn(4, 3, 224, 224)
    
    # Forward pass
    emb1, emb2, attn_map = model(img1, img2)
    
    print(f"Embedding 1 shape: {emb1.shape}")
    print(f"Embedding 2 shape: {emb2.shape}")
    print(f"Attention map shape: {attn_map.shape}")
    print(f"Cosine similarity: {F.cosine_similarity(emb1, emb2, dim=1)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test classifier wrapper
    classifier = ViTFaCoRClassifier(model)
    logits, _, _, _ = classifier(img1, img2)
    print(f"Logits shape: {logits.shape}")
