"""
ConvNeXt + ViT Hybrid Model for Kinship Verification.

Key Innovation:
- Dual-backbone architecture combining CNN and Transformer strengths
- ConvNeXt extracts local/texture features
- ViT extracts global/structural features
- Learnable fusion of complementary representations

Architecture:
    Input Image → ┌─ ConvNeXt (Local) ─┐
                  │                     │ → Fusion → Embedding
                  └─ ViT (Global) ─────┘
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Optional, Tuple
import timm


class FeatureFusion(nn.Module):
    """
    Fusion module to combine ConvNeXt and ViT features.
    Supports multiple fusion strategies.
    """
    
    def __init__(
        self,
        convnext_dim: int = 1024,
        vit_dim: int = 768,
        output_dim: int = 512,
        fusion_type: str = "concat",  # "concat", "attention", "gated", "bilinear"
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.output_dim = output_dim

        if fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(convnext_dim + vit_dim, output_dim * 2),
                nn.LayerNorm(output_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim * 2, output_dim),
            )
        
        elif fusion_type == "attention":
            # Cross-attention fusion
            self.q_proj = nn.Linear(convnext_dim, output_dim)
            self.k_proj = nn.Linear(vit_dim, output_dim)
            self.v_proj = nn.Linear(vit_dim, output_dim)
            self.out_proj = nn.Linear(output_dim, output_dim)
            self.norm = nn.LayerNorm(output_dim)
            
        elif fusion_type == "gated":
            # Gated fusion with learned weights
            self.gate_conv = nn.Sequential(
                nn.Linear(convnext_dim, output_dim),
                nn.Sigmoid(),
            )
            self.gate_vit = nn.Sequential(
                nn.Linear(vit_dim, output_dim),
                nn.Sigmoid(),
            )
            self.proj_conv = nn.Linear(convnext_dim, output_dim)
            self.proj_vit = nn.Linear(vit_dim, output_dim)
            self.norm = nn.LayerNorm(output_dim)
            
        elif fusion_type == "bilinear":
            # Bilinear pooling (compact)
            self.proj_conv = nn.Linear(convnext_dim, output_dim)
            self.proj_vit = nn.Linear(vit_dim, output_dim)
            self.bilinear = nn.Bilinear(output_dim, output_dim, output_dim)
            self.norm = nn.LayerNorm(output_dim)
    
    def forward(
        self,
        conv_feat: torch.Tensor,
        vit_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse ConvNeXt and ViT features.
        
        Args:
            conv_feat: ConvNeXt features [B, D1]
            vit_feat: ViT features [B, D2]
        
        Returns:
            Fused features [B, output_dim]
        """
        if self.fusion_type == "concat":
            combined = torch.cat([conv_feat, vit_feat], dim=1)
            return self.fusion(combined)
        
        elif self.fusion_type == "attention":
            q = self.q_proj(conv_feat).unsqueeze(1)  # [B, 1, D]
            k = self.k_proj(vit_feat).unsqueeze(1)   # [B, 1, D]
            v = self.v_proj(vit_feat).unsqueeze(1)   # [B, 1, D]
            
            attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) / (self.output_dim ** 0.5), dim=-1)
            out = torch.bmm(attn, v).squeeze(1)
            out = self.out_proj(out)
            return self.norm(out + self.q_proj(conv_feat))
        
        elif self.fusion_type == "gated":
            g_conv = self.gate_conv(conv_feat)
            g_vit = self.gate_vit(vit_feat)
            
            p_conv = self.proj_conv(conv_feat)
            p_vit = self.proj_vit(vit_feat)
            
            fused = g_conv * p_conv + g_vit * p_vit
            return self.norm(fused)
        
        elif self.fusion_type == "bilinear":
            p_conv = self.proj_conv(conv_feat)
            p_vit = self.proj_vit(vit_feat)
            fused = self.bilinear(p_conv, p_vit)
            return self.norm(fused)


class ConvNeXtViTHybrid(nn.Module):
    """
    Hybrid model combining ConvNeXt and ViT for kinship verification.
    
    ConvNeXt: Excellent for local features, textures, fine-grained details
    ViT: Excellent for global structure, spatial relationships
    
    Together: Complementary features for robust kinship recognition
    """
    
    def __init__(
        self,
        convnext_model: str = "convnext_base",
        vit_model: str = "vit_base_patch16_224",
        pretrained: bool = True,
        embedding_dim: int = 512,
        fusion_type: str = "concat",
        freeze_backbones: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # ConvNeXt backbone (local features)
        self.convnext = timm.create_model(
            convnext_model,
            pretrained=pretrained,
            num_classes=0,
        )
        convnext_dim = self.convnext.num_features  # 1024 for base
        
        # ViT backbone (global features)
        self.vit = timm.create_model(
            vit_model,
            pretrained=pretrained,
            num_classes=0,
        )
        vit_dim = self.vit.embed_dim  # 768 for base
        
        if freeze_backbones:
            for param in self.convnext.parameters():
                param.requires_grad = False
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Feature fusion
        self.fusion = FeatureFusion(
            convnext_dim=convnext_dim,
            vit_dim=vit_dim,
            output_dim=embedding_dim,
            fusion_type=fusion_type,
            dropout=dropout,
        )
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )
        
        self.embedding_dim = embedding_dim
        self.convnext_dim = convnext_dim
        self.vit_dim = vit_dim
    
    def _run_convnext(self, x: torch.Tensor) -> torch.Tensor:
        return self.convnext(x)

    def _run_vit(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing on both backbones to save VRAM."""
        if hasattr(self.convnext, 'set_grad_checkpointing'):
            self.convnext.set_grad_checkpointing(enable=True)
        if hasattr(self.vit, 'set_grad_checkpointing'):
            self.vit.set_grad_checkpointing(enable=True)
        self._use_grad_checkpoint = True

    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from both backbones.

        Args:
            x: Input image [B, 3, H, W]

        Returns:
            conv_feat: ConvNeXt features [B, convnext_dim]
            vit_feat: ViT features [B, vit_dim]
        """
        use_ckpt = getattr(self, '_use_grad_checkpoint', False) and self.training
        if use_ckpt:
            conv_feat = grad_checkpoint(self._run_convnext, x, use_reentrant=False)
            vit_feat = grad_checkpoint(self._run_vit, x, use_reentrant=False)
        else:
            conv_feat = self.convnext(x)
            vit_feat = self.vit(x)

        return conv_feat, vit_feat
    
    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass with hybrid feature extraction.
        
        Args:
            img1: First face image [B, 3, H, W]
            img2: Second face image [B, 3, H, W]
        
        Returns:
            emb1: Embedding for first face [B, D]
            emb2: Embedding for second face [B, D]
            aux: Dictionary with auxiliary outputs (individual features)
        """
        # Extract features for both images
        conv1, vit1 = self.extract_features(img1)
        conv2, vit2 = self.extract_features(img2)
        
        # Fuse features
        fused1 = self.fusion(conv1, vit1)
        fused2 = self.fusion(conv2, vit2)
        
        # Project to embedding space
        emb1 = self.projection(fused1)
        emb2 = self.projection(fused2)
        
        # L2 normalize
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)
        
        # Auxiliary outputs for analysis
        aux = {
            "conv1": conv1, "conv2": conv2,
            "vit1": vit1, "vit2": vit2,
            "fused1": fused1, "fused2": fused2,
        }
        
        return emb1, emb2, aux
    
    def get_similarity(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> torch.Tensor:
        """Get cosine similarity between two faces."""
        emb1, emb2, _ = self.forward(img1, img2)
        return F.cosine_similarity(emb1, emb2, dim=1)


class ConvNeXtViTClassifier(nn.Module):
    """
    Hybrid model with classification head for direct kinship prediction.
    """
    
    def __init__(self, base_model: ConvNeXtViTHybrid):
        super().__init__()
        self.base_model = base_model
        
        # Classification head with multiple interaction types
        self.classifier = nn.Sequential(
            nn.Linear(base_model.embedding_dim * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    
    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass with classification.
        
        Returns:
            logits: Kinship prediction [B, 1]
            emb1, emb2: Embeddings
            aux: Auxiliary outputs
        """
        emb1, emb2, aux = self.base_model(img1, img2)
        
        # Multiple interaction features
        diff = emb1 - emb2
        product = emb1 * emb2
        combined = torch.cat([emb1, emb2, diff, product], dim=1)
        
        logits = self.classifier(combined)
        
        return logits, emb1, emb2, aux


class AblationModel(nn.Module):
    """
    Model for ablation studies - can use only ConvNeXt or only ViT.
    """
    
    def __init__(
        self,
        mode: str = "hybrid",  # "hybrid", "convnext_only", "vit_only"
        convnext_model: str = "convnext_base",
        vit_model: str = "vit_base_patch16_224",
        embedding_dim: int = 512,
    ):
        super().__init__()
        self.mode = mode
        
        if mode in ["hybrid", "convnext_only"]:
            self.convnext = timm.create_model(convnext_model, pretrained=True, num_classes=0)
            self.proj_conv = nn.Linear(self.convnext.num_features, embedding_dim)
        
        if mode in ["hybrid", "vit_only"]:
            self.vit = timm.create_model(vit_model, pretrained=True, num_classes=0)
            self.proj_vit = nn.Linear(self.vit.embed_dim, embedding_dim)
        
        if mode == "hybrid":
            self.fusion = nn.Linear(embedding_dim * 2, embedding_dim)
    
    def forward(self, img1, img2):
        if self.mode == "convnext_only":
            emb1 = F.normalize(self.proj_conv(self.convnext(img1)), dim=1)
            emb2 = F.normalize(self.proj_conv(self.convnext(img2)), dim=1)
        elif self.mode == "vit_only":
            emb1 = F.normalize(self.proj_vit(self.vit(img1)), dim=1)
            emb2 = F.normalize(self.proj_vit(self.vit(img2)), dim=1)
        else:
            conv1 = self.proj_conv(self.convnext(img1))
            vit1 = self.proj_vit(self.vit(img1))
            conv2 = self.proj_conv(self.convnext(img2))
            vit2 = self.proj_vit(self.vit(img2))
            
            emb1 = F.normalize(self.fusion(torch.cat([conv1, vit1], dim=1)), dim=1)
            emb2 = F.normalize(self.fusion(torch.cat([conv2, vit2], dim=1)), dim=1)
        
        return emb1, emb2, {}


def create_model(config=None) -> ConvNeXtViTHybrid:
    """Factory function to create model from config."""
    if config is None:
        return ConvNeXtViTHybrid()
    
    return ConvNeXtViTHybrid(
        convnext_model=getattr(config, "convnext_model", "convnext_base"),
        vit_model=getattr(config, "vit_model", "vit_base_patch16_224"),
        pretrained=getattr(config, "pretrained", True),
        embedding_dim=getattr(config, "embedding_dim", 512),
        fusion_type=getattr(config, "fusion_type", "concat"),
    )


if __name__ == "__main__":
    # Test model
    model = ConvNeXtViTHybrid()
    
    # Dummy input
    img1 = torch.randn(4, 3, 224, 224)
    img2 = torch.randn(4, 3, 224, 224)
    
    # Forward pass
    emb1, emb2, aux = model(img1, img2)
    
    print(f"Embedding shape: {emb1.shape}")
    print(f"ConvNeXt feat shape: {aux['conv1'].shape}")
    print(f"ViT feat shape: {aux['vit1'].shape}")
    print(f"Cosine similarity: {F.cosine_similarity(emb1, emb2, dim=1)}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test different fusion types
    for fusion in ["concat", "attention", "gated", "bilinear"]:
        m = ConvNeXtViTHybrid(fusion_type=fusion)
        e1, e2, _ = m(img1, img2)
        print(f"Fusion '{fusion}': output shape {e1.shape}")
