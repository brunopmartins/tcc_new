"""
Unified Kinship Model - Combines All Techniques.

This model integrates:
1. Age Synthesis + All-vs-All Comparison
2. Hybrid ConvNeXt + ViT Backbone
3. FaCoR-style Cross-Attention
4. Learnable Multi-Age Aggregation

Architecture:
    Input Pair → Age Synthesis → Multiple Age Variants
                      ↓
              Hybrid Backbone (ConvNeXt + ViT)
                      ↓
              Cross-Attention Module
                      ↓
              Multi-Age Aggregation
                      ↓
                 Kinship Score
"""
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Optional, List, Tuple, Dict
import timm


class AgeEncoder(nn.Module):
    """Age synthesis module (placeholder for pretrained model)."""
    
    def __init__(self, model_type: str = "sam"):
        super().__init__()
        self.model_type = model_type
        self._initialized = False
    
    def load_pretrained(self, checkpoint_path: Optional[str] = None):
        """Load pretrained age synthesis model."""
        self._initialized = True
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor, target_age: int) -> torch.Tensor:
        """Generate face at target age."""
        if not self._initialized:
            return x  # Identity mapping if not initialized
        return x


class HybridBackbone(nn.Module):
    """
    Combined ConvNeXt + ViT backbone for rich feature extraction.
    """
    
    def __init__(
        self,
        convnext_model: str = "convnext_base",
        vit_model: str = "vit_base_patch16_224",
        pretrained: bool = True,
        output_dim: int = 512,
        fusion_type: str = "concat",
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # ConvNeXt for local features
        self.convnext = timm.create_model(convnext_model, pretrained=pretrained, num_classes=0)
        self.convnext_dim = self.convnext.num_features

        # ViT for global features
        self.vit = timm.create_model(vit_model, pretrained=pretrained, num_classes=0)
        self.vit_dim = self.vit.embed_dim

        # Enable gradient checkpointing on backbones
        if use_gradient_checkpointing:
            if hasattr(self.convnext, 'set_grad_checkpointing'):
                self.convnext.set_grad_checkpointing(enable=True)
            if hasattr(self.vit, 'set_grad_checkpointing'):
                self.vit.set_grad_checkpointing(enable=True)
        
        # Fusion
        self.fusion_type = fusion_type
        if fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(self.convnext_dim + self.vit_dim, output_dim * 2),
                nn.LayerNorm(output_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim * 2, output_dim),
            )
        else:  # gated
            self.gate_conv = nn.Sequential(nn.Linear(self.convnext_dim, output_dim), nn.Sigmoid())
            self.gate_vit = nn.Sequential(nn.Linear(self.vit_dim, output_dim), nn.Sigmoid())
            self.proj_conv = nn.Linear(self.convnext_dim, output_dim)
            self.proj_vit = nn.Linear(self.vit_dim, output_dim)
        
        self.output_dim = output_dim
    
    def _run_convnext(self, x: torch.Tensor) -> torch.Tensor:
        return self.convnext(x)

    def _run_vit(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract fused features."""
        if self.use_gradient_checkpointing and self.training:
            conv_feat = grad_checkpoint(self._run_convnext, x, use_reentrant=False)
            vit_feat = grad_checkpoint(self._run_vit, x, use_reentrant=False)
        else:
            conv_feat = self.convnext(x)
            vit_feat = self.vit(x)
        
        if self.fusion_type == "concat":
            combined = torch.cat([conv_feat, vit_feat], dim=1)
            return self.fusion(combined)
        else:
            g_conv = self.gate_conv(conv_feat)
            g_vit = self.gate_vit(vit_feat)
            return g_conv * self.proj_conv(conv_feat) + g_vit * self.proj_vit(vit_feat)
    
    def get_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Get ViT patch tokens for cross-attention."""
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        
        for block in self.vit.blocks:
            x = block(x)
        
        x = self.vit.norm(x)
        return x[:, 1:]  # Exclude CLS token


class CrossAttentionModule(nn.Module):
    """Cross-attention between two faces."""
    
    def __init__(self, dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
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
        """Bidirectional cross-attention."""
        B, N, D = x1.shape
        
        # x1 attends to x2
        q1 = self.q_proj(x1).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k2 = self.k_proj(x2).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v2 = self.v_proj(x2).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn1 = F.softmax((q1 @ k2.transpose(-2, -1)) * self.scale, dim=-1)
        attn1 = self.dropout(attn1)
        out1 = (attn1 @ v2).transpose(1, 2).reshape(B, N, D)
        out1 = self.norm1(x1 + self.out_proj(out1))
        out1 = self.norm2(out1 + self.ffn(out1))
        
        # x2 attends to x1
        q2 = self.q_proj(x2).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k1 = self.k_proj(x1).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v1 = self.v_proj(x1).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn2 = F.softmax((q2 @ k1.transpose(-2, -1)) * self.scale, dim=-1)
        attn2 = self.dropout(attn2)
        out2 = (attn2 @ v1).transpose(1, 2).reshape(B, N, D)
        out2 = self.norm1(x2 + self.out_proj(out2))
        out2 = self.norm2(out2 + self.ffn(out2))
        
        attn_map = (attn1 + attn2.transpose(-2, -1)) / 2
        
        return out1, out2, attn_map


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention."""
    
    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            weights = self.fc(x.mean(dim=1)).unsqueeze(1)
        else:
            weights = self.fc(x)
        return x * weights


class MultiAgeAggregator(nn.Module):
    """Learnable aggregation of multi-age comparison scores."""
    
    def __init__(
        self,
        num_comparisons: int = 9,
        embedding_dim: int = 512,
        aggregation: str = "attention",
    ):
        super().__init__()
        self.aggregation = aggregation
        
        if aggregation == "attention":
            self.attention = nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        
        self.combine = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim),
        )
    
    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate embeddings from multiple age comparisons.
        
        Args:
            embeddings: List of [B, D] embeddings
        
        Returns:
            Aggregated embedding [B, D]
        """
        # Stack: [B, num_comparisons, D]
        stacked = torch.stack(embeddings, dim=1)
        
        if self.aggregation == "max":
            aggregated = stacked.max(dim=1)[0]
        elif self.aggregation == "mean":
            aggregated = stacked.mean(dim=1)
        else:  # attention
            weights = F.softmax(self.attention(stacked), dim=1)  # [B, num_comp, 1]
            aggregated = (stacked * weights).sum(dim=1)  # [B, D]
        
        return self.combine(aggregated)


class UnifiedKinshipModel(nn.Module):
    """
    Unified model combining all kinship verification techniques.
    
    Components:
    1. Age Synthesis: Generate multiple age variants
    2. Hybrid Backbone: ConvNeXt + ViT for local+global features
    3. Cross-Attention: FaCoR-style interaction between faces
    4. Multi-Age Aggregation: Learnable combination of age comparisons
    """
    
    def __init__(
        self,
        # Age synthesis
        use_age_synthesis: bool = True,
        target_ages: List[int] = [20, 40, 60],

        # Hybrid backbone
        convnext_model: str = "convnext_base",
        vit_model: str = "vit_base_patch16_224",
        fusion_type: str = "concat",

        # Cross-attention
        use_cross_attention: bool = True,
        num_cross_attn_layers: int = 2,
        cross_attn_heads: int = 8,

        # General
        embedding_dim: int = 512,
        dropout: float = 0.1,
        aggregation: str = "attention",
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        
        self.use_age_synthesis = use_age_synthesis
        self.use_cross_attention = use_cross_attention
        self.target_ages = target_ages
        self.num_ages = len(target_ages)
        
        # Age synthesis (frozen)
        if use_age_synthesis:
            self.age_encoder = AgeEncoder()
            for param in self.age_encoder.parameters():
                param.requires_grad = False
        
        # Hybrid backbone
        self.backbone = HybridBackbone(
            convnext_model=convnext_model,
            vit_model=vit_model,
            output_dim=embedding_dim,
            fusion_type=fusion_type,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        
        # Cross-attention layers
        if use_cross_attention:
            self.cross_attn_layers = nn.ModuleList([
                CrossAttentionModule(
                    dim=embedding_dim,
                    num_heads=cross_attn_heads,
                    dropout=dropout,
                )
                for _ in range(num_cross_attn_layers)
            ])
        
        # Channel attention
        self.channel_attn = ChannelAttention(embedding_dim)
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )
        
        # Multi-age aggregator
        if use_age_synthesis:
            num_comparisons = self.num_ages * self.num_ages
            self.age_aggregator = MultiAgeAggregator(
                num_comparisons=num_comparisons,
                embedding_dim=embedding_dim,
                aggregation=aggregation,
            )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        
        self.embedding_dim = embedding_dim
    
    @torch.no_grad()
    def generate_age_variants(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Generate multiple age variants of input face."""
        if not self.use_age_synthesis:
            return [x]
        
        variants = []
        for age in self.target_ages:
            variant = self.age_encoder(x, age)
            variants.append(variant)
        return variants
    
    def process_pair(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Process a single pair through backbone and cross-attention."""
        # Extract features
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)
        
        attn_map = None
        
        if self.use_cross_attention:
            # Get patch tokens for cross-attention
            # For simplicity, treat features as single-token sequences
            feat1 = feat1.unsqueeze(1)  # [B, 1, D]
            feat2 = feat2.unsqueeze(1)
            
            for cross_attn in self.cross_attn_layers:
                feat1, feat2, attn_map = cross_attn(feat1, feat2)
            
            feat1 = feat1.squeeze(1)
            feat2 = feat2.squeeze(1)
        
        # Channel attention
        feat1 = self.channel_attn(feat1)
        feat2 = self.channel_attn(feat2)
        
        # Project and normalize
        emb1 = F.normalize(self.projection(feat1), dim=1)
        emb2 = F.normalize(self.projection(feat2), dim=1)
        
        return emb1, emb2, attn_map
    
    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass with all components.
        
        Returns:
            Dictionary containing:
            - logits: Final kinship prediction [B, 1]
            - emb1, emb2: Final embeddings [B, D]
            - attn_map: Cross-attention map (if enabled)
            - comparison_scores: All age pair scores (if age synthesis)
        """
        batch_size = img1.size(0)
        
        # Generate age variants
        variants1 = self.generate_age_variants(img1)
        variants2 = self.generate_age_variants(img2)
        
        # Process all age-matched pairs
        pair_embeddings1 = []
        pair_embeddings2 = []
        all_attn_maps = []
        
        for v1 in variants1:
            for v2 in variants2:
                emb1, emb2, attn = self.process_pair(v1, v2)
                pair_embeddings1.append(emb1)
                pair_embeddings2.append(emb2)
                if attn is not None:
                    all_attn_maps.append(attn)
        
        # Aggregate across age variants
        if self.use_age_synthesis and len(pair_embeddings1) > 1:
            final_emb1 = self.age_aggregator(pair_embeddings1)
            final_emb2 = self.age_aggregator(pair_embeddings2)
        else:
            final_emb1 = pair_embeddings1[0]
            final_emb2 = pair_embeddings2[0]
        
        # Final normalize
        final_emb1 = F.normalize(final_emb1, dim=1)
        final_emb2 = F.normalize(final_emb2, dim=1)
        
        # Classification
        diff = final_emb1 - final_emb2
        product = final_emb1 * final_emb2
        combined = torch.cat([final_emb1, final_emb2, diff, product], dim=1)
        logits = self.classifier(combined)
        
        # Build output dictionary
        output = {
            "logits": logits,
            "emb1": final_emb1,
            "emb2": final_emb2,
        }
        
        if all_attn_maps:
            output["attn_map"] = torch.stack(all_attn_maps, dim=0).mean(dim=0)
        
        # Compute comparison scores for analysis
        if len(pair_embeddings1) > 1:
            scores = [
                F.cosine_similarity(e1, e2, dim=1)
                for e1, e2 in zip(pair_embeddings1, pair_embeddings2)
            ]
            output["comparison_scores"] = torch.stack(scores, dim=1)
        
        return output
    
    def get_embeddings(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get normalized embeddings for contrastive loss."""
        output = self.forward(img1, img2)
        return output["emb1"], output["emb2"]


def create_model(config=None) -> UnifiedKinshipModel:
    """Factory function to create model."""
    if config is None:
        return UnifiedKinshipModel(use_age_synthesis=False)

    return UnifiedKinshipModel(
        use_age_synthesis=getattr(config, "use_age_synthesis", False),
        target_ages=getattr(config, "target_ages", [20, 40, 60]),
        convnext_model=getattr(config, "convnext_model", "convnext_base"),
        vit_model=getattr(config, "vit_model", "vit_base_patch16_224"),
        fusion_type=getattr(config, "fusion_type", "concat"),
        use_cross_attention=getattr(config, "use_cross_attention", True),
        num_cross_attn_layers=getattr(config, "cross_attn_layers", 2),
        cross_attn_heads=getattr(config, "cross_attn_heads", 8),
        embedding_dim=getattr(config, "embedding_dim", 512),
        aggregation=getattr(config, "age_aggregation", "attention"),
        use_gradient_checkpointing=getattr(config, "use_gradient_checkpointing", True),
    )


if __name__ == "__main__":
    # Test model
    model = UnifiedKinshipModel(use_age_synthesis=False)
    
    img1 = torch.randn(4, 3, 224, 224)
    img2 = torch.randn(4, 3, 224, 224)
    
    output = model(img1, img2)
    
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Embedding shape: {output['emb1'].shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with age synthesis
    model_full = UnifiedKinshipModel(use_age_synthesis=True)
    output_full = model_full(img1, img2)
    print(f"\nWith age synthesis:")
    print(f"Logits shape: {output_full['logits'].shape}")
    if "comparison_scores" in output_full:
        print(f"Comparison scores shape: {output_full['comparison_scores'].shape}")
