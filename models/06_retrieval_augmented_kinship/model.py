"""
Model 06 — Retrieval-Augmented Kinship Verification.

For each test pair (a, b) we retrieve the K most similar *positive* pairs
from the training gallery (measured by cosine similarity of pair
signatures), then a small transformer cross-attends the query pair to
those retrieved supports to produce the final kinship score. This adds a
non-parametric memory that is unusual in the kinship literature and gives
the model explicit examples to compare against instead of forcing
everything into the weights.

Memory notes (AMD RX 6750 XT, 12 GB VRAM, 32 GB RAM):
- The encoder is a pre-trained ViT backbone loaded via timm. By default
  it is FROZEN (transfer-learning mode) — the only trainable parts are
  the small projection head, the retrieval aggregator and the classifier.
- The gallery is built once at the start of training (or loaded from
  disk). FIW has ~99k positive train pairs. At embedding_dim = 512 and
  FP16 this is ~200 MB — fits in VRAM; otherwise it can live on CPU.
- Retrieval is done in chunks with torch.topk, never materialising the
  full (query × gallery) similarity matrix at once.

Architecture
------------
                ┌─────────────────────────────┐
                │  Encoder f_θ (ViT, frozen)  │
                └───────────────┬─────────────┘
                                │
     ┌──────────────────────────┴──────────────────────────┐
     ▼                                                     ▼
  (e_a, e_b) query pair                         (e_1..e_N) gallery pairs
     │                                                     │
     ▼                                                     │
  pair signature s_q = [e_a, e_b, |e_a-e_b|, e_a·e_b]      │
     │                                                     │
     └────► cosine retrieval (top-K)  ◄────────────────────┘
                       │
                       ▼
     K support tokens [s_i + relation_embed(r_i)]
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
      Cross-attention      Support → aggregation
         query ↔ supports        (attention pooled)
             │                   │
             └─────────┬─────────┘
                       ▼
             [s_q, attended] → MLP → kinship logit
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

try:
    import timm
except ImportError as exc:
    raise ImportError(
        "timm is required for Model 06 encoder. "
        "Install via `pip install 'timm>=0.9.0'`."
    ) from exc


RELATION_SETS: Dict[str, Tuple[str, ...]] = {
    "fiw": ("bb", "ss", "sibs", "fd", "fs", "md", "ms", "gfgd", "gfgs", "gmgd", "gmgs"),
    "kinface": ("fd", "fs", "md", "ms"),
}


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class FrozenEncoder(nn.Module):
    """
    Frozen timm ViT encoder producing a single global embedding per image.
    The default backbone is the ViT-B/16 used by Model 02, but DINOv2 or
    any other timm model may be swapped in via `backbone_name`.
    """

    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        img_size: int = 224,
        pretrained: bool = True,
        freeze: bool = True,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.vit = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
        )
        self.feature_dim = self.vit.num_features
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False
        if use_gradient_checkpointing and hasattr(self.vit, "set_grad_checkpointing"):
            self.vit.set_grad_checkpointing(enable=True)

        self.use_gradient_checkpointing = use_gradient_checkpointing and not freeze
        self.frozen = freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) → global embedding (B, D)."""
        if self.frozen:
            with torch.no_grad():
                return self.vit(x)
        if self.use_gradient_checkpointing and self.training:
            return grad_checkpoint(self.vit, x, use_reentrant=False)
        return self.vit(x)


# ---------------------------------------------------------------------------
# Pair signature
# ---------------------------------------------------------------------------
def pair_signature(emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
    """
    Signature used for retrieval and for the final classifier.
    Symmetric in (a, b) up to the absolute difference and elementwise
    product — i.e. swapping a and b yields the same signature.
    """
    a = F.normalize(emb_a, dim=-1)
    b = F.normalize(emb_b, dim=-1)
    diff = (a - b).abs()
    prod = a * b
    sig = torch.cat([a + b, diff, prod], dim=-1)  # (B, 3D)
    return F.normalize(sig, dim=-1)


# ---------------------------------------------------------------------------
# Gallery (non-parametric memory)
# ---------------------------------------------------------------------------
class Gallery(nn.Module):
    """
    Fixed memory bank of (pair_signature, pair_embeddings, relation) for
    positive training pairs. Registered as buffers so they move with the
    model but carry no gradients.
    """

    def __init__(
        self,
        signature_dim: int,
        embedding_dim: int,
        max_capacity: int = 200_000,
        store_on_cpu: bool = False,
    ):
        super().__init__()
        self.signature_dim = signature_dim
        self.embedding_dim = embedding_dim
        self.max_capacity = max_capacity
        self.store_on_cpu = store_on_cpu

        # buffer tensors get allocated lazily by `populate`
        self.register_buffer("signatures", torch.empty(0, signature_dim), persistent=False)
        self.register_buffer("emb_a", torch.empty(0, embedding_dim), persistent=False)
        self.register_buffer("emb_b", torch.empty(0, embedding_dim), persistent=False)
        self.register_buffer("relation_idx", torch.empty(0, dtype=torch.long), persistent=False)

    @torch.no_grad()
    def populate(
        self,
        signatures: torch.Tensor,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        relation_idx: torch.Tensor,
    ) -> None:
        n = signatures.size(0)
        if n > self.max_capacity:
            idx = torch.randperm(n)[: self.max_capacity]
            signatures = signatures[idx]
            emb_a = emb_a[idx]
            emb_b = emb_b[idx]
            relation_idx = relation_idx[idx]

        device = torch.device("cpu") if self.store_on_cpu else signatures.device

        self.signatures = signatures.detach().to(device)
        self.emb_a = emb_a.detach().to(device)
        self.emb_b = emb_b.detach().to(device)
        self.relation_idx = relation_idx.detach().to(device)

    def __len__(self) -> int:
        return self.signatures.size(0)

    @torch.no_grad()
    def retrieve(
        self,
        query_signature: torch.Tensor,
        k: int,
        exclude_mask: Optional[torch.Tensor] = None,
        chunk_size: int = 4096,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (topk_idx, topk_scores) for each query row.

        query_signature: (B, 3D), already L2-normalised.
        The similarity is computed chunk-wise against the gallery to keep
        peak memory low on 12 GB VRAM.
        """
        assert len(self) > 0, "Gallery is empty — call populate() first."
        device = query_signature.device
        gallery_sigs = self.signatures.to(device, non_blocking=True)

        n_gallery = gallery_sigs.size(0)
        best_scores = torch.full(
            (query_signature.size(0), k), float("-inf"),
            device=device, dtype=query_signature.dtype,
        )
        best_indices = torch.full(
            (query_signature.size(0), k), -1,
            device=device, dtype=torch.long,
        )

        for start in range(0, n_gallery, chunk_size):
            end = min(start + chunk_size, n_gallery)
            scores = query_signature @ gallery_sigs[start:end].transpose(0, 1)  # (B, chunk)
            if exclude_mask is not None:
                local_mask = exclude_mask[:, start:end]
                scores = scores.masked_fill(local_mask, float("-inf"))

            combined_scores = torch.cat([best_scores, scores], dim=1)
            local_indices = torch.arange(start, end, device=device).unsqueeze(0).expand(
                query_signature.size(0), -1,
            )
            combined_indices = torch.cat([best_indices, local_indices], dim=1)
            top_scores, top_pos = combined_scores.topk(k, dim=1)
            best_scores = top_scores
            best_indices = combined_indices.gather(1, top_pos)

        return best_indices, best_scores


# ---------------------------------------------------------------------------
# Retrieval-augmented cross-attention
# ---------------------------------------------------------------------------
class RetrievalCrossAttention(nn.Module):
    """
    Query = the current pair's hidden state (1 token).
    K, V  = K retrieved support tokens.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """q: (B, 1, D),   kv: (B, K, D)  →  (B, 1, D)."""
        out, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv), need_weights=False)
        q = q + out
        q = q + self.ffn(q)
        return q


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class RetrievalAugmentedKinship(nn.Module):
    """
    Full Model 06 architecture. See the module-level docstring for the
    high-level picture.
    """

    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        img_size: int = 224,
        freeze_backbone: bool = True,
        backbone_pretrained: bool = True,
        embedding_dim: int = 512,
        retrieval_k: int = 32,
        retrieval_attn_layers: int = 2,
        retrieval_attn_heads: int = 4,
        dropout: float = 0.1,
        relation_set: str = "fiw",
        relation_loss_weight: float = 0.15,
        max_gallery: int = 200_000,
        store_gallery_on_cpu: bool = False,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.encoder = FrozenEncoder(
            backbone_name=backbone_name,
            img_size=img_size,
            pretrained=backbone_pretrained,
            freeze=freeze_backbone,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        self.project = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.embedding_dim = embedding_dim
        self.signature_dim = 3 * embedding_dim
        self.retrieval_k = retrieval_k

        # Project signature → token dim for the cross-attn stack.
        self.sig_to_token = nn.Sequential(
            nn.Linear(self.signature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        if relation_set not in RELATION_SETS:
            raise ValueError(f"unknown relation_set '{relation_set}'")
        self.relation_classes = RELATION_SETS[relation_set]
        self.relation_set = relation_set
        self.relation_loss_weight = relation_loss_weight

        # "Unknown"/"non-kin" slot included so we can safely embed any sample.
        self.relation_embed = nn.Embedding(len(self.relation_classes) + 1, embedding_dim)

        self.cross_layers = nn.ModuleList([
            RetrievalCrossAttention(embedding_dim, num_heads=retrieval_attn_heads, dropout=dropout)
            for _ in range(retrieval_attn_layers)
        ])

        self.binary_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        # Auxiliary relation head (also reads the cross-attended token).
        self.relation_head = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, len(self.relation_classes)),
        )

        self.gallery = Gallery(
            signature_dim=self.signature_dim,
            embedding_dim=embedding_dim,
            max_capacity=max_gallery,
            store_on_cpu=store_gallery_on_cpu,
        )

    # -- building blocks ----------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        return F.normalize(self.project(feat), dim=-1)

    def encode_pair(self, img_a: torch.Tensor, img_b: torch.Tensor):
        emb_a = self.encode(img_a)
        emb_b = self.encode(img_b)
        sig = pair_signature(emb_a, emb_b)
        return emb_a, emb_b, sig

    def relation_to_index(self, relation_name: str) -> int:
        try:
            return self.relation_classes.index(relation_name)
        except ValueError:
            return -1

    def relations_to_indices(self, relations: List[str]) -> torch.Tensor:
        return torch.tensor(
            [self.relation_to_index(r) for r in relations],
            dtype=torch.long,
        )

    # -- gallery population --------------------------------------------------
    @torch.no_grad()
    def build_gallery(
        self,
        loader,
        device: torch.device,
        positive_only: bool = True,
        verbose: bool = True,
    ) -> int:
        """
        Run the frozen encoder over a dataloader of training pairs and
        populate the gallery with positive-pair signatures + embeddings.
        Returns the number of stored pairs.
        """
        self.eval()
        sigs: List[torch.Tensor] = []
        embs_a: List[torch.Tensor] = []
        embs_b: List[torch.Tensor] = []
        rels: List[torch.Tensor] = []

        if verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(loader, desc="Building retrieval gallery")
            except ImportError:
                iterator = loader
        else:
            iterator = loader

        for batch in iterator:
            img_a = batch["img1"].to(device, non_blocking=True)
            img_b = batch["img2"].to(device, non_blocking=True)
            labels = batch["label"].to(device)
            relations = batch.get("relation", ["unknown"] * labels.size(0))

            if positive_only:
                mask = labels > 0.5
                if not mask.any():
                    continue
                img_a = img_a[mask]
                img_b = img_b[mask]
                relations = [r for r, keep in zip(relations, mask.cpu().tolist()) if keep]

            emb_a, emb_b, sig = self.encode_pair(img_a, img_b)
            sigs.append(sig.detach())
            embs_a.append(emb_a.detach())
            embs_b.append(emb_b.detach())

            rel_idx = self.relations_to_indices(relations).to(device)
            # Use "unknown" slot (len(classes)) for any relation not in the set.
            rel_idx = rel_idx.clamp_min_(0)
            rels.append(rel_idx)

        if not sigs:
            return 0

        self.gallery.populate(
            torch.cat(sigs, dim=0),
            torch.cat(embs_a, dim=0),
            torch.cat(embs_b, dim=0),
            torch.cat(rels, dim=0),
        )
        return len(self.gallery)

    # -- forward -------------------------------------------------------------
    def forward(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        exclude_gallery_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        emb_a, emb_b, sig = self.encode_pair(img_a, img_b)
        B = sig.size(0)
        k = min(self.retrieval_k, max(len(self.gallery), 1))

        if len(self.gallery) == 0:
            # Warm-up path: no gallery yet → fall back to pair signature only.
            q_token = self.sig_to_token(sig).unsqueeze(1)
            attended = q_token
        else:
            topk_idx, _ = self.gallery.retrieve(sig, k=k, exclude_mask=exclude_gallery_idx)
            device = sig.device
            support_sigs = self.gallery.signatures.to(device, non_blocking=True)[topk_idx]  # (B, K, 3D)
            support_rels = self.gallery.relation_idx.to(device, non_blocking=True)[topk_idx]  # (B, K)

            rel_tokens = self.relation_embed(support_rels)                    # (B, K, D)
            sup_tokens = self.sig_to_token(support_sigs) + rel_tokens         # (B, K, D)

            q_token = self.sig_to_token(sig).unsqueeze(1)                     # (B, 1, D)
            attended = q_token
            for layer in self.cross_layers:
                attended = layer(attended, sup_tokens)

        attended = attended.squeeze(1)                                         # (B, D)

        combined = torch.cat([self.sig_to_token(sig), attended], dim=-1)
        logits = self.binary_head(combined)
        rel_logits = self.relation_head(attended)

        return {
            "logits": logits,
            "emb1": emb_a,
            "emb2": emb_b,
            "signature": sig,
            "attended": attended,
            "relation_logits": rel_logits,
        }

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def count_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_model(config=None) -> RetrievalAugmentedKinship:
    if config is None:
        return RetrievalAugmentedKinship()
    return RetrievalAugmentedKinship(
        backbone_name=getattr(config, "backbone_name", "vit_base_patch16_224"),
        img_size=getattr(config, "img_size", 224),
        freeze_backbone=getattr(config, "freeze_backbone", True),
        backbone_pretrained=getattr(config, "backbone_pretrained", True),
        embedding_dim=getattr(config, "embedding_dim", 512),
        retrieval_k=getattr(config, "retrieval_k", 32),
        retrieval_attn_layers=getattr(config, "retrieval_attn_layers", 2),
        retrieval_attn_heads=getattr(config, "retrieval_attn_heads", 4),
        dropout=getattr(config, "dropout", 0.1),
        relation_set=getattr(config, "relation_set", "fiw"),
        relation_loss_weight=getattr(config, "relation_loss_weight", 0.15),
        max_gallery=getattr(config, "max_gallery", 200_000),
        store_gallery_on_cpu=getattr(config, "store_gallery_on_cpu", False),
        use_gradient_checkpointing=getattr(config, "use_gradient_checkpointing", True),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = RetrievalAugmentedKinship(backbone_pretrained=False, freeze_backbone=True)
    x1 = torch.randn(2, 3, 224, 224)
    x2 = torch.randn(2, 3, 224, 224)
    out = model(x1, x2)
    print(f"logits:  {out['logits'].shape}")
    print(f"emb1:    {out['emb1'].shape}")
    print(f"attended:{out['attended'].shape}")
    total = sum(p.numel() for p in model.parameters())
    train = model.count_trainable()
    print(f"total={total:,}  trainable={train:,} ({100*train/total:.2f}%)")
