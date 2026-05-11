"""
Model 08 — ArcFace + Retrieval-Augmented Kinship Verification.

Combines:
- ArcFace IResNet-100 encoder (frozen, pretrained MS1MV2)
- Non-parametric gallery of positive training pair signatures
- Cross-attention over top-K retrieved supports (with relation tags)
- Binary head + auxiliary relation head

Retrieval architecture is identical to Model 06 R001. Only the encoder
differs: ArcFace face-discriminative features instead of frozen ViT/DINOv2.

Memory (AMD RX 6750 XT, 12 GB):
- Frozen backbone (no Adam state) → much lower VRAM than Model 06
- Trainable: sig_to_token + cross-attn + heads ≈ 6.3 M
- Input is 112×112 (vs 224 in M06) → faster forward
- Gallery: 33k positive train pairs × 1536-dim signatures ≈ 200 MB
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

# Local IResNet implementation (InsightFace MIT-licensed code, vendored).
sys.path.insert(0, str(Path(__file__).parent))
from iresnet import iresnet100, iresnet50  # noqa: E402


RELATION_SETS: Dict[str, Tuple[str, ...]] = {
    "fiw": ("bb", "ss", "sibs", "fd", "fs", "md", "ms", "gfgd", "gfgs", "gmgd", "gmgs"),
    "kinface": ("fd", "fs", "md", "ms"),
}


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class ArcFaceEncoder(nn.Module):
    """
    Frozen ArcFace IResNet encoder.

    Input is 112×112 RGB images pre-normalized by the ArcFace pipeline
    (i.e. (img / 255 - 0.5) / 0.5). Output is a 512-dim L2-normalised
    embedding.

    Supports two checkpoint formats:
    - State dict (InsightFace-style keys: conv1.weight, layer1.0.bn1.weight,
      etc.) — loaded into the local iresnet50/100 class.
    - Full pickled torch.nn.Module — e.g. an onnx2torch.GraphModule produced
      by converting InsightFace's official .onnx releases (glintr100, etc.).
      Used directly as the backbone.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        arch: str = "r100",
        freeze: bool = True,
    ):
        super().__init__()
        self._is_pickled_module = False

        if weights_path is not None and os.path.exists(weights_path):
            loaded = torch.load(weights_path, map_location="cpu", weights_only=False)

            if isinstance(loaded, nn.Module):
                # Format B: full pickled module (e.g. onnx2torch GraphModule).
                self.backbone = loaded
                self._is_pickled_module = True
                print(f"  [ArcFace] loaded pickled module from {weights_path}")
                print(f"  [ArcFace]   type={type(self.backbone).__name__}, "
                      f"params={sum(p.numel() for p in self.backbone.parameters()):,}")
            else:
                # Format A: state dict for our iresnet50/100 class.
                if arch == "r100":
                    self.backbone = iresnet100()
                elif arch == "r50":
                    self.backbone = iresnet50()
                else:
                    raise ValueError(f"unknown arch '{arch}'")

                state = loaded
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                cleaned = {(k[len("module."):] if k.startswith("module.") else k): v
                           for k, v in state.items()}
                missing, unexpected = self.backbone.load_state_dict(cleaned, strict=False)
                if missing:
                    print(f"  [ArcFace] missing keys ({len(missing)}): {missing[:5]}...")
                if unexpected:
                    print(f"  [ArcFace] unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
                print(f"  [ArcFace] loaded state_dict from {weights_path} (arch={arch})")
        elif weights_path is not None:
            raise FileNotFoundError(
                f"ArcFace weights not found at {weights_path}. "
                f"Download from InsightFace model zoo (see README)."
            )
        else:
            if arch == "r100":
                self.backbone = iresnet100()
            elif arch == "r50":
                self.backbone = iresnet50()
            else:
                raise ValueError(f"unknown arch '{arch}'")
            print("  [ArcFace] no weights_path given — backbone is RANDOMLY INITIALIZED. "
                  "This is only valid for smoke tests.")

        self.feature_dim = 512
        self.embed_dim = 512
        self.frozen = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 112, 112) → (B, 512) L2-normalised."""
        if self.frozen:
            with torch.no_grad():
                feat = self.backbone(x)
        else:
            feat = self.backbone(x)
        return F.normalize(feat, dim=-1)


# ---------------------------------------------------------------------------
# Pair signature (identical to Model 06)
# ---------------------------------------------------------------------------
def pair_signature(emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(emb_a, dim=-1)
    b = F.normalize(emb_b, dim=-1)
    diff = (a - b).abs()
    prod = a * b
    sig = torch.cat([a + b, diff, prod], dim=-1)
    return F.normalize(sig, dim=-1)


# ---------------------------------------------------------------------------
# Gallery (verbatim from Model 06)
# ---------------------------------------------------------------------------
class Gallery(nn.Module):
    def __init__(self, signature_dim: int, embedding_dim: int,
                 max_capacity: int = 200_000, store_on_cpu: bool = False):
        super().__init__()
        self.signature_dim = signature_dim
        self.embedding_dim = embedding_dim
        self.max_capacity = max_capacity
        self.store_on_cpu = store_on_cpu
        self.register_buffer("signatures", torch.empty(0, signature_dim), persistent=False)
        self.register_buffer("emb_a", torch.empty(0, embedding_dim), persistent=False)
        self.register_buffer("emb_b", torch.empty(0, embedding_dim), persistent=False)
        self.register_buffer("relation_idx", torch.empty(0, dtype=torch.long), persistent=False)

    @torch.no_grad()
    def populate(self, signatures, emb_a, emb_b, relation_idx) -> None:
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
    def retrieve(self, query_signature, k, exclude_mask=None, chunk_size=4096):
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
            scores = query_signature @ gallery_sigs[start:end].transpose(0, 1)
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
# Retrieval cross-attention (verbatim from Model 06)
# ---------------------------------------------------------------------------
class RetrievalCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv):
        out, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv),
                            need_weights=False)
        q = q + out
        q = q + self.ffn(q)
        return q


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ArcFaceRetrievalKinship(nn.Module):
    """
    Full Model 08 architecture.
    """

    def __init__(
        self,
        arcface_weights: Optional[str] = None,
        arcface_arch: str = "r100",
        embedding_dim: int = 512,
        retrieval_k: int = 32,
        retrieval_attn_layers: int = 2,
        retrieval_attn_heads: int = 4,
        dropout: float = 0.1,
        relation_set: str = "fiw",
        relation_loss_weight: float = 0.15,
        max_gallery: int = 200_000,
        store_gallery_on_cpu: bool = False,
    ):
        super().__init__()
        self.encoder = ArcFaceEncoder(
            weights_path=arcface_weights,
            arch=arcface_arch,
            freeze=True,
        )

        # ArcFace already outputs 512-dim. We pass through as-is (no projection
        # needed since the embedding_dim matches).
        if self.encoder.feature_dim != embedding_dim:
            self.project = nn.Sequential(
                nn.Linear(self.encoder.feature_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, embedding_dim),
            )
        else:
            self.project = nn.Identity()

        self.embedding_dim = embedding_dim
        self.signature_dim = 3 * embedding_dim
        self.retrieval_k = retrieval_k

        self.sig_to_token = nn.Sequential(
            nn.Linear(self.signature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        if relation_set not in RELATION_SETS:
            raise ValueError(f"unknown relation_set '{relation_set}'")
        self.relation_classes = RELATION_SETS[relation_set]
        self.relation_set = relation_set
        self.relation_loss_weight = relation_loss_weight

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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        return F.normalize(self.project(feat), dim=-1)

    def encode_pair(self, img_a, img_b):
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

    @torch.no_grad()
    def build_gallery(self, loader, device, positive_only: bool = True,
                      verbose: bool = True) -> int:
        self.eval()
        sigs, embs_a, embs_b, rels = [], [], [], []

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
            rel_idx = self.relations_to_indices(relations).to(device).clamp_min_(0)
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

    def forward(self, img_a, img_b, exclude_gallery_idx=None) -> Dict[str, torch.Tensor]:
        emb_a, emb_b, sig = self.encode_pair(img_a, img_b)
        B = sig.size(0)
        k = min(self.retrieval_k, max(len(self.gallery), 1))

        if len(self.gallery) == 0:
            q_token = self.sig_to_token(sig).unsqueeze(1)
            attended = q_token
        else:
            topk_idx, _ = self.gallery.retrieve(sig, k=k, exclude_mask=exclude_gallery_idx)
            device = sig.device
            support_sigs = self.gallery.signatures.to(device, non_blocking=True)[topk_idx]
            support_rels = self.gallery.relation_idx.to(device, non_blocking=True)[topk_idx]
            rel_tokens = self.relation_embed(support_rels)
            sup_tokens = self.sig_to_token(support_sigs) + rel_tokens
            q_token = self.sig_to_token(sig).unsqueeze(1)
            attended = q_token
            for layer in self.cross_layers:
                attended = layer(attended, sup_tokens)

        attended = attended.squeeze(1)
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


def create_model(config=None) -> ArcFaceRetrievalKinship:
    if config is None:
        return ArcFaceRetrievalKinship()
    return ArcFaceRetrievalKinship(
        arcface_weights=getattr(config, "arcface_weights", None),
        arcface_arch=getattr(config, "arcface_arch", "r100"),
        embedding_dim=getattr(config, "embedding_dim", 512),
        retrieval_k=getattr(config, "retrieval_k", 32),
        retrieval_attn_layers=getattr(config, "retrieval_attn_layers", 2),
        retrieval_attn_heads=getattr(config, "retrieval_attn_heads", 4),
        dropout=getattr(config, "dropout", 0.1),
        relation_set=getattr(config, "relation_set", "fiw"),
        relation_loss_weight=getattr(config, "relation_loss_weight", 0.15),
        max_gallery=getattr(config, "max_gallery", 200_000),
        store_gallery_on_cpu=getattr(config, "store_gallery_on_cpu", False),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = ArcFaceRetrievalKinship()
    x1 = torch.randn(2, 3, 112, 112)
    x2 = torch.randn(2, 3, 112, 112)
    out = m(x1, x2)
    print(f"logits:   {out['logits'].shape}")
    print(f"emb1:     {out['emb1'].shape}")
    print(f"attended: {out['attended'].shape}")
    total = sum(p.numel() for p in m.parameters())
    train = m.count_trainable()
    print(f"total={total:,}  trainable={train:,} ({100*train/total:.2f}%)")
