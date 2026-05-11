"""
Model 09 — DINOv2-Face + Retrieval-Augmented Kinship Verification.

Hypothesis: Model 08 confirmed that **ArcFace frozen is anti-kinship** — its
identity-discrimination objective actively pushes family members apart in
embedding space (M08 Test AUC 0.693, the worst in the project, below M06's
0.776 with a generic ImageNet ViT). The bottleneck is *not* face
specialization per se — it is the **identity-separation pretext**.

Model 09 tests the complementary hypothesis: a face encoder pretrained with a
**self-supervised** objective (DINO/contrastive on faces) should preserve
face specialization **without** the anti-kinship pressure. Such weights are
commonly called "DINOv2-Face" in the literature (e.g. FRoundation, Liu et al.
2024). DINOv2 was pretrained on LVD-142M (no identity labels), so it has
neither been trained "for" nor "against" kinship — it just sees faces.

Encoder choice (priority order, set via `dinov2_weights`):
  (1) DINOv2-Face weights (state_dict matching timm's vit_base_patch14_dinov2
      key names) loaded on top of the timm base. None of the public HF repos
      surveyed (2026-05) host such weights directly under that name; if you
      have a private checkpoint, point `dinov2_weights` at it.
  (2) Fallback: base `vit_base_patch14_dinov2.lvd142m` from timm — same
      backbone Model 05 (R002) and M06 explored. Useful as a clean
      retrieval-architecture comparison with M06 R002, isolating the
      retrieval-vs-LoRA design from the backbone choice.

Combines:
- DINOv2 ViT-B/14 encoder (frozen, optionally face-finetuned)
- 512-dim projection of CLS token (since DINOv2 is 768-d, M06/M08 use 512)
- Non-parametric gallery of positive training pair signatures
- Cross-attention over top-K retrieved supports (with relation tags)
- Binary head + auxiliary relation head

Retrieval architecture is identical to Model 06 R001 / Model 08. Only the
encoder differs.

Memory (AMD RX 6750 XT, 12 GB):
- Frozen ViT-B/14 backbone (~86 M params, no Adam state) → low VRAM.
- Trainable: projection + sig_to_token + cross-attn + heads ≈ 7.2 M.
- Input is 224×224 (vs 112 in M08) → slightly slower forward, but
  DINOv2 was pretrained at this resolution.
- Gallery: 33k positive train pairs × 1536-d signatures ≈ 200 MB.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as exc:
    raise ImportError(
        "timm is required for Model 09 encoder. "
        "Install via `pip install 'timm>=0.9.0'`."
    ) from exc


RELATION_SETS: Dict[str, Tuple[str, ...]] = {
    "fiw": ("bb", "ss", "sibs", "fd", "fs", "md", "ms", "gfgd", "gfgs", "gmgd", "gmgs"),
    "kinface": ("fd", "fs", "md", "ms"),
}


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class DINOv2FaceEncoder(nn.Module):
    """
    Frozen DINOv2 ViT-B/14 encoder, optionally overlaid with a face-pretrained
    state_dict ("DINOv2-Face").

    Input is 224×224 RGB pre-normalized with ImageNet mean/std (the same
    normalization DINOv2 was pretrained with). Output is a 768-dim CLS token
    pulled from `forward_features()[:, 0]` (timm convention).

    The pooling strategy can be switched to mean-pool over patch tokens via
    `pool='mean'`. CLS pooling is the DINOv2 default and works best for
    classification-adjacent tasks.

    Two ways to load weights:
    - `model_name`: timm identifier. Default is `vit_base_patch14_dinov2.lvd142m`,
      which downloads the public DINOv2-base weights (no face fine-tune).
    - `weights_path`: optional path to a state_dict that **overlays** the timm
      DINOv2 base. Keys are expected to match timm's naming (e.g.
      `cls_token`, `pos_embed`, `blocks.0.norm1.weight`, ...). Missing /
      unexpected keys are printed but not fatal — for face-finetuned
      DINOv2-Face variants, only attention/FFN weights typically change while
      the patch embedding / positional embedding stay shared. Set
      `strict_load=True` to fail on mismatch instead.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch14_dinov2.lvd142m",
        weights_path: Optional[str] = None,
        img_size: int = 224,
        pool: str = "cls",
        pretrained: bool = True,
        freeze: bool = True,
        strict_load: bool = False,
    ):
        super().__init__()
        if pool not in ("cls", "mean"):
            raise ValueError(f"pool must be 'cls' or 'mean', got '{pool}'")
        self.pool = pool
        self.model_name = model_name

        # DINOv2 uses patch size 14, which only divides 224, 252, 280, ...
        # We keep img_size = 224 by default.
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
        )

        self.feature_dim = self.vit.num_features  # 768 for vit_base
        self.num_prefix_tokens = getattr(self.vit, "num_prefix_tokens", 1)

        # Optional overlay of DINOv2-Face weights on top of the base ViT.
        if weights_path is not None:
            if not os.path.exists(weights_path):
                raise FileNotFoundError(
                    f"DINOv2-Face weights not found at {weights_path}. "
                    f"See weights/README.md for download options. To run with "
                    f"the public DINOv2 base (no face fine-tune), pass "
                    f"--dinov2_weights '' (empty string) on the CLI."
                )
            loaded = torch.load(weights_path, map_location="cpu", weights_only=False)
            state = loaded
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            elif isinstance(state, dict) and "model" in state:
                state = state["model"]
            cleaned: Dict[str, torch.Tensor] = {}
            for k, v in state.items():
                if not isinstance(v, torch.Tensor):
                    continue
                key = k
                for prefix in ("module.", "backbone.", "vit.", "encoder.", "teacher.", "student."):
                    if key.startswith(prefix):
                        key = key[len(prefix):]
                cleaned[key] = v
            missing, unexpected = self.vit.load_state_dict(cleaned, strict=False)
            if strict_load and (missing or unexpected):
                raise RuntimeError(
                    f"DINOv2-Face strict load failed: "
                    f"{len(missing)} missing, {len(unexpected)} unexpected"
                )
            if missing:
                print(f"  [DINOv2-Face] missing keys ({len(missing)}): {missing[:5]}...")
            if unexpected:
                print(f"  [DINOv2-Face] unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
            print(f"  [DINOv2-Face] overlaid weights from {weights_path}")
        else:
            print(f"  [DINOv2-Face] using base timm weights ({model_name}); "
                  f"no face fine-tune overlay applied.")

        self.frozen = freeze
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False
            self.vit.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) → (B, feature_dim)."""
        if self.frozen:
            with torch.no_grad():
                feats = self.vit.forward_features(x)
        else:
            feats = self.vit.forward_features(x)
        # timm ViT returns (B, N_prefix + N_patches, D).
        if self.pool == "cls":
            return feats[:, 0]
        # mean-pool patch tokens (exclude CLS / register tokens).
        return feats[:, self.num_prefix_tokens:].mean(dim=1)


# ---------------------------------------------------------------------------
# Pair signature (identical to Model 06 / Model 08)
# ---------------------------------------------------------------------------
def pair_signature(emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(emb_a, dim=-1)
    b = F.normalize(emb_b, dim=-1)
    diff = (a - b).abs()
    prod = a * b
    sig = torch.cat([a + b, diff, prod], dim=-1)
    return F.normalize(sig, dim=-1)


# ---------------------------------------------------------------------------
# Gallery (verbatim from Model 06 / Model 08)
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
# Retrieval cross-attention (verbatim from Model 06 / Model 08)
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
class DINOv2FaceRetrievalKinship(nn.Module):
    """
    Full Model 09 architecture.

    Embedding dim defaults to 512 (matches M06/M08). Since DINOv2 outputs
    768-d, a learned projection (768 → 512) is *always* applied unless
    `embedding_dim` is set to 768.
    """

    def __init__(
        self,
        dinov2_weights: Optional[str] = None,
        dinov2_model_name: str = "vit_base_patch14_dinov2.lvd142m",
        img_size: int = 224,
        pool: str = "cls",
        embedding_dim: int = 512,
        retrieval_k: int = 32,
        retrieval_attn_layers: int = 2,
        retrieval_attn_heads: int = 4,
        dropout: float = 0.1,
        relation_set: str = "fiw",
        relation_loss_weight: float = 0.15,
        max_gallery: int = 200_000,
        store_gallery_on_cpu: bool = False,
        strict_load: bool = False,
    ):
        super().__init__()
        self.encoder = DINOv2FaceEncoder(
            model_name=dinov2_model_name,
            weights_path=dinov2_weights or None,
            img_size=img_size,
            pool=pool,
            pretrained=True,
            freeze=True,
            strict_load=strict_load,
        )

        # Always project to embedding_dim (default 512). DINOv2 base is 768-d
        # so a project layer is mandatory unless embedding_dim == 768.
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


def create_model(config=None) -> DINOv2FaceRetrievalKinship:
    if config is None:
        return DINOv2FaceRetrievalKinship()
    return DINOv2FaceRetrievalKinship(
        dinov2_weights=getattr(config, "dinov2_weights", None),
        dinov2_model_name=getattr(config, "dinov2_model_name",
                                  "vit_base_patch14_dinov2.lvd142m"),
        img_size=getattr(config, "img_size", 224),
        pool=getattr(config, "pool", "cls"),
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
    # Smoke test: don't download timm weights here (avoid network calls in CI).
    # The model body must instantiate from scratch without pretrained weights.
    print("Model 09 smoke test (no pretrained weights download)")
    print("-" * 60)
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # Force pretrained=False for smoke by monkey-patching the encoder default.
    class _SmokeEnc(DINOv2FaceEncoder):
        def __init__(self, *a, **kw):
            kw["pretrained"] = False
            super().__init__(*a, **kw)
    DINOv2FaceRetrievalKinship.__init__.__globals__["DINOv2FaceEncoder"] = _SmokeEnc
    m = DINOv2FaceRetrievalKinship()
    x1 = torch.randn(2, 3, 224, 224)
    x2 = torch.randn(2, 3, 224, 224)
    out = m(x1, x2)
    print(f"logits:   {out['logits'].shape}")
    print(f"emb1:     {out['emb1'].shape}")
    print(f"signature:{out['signature'].shape}")
    print(f"attended: {out['attended'].shape}")
    print(f"rel_log:  {out['relation_logits'].shape}")
    total = sum(p.numel() for p in m.parameters())
    train = m.count_trainable()
    print(f"total={total:,}  trainable={train:,} ({100*train/total:.2f}%)")
    print(f"encoder feature_dim: {m.encoder.feature_dim}")
    print(f"embedding_dim:       {m.embedding_dim}")
    print(f"signature_dim:       {m.signature_dim}")
