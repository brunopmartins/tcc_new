"""
Model 13 — LGKT-Net (Landmark Graph Kinship Transformer).

Implements the architecture contract in `ARCHITECTURE.md`. R001 recipe:

    aligned 224×224 face_A, face_B
        │
        ├─► resize to 112 → AdaFace IR-101 forward_spatial → (B, 512, 7, 7)
        │
        └─► ROIAlign with landmark-derived boxes → 8 component tokens per face
              global, left_eye, right_eye, nose, mouth, jaw, left_cheek, right_cheek

    component tokens × face_A, face_B  →  K=8 nodes per face

        │
        ▼

    Joint pair graph: V_A ∪ V_B, edges = intra-face anatomical
                                       ∪ homologous cross-face
                                       ∪ global ↔ all

        │
        ▼

    Graph Transformer (2 layers, edge-type-aware multi-head attention)

        │
        ▼

    Native symmetric pooling: per node-type, compute [mean, |diff|, prod]
                              of (A_i, B_i) — order-invariant by construction

        │
        ▼

    Attention pooling → fused pair representation → BCE classifier
                                                   → optional 11-way relation aux head
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


# ---------------------------------------------------------------------------
# Canonical landmark positions for aligned FIW faces (112×112 ArcFace template)
# ---------------------------------------------------------------------------
#
# tools/align_fiw_dataset.py aligns every FIW face so that the 5 MTCNN
# landmarks land at fixed pixel coordinates. At 224×224 these are 2× the
# standard ArcFace 112 template. M13 derives all anatomical ROIs from
# these canonical positions, so no per-image landmark detection is needed.

CANONICAL_LANDMARKS_112: Dict[str, Tuple[float, float]] = {
    # (x, y) in 112-pixel coordinates
    "left_eye":    (38.30, 51.70),
    "right_eye":   (73.53, 51.50),
    "nose":        (56.03, 71.74),
    "left_mouth":  (41.55, 92.37),
    "right_mouth": (70.73, 92.20),
}


# Inter-ocular distance at 112 (~35 pixels) sets the scale for box sizing.
_LE = CANONICAL_LANDMARKS_112["left_eye"]
_RE = CANONICAL_LANDMARKS_112["right_eye"]
_NOSE = CANONICAL_LANDMARKS_112["nose"]
_LM = CANONICAL_LANDMARKS_112["left_mouth"]
_RM = CANONICAL_LANDMARKS_112["right_mouth"]
_IOD = ((_RE[0] - _LE[0]) ** 2 + (_RE[1] - _LE[1]) ** 2) ** 0.5  # ≈ 35.2


def _box_around(center_x: float, center_y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """Return (x0, y0, x1, y1) clamped to the 112 canvas."""
    x0 = max(0.0, center_x - w / 2)
    y0 = max(0.0, center_y - h / 2)
    x1 = min(112.0, center_x + w / 2)
    y1 = min(112.0, center_y + h / 2)
    return x0, y0, x1, y1


# Component boxes in 112-pixel coordinates. Sized relative to inter-ocular
# distance so they're robust to scale variations (though aligned faces are
# already canonical scale).
COMPONENT_BOXES_112: List[Tuple[str, Tuple[float, float, float, float]]] = [
    ("global",      (0.0, 0.0, 112.0, 112.0)),
    ("left_eye",    _box_around(_LE[0], _LE[1], 0.55 * _IOD, 0.35 * _IOD)),
    ("right_eye",   _box_around(_RE[0], _RE[1], 0.55 * _IOD, 0.35 * _IOD)),
    ("nose",        _box_around(_NOSE[0], _NOSE[1], 0.70 * _IOD, 0.95 * _IOD)),
    ("mouth",       _box_around((_LM[0] + _RM[0]) / 2, (_LM[1] + _RM[1]) / 2, 1.15 * _IOD, 0.50 * _IOD)),
    ("jaw",         _box_around((_LM[0] + _RM[0]) / 2, min(108.0, (_LM[1] + _RM[1]) / 2 + 0.55 * _IOD), 1.50 * _IOD, 0.55 * _IOD)),
    ("left_cheek",  _box_around((_LE[0] + _LM[0]) / 2 - 0.10 * _IOD, (_LE[1] + _LM[1]) / 2, 0.55 * _IOD, 0.75 * _IOD)),
    ("right_cheek", _box_around((_RE[0] + _RM[0]) / 2 + 0.10 * _IOD, (_RE[1] + _RM[1]) / 2, 0.55 * _IOD, 0.75 * _IOD)),
]
NODE_NAMES: List[str] = [name for name, _ in COMPONENT_BOXES_112]
NUM_NODES = len(NODE_NAMES)


def _build_intra_edges() -> List[Tuple[int, int]]:
    """Anatomical adjacency edges (undirected, expanded to directed pairs)."""
    n2i = {name: i for i, name in enumerate(NODE_NAMES)}
    edges_undir = [
        ("global", "left_eye"), ("global", "right_eye"), ("global", "nose"),
        ("global", "mouth"), ("global", "jaw"),
        ("global", "left_cheek"), ("global", "right_cheek"),
        ("left_eye", "nose"), ("right_eye", "nose"),
        ("left_eye", "left_cheek"), ("right_eye", "right_cheek"),
        ("nose", "mouth"),
        ("left_cheek", "nose"), ("right_cheek", "nose"),
        ("left_cheek", "mouth"), ("right_cheek", "mouth"),
        ("mouth", "jaw"),
        ("left_cheek", "jaw"), ("right_cheek", "jaw"),
    ]
    edges: List[Tuple[int, int]] = []
    for a, b in edges_undir:
        ia, ib = n2i[a], n2i[b]
        edges.append((ia, ib))
        edges.append((ib, ia))
    # Self-loops so each node always attends to itself.
    for i in range(NUM_NODES):
        edges.append((i, i))
    return edges


# ---------------------------------------------------------------------------
# Landmark ROIAlign tokenizer
# ---------------------------------------------------------------------------

class LandmarkROITokenizer(nn.Module):
    """One AdaFace pass per face → spatial feature map → ROIAlign over
    landmark-derived component boxes → (B, K, embedding_dim) node tokens.

    Supports two feature-stage configurations (R002 introduces stage3):

    - ``feature_stage="stage4"`` (R001 default): forward_spatial → (B, 512, 7, 7).
      Coarse spatial grid (16-pixel cells in 112-pixel space) — small ROIs
      may sample only 1-2 cells.
    - ``feature_stage="stage3"``: manual forward through ``body[0:46]`` →
      (B, 256, 14, 14). 4× more spatial cells, but 256 channels instead of
      512 — a learned projection brings the per-node tokens back to
      ``embedding_dim``.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int = 512,
        boxes_112: List[Tuple[str, Tuple[float, float, float, float]]] = None,
        feature_stage: str = "stage4",
        roi_output_size: int = 3,
        pool: str = "avg",
    ):
        super().__init__()
        if feature_stage not in ("stage4", "stage3"):
            raise ValueError(f"Unknown feature_stage: {feature_stage}")
        self.feature_stage = feature_stage
        self.backbone = backbone
        boxes_112 = boxes_112 if boxes_112 is not None else COMPONENT_BOXES_112
        self.box_names = [name for name, _ in boxes_112]
        self.num_nodes = len(boxes_112)
        self.embedding_dim = embedding_dim
        self.roi_output_size = roi_output_size

        boxes_tensor = torch.tensor(
            [list(b) for _, b in boxes_112], dtype=torch.float32
        )  # (K, 4) — (x0, y0, x1, y1)
        self.register_buffer("boxes_112", boxes_tensor, persistent=False)

        # IR-101 stage outputs at 112-pixel input:
        #   stage4 → (B, 512, 7, 7),  scale = 7/112
        #   stage3 → (B, 256, 14, 14), scale = 14/112
        if feature_stage == "stage4":
            self.feature_map_size = 7
            self._feature_channels = 512
        else:  # stage3
            self.feature_map_size = 14
            self._feature_channels = 256

        self.spatial_scale = self.feature_map_size / 112.0

        if pool == "avg":
            self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        elif pool == "max":
            self.spatial_pool = nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError(f"Unknown pool: {pool}")

        # Stage 3 outputs 256-channel features; project back to embedding_dim
        # so the rest of the network (graph transformer, pooler, classifier)
        # is unchanged. For stage 4 this is an identity.
        if self._feature_channels != embedding_dim:
            self.channel_projection = nn.Linear(self._feature_channels, embedding_dim)
        else:
            self.channel_projection = nn.Identity()

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone up to the selected stage and return its spatial map."""
        if self.feature_stage == "stage4":
            feat = self.backbone.forward_spatial(x)
            if isinstance(feat, tuple):
                feat = feat[0]
            return feat
        # stage3: input_layer + body[0:46], skipping body[46:49] (stage 4)
        h = self.backbone.input_layer(x)
        for i in range(46):
            h = self.backbone.body[i](h)
        return h

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, 3, 224, 224) → (B, K, embedding_dim)."""
        B = img.shape[0]

        # AdaFace was trained on 112×112; resize once.
        if img.shape[-1] != 112 or img.shape[-2] != 112:
            x = F.interpolate(img, size=(112, 112), mode="bilinear", align_corners=False)
        else:
            x = img

        feat = self._extract_features(x)  # (B, C, H, H), H ∈ {7, 14}

        K = self.num_nodes
        batch_idx = (
            torch.arange(B, device=feat.device, dtype=feat.dtype)
            .unsqueeze(1)
            .expand(B, K)
            .reshape(B * K, 1)
        )
        boxes = self.boxes_112.unsqueeze(0).expand(B, K, 4).reshape(B * K, 4)
        rois = torch.cat([batch_idx, boxes], dim=1)  # (B*K, 5)

        pooled = roi_align(
            feat,
            rois,
            output_size=(self.roi_output_size, self.roi_output_size),
            spatial_scale=self.spatial_scale,
            sampling_ratio=2,
            aligned=True,
        )

        tokens = self.spatial_pool(pooled).squeeze(-1).squeeze(-1)  # (B*K, C)
        tokens = self.channel_projection(tokens)  # (B*K, embedding_dim)
        tokens = tokens.view(B, K, -1)
        return tokens


# ---------------------------------------------------------------------------
# Graph Transformer
# ---------------------------------------------------------------------------

EDGE_TYPES = {
    "intra":      0,  # within the same face
    "homologous": 1,  # node_A_i ↔ node_B_i (same anatomical component)
    "global":     2,  # global_A ↔ any node_B, global_B ↔ any node_A
    "self":       3,  # self-loop
}


def _build_pair_edge_index(num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (edge_index, edge_type) for the joint pair graph.

    Node indexing: 0..K-1 = face_A nodes; K..2K-1 = face_B nodes.
    edge_index has shape (2, E) (source, target). edge_type has shape (E,).
    """
    intra = _build_intra_edges()
    n2i = {name: i for i, name in enumerate(NODE_NAMES)}
    global_idx = n2i["global"]

    edges: List[Tuple[int, int]] = []
    types: List[int] = []

    # Intra-face edges, replicated for face A and face B.
    for src, dst in intra:
        if src == dst:
            edges.append((src, dst));     types.append(EDGE_TYPES["self"])
            edges.append((num_nodes + src, num_nodes + dst)); types.append(EDGE_TYPES["self"])
        else:
            edges.append((src, dst));     types.append(EDGE_TYPES["intra"])
            edges.append((num_nodes + src, num_nodes + dst)); types.append(EDGE_TYPES["intra"])

    # Homologous cross-face edges (i_A ↔ i_B) for every component.
    for i in range(num_nodes):
        # exclude global from homologous; global handled separately
        if i == global_idx:
            continue
        edges.append((i, num_nodes + i)); types.append(EDGE_TYPES["homologous"])
        edges.append((num_nodes + i, i)); types.append(EDGE_TYPES["homologous"])

    # Global cross-face edges: global_A ↔ each node_B, global_B ↔ each node_A.
    for j in range(num_nodes):
        edges.append((global_idx, num_nodes + j)); types.append(EDGE_TYPES["global"])
        edges.append((num_nodes + j, global_idx)); types.append(EDGE_TYPES["global"])
        edges.append((num_nodes + global_idx, j)); types.append(EDGE_TYPES["global"])
        edges.append((j, num_nodes + global_idx)); types.append(EDGE_TYPES["global"])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2, E)
    edge_type = torch.tensor(types, dtype=torch.long)  # (E,)
    return edge_index, edge_type


class EdgeAwareGraphAttention(nn.Module):
    """One graph attention block with edge-type-aware bias.

    Implementation: dense attention over the union of nodes with a mask that
    only allows edges defined in the joint pair graph. The mask injects an
    edge-type embedding into the attention logits.
    """

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 4,
        ffn_expansion: int = 2,
        dropout: float = 0.1,
        num_edge_types: int = 4,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.edge_bias = nn.Embedding(num_edge_types + 1, num_heads)  # +1 for "no edge"

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_expansion, dim),
        )

    def forward(
        self, x: torch.Tensor, edge_type_mat: torch.Tensor, edge_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        x:               (B, N, D) where N = 2*K nodes
        edge_type_mat:   (N, N) long — edge type per (src, dst); -1 for no edge
                         but we encode "no edge" as ``num_edge_types`` here.
        edge_mask:       (N, N) bool — True where attention is allowed
        Returns: (B, N, D)
        """
        B, N, D = x.shape

        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, head_dim)

        attn = torch.einsum("bhnd,bhmd->bhnm", q, k) / (self.head_dim ** 0.5)  # (B, H, N, N)

        # Inject edge-type bias per head.
        # edge_type_mat: (N, N) → look up bias (N, N, H) → permute (H, N, N) → broadcast.
        bias = self.edge_bias(edge_type_mat)  # (N, N, H)
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # (1, H, N, N)
        attn = attn + bias

        # Mask: forbid attention where edge_mask=False.
        attn = attn.masked_fill(~edge_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.einsum("bhnm,bhmd->bhnd", attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)
        out = self.out_proj(out)

        x = residual + out
        x = x + self.ffn(self.norm2(x))
        return x


class GraphTransformer(nn.Module):
    """Stack of EdgeAwareGraphAttention blocks. Edges are fixed for the joint
    pair graph; only node features are batched."""

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_expansion: int = 2,
        dropout: float = 0.1,
        num_nodes_per_face: int = NUM_NODES,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            EdgeAwareGraphAttention(dim, num_heads, ffn_expansion, dropout)
            for _ in range(num_layers)
        ])

        edge_index, edge_type = _build_pair_edge_index(num_nodes_per_face)
        N = 2 * num_nodes_per_face
        edge_type_mat = torch.full((N, N), len(EDGE_TYPES), dtype=torch.long)  # "no edge" code
        edge_mask = torch.zeros(N, N, dtype=torch.bool)
        for e in range(edge_index.shape[1]):
            s, d = edge_index[0, e].item(), edge_index[1, e].item()
            edge_type_mat[s, d] = edge_type[e].item()
            edge_mask[s, d] = True

        self.register_buffer("edge_type_mat", edge_type_mat, persistent=False)
        self.register_buffer("edge_mask", edge_mask, persistent=False)

    def forward(self, tokens_a: torch.Tensor, tokens_b: torch.Tensor) -> torch.Tensor:
        """tokens_a, tokens_b: (B, K, D) → (B, 2K, D)."""
        x = torch.cat([tokens_a, tokens_b], dim=1)  # (B, 2K, D)
        for layer in self.layers:
            x = layer(x, self.edge_type_mat, self.edge_mask)
        return x


# ---------------------------------------------------------------------------
# Symmetric pooling + classifier
# ---------------------------------------------------------------------------

class SymmetricPairPooler(nn.Module):
    """For each homologous node pair (A_i, B_i), compute order-invariant
    features [mean, |diff|, prod], then aggregate over nodes with an attention
    pool. The output is invariant to swapping A and B by construction.

    When ``comparison_only_pooling=True`` (M12 R009 analog), the global node
    (index 0) is excluded from the pooled output to suppress identity leakage,
    while it still participates in the graph attention as context.
    """

    def __init__(
        self,
        dim: int = 512,
        num_nodes: int = NUM_NODES,
        gate_hidden: int = 128,
        dropout: float = 0.1,
        comparison_only_pooling: bool = False,
        global_node_idx: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.num_nodes = num_nodes
        self.comparison_only_pooling = comparison_only_pooling
        self.global_node_idx = global_node_idx
        # Gate: per-node importance from the symmetric features.
        self.gate = nn.Sequential(
            nn.Linear(3 * dim, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1),
        )

    def forward(self, x_pair: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x_pair: (B, 2K, D). Returns (pooled: (B, 3D), gate_weights: (B, K')).

        K' = K-1 when comparison_only_pooling, else K.
        """
        B, twoK, D = x_pair.shape
        K = self.num_nodes
        assert twoK == 2 * K, f"Expected 2*{K} nodes, got {twoK}"

        a = x_pair[:, :K, :]   # (B, K, D)
        b = x_pair[:, K:, :]   # (B, K, D)

        if self.comparison_only_pooling:
            keep = [i for i in range(K) if i != self.global_node_idx]
            keep_idx = torch.tensor(keep, device=x_pair.device, dtype=torch.long)
            a = a.index_select(1, keep_idx)  # (B, K-1, D)
            b = b.index_select(1, keep_idx)

        mean = 0.5 * (a + b)
        diff = (a - b).abs()
        prod = a * b
        symm = torch.cat([mean, diff, prod], dim=-1)  # (B, K', 3D)

        gate_logits = self.gate(symm).squeeze(-1)  # (B, K')
        gate = torch.softmax(gate_logits, dim=-1)
        pooled = (symm * gate.unsqueeze(-1)).sum(dim=1)  # (B, 3D)
        return pooled, gate


# ---------------------------------------------------------------------------
# Full LGKT-Net
# ---------------------------------------------------------------------------

class LGKTNet(nn.Module):
    """Landmark Graph Kinship Transformer — full model.

    Forward signature: ``(img_a, img_b) -> dict`` with keys ``logits``,
    ``gate``, ``tokens_a``, ``tokens_b``, and optionally ``relation_logits``.
    """

    def __init__(
        self,
        adaface_backbone: nn.Module,
        embedding_dim: int = 512,
        num_heads: int = 4,
        num_graph_layers: int = 2,
        gate_hidden: int = 128,
        classifier_hidden: int = 512,
        dropout: float = 0.2,
        freeze_backbone: bool = True,
        unfreeze_last_stage: bool = False,
        aux_relation_head: bool = False,
        num_relation_classes: int = 11,
        roi_output_size: int = 3,
        feature_stage: str = "stage4",
        comparison_only_pooling: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_stage = unfreeze_last_stage
        self.aux_relation_head = aux_relation_head
        self.num_relation_classes = num_relation_classes
        self.feature_stage = feature_stage
        self.comparison_only_pooling = comparison_only_pooling

        self.tokenizer = LandmarkROITokenizer(
            backbone=adaface_backbone,
            embedding_dim=embedding_dim,
            feature_stage=feature_stage,
            roi_output_size=roi_output_size,
        )
        self.node_names = self.tokenizer.box_names
        self.num_nodes = self.tokenizer.num_nodes
        # Determine global node index (first "global" entry, typically 0).
        if "global" in self.node_names:
            self._global_node_idx = self.node_names.index("global")
        else:
            self._global_node_idx = 0

        # Per-node embedding to disambiguate components (a "node-type" prior).
        self.node_type_embed = nn.Embedding(self.num_nodes, embedding_dim)

        if freeze_backbone:
            for p in adaface_backbone.parameters():
                p.requires_grad = False

            if unfreeze_last_stage:
                # AdaFace IR-101 body has 49 BasicBlockIR's grouped as:
                #   stage1 [0:3], stage2 [3:16], stage3 [16:46], stage4 [46:49].
                # output_layer = BN + Flatten + FC.
                #
                # stage4: unfreeze body[46:49] + output_layer (R001 / M12 pattern;
                #         output_layer is consumed by forward_spatial path indirectly,
                #         but matching the M12 Phase 2 unfreeze set).
                # stage3: stage 4 and output_layer are dead code (we cut at body[46]),
                #         so unfreeze the last 3 blocks of stage 3 instead — body[43:46].
                if feature_stage == "stage4":
                    for p in adaface_backbone.body[46:49].parameters():
                        p.requires_grad = True
                    for p in adaface_backbone.output_layer.parameters():
                        p.requires_grad = True
                else:  # stage3
                    for p in adaface_backbone.body[43:46].parameters():
                        p.requires_grad = True

        self.graph = GraphTransformer(
            dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_graph_layers,
            dropout=dropout,
            num_nodes_per_face=self.num_nodes,
        )

        self.pooler = SymmetricPairPooler(
            dim=embedding_dim,
            num_nodes=self.num_nodes,
            gate_hidden=gate_hidden,
            dropout=dropout,
            comparison_only_pooling=comparison_only_pooling,
            global_node_idx=self._global_node_idx,
        )

        # Classifier consumes the pooled [mean, |diff|, prod] features (3*D).
        self.classifier = nn.Sequential(
            nn.Linear(3 * embedding_dim, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, classifier_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden // 2, 1),
        )

        if aux_relation_head:
            self.relation_head = nn.Linear(embedding_dim, num_relation_classes)
        else:
            self.relation_head = None

    def _encode(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, 3, 224, 224) → (B, K, D) node tokens + node-type embed."""
        tokens = self.tokenizer(img)  # (B, K, D)
        node_ids = torch.arange(self.num_nodes, device=tokens.device)
        type_embed = self.node_type_embed(node_ids).unsqueeze(0)  # (1, K, D)
        return tokens + type_embed

    def forward(
        self, img_a: torch.Tensor, img_b: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        tokens_a = self._encode(img_a)
        tokens_b = self._encode(img_b)

        x_pair = self.graph(tokens_a, tokens_b)  # (B, 2K, D)
        pooled, gate = self.pooler(x_pair)       # (B, 3D), (B, K)

        logits = self.classifier(pooled).squeeze(-1)  # (B,)

        out = {
            "logits": logits,
            "gate": gate,
            "tokens_a": tokens_a,
            "tokens_b": tokens_b,
        }

        if self.relation_head is not None:
            # Relation head on the mean of A and B global tokens.
            global_idx = self.node_names.index("global")
            rel_input = 0.5 * (tokens_a[:, global_idx, :] + tokens_b[:, global_idx, :])
            out["relation_logits"] = self.relation_head(rel_input)

        return out


def build_lgkt_model(
    adaface_weights_path: str,
    embedding_dim: int = 512,
    num_heads: int = 4,
    num_graph_layers: int = 2,
    gate_hidden: int = 128,
    classifier_hidden: int = 512,
    dropout: float = 0.2,
    freeze_backbone: bool = True,
    unfreeze_last_stage: bool = False,
    aux_relation_head: bool = False,
    num_relation_classes: int = 11,
    roi_output_size: int = 3,
    feature_stage: str = "stage4",
    comparison_only_pooling: bool = False,
) -> LGKTNet:
    """Factory that loads AdaFace IR-101 weights then wraps it in LGKT-Net."""
    # Reuse the M12 AdaFace loader (single ``adaface_ir101(weights_path)`` factory).
    import sys
    from pathlib import Path
    m12_root = Path(__file__).parent.parent / "12_rgck_net"
    sys.path.insert(0, str(m12_root))
    from adaface_iresnet import adaface_ir101

    backbone = adaface_ir101(weights_path=adaface_weights_path)
    print(f"  [M13] AdaFace IR-101 weights loaded from {adaface_weights_path}")

    return LGKTNet(
        adaface_backbone=backbone,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_graph_layers=num_graph_layers,
        gate_hidden=gate_hidden,
        classifier_hidden=classifier_hidden,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        unfreeze_last_stage=unfreeze_last_stage,
        aux_relation_head=aux_relation_head,
        num_relation_classes=num_relation_classes,
        roi_output_size=roi_output_size,
        feature_stage=feature_stage,
        comparison_only_pooling=comparison_only_pooling,
    )


if __name__ == "__main__":
    # Quick smoke test — R002 config: stage3 features + comparison-only pooling.
    import sys
    from pathlib import Path
    proj = Path(__file__).parent.parent.parent
    weights = proj / "models/12_rgck_net/weights/adaface_ir101_webface4m.pth"
    model = build_lgkt_model(
        str(weights),
        aux_relation_head=True,
        unfreeze_last_stage=True,
        feature_stage="stage3",
        comparison_only_pooling=True,
    )
    img_a = torch.randn(2, 3, 224, 224)
    img_b = torch.randn(2, 3, 224, 224)
    out = model(img_a, img_b)
    print(f"logits:           {tuple(out['logits'].shape)}")
    print(f"gate:            {tuple(out['gate'].shape)}")
    print(f"relation_logits: {tuple(out['relation_logits'].shape)}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total params:     {total:,}")
    print(f"trainable params: {trainable:,} ({100*trainable/total:.2f}%)")
