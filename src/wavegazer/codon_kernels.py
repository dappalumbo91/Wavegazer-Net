"""64-codon trinary kernels. These replace learned 3×3 convolution weights.

PRIMARY: A,G = +1, C,T = −1  (FSOT-Genetics / neuron-zig law)
SECONDARY: A = +1, T = −1, G,C = 0

Each codon becomes one 3×3 spatial filter (64 output channels at the stem).
No trained entries.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .fsot_seeds import CODON_CHANNELS

CODONS = (
    "AAA", "AAC", "AAG", "AAT", "ACA", "ACC", "ACG", "ACT",
    "AGA", "AGC", "AGG", "AGT", "ATA", "ATC", "ATG", "ATT",
    "CAA", "CAC", "CAG", "CAT", "CCA", "CCC", "CCG", "CCT",
    "CGA", "CGC", "CGG", "CGT", "CTA", "CTC", "CTG", "CTT",
    "GAA", "GAC", "GAG", "GAT", "GCA", "GCC", "GCG", "GCT",
    "GGA", "GGC", "GGG", "GGT", "GTA", "GTC", "GTG", "GTT",
    "TAA", "TAC", "TAG", "TAT", "TCA", "TCC", "TCG", "TCT",
    "TGA", "TGC", "TGG", "TGT", "TTA", "TTC", "TTG", "TTT",
)

_PRIMARY = {"A": 1.0, "G": 1.0, "C": -1.0, "T": -1.0}
_SECONDARY = {"A": 1.0, "T": -1.0, "G": 0.0, "C": 0.0}


def primary(codon: str) -> tuple[float, float, float]:
    return tuple(_PRIMARY[b] for b in codon)  # type: ignore[return-value]


def secondary(codon: str) -> tuple[float, float, float]:
    return tuple(_SECONDARY[b] for b in codon)  # type: ignore[return-value]


def codon_kernel_3x3(codon: str) -> torch.Tensor:
    """Plus-shaped primary, corner secondary. Zero kernels get an identity center."""
    p = primary(codon)
    s = secondary(codon)
    k = torch.tensor(
        [
            [s[0], p[0], s[1]],
            [p[1], 0.0, p[2]],
            [s[2], p[0], s[1]],
        ],
        dtype=torch.float32,
    )
    if torch.count_nonzero(k) == 0:
        k[1, 1] = 1.0
    norm = k.abs().sum().clamp_min(1.0)
    return k / norm


def stem_weight(in_channels: int) -> torch.Tensor:
    """(64, in_channels, 3, 3) stem: each codon filter is copied across input channels."""
    assert len(CODONS) == CODON_CHANNELS
    filters = torch.stack([codon_kernel_3x3(c) for c in CODONS], dim=0)
    return filters.unsqueeze(1).repeat(1, in_channels, 1, 1) / float(in_channels)


def trit_similarity_matrix() -> torch.Tensor:
    """64×64 1×1 mix: mean PRIMARY agreement. Diagonal is 1."""
    prim = torch.tensor([primary(c) for c in CODONS], dtype=torch.float32)
    sim = (prim @ prim.T) / 3.0
    return sim


def load_published_map(path: Path | None = None) -> list[str]:
    """Optional cross-check against data/64_codon_trinary_map.txt."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "data" / "64_codon_trinary_map.txt"
    if not path.is_file():
        return list(CODONS)
    found: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        tok = line.strip().split()
        if tok and len(tok[0]) == 3 and tok[0].isalpha() and tok[0].isupper():
            found.append(tok[0])
    return found or list(CODONS)
