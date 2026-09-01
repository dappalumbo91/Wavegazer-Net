"""Render the canonical U-Net schematic as SVG (exact labels, no image model)."""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "schematics"


def _box(x, y, w, h, fill, stroke, title, sub, extra=""):
    return f"""
  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <text x="{x + w/2}" y="{y + 22}" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="13" font-weight="700" fill="#0f172a">{title}</text>
  <text x="{x + w/2}" y="{y + 40}" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11" fill="#334155">{sub}</text>
  {extra}
"""


def _arrow(x1, y1, x2, y2, color="#0f172a", dashed=False):
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2.2"{dash} marker-end="url(#arrow)"/>'


def canonical() -> str:
    # Encoder column x=70, decoder column x=860, bottleneck center.
    enc = [
        (70, 40, "L0 encoder", "2 x 3x3 conv + ReLU", "64 ch  ·  568 x 568", "#dbeafe", "#1d4ed8"),
        (110, 150, "L1 encoder", "pool 2x2, then 2 x 3x3", "128 ch  ·  280 x 280", "#bfdbfe", "#1d4ed8"),
        (150, 260, "L2 encoder", "pool 2x2, then 2 x 3x3", "256 ch  ·  136 x 136", "#93c5fd", "#1e40af"),
        (190, 370, "L3 encoder", "pool 2x2, then 2 x 3x3", "512 ch  ·  64 x 64", "#60a5fa", "#1e3a8a"),
    ]
    bottleneck = (360, 480, "Bottleneck", "pool 2x2, then 2 x 3x3", "1024 ch  ·  28 x 28", "#c4b5fd", "#5b21b6")
    dec = [
        (530, 370, "L3 decoder", "up-conv 2x2, concat, 2 x 3x3", "512 ch  ·  52 x 52", "#6ee7b7", "#047857"),
        (570, 260, "L2 decoder", "up-conv 2x2, concat, 2 x 3x3", "256 ch  ·  100 x 100", "#34d399", "#047857"),
        (610, 150, "L1 decoder", "up-conv 2x2, concat, 2 x 3x3", "128 ch  ·  196 x 196", "#a7f3d0", "#065f46"),
        (650, 40, "L0 decoder", "up-conv 2x2, concat, 2 x 3x3", "64 ch  ·  388 x 388", "#d1fae5", "#065f46"),
    ]
    parts = [
        '''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="720" viewBox="0 0 1400 720">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#0f172a"/>
    </marker>
    <marker id="arrow-orange" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#c2410c"/>
    </marker>
  </defs>
  <rect width="1400" height="720" fill="#f8fafc"/>
  <text x="700" y="28" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="20" font-weight="700" fill="#0f172a">Canonical U-Net (Ronneberger, Fischer, Brox 2015) — paper figure sizes</text>
  <text x="700" y="48" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#475569">Valid 3x3 convolutions. Input 572x572 x C  →  output 388x388 x K. 23 convolutional layers. Skip = copy + crop + concatenate.</text>
'''
    ]
    # input
    parts.append(_box(70, 40 - 0, 0, 0, "none", "none", "", ""))  # placeholder skipped
    parts.append(
        _box(
            -10 + 20,
            40,
            50,
            64,
            "#e2e8f0",
            "#334155",
            "in",
            "572²",
            f'<text x="45" y="122" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="10" fill="#64748b">C ch</text>',
        )
    )
    for x, y, title, sub, size, fill, stroke in enc:
        extra = f'<text x="{x + 95}" y="{y + 56}" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11" fill="#1e3a8a">{size}</text>'
        parts.append(_box(x, y, 190, 70, fill, stroke, title, sub, extra))
    x, y, title, sub, size, fill, stroke = bottleneck
    extra = f'<text x="{x + 120}" y="{y + 56}" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11" fill="#5b21b6">{size}</text>'
    parts.append(_box(x, y, 240, 70, fill, stroke, title, sub, extra))
    for x, y, title, sub, size, fill, stroke in dec:
        extra = f'<text x="{x + 110}" y="{y + 56}" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11" fill="#065f46">{size}</text>'
        parts.append(_box(x, y, 220, 70, fill, stroke, title, sub, extra))
    parts.append(
        _box(
            900,
            40,
            70,
            70,
            "#fee2e2",
            "#991b1b",
            "1x1",
            "K classes",
            '<text x="935" y="122" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="10" fill="#991b1b">388² logits</text>',
        )
    )

    # down arrows
    parts.append(_arrow(165, 110, 205, 150))
    parts.append(_arrow(205, 220, 245, 260))
    parts.append(_arrow(245, 330, 285, 370))
    parts.append(_arrow(285, 440, 430, 480))
    # up arrows
    parts.append(_arrow(480, 480, 640, 440))
    parts.append(_arrow(640, 370, 680, 330))
    parts.append(_arrow(680, 260, 720, 220))
    parts.append(_arrow(720, 150, 760, 110))
    parts.append(_arrow(870, 75, 900, 75))

    # skip connections (dashed orange)
    skips = [
        (260, 75, 650, 75),
        (300, 185, 610, 185),
        (340, 295, 570, 295),
        (380, 405, 530, 405),
    ]
    for x1, y1, x2, y2 in skips:
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#c2410c" stroke-width="2" stroke-dasharray="7 5" marker-end="url(#arrow-orange)"/>'
        )
        parts.append(
            f'<text x="{(x1+x2)/2}" y="{y1 - 8}" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11" fill="#c2410c">skip: copy → crop → concat</text>'
        )

    parts.append(
        """
  <rect x="1000" y="160" width="370" height="500" rx="10" fill="#ffffff" stroke="#cbd5e1"/>
  <text x="1185" y="188" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="14" font-weight="700" fill="#0f172a">What each arrow is</text>
  <text x="1020" y="220" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#0f172a">Contracting path (left, blue)</text>
  <text x="1020" y="240" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#334155">context / “what”. Channels double.</text>
  <text x="1020" y="270" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#0f172a">Bottleneck (purple)</text>
  <text x="1020" y="290" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#334155">lowest resolution, 1024 channels.</text>
  <text x="1020" y="320" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#0f172a">Expansive path (right, green)</text>
  <text x="1020" y="340" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#334155">localization / “where”. Channels halve.</text>
  <text x="1020" y="370" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#c2410c">Skip (dashed)</text>
  <text x="1020" y="390" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#334155">re-injects high-res edges lost in pooling.</text>
  <text x="1020" y="420" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#0f172a">Modern same-padding variant</text>
  <text x="1020" y="440" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#334155">pad=1 so H_out = H_in at every level.</text>
  <text x="1020" y="456" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#334155">Typical 256² input stays 256² output.</text>
  <text x="1020" y="472" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#334155">Channels still 64-128-256-512-1024.</text>
  <text x="1020" y="504" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#0f172a">This schematic is the baseline.</text>
  <text x="1020" y="524" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#334155">A Wavegazer Net equivalent must state which</text>
  <text x="1020" y="540" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#334155">boxes it replaces (conv, pool, skip, loss)</text>
  <text x="1020" y="556" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#334155">and keep the same in/out contract to compare.</text>
  <text x="1020" y="588" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11" fill="#64748b">Sources: arXiv:1505.04597 Fig. 1;</text>
  <text x="1020" y="604" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11" fill="#64748b">milesial/Pytorch-UNet; MIC-DKFZ PlainConvUNet.</text>
</svg>
"""
    )
    # The dummy empty box at the start is messy. Rebuild more cleanly below.
    return "".join(parts)


def canonical_clean() -> str:
    W, H = 1480, 760
    # geometry
    enc_x = [80, 130, 180, 230]
    enc_y = [80, 200, 320, 440]
    dec_x = [820, 870, 920, 970]
    dec_y = [440, 320, 200, 80]
    bw, bh = 200, 84
    dbw = 230

    def box(x, y, w, h, fill, stroke, t1, t2, t3):
        return f"""  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <text x="{x + w/2}" y="{y + 24}" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="14" font-weight="700" fill="#0f172a">{t1}</text>
  <text x="{x + w/2}" y="{y + 46}" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#334155">{t2}</text>
  <text x="{x + w/2}" y="{y + 66}" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#1e3a8a">{t3}</text>
"""

    def line(x1, y1, x2, y2, color="#0f172a", dashed=False, marker="url(#arrow)"):
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        return f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2.4"{dash} marker-end="{marker}"/>\n'

    s = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#0f172a"/>
    </marker>
    <marker id="arrow-orange" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#c2410c"/>
    </marker>
  </defs>
  <rect width="{W}" height="{H}" fill="#f8fafc"/>
  <text x="{W/2}" y="32" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="22" font-weight="700" fill="#0f172a">Canonical U-Net schematic — observable baseline</text>
  <text x="{W/2}" y="54" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="13" fill="#475569">Ronneberger, Fischer, Brox, MICCAI 2015 (arXiv:1505.04597). Valid 3×3 convs. 572×572 in → 388×388 out. 23 conv layers.</text>
'''
    # input
    s += box(8, 80, 64, 84, "#e2e8f0", "#334155", "in", "C ch", "572×572")
    s += line(72, 122, 80, 122)
    enc_meta = [
        ("L0  encoder", "2× (3×3 conv, ReLU)", "64 ch · 568×568", "#dbeafe", "#1d4ed8"),
        ("L1  encoder", "max-pool 2×2 + 2×conv", "128 ch · 280×280", "#bfdbfe", "#1d4ed8"),
        ("L2  encoder", "max-pool 2×2 + 2×conv", "256 ch · 136×136", "#93c5fd", "#1e40af"),
        ("L3  encoder", "max-pool 2×2 + 2×conv", "512 ch · 64×64", "#60a5fa", "#1e3a8a"),
    ]
    for (x, y), (t1, t2, t3, fill, stroke) in zip(zip(enc_x, enc_y), enc_meta):
        s += box(x, y, bw, bh, fill, stroke, t1, t2, t3)
    s += box(390, 560, 260, 84, "#ddd6fe", "#5b21b6", "Bottleneck", "max-pool 2×2 + 2×conv", "1024 ch · 28×28")
    dec_meta = [
        ("L3  decoder", "up-conv 2×2, concat, 2×conv", "512 ch · 52×52", "#6ee7b7", "#047857"),
        ("L2  decoder", "up-conv 2×2, concat, 2×conv", "256 ch · 100×100", "#34d399", "#047857"),
        ("L1  decoder", "up-conv 2×2, concat, 2×conv", "128 ch · 196×196", "#a7f3d0", "#065f46"),
        ("L0  decoder", "up-conv 2×2, concat, 2×conv", "64 ch · 388×388", "#d1fae5", "#065f46"),
    ]
    # decoder listed L3..L0 matching dec_x/dec_y
    for (x, y), (t1, t2, t3, fill, stroke) in zip(zip(dec_x, dec_y), dec_meta):
        s += box(x, y, dbw, bh, fill, stroke, t1, t2, t3)
    s += line(1200, 122, 1220, 122)
    s += box(1220, 80, 90, 84, "#fecaca", "#991b1b", "1×1", "K logits", "388×388")

    # vertical encoder arrows
    for i in range(3):
        s += line(enc_x[i] + bw / 2, enc_y[i] + bh, enc_x[i + 1] + bw / 2, enc_y[i + 1])
    s += line(enc_x[3] + bw / 2, enc_y[3] + bh, 390 + 80, 560)
    s += line(390 + 180, 560, dec_x[0] + 40, dec_y[0] + bh)
    for i in range(3):
        s += line(dec_x[i] + dbw / 2, dec_y[i], dec_x[i + 1] + dbw / 2, dec_y[i + 1] + bh)

    # skips
    skip_y = [80 + 42, 200 + 42, 320 + 42, 440 + 42]
    skip_x1 = [enc_x[i] + bw for i in range(4)]
    skip_x2 = [dec_x[3 - i] for i in range(4)]  # L0 decoder is last in dec list? Wait
    # enc 0 -> dec L0 which is dec_y[3] = 80, dec_x[3]
    # enc 1 -> dec L1 dec_y[2]
    # enc 2 -> dec L2 dec_y[1]
    # enc 3 -> dec L3 dec_y[0]
    skip_x2 = [dec_x[3], dec_x[2], dec_x[1], dec_x[0]]
    for i in range(4):
        y = skip_y[i]
        s += line(skip_x1[i], y, skip_x2[i], y, color="#c2410c", dashed=True, marker="url(#arrow-orange)")
        s += f'  <text x="{(skip_x1[i]+skip_x2[i])/2}" y="{y - 10}" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#c2410c">skip copy · crop · concat</text>\n'

    # legend
    s += """  <rect x="80" y="668" width="1320" height="70" rx="8" fill="#ffffff" stroke="#cbd5e1"/>
  <rect x="100" y="688" width="22" height="14" fill="#93c5fd" stroke="#1e40af"/>
  <text x="128" y="700" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="13" fill="#0f172a">Contracting path — context (“what”), channels double, space halves</text>
  <rect x="620" y="688" width="22" height="14" fill="#6ee7b7" stroke="#047857"/>
  <text x="648" y="700" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="13" fill="#0f172a">Expansive path — localization (“where”), channels halve, space doubles</text>
  <line x1="100" y1="722" x2="150" y2="722" stroke="#c2410c" stroke-width="2.4" stroke-dasharray="7 5"/>
  <text x="160" y="726" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="13" fill="#0f172a">Skip — high-res features re-enter so boundaries survive pooling</text>
  <text x="900" y="726" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#64748b">Modern variant: same topology, pad=1, H_out = H_in (typical 256²). See docs/05_SCHEMATIC.md</text>
</svg>
"""
    return s


def pipeline() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="1480" height="560" viewBox="0 0 1480 560">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#0f172a"/>
    </marker>
  </defs>
  <rect width="1480" height="560" fill="#f8fafc"/>
  <text x="740" y="36" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="22" font-weight="700" fill="#0f172a">Standard U-Net build / train / infer pipeline</text>
  <text x="740" y="58" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="13" fill="#475569">Paper recipe on the top row. nnU-Net (the current open-source standard) additions on the bottom row.</text>

  <rect x="40" y="90" width="1400" height="200" rx="10" fill="#ffffff" stroke="#cbd5e1"/>
  <text x="60" y="118" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="14" font-weight="700" fill="#1d4ed8">2015 paper (Caffe)</text>

  <rect x="60" y="140" width="150" height="70" rx="8" fill="#dbeafe" stroke="#1d4ed8"/>
  <text x="135" y="170" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="13" font-weight="700">Image + mask</text>
  <text x="135" y="190" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11">few annotated tiles</text>

  <line x1="210" y1="175" x2="240" y2="175" stroke="#0f172a" stroke-width="2.2" marker-end="url(#arrow)"/>

  <rect x="240" y="140" width="180" height="70" rx="8" fill="#dbeafe" stroke="#1d4ed8"/>
  <text x="330" y="170" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="13" font-weight="700">Elastic augment</text>
  <text x="330" y="190" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11">3×3 grid, σ=10 px</text>

  <line x1="420" y1="175" x2="450" y2="175" stroke="#0f172a" stroke-width="2.2" marker-end="url(#arrow)"/>

  <rect x="450" y="140" width="160" height="70" rx="8" fill="#dbeafe" stroke="#1d4ed8"/>
  <text x="530" y="170" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="13" font-weight="700">U-Net forward</text>
  <text x="530" y="190" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11">batch = 1 tile</text>

  <line x1="610" y1="175" x2="640" y2="175" stroke="#0f172a" stroke-width="2.2" marker-end="url(#arrow)"/>

  <rect x="640" y="140" width="170" height="70" rx="8" fill="#dbeafe" stroke="#1d4ed8"/>
  <text x="725" y="170" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="13" font-weight="700">Softmax + wCE</text>
  <text x="725" y="190" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11">eq. (1)(2), w0=10</text>

  <line x1="810" y1="175" x2="840" y2="175" stroke="#0f172a" stroke-width="2.2" marker-end="url(#arrow)"/>

  <rect x="840" y="140" width="170" height="70" rx="8" fill="#dbeafe" stroke="#1d4ed8"/>
  <text x="925" y="170" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="13" font-weight="700">SGD + momentum</text>
  <text x="925" y="190" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11">μ = 0.99, He init</text>

  <line x1="1010" y1="175" x2="1040" y2="175" stroke="#0f172a" stroke-width="2.2" marker-end="url(#arrow)"/>

  <rect x="1040" y="140" width="180" height="70" rx="8" fill="#dbeafe" stroke="#1d4ed8"/>
  <text x="1130" y="170" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="13" font-weight="700">Overlap-tile infer</text>
  <text x="1130" y="190" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11">mirror borders</text>

  <line x1="1220" y1="175" x2="1250" y2="175" stroke="#0f172a" stroke-width="2.2" marker-end="url(#arrow)"/>

  <rect x="1250" y="140" width="160" height="70" rx="8" fill="#fecaca" stroke="#991b1b"/>
  <text x="1330" y="170" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="13" font-weight="700">Seg map</text>
  <text x="1330" y="190" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11">IOU / warping / Rand</text>

  <rect x="40" y="320" width="1400" height="210" rx="10" fill="#ffffff" stroke="#cbd5e1"/>
  <text x="60" y="348" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="14" font-weight="700" fill="#047857">2018–now standard (nnU-Net / PlainConvUNet)</text>

  <rect x="60" y="370" width="180" height="80" rx="8" fill="#d1fae5" stroke="#047857"/>
  <text x="150" y="400" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="13" font-weight="700">Dataset fingerprint</text>
  <text x="150" y="420" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11">spacing, median shape</text>
  <text x="150" y="436" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11">crop / resample / z-score</text>

  <line x1="240" y1="410" x2="270" y2="410" stroke="#0f172a" stroke-width="2.2" marker-end="url(#arrow)"/>

  <rect x="270" y="370" width="200" height="80" rx="8" fill="#d1fae5" stroke="#047857"/>
  <text x="370" y="400" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="13" font-weight="700">Auto topology</text>
  <text x="370" y="420" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11">patch size, #pools, 2D/3D</text>
  <text x="370" y="436" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11">IN + leaky ReLU, pad=1</text>

  <line x1="470" y1="410" x2="500" y2="410" stroke="#0f172a" stroke-width="2.2" marker-end="url(#arrow)"/>

  <rect x="500" y="370" width="210" height="80" rx="8" fill="#d1fae5" stroke="#047857"/>
  <text x="605" y="400" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="13" font-weight="700">Train 1000 epochs</text>
  <text x="605" y="420" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11">L = Dice + CE</text>
  <text x="605" y="436" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11">SGD 0.01, poly LR, DS</text>

  <line x1="710" y1="410" x2="740" y2="410" stroke="#0f172a" stroke-width="2.2" marker-end="url(#arrow)"/>

  <rect x="740" y="370" width="210" height="80" rx="8" fill="#d1fae5" stroke="#047857"/>
  <text x="845" y="400" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="13" font-weight="700">Sliding window</text>
  <text x="845" y="420" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11">Gaussian weight, TTA flip</text>
  <text x="845" y="436" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11">5-fold ensemble</text>

  <line x1="950" y1="410" x2="980" y2="410" stroke="#0f172a" stroke-width="2.2" marker-end="url(#arrow)"/>

  <rect x="980" y="370" width="200" height="80" rx="8" fill="#d1fae5" stroke="#047857"/>
  <text x="1080" y="400" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="13" font-weight="700">Post-process</text>
  <text x="1080" y="420" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11">keep largest component</text>
  <text x="1080" y="436" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11">if GT is always 1 CC</text>

  <line x1="1180" y1="410" x2="1210" y2="410" stroke="#0f172a" stroke-width="2.2" marker-end="url(#arrow)"/>

  <rect x="1210" y="370" width="200" height="80" rx="8" fill="#fecaca" stroke="#991b1b"/>
  <text x="1310" y="400" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="13" font-weight="700">Dice / IoU</text>
  <text x="1310" y="420" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11">foreground mean Dice</text>
  <text x="1310" y="436" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="11">the number we beat</text>
</svg>
'''


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "unet_canonical.svg").write_text(canonical_clean(), encoding="utf-8")
    (OUT / "unet_pipeline.svg").write_text(pipeline(), encoding="utf-8")
    print(f"wrote {OUT / 'unet_canonical.svg'}")
    print(f"wrote {OUT / 'unet_pipeline.svg'}")


if __name__ == "__main__":
    main()
