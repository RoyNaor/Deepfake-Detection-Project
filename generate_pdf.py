#!/usr/bin/env python3
"""
Generates a professional PDF of the FusionGuardNet project book.
Converts project_book2.md to styled HTML with embedded figures, then to PDF.
"""

import base64
import os
import re
import markdown
from weasyprint import HTML as WHTML, CSS as WCSS

BASE_DIR = "/home/user/Deepfake-Detection-Project"
MD_PATH  = os.path.join(BASE_DIR, "project_book2.md")
PDF_PATH = os.path.join(BASE_DIR, "FusionGuardNet_Project_Book.pdf")

IMAGES = {
    "d1_acc":    "results/results-d1/accuracy_curve.png",
    "d1_loss":   "results/results-d1/loss_curve.png",
    "d1_cm":     "results/results-d1/test_runs/test_epoch_08_2026-03-26_19-46-37/confusion_matrix_epoch_08.png",
    "d2_acc":    "results/results-d2/accuracy_curve.png",
    "d2_loss":   "results/results-d2/loss_curve.png",
    "d2_cm":     "results/results-d2/test_runs/test_epoch_07_2026-04-12_09-31-23/confusion_matrix_epoch_07.png",
    "d2_roc":    "results/results-d2/test_runs/test_epoch_07_2026-04-12_09-31-23/roc_curve_epoch_07.png",
    "wavlm":     "figures/wavlm_architecture.png",
    "whisper":   "figures/whisper_architecture.png",
    "spectro":   "figures/spectrogram_real_vs_fake.png",
}

def b64img(rel_path):
    with open(os.path.join(BASE_DIR, rel_path), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

def figure_block(key, caption, num):
    return (
        f'<figure class="fig">'
        f'<img src="{b64img(IMAGES[key])}" alt="{caption}">'
        f'<figcaption><strong>Figure {num}.</strong> {caption}</figcaption>'
        f'</figure>'
    )

# ── 1. Read markdown ──────────────────────────────────────────────────────────
with open(MD_PATH, "r") as f:
    md_text = f.read()

# ── 2. Convert to HTML ────────────────────────────────────────────────────────
html_body = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "attr_list", "sane_lists"],
)

# ── 3. Style inline citations as superscripts ─────────────────────────────────
# Avoid replacing inside <code> or <pre> blocks
def add_cite_superscripts(html):
    parts = re.split(r'(<(?:code|pre)[^>]*>.*?</(?:code|pre)>)', html, flags=re.DOTALL)
    result = []
    for i, part in enumerate(parts):
        if i % 2 == 1:          # inside code/pre — leave untouched
            result.append(part)
        else:
            # Replace [N] or [NN] citation markers with superscript spans
            part = re.sub(r'\[(\d{1,2})\]', r'<sup class="cite">[\1]</sup>', part)
            result.append(part)
    return "".join(result)

html_body = add_cite_superscripts(html_body)

# ── 4. Inject architecture figures ───────────────────────────────────────────
WAVLM_ANCHOR = "which voice-conversion and TTS systems must approximate but rarely replicate perfectly.</p>"
WAVLM_FIG = figure_block("wavlm", "WavLM pre-training architecture — masked prediction over utterance-mixed audio. CNN encoders extract local features, the Transformer encoder with Gated Relative Position Bias contextualises them, and the Mask Prediction Loss trains the model to reconstruct masked frames [M] from context.", "A")
if WAVLM_ANCHOR in html_body:
    html_body = html_body.replace(WAVLM_ANCHOR, WAVLM_ANCHOR + "\n" + WAVLM_FIG)
else:
    print("WARNING: WavLM anchor not found — figure not inserted.")

WHISPER_ANCHOR = "which is the core idea behind our fusion approach"
WHISPER_FIG = figure_block("whisper", "Whisper sequence-to-sequence architecture. The encoder (left) processes a Log-Mel Spectrogram through 2×Conv1D+GELU and stacked Transformer blocks; the decoder (right) generates text tokens autoregressively via cross-attention. In FusionGuardNet only the encoder is used [16].", "B")
# find end of the paragraph containing this anchor
if WHISPER_ANCHOR in html_body:
    idx = html_body.index(WHISPER_ANCHOR)
    close_p = html_body.index("</p>", idx)
    insert_at = close_p + len("</p>")
    html_body = html_body[:insert_at] + "\n" + WHISPER_FIG + html_body[insert_at:]
else:
    print("WARNING: Whisper anchor not found — figure not inserted.")

SPECTRO_ANCHOR = "rather than raw acoustic signal properties.</p>"
SPECTRO_FIG = figure_block("spectro", "Log-Mel spectrogram comparison: real speech (left) shows denser, more irregular low-frequency energy; synthetic speech (right) displays smoother, more uniform spectral structure with visible silence bands — differences that motivate the Whisper encoder branch.", "C")
# There may be multiple occurrences; target the one in §4.3 (second occurrence)
occurrences = [m.start() for m in __import__('re').finditer(re.escape(SPECTRO_ANCHOR), html_body)]
if len(occurrences) >= 1:
    idx = occurrences[-1]  # use last occurrence (§4.3)
    insert_at = idx + len(SPECTRO_ANCHOR)
    html_body = html_body[:insert_at] + "\n" + SPECTRO_FIG + html_body[insert_at:]
else:
    print("WARNING: Spectrogram anchor not found — figure not inserted.")

# ── 5. Inject result figures — spread individually across §6.2 ───────────────
# D1 accuracy: after the sentence describing 8-epoch convergence
D1_ACC_ANCHOR = "with best dev accuracy of <strong>99.25%</strong>.</p>"
if D1_ACC_ANCHOR in html_body:
    html_body = html_body.replace(
        D1_ACC_ANCHOR,
        D1_ACC_ANCHOR + "\n" + figure_block("d1_acc", "Dataset 1 — Training vs. Validation Accuracy over 8 epochs. The model converges stably, reaching 99.49% training accuracy by epoch 8.", 1)
    )
else:
    print("WARNING: D1 accuracy anchor not found.")

# D1 confusion matrix: after the D1 counts sentence
D1_CM_ANCHOR = "TN=5,225, FP=43, FN=43, TP=5,225</strong>)."
D1_CM_CLOSE = html_body.find("</p>", html_body.find(D1_CM_ANCHOR)) if D1_CM_ANCHOR in html_body else -1
if D1_CM_CLOSE != -1:
    insert = D1_CM_CLOSE + len("</p>")
    html_body = html_body[:insert] + "\n" + figure_block("d1_cm", "Dataset 1 — Confusion Matrix on the Test Set (Epoch 8). Perfectly symmetric errors: 43 false positives and 43 false negatives out of 10,536 samples.", 2) + html_body[insert:]
else:
    print("WARNING: D1 confusion matrix anchor not found.")

# D2 accuracy: after the sentence describing 99.34% on 17,467 samples
D2_ACC_ANCHOR = "indicating that the integration remains robust under broader and more modern spoofing conditions"
D2_ACC_CLOSE = html_body.find("</p>", html_body.find(D2_ACC_ANCHOR)) if D2_ACC_ANCHOR in html_body else -1
if D2_ACC_CLOSE != -1:
    insert = D2_ACC_CLOSE + len("</p>")
    html_body = html_body[:insert] + "\n" + figure_block("d2_acc", "Dataset 2 — Training vs. Validation Accuracy over 8 epochs. The fused model maintains stable convergence on the larger mixed-source dataset.", 3) + html_body[insert:]
else:
    print("WARNING: D2 accuracy anchor not found.")

# D2 confusion matrix: after the sentence about remains stable / broader conditions (intro summary)
D2_CM_ANCHOR = "remains stable when the data becomes more varied.</p>"
if D2_CM_ANCHOR in html_body:
    html_body = html_body.replace(
        D2_CM_ANCHOR,
        D2_CM_ANCHOR + "\n" + figure_block("d2_cm", "Dataset 2 — Confusion Matrix on the Test Set (Epoch 7). TN=8,652, FP=77, FN=38, TP=8,700 across 17,467 samples.", 4)
    )
else:
    print("WARNING: D2 confusion matrix anchor not found.")

# D2 ROC: after the EER sentence in the intro results summary
D2_ROC_ANCHOR = "an <strong>Equal Error Rate (EER) of 0.60%</strong>"
D2_ROC_CLOSE = html_body.find("</p>", html_body.find(D2_ROC_ANCHOR)) if D2_ROC_ANCHOR in html_body else -1
if D2_ROC_CLOSE != -1:
    insert = D2_ROC_CLOSE + len("</p>")
    html_body = html_body[:insert] + "\n" + figure_block("d2_roc", "Dataset 2 — ROC Curve (AUC = 0.9991, EER = 0.60%, Epoch 7). Near-perfect discrimination between real and fake samples across all operating thresholds.", 5) + html_body[insert:]
else:
    print("WARNING: D2 ROC anchor not found.")

# ── 7. CSS ────────────────────────────────────────────────────────────────────
CSS = """
@page {
    size: A4;
    margin: 2.8cm 2.6cm 2.8cm 2.6cm;
    @bottom-center {
        content: counter(page);
        font-size: 9pt;
        color: #666;
    }
}

body {
    font-family: "Georgia", "Times New Roman", serif;
    font-size: 11pt;
    line-height: 1.7;
    color: #1c1c1c;
}

/* ── Headings ── */
h1 {
    font-size: 22pt;
    color: #102540;
    text-align: center;
    border-bottom: 3px solid #102540;
    padding-bottom: 10pt;
    margin-bottom: 20pt;
    page-break-after: avoid;
}
h2 {
    font-size: 15pt;
    color: #102540;
    border-bottom: 1.5px solid #9eb3cc;
    padding-bottom: 4pt;
    margin-top: 28pt;
    margin-bottom: 10pt;
    page-break-after: avoid;
}
h3 {
    font-size: 12.5pt;
    color: #1e3a5f;
    margin-top: 20pt;
    margin-bottom: 6pt;
    page-break-after: avoid;
}
h4 {
    font-size: 11.5pt;
    color: #1e3a5f;
    font-style: italic;
    margin-top: 14pt;
    margin-bottom: 4pt;
    page-break-after: avoid;
}

/* ── Body text ── */
p {
    margin: 0 0 9pt 0;
    text-align: justify;
}

/* ── Tables ── */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 14pt 0;
    font-size: 10pt;
    page-break-inside: avoid;
}
thead tr th {
    background-color: #102540;
    color: #ffffff;
    padding: 7pt 10pt;
    text-align: left;
    font-weight: bold;
    letter-spacing: 0.3pt;
}
tbody tr td {
    padding: 6pt 10pt;
    border-bottom: 1px solid #dde2e8;
    vertical-align: top;
}
tbody tr:nth-child(even) td {
    background-color: #f3f6fa;
}

/* ── Code ── */
code {
    font-family: "Courier New", monospace;
    font-size: 9.5pt;
    background-color: #eef1f5;
    padding: 1pt 4pt;
    border-radius: 2pt;
}
pre {
    background-color: #eef1f5;
    border-left: 4pt solid #102540;
    padding: 10pt 14pt;
    font-size: 9pt;
    margin: 12pt 0;
    page-break-inside: avoid;
}
pre code {
    background: none;
    padding: 0;
}

/* ── Citations ── */
sup.cite {
    font-size: 7.5pt;
    color: #102540;
    font-weight: bold;
    vertical-align: super;
    line-height: 0;
}

/* ── Figures ── */
figure.fig {
    text-align: center;
    margin: 22pt 0;
    page-break-inside: avoid;
}
figure.fig img {
    max-width: 78%;
    height: auto;
    border: 1px solid #c8d2de;
    border-radius: 3pt;
    box-shadow: 0 1pt 4pt rgba(0,0,0,0.12);
}
figcaption {
    font-size: 9.5pt;
    color: #444;
    font-style: italic;
    margin-top: 6pt;
    text-align: center;
}

/* ── Horizontal rules ── */
hr {
    border: none;
    border-top: 1px solid #c8d2de;
    margin: 22pt 0;
}

/* ── Lists ── */
ul, ol {
    margin: 6pt 0 10pt 22pt;
    padding: 0;
}
li {
    margin-bottom: 4pt;
}

/* ── Links ── */
a {
    color: #1e3a5f;
    text-decoration: none;
}

/* ── Block quotes ── */
blockquote {
    border-left: 4pt solid #9eb3cc;
    margin: 10pt 0;
    padding: 4pt 0 4pt 16pt;
    color: #444;
}

/* ── References section ── */
h2#references + ol li,
h2 + ol li {
    font-size: 10pt;
    margin-bottom: 7pt;
    text-align: justify;
}
"""

# ── 8. Assemble full HTML ──────────────────────────────────────────────────────
full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FusionGuardNet — Project Book</title>
</head>
<body>
{html_body}
</body>
</html>"""

# ── 9. Render PDF ─────────────────────────────────────────────────────────────
print("Rendering PDF …")
WHTML(string=full_html).write_pdf(PDF_PATH, stylesheets=[WCSS(string=CSS)])
print(f"Done → {PDF_PATH}")
