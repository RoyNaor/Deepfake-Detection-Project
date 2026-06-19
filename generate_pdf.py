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
    "d1_acc":  "results/results-d1/accuracy_curve.png",
    "d1_loss": "results/results-d1/loss_curve.png",
    "d2_acc":  "results/results-d2/accuracy_curve.png",
    "d2_loss": "results/results-d2/loss_curve.png",
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

# ── 4. Inject Dataset 1 figures ───────────────────────────────────────────────
D1_ANCHOR = "well balanced for this dataset configuration.</p>"
D1_FIGS = "\n".join([
    figure_block("d1_acc",  "Dataset 1 (ASVsp-C) — Training vs. Validation Accuracy over 8 epochs", 2),
    figure_block("d1_loss", "Dataset 1 (ASVsp-C) — Training vs. Validation Loss over 8 epochs",     3),
])
if D1_ANCHOR in html_body:
    html_body = html_body.replace(D1_ANCHOR, D1_ANCHOR + "\n" + D1_FIGS)
else:
    print("WARNING: Dataset 1 anchor not found — figures not inserted.")

# ── 5. Inject Dataset 2 figures ───────────────────────────────────────────────
D2_ANCHOR = "remains stable when the data becomes more varied.</p>"
D2_FIGS = "\n".join([
    figure_block("d2_acc",  "Dataset 2 (ASVsp-FoR) — Training vs. Validation Accuracy over 8 epochs", 4),
    figure_block("d2_loss", "Dataset 2 (ASVsp-FoR) — Training vs. Validation Loss over 8 epochs",     5),
])
if D2_ANCHOR in html_body:
    html_body = html_body.replace(D2_ANCHOR, D2_ANCHOR + "\n" + D2_FIGS)
else:
    print("WARNING: Dataset 2 anchor not found — figures not inserted.")

# ── 6. CSS ────────────────────────────────────────────────────────────────────
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

# ── 7. Assemble full HTML ──────────────────────────────────────────────────────
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

# ── 8. Render PDF ─────────────────────────────────────────────────────────────
print("Rendering PDF …")
WHTML(string=full_html).write_pdf(PDF_PATH, stylesheets=[WCSS(string=CSS)])
print(f"Done → {PDF_PATH}")
