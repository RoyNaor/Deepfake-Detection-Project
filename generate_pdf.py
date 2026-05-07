import markdown
import re
from weasyprint import HTML, CSS

with open("project_book2.md", "r", encoding="utf-8") as f:
    md_text = f.read()

md = markdown.Markdown(extensions=["tables", "fenced_code", "toc"])
body_html = md.convert(md_text)

css = CSS(string="""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Source+Code+Pro&display=swap');

@page {
    size: A4;
    margin: 2.5cm 2.2cm 2.5cm 2.2cm;
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-family: 'Inter', Arial, sans-serif;
        font-size: 9pt;
        color: #888;
    }
}

* {
    box-sizing: border-box;
}

body {
    font-family: 'Inter', Arial, Helvetica, sans-serif;
    font-size: 10.5pt;
    line-height: 1.7;
    color: #1a1a2e;
    background: white;
}

/* ── Headings ── */
h1 {
    font-size: 20pt;
    font-weight: 700;
    color: #0d1b2a;
    text-align: center;
    margin: 0 0 8pt 0;
    padding-bottom: 10pt;
    border-bottom: 3px solid #2563eb;
    letter-spacing: -0.5px;
}

h2 {
    font-size: 14pt;
    font-weight: 700;
    color: #1e3a5f;
    margin: 22pt 0 6pt 0;
    padding: 6pt 10pt;
    background: #f0f4ff;
    border-left: 4px solid #2563eb;
    border-radius: 3px;
    page-break-after: avoid;
}

h3 {
    font-size: 11.5pt;
    font-weight: 600;
    color: #1e3a5f;
    margin: 14pt 0 4pt 0;
    border-bottom: 1px solid #d1daf0;
    padding-bottom: 3pt;
    page-break-after: avoid;
}

h4 {
    font-size: 10.5pt;
    font-weight: 600;
    color: #2563eb;
    margin: 10pt 0 3pt 0;
    page-break-after: avoid;
}

/* ── Paragraphs ── */
p {
    margin: 0 0 8pt 0;
    text-align: justify;
    orphans: 3;
    widows: 3;
}

/* ── Tables ── */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 10pt 0 14pt 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}

thead tr {
    background: #1e3a5f;
    color: white;
}

thead th {
    padding: 6pt 10pt;
    text-align: left;
    font-weight: 600;
    letter-spacing: 0.2px;
}

thead th:last-child,
tbody td:last-child {
    text-align: right;
}

tbody tr:nth-child(even) {
    background: #f5f7ff;
}

tbody tr:nth-child(odd) {
    background: #ffffff;
}

tbody td {
    padding: 5pt 10pt;
    border-bottom: 1px solid #dde3f0;
    vertical-align: top;
}

/* ── Lists ── */
ul, ol {
    margin: 4pt 0 8pt 0;
    padding-left: 18pt;
}

li {
    margin-bottom: 3pt;
}

/* ── Strong / Em ── */
strong {
    font-weight: 700;
    color: #0d1b2a;
}

em {
    font-style: italic;
    color: #374151;
}

/* ── Code ── */
code {
    font-family: 'Source Code Pro', 'Courier New', monospace;
    font-size: 9pt;
    background: #eef1f8;
    padding: 1pt 4pt;
    border-radius: 3px;
    color: #1e3a5f;
}

/* ── HR ── */
hr {
    border: none;
    border-top: 1.5px solid #d1daf0;
    margin: 18pt 0;
}

/* ── Blockquotes ── */
blockquote {
    border-left: 4px solid #2563eb;
    background: #f0f4ff;
    margin: 10pt 0;
    padding: 6pt 12pt;
    font-style: italic;
    color: #374151;
}
""")

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FusionGuardNet – Audio Deepfake Detection</title>
</head>
<body>
{body_html}
</body>
</html>"""

HTML(string=full_html, base_url=".").write_pdf(
    "project_book2.pdf",
    stylesheets=[css],
    presentational_hints=True,
)
print("PDF created: project_book2.pdf")
