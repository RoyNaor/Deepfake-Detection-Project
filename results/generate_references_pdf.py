from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate, Frame
from reportlab.lib.enums import TA_LEFT, TA_CENTER

PAPERS = [
    {
        "num": 1,
        "title": "Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-Spoofing",
        "authors": "Liu, T., Truong, D. T., Das, R. K., Lee, K. A., & Li, H.",
        "year": "2025",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2504.05657",
    },
    {
        "num": 2,
        "title": "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing",
        "authors": "Chen, S., Wang, C., Chen, Z., Wu, Y., Liu, S. et al.",
        "year": "2022",
        "venue": "IEEE Journal of Selected Topics in Signal Processing",
        "url": "https://arxiv.org/abs/2110.13900",
    },
    {
        "num": 3,
        "title": "Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)",
        "authors": "Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I.",
        "year": "2023",
        "venue": "ICML 2023",
        "url": "https://arxiv.org/abs/2212.04356",
    },
    {
        "num": 4,
        "title": "ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection",
        "authors": "Kinnunen, T., Yamagishi, J., Todisco, M., Delgado, H. et al.",
        "year": "2019",
        "venue": "Interspeech 2019",
        "url": "https://arxiv.org/abs/1904.05441",
    },
    {
        "num": 5,
        "title": "ASVspoof 5: Crowdsourced Data, Deepfakes, and Adversarial Attacks at Scale",
        "authors": "Wang, X., Tak, H., Patino, J., Todisco, M., Nautsch, A. et al.",
        "year": "2024",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2408.08703",
    },
    {
        "num": 6,
        "title": "Fake or Real Audio Dataset",
        "authors": "Kaggle / Mohammed Abdel Dayem",
        "year": "n.d.",
        "venue": "Kaggle",
        "url": "https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset",
    },
    {
        "num": 7,
        "title": "A Survey on Speech Deepfake Detection",
        "authors": "Khanjani, Z., Watson, G., & Janeja, V. P.",
        "year": "2024",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2404.13914",
    },
    {
        "num": 8,
        "title": "Where Are We in Audio Deepfake Detection? A Systematic Analysis over Generative and Detection Models",
        "authors": "Li, X., Chen, P-Y., & Wei, W.",
        "year": "2025",
        "venue": "ACM Transactions on Internet Technology",
        "url": "https://arxiv.org/abs/2410.04324",
    },
    {
        "num": 9,
        "title": "Vulnerabilities of Audio-Based Biometric Authentication Systems Against Deepfake Speech Synthesis",
        "authors": "Yi, J. et al.",
        "year": "2025",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2601.02914",
    },
    {
        "num": 10,
        "title": "A Survey of Threats Against Voice Authentication and Anti-Spoofing Systems",
        "authors": "(Multiple authors)",
        "year": "2025",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2508.16843",
    },
    {
        "num": 11,
        "title": "Deepfake-Eval-2024: A Multi-Modal In-the-Wild Benchmark of Deepfakes Circulated in 2024",
        "authors": "Cai, Z. et al.",
        "year": "2025",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2503.02857",
    },
    {
        "num": 12,
        "title": "Benchmarking Audio Deepfake Detection Robustness in Real-World Communication Scenarios",
        "authors": "(Multiple authors)",
        "year": "2025",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2504.12423",
    },
    {
        "num": 13,
        "title": "Audio Anti-Spoofing Detection: A Survey",
        "authors": "(Multiple authors)",
        "year": "2023",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2308.14970",
    },
    {
        "num": 14,
        "title": "Harder or Different? Understanding Generalization of Audio Deepfake Detection",
        "authors": "(Multiple authors)",
        "year": "2024",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2406.03512",
    },
    {
        "num": 15,
        "title": "Attentive Merging of Hidden Embeddings from Pre-Trained Speech Model for Anti-Spoofing Detection",
        "authors": "(Multiple authors)",
        "year": "2024",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2406.10283",
    },
    {
        "num": 16,
        "title": "Two Views, One Truth: Spectral and Self-Supervised Features Fusion for Robust Speech Deepfake Detection",
        "authors": "(Multiple authors)",
        "year": "2025",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2507.20417",
    },
    {
        "num": 17,
        "title": "From Sharpness to Better Generalization for Speech Deepfake Detection",
        "authors": "(Multiple authors)",
        "year": "2025",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2506.11532",
    },
    {
        "num": 18,
        "title": "Comprehensive Layer-Wise Analysis of SSL Models for Audio Deepfake Detection",
        "authors": "(Multiple authors)",
        "year": "2025",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2502.03559",
    },
    {
        "num": 19,
        "title": "SUPERB: Speech Processing Universal PERformance Benchmark",
        "authors": "Yang, S-W. et al.",
        "year": "2021",
        "venue": "Interspeech 2021",
        "url": "https://arxiv.org/abs/2105.01051",
    },
    {
        "num": 20,
        "title": "Audio Deepfake Detection with Self-Supervised WavLM and Multi-Fusion Attentive Classifier",
        "authors": "(Multiple authors)",
        "year": "2023",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2312.08089",
    },
    {
        "num": 21,
        "title": "WavLM Model Ensemble for Audio Deepfake Detection",
        "authors": "(Multiple authors)",
        "year": "2024",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2408.07414",
    },
    {
        "num": 22,
        "title": "Exploring WavLM Back-Ends for Speech Spoofing and Deepfake Detection",
        "authors": "(Multiple authors)",
        "year": "2024",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2409.05032",
    },
    {
        "num": 23,
        "title": "Experimental Study: Enhancing Voice Spoofing Detection Models with wav2vec 2.0",
        "authors": "(Multiple authors)",
        "year": "2024",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2402.17127",
    },
    {
        "num": 24,
        "title": "Deep Residual Learning for Image Recognition (ResNet)",
        "authors": "He, K., Zhang, X., Ren, S., & Sun, J.",
        "year": "2016",
        "venue": "CVPR 2016",
        "url": "https://arxiv.org/abs/1512.03385",
    },
    {
        "num": 25,
        "title": "Res2Net: A New Multi-Scale Backbone Architecture",
        "authors": "Gao, S-H., Cheng, M-M., Zhao, K., Zhang, X-Y. et al.",
        "year": "2021",
        "venue": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
        "url": "https://arxiv.org/abs/1904.01169",
    },
    {
        "num": 26,
        "title": "Replay and Synthetic Speech Detection with Res2Net Architecture",
        "authors": "Li, X. et al.",
        "year": "2021",
        "venue": "ICASSP 2021",
        "url": "https://arxiv.org/abs/2010.15006",
    },
    {
        "num": 27,
        "title": "Spatial Reconstructed Local Attention Res2Net with F0 Subband for Fake Speech Detection",
        "authors": "(Multiple authors)",
        "year": "2023",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2308.09944",
    },
    {
        "num": 28,
        "title": "Improving Short Utterance Anti-Spoofing with AASIST2",
        "authors": "(Multiple authors)",
        "year": "2024",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2309.08279",
    },
    {
        "num": 29,
        "title": "AASIST3: KAN-Enhanced AASIST Speech Deepfake Detection Using SSL Features for ASVspoof 2024",
        "authors": "(Multiple authors)",
        "year": "2024",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2408.17352",
    },
    {
        "num": 30,
        "title": "Beyond Identity: A Generalizable Approach for Deepfake Audio Detection",
        "authors": "(Multiple authors)",
        "year": "2025",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2505.06766",
    },
]


def build_pdf(output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()

    header_style = ParagraphStyle(
        "header", fontSize=16, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1e3c78"), spaceAfter=4, alignment=TA_CENTER,
    )
    sub_style = ParagraphStyle(
        "sub", fontSize=9, fontName="Helvetica",
        textColor=colors.HexColor("#555555"), spaceAfter=14, alignment=TA_CENTER,
    )
    num_style = ParagraphStyle(
        "num", fontSize=10, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1e3c78"),
    )
    title_style = ParagraphStyle(
        "title", fontSize=10, fontName="Helvetica-Bold",
        textColor=colors.black, spaceAfter=2,
    )
    meta_style = ParagraphStyle(
        "meta", fontSize=8.5, fontName="Helvetica-Oblique",
        textColor=colors.HexColor("#444444"), spaceAfter=2, leftIndent=10,
    )
    link_style = ParagraphStyle(
        "link", fontSize=8.5, fontName="Helvetica",
        textColor=colors.HexColor("#0050b4"), spaceAfter=6, leftIndent=10,
    )

    story = []

    story.append(Paragraph("FusionGuardNet — Reference List", header_style))
    story.append(Paragraph("Audio Deepfake Detection · 30 cited papers with links", sub_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1e3c78")))
    story.append(Spacer(1, 10))

    for p in PAPERS:
        story.append(Paragraph(f"[{p['num']}]  {p['title']}", title_style))
        story.append(Paragraph(f"{p['authors']}  ({p['year']})  ·  {p['venue']}", meta_style))
        story.append(Paragraph(f'<a href="{p["url"]}" color="#0050b4">{p["url"]}</a>', link_style))
        story.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#dddddd")))
        story.append(Spacer(1, 5))

    doc.build(story)
    print(f"PDF saved: {output_path}  ({len(PAPERS)} references)")


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(__file__), "FusionGuardNet_References.pdf")
    build_pdf(out)
