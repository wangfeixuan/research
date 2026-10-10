"""Build the Week 5 final report from measured experiment outputs."""
from pathlib import Path
import csv
from matplotlib import font_manager
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle

root = Path(__file__).resolve().parent
results = root / "results"
out = root / "week5_final_report.pdf"
font = Path(font_manager.findfont('DejaVu Sans'))
pdfmetrics.registerFont(TTFont('Harmony', str(font)))
with (results / 'representation_comparison.csv').open() as f:
    representation = list(csv.DictReader(f))
with (results / 'vae_dimension_comparison.csv').open() as f:
    dimensions = list(csv.DictReader(f))

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCustom', parent=styles['Title'], fontName='Harmony', fontSize=17, leading=23, alignment=TA_CENTER, textColor=HexColor('#1f2937'), spaceAfter=5))
styles.add(ParagraphStyle(name='SubCustom', parent=styles['Normal'], fontName='Harmony', fontSize=9, leading=14, alignment=TA_CENTER, textColor=HexColor('#6b7280'), spaceAfter=9))
styles.add(ParagraphStyle(name='HeadingCustom', parent=styles['Heading2'], fontName='Harmony', fontSize=11.5, leading=16, textColor=HexColor('#1769aa'), spaceBefore=7, spaceAfter=4))
styles.add(ParagraphStyle(name='BodyCustom', parent=styles['BodyText'], fontName='Harmony', fontSize=9.4, leading=14.5, textColor=HexColor('#263238'), spaceAfter=6))
styles.add(ParagraphStyle(name='SmallCustom', parent=styles['BodyText'], fontName='Harmony', fontSize=8.6, leading=13, textColor=HexColor('#394956'), spaceAfter=5))

def p(text, kind='BodyCustom'):
    return Paragraph(text, styles[kind])

def table(rows, widths):
    t = Table(rows, colWidths=widths, repeatRows=1, hAlign='CENTER')
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),HexColor('#e8f1fb')),
        ('TEXTCOLOR',(0,0),(-1,0),HexColor('#17365d')),
        ('FONTNAME',(0,0),(-1,-1),'Harmony'),
        ('FONTSIZE',(0,0),(-1,-1),8.7),
        ('GRID',(0,0),(-1,-1),0.4,HexColor('#b8c7d9')),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),5),
        ('BOTTOMPADDING',(0,0),(-1,-1),5),
    ]))
    return t

def img(name, width, height):
    return Image(str(results / name), width=width*mm, height=height*mm)

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Harmony',8)
    canvas.setFillColor(HexColor('#64748b'))
    canvas.drawRightString(190*mm, 10*mm, f'Week 5  |  {doc.page}')
    canvas.restoreState()

story = [p('Learning Image Representations for Clustering', 'TitleCustom'),
         p('Week 5 final mini research report  |  Fashion-MNIST', 'SubCustom'),
         p('1. Question and dataset', 'HeadingCustom'),
         p('I wanted to see whether changing the image representation helps K-means group Fashion-MNIST images. I compared raw pixels, PCA, an autoencoder (AE), and a variational autoencoder (VAE). Fashion-MNIST contains 28 × 28 grayscale images from ten clothing categories. This is a clustering experiment: class labels were used only after clustering to calculate NMI and ARI.'),
         p('2. Methods', 'HeadingCustom'),
         p('<b>Raw pixels:</b> flatten each image to 784 values. <b>PCA:</b> reduce those values to 20 components. <b>AE:</b> encode the image into a 20-value bottleneck, then train a decoder to reconstruct it. <b>VAE:</b> encode each image into a mean and log variance, sample a latent vector using reparameterization, and train with reconstruction plus KL loss. For VAE clustering, I used the mean vector because it is deterministic.'),
         p('3. Experimental setup', 'HeadingCustom'),
         table([['Setting','Value'],['Training images','20,000 Fashion-MNIST training images'],['Evaluation images','10,000 Fashion-MNIST test images'],['AE and VAE structure','784 → 256 → 64 → latent → 64 → 256 → 784'],['Optimizer and learning rate','Adam, 0.001'],['Batch size and epochs','256, 5 epochs'],['Latent dimensions','2, 10, 20, 50'],['Clustering','K-means, K=10, n_init=10, random_state=42'],['Metrics','NMI and ARI, labels used only for evaluation']],[52*mm,120*mm]),
         Spacer(1,5),
         p('I used the same random seed and the same test subset as the Week 4 AE experiment. The Week 4 AE values were reused from its saved results; raw and PCA baselines were recomputed here on the same 10,000 test images. The VAE results come from this Week 5 run.', 'SmallCustom'),
         PageBreak(),
         p('Reconstruction and generation', 'TitleCustom'),
         p('4. Ten original images and VAE reconstructions', 'HeadingCustom'),
         img('vae_reconstructions_10.png', 174, 31),
         p('The first row shows the input images and the second row shows reconstructions. The VAE usually keeps the broad shape, but fine detail is blurred. This is a visible limit of this small fully connected model after five epochs.'),
         p('5. Twenty images generated from the prior', 'HeadingCustom'),
         img('vae_generated_20.png', 102, 82),
         p('For generation, I sampled twenty new 20-dimensional vectors from N(0, I) and passed them through the decoder. These images have no input image. Many look like clothing silhouettes, but they remain soft and some classes are hard to identify.'),
         p('<b>Difference:</b> reconstruction starts with a real image and tries to reproduce it. Generation starts with a random latent vector and makes a new image.', 'SmallCustom'),
         PageBreak(),
         p('Clustering results', 'TitleCustom'),
         p('6. Raw pixels, PCA, AE and VAE', 'HeadingCustom'),
         table([['Representation','Dimension','NMI','ARI']] + [[r['Representation'],r['Dimension'],f"{float(r['NMI']):.4f}",f"{float(r['ARI']):.4f}"] for r in representation],[62*mm,31*mm,31*mm,31*mm]),
         Spacer(1,6),
         p('In this run, VAE-20 had the highest NMI and ARI (0.5713 and 0.4038). PCA-20 was close to raw pixels. AE-20 was lower than both. I did not assume learned features must win; the AE result shows that reconstructing pixels alone does not guarantee that clothing categories will form clearer groups.'),
         p('7. Latent dimension study', 'HeadingCustom'),
         table([['Dimension','Recon BCE/image','VAE loss/image','NMI','ARI']] + [[r['latent_dim'],f"{float(r['test_mean_recon_bce_per_image']):.2f}",f"{float(r['test_vae_loss_per_image']):.2f}",f"{float(r['NMI']):.4f}",f"{float(r['ARI']):.4f}"] for r in dimensions],[28*mm,39*mm,39*mm,25*mm,25*mm]),
         Spacer(1,5),
         img('vae_loss_curves.png', 130, 73),
         p('Here the lowest deterministic reconstruction BCE and the lowest VAE loss were both at dimension 10, while the best clustering was at dimension 20. So lower training objective value did not mean better NMI or ARI. Recon BCE uses the decoded mean vector; VAE loss combines reconstruction and KL divergence. Neither is numerically comparable with the AE mean squared error from Week 4.', 'SmallCustom'),
         PageBreak(),
         p('Latent space and interpolation', 'TitleCustom'),
         p('8. Two-dimensional latent means', 'HeadingCustom'),
         img('vae_latent_2.png', 124, 96),
         p('The colors are true Fashion-MNIST labels added after training for visualization. Some groups occupy recognizable areas, while several colors overlap near the center and lower-right. This helps explain why perfect clustering is unrealistic for this representation.'),
         p('9. Interpolation', 'HeadingCustom'),
         img('vae_interpolation.png', 174, 20),
         p('I interpolated between the latent means of a sneaker (class 7) and a T-shirt/top (class 0). The decoder changes the image gradually rather than switching at one point. The middle images are ambiguous, which fits the overlap seen in the latent plot.'),
         PageBreak(),
         p('What I learned and what I would improve', 'TitleCustom'),
         p('10. Answers to the main comparison questions', 'HeadingCustom'),
         p('<b>Does PCA improve clustering?</b> Not clearly here. PCA-20 NMI was 0.5138 versus 0.5145 for raw pixels, while ARI was 0.3506 versus 0.3497. The changes were small.'),
         p('<b>Does the AE improve clustering?</b> No. Its 20-dimensional features had NMI 0.4471 and ARI 0.2633, below the raw and PCA baselines.'),
         p('<b>Does the VAE improve clustering?</b> Yes in this particular run. VAE-20 reached NMI 0.5713 and ARI 0.4038, above the other three representations.'),
         p('<b>How does dimension affect results?</b> VAE clustering improved from 2 to 20 dimensions, then fell at 50. The reconstruction BCE and VAE loss were lowest at 10 dimensions. More dimensions did not automatically help.'),
         p('<b>Are reconstruction and clustering the same goal?</b> No. The decoder can reconstruct image details that are not useful for grouping clothing categories. The Week 4 AE-20 result and this VAE dimension study both show this.'),
         p('11. Problems and limitations', 'HeadingCustom'),
         p('The generated images are blurry. In the 2D plot, several clothing classes overlap, especially around the central region. The MLP encoder and decoder only see flattened pixels, so they do not use local image structure like a CNN. I also trained each setting only once for five epochs. This makes the comparison a useful first experiment, but not a final claim that VAE always beats the other methods.'),
         p('For a stronger follow-up, I would run several random seeds, train longer, try a convolutional VAE, and tune the balance between reconstruction and KL loss. I would also inspect cluster assignments for easily confused clothing categories and keep a separate validation set for model selection.'),
         p('12. Personal takeaway', 'HeadingCustom'),
         p('This project made the role of the representation concrete for me. K-means itself stayed the same, but its NMI and ARI changed when I changed the features. I also learned to judge reconstruction images and clustering numbers separately rather than treating one good-looking result as proof that the whole model is better.')]

doc = SimpleDocTemplate(str(out), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=15*mm, bottomMargin=16*mm, title='Learning Image Representations for Clustering')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(out)
