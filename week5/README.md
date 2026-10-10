# Week 5: VAE and image representation clustering

This folder contains the Week 5 VAE experiment and the five-week final research summary. Use the `Python (research)` kernel and run the notebook from top to bottom.

## Deliverables

- `week5_vae.ipynb`: VAE model, training, reconstruction, generation, clustering, latent visualization, and interpolation.
- `week5_final_report.pdf`: five-page final mini research report.
- `week5_presentation.pptx`: ten-slide presentation.
- `results/representation_comparison.csv`: raw pixels, PCA-20, AE-20, and VAE-20 clustering results.
- `results/vae_dimension_comparison.csv`: VAE dimensions 2, 10, 20, and 50.
- `results/vae_reconstructions_10.png`: ten original/reconstructed image pairs.
- `results/vae_generated_20.png`: twenty generated images.
- `results/vae_latent_2.png` and `results/vae_interpolation.png`: latent space and interpolation figures.

## Reproduce

The notebook uses Fashion-MNIST data from `../data` and the Week 4 AE record from `../week4/results/ae_comparison.json`. It trains on 20,000 images and evaluates on 10,000 test images, with random seed 42. The main settings are Adam, learning rate 0.001, batch size 256, five epochs, and K-means with ten clusters.

```bash
# Run these commands from the research project folder after creating the environment described in the root README.
source research/bin/activate
python -m pip install -r requirements.txt
```

Open `week5/week5_vae.ipynb` and select `Python (research)`. Re-running the notebook refreshes the figures, CSV files, and model checkpoints in `results/`. Rebuild the report and presentation with `python week5/make_report.py` and `python week5/make_presentation.py`.

## Main result

| Representation | Dimension | NMI | ARI |
|---|---:|---:|---:|
| Raw pixels | 784 | 0.5145 | 0.3497 |
| PCA-20 | 20 | 0.5138 | 0.3506 |
| AE-20 | 20 | 0.4471 | 0.2633 |
| VAE-20 | 20 | 0.5713 | 0.4038 |

These are one-run results, not a claim that VAE always wins. The generated images are visibly blurry, and several classes overlap in the 2D latent plot.
