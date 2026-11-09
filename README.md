# Research preparation

How does the representation of a Fashion-MNIST image affect K-means clustering? The five-week project compares raw pixels, PCA, autoencoder codes, and VAE latent means. With ten clusters, VAE-20 achieved NMI 0.5713 and ARI 0.4038 in the recorded five-epoch run. The 10-dimensional VAE had the lowest reconstruction error. These results are from one seed.

[Final report](report/final_report.pdf) · [Presentation](report/week5_presentation.pptx) · [Experiment log](results/experiments.csv) · [Figures](results/figures/)

## Project layout

| Directory | Contents |
|---|---|
| [notebooks](notebooks/) | Eight notebooks, ordered by week. Week 4 combines autoencoder training and the latent-clustering exercises in one notebook. |
| [src](src/) | Model definitions, training routines, classifier preprocessing and random seeds used by the notebooks. |
| [results](results/) | Recorded experiments and figures. |
| [report](report/) | Final PDF, editable slides and weekly PDF summaries. |

## Run the notebooks

From the repository root, create a Python 3.12 environment and select its kernel in Jupyter.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name research --display-name "Python (research)"
python -m pip install jupyterlab
python -m jupyterlab
```

Open the notebooks in this order and run cells from top to bottom.

1. [NumPy practice](notebooks/week1_numpy_practice.ipynb) and [ML basics](notebooks/week1_ml_basics.ipynb)
2. [PyTorch basics](notebooks/week2_pytorch_basics.ipynb) and [MLP classifier](notebooks/week2_mlp_fashion_mnist.ipynb)
3. [CNN classifier](notebooks/week3_cnn.ipynb) and [pixel and PCA clustering](notebooks/week3_clustering.ipynb)
4. [Autoencoder and latent clustering](notebooks/week4_autoencoder.ipynb)
5. [VAE and representation comparison](notebooks/week5_vae.ipynb)

Fashion-MNIST downloads to the ignored data/ directory on first use. The optional Kaggle Digit Recognizer section of the CNN notebook expects train.csv and test.csv under data/kaggle/. The original platform score was 0.98353; the separate local rerun reached 0.98488 validation accuracy. Week 5 reads the recorded Week 4 AE-20 scores from results/experiments.csv. Running experiment cells again can replace result files. The historical experiment log describes the recorded runs.

Runtime experiments create additional result files under results/week2 through results/week5. These directories are generated when needed.
