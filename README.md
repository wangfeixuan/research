# Research preparation: image representations and clustering

This workspace follows the five-week research preparation assignment. The main research question is how raw pixels, PCA, an autoencoder (AE), and a variational autoencoder (VAE) change Fashion-MNIST clustering results.

## Weekly work

| Week | Main deliverables |
|---|---|
| 1 | `week1/week1_numpy_practice.ipynb`, `week1/week1_ml_basics.ipynb`, `week1/week1_ml_summary.pdf` |
| 2 | `week2/week2_pytorch_basics.ipynb`, `week2/week2_mlp_fashion_mnist.ipynb`, learning-rate PDF and plot |
| 3 | `week3/week3_cnn.ipynb`, `week3/week3_clustering.ipynb`, Kaggle submission CSV and report |
| 4 | `week4/week4_autoencoder.ipynb`, `week4/week4_latent_clustering.ipynb`, AE comparison table and report |
| 5 | `week5/week5_vae.ipynb`, five-page final report, ten-slide presentation and figures |

The combined experiment log is [results/experiments.csv](results/experiments.csv). Each week also contains its own result files. The final report is [week5/week5_final_report.pdf](week5/week5_final_report.pdf).

## Run the notebooks

Create the environment from the project folder:

```bash
python3 -m venv research
source research/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name research --display-name "Python (research)"
```

Start Jupyter from this project folder or one of its subfolders, select the `Python (research)` kernel, then run notebook cells from top to bottom. The notebooks locate the project folder automatically and store Fashion-MNIST under `data/`. Kaggle `train.csv` and `test.csv` are kept in `week3/` and are required only for its Kaggle cells; download these two files from Kaggle before running that section.

The original local `research` environment was created with access to system Python packages. The command above creates a fully isolated environment on a fresh machine. `requirements.txt` lists the packages used by these notebooks and the report scripts.

## Interpretation

The four-way Fashion-MNIST comparison used the same 10,000 test images and K-means with ten clusters. VAE-20 reached NMI 0.5713 and ARI 0.4038 in one five-epoch run. This is an initial result; the experiments were not repeated across several random seeds.

The Kaggle platform score was 0.98353. A separate local validation rerun reached 0.98488 accuracy; it is not claimed to be the exact submitted model.
