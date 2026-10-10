# Research Preparation Assignment

## Machine Learning, PyTorch, Image Representation Learning, and Deep Generative Models

**Duration: 5 Weeks**

**Recommended workload: 10-12 hours per week**

---

# 1. Goal of This Assignment

This assignment is designed to prepare you for future research on:

- deep learning
- image representation learning
- image classification
- clustering
- autoencoders
- variational autoencoders
- deep generative models

You already have experience in mathematics, statistics, and programming. However, before starting a research project, you need a stronger foundation in machine learning and deep learning, as well as more experience with PyTorch and experimental work.

You are **not expected to understand all theory before you start coding**.

The learning process should follow this cycle:

**Learn a concept → implement it → run experiments → inspect the results → explain what happened**

You should not spend an entire week only watching videos or reading tutorials.

For every topic, you are expected to write code and produce experimental results.

---

# 2. Expected Skills After Five Weeks

After completing this assignment, you should be able to:

1. Use Python, NumPy, Matplotlib, scikit-learn, and PyTorch for machine learning experiments.
2. Explain the basic machine learning workflow.
3. Understand training, validation, and testing.
4. Understand loss functions, optimization, gradients, and backpropagation.
5. Build and train a multilayer perceptron (MLP).
6. Build and train a convolutional neural network (CNN).
7. Understand the difference between classification and clustering.
8. Apply K-means and principal component analysis (PCA).
9. Build a basic autoencoder (AE).
10. Extract and analyze latent representations.
11. Build and understand a basic variational autoencoder (VAE).
12. Use learned representations for clustering.
13. Compare models using controlled experiments.
14. Read basic PyTorch research code.
15. Organize research code and experimental results clearly.

---

# 3. Main Learning Resources

You do not need to finish every page of these resources.

Use them when the related topic appears in your weekly tasks.

## Machine Learning Fundamentals

### Google Machine Learning Crash Course

Use this as one of your main references for basic machine learning.

Focus on:

- Introduction to Machine Learning
- Linear Regression
- Loss
- Gradient Descent
- Logistic Regression
- Classification
- Datasets
- Generalization
- Overfitting
- Neural Networks

[Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)

The course also contains short interactive exercises and programming exercises.

[Google ML Crash Course Exercises](https://developers.google.com/machine-learning/crash-course/exercises)

---

## Deep Learning Reference

### Dive into Deep Learning

Use selected chapters when you need another explanation of neural networks.

For this assignment, focus mainly on:

- linear models
- softmax classification
- multilayer perceptrons
- overfitting and regularization
- convolutional neural networks

[Dive into Deep Learning: Multilayer Perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/index.html)

[Dive into Deep Learning: Convolutional Neural Networks](https://www.d2l.ai/chapter_convolutional-neural-networks/index.html)

Do not try to read the entire book during these five weeks.

---

## PyTorch

Use the official PyTorch tutorials as your main PyTorch reference.

Focus on:

- tensors
- datasets
- data loaders
- model construction
- automatic differentiation
- optimization
- saving and loading models

[PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro)

A shorter overview is also available here:

[PyTorch 60 Minute Blitz](https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz)

For image classification:

[PyTorch Training a Classifier Tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

---

# 4. Week 1: Python, NumPy, and Machine Learning Foundations

## Main Goal

The goal of Week 1 is to understand the basic language of machine learning before using deep neural networks.

Do not rush into PyTorch.

You first need to understand what a machine learning experiment is doing.

---

## 4.1 Python and NumPy Review

Review:

### Python

- variables
- lists
- dictionaries
- loops
- conditions
- functions
- classes
- modules
- reading files

You do not need advanced Python knowledge at this stage.

### NumPy

You should understand:

- arrays
- shapes
- indexing
- slicing
- reshape
- transpose
- matrix multiplication
- broadcasting
- mean
- variance
- standard deviation
- random numbers
- vectorized operations

---

## Task 1A: NumPy Practice

Create:

`week1_numpy_practice.ipynb`

Generate a random matrix:

```text
X.shape = (1000, 50)
```

Complete the following:

1. Calculate the mean of every feature.
2. Calculate the variance of every feature.
3. Normalize every feature.
4. Calculate the Euclidean distance between two samples.
5. Calculate cosine similarity between two samples.
6. Calculate matrix multiplication between two compatible matrices.
7. Select the first 100 samples without using a loop.
8. Find the sample with the largest L2 norm.

---

## Task 1B: Implement Basic Functions

Implement the following using NumPy:

```text
mean_squared_error()
accuracy()
sigmoid()
softmax()
cross_entropy()
```

You may compare your result with library implementations after finishing your own version.

---

# 4.2 Basic Machine Learning Concepts

Study the following concepts carefully.

## Supervised Learning

The training data contains:

```text
input x
+
label y
```

Example:

```text
image → cat
image → dog
```

The model learns a mapping:

\[
f(x) \rightarrow y
\]

---

## Unsupervised Learning

The data does not provide class labels during training.

Example:

```text
images → discover groups
```

Clustering is one form of unsupervised learning.

---

## Training, Validation, and Test Sets

Understand why data is divided into different sets.

### Training Set

Used to update model parameters.

### Validation Set

Used to make decisions about the model and training settings.

### Test Set

Used for final evaluation.

The test set should not be repeatedly used to tune the model.

---

## Features and Labels

Example:

For an image classification problem:

```text
image pixels = features
image category = label
```

Later, neural networks will learn new features automatically.

---

## Loss Function

The loss function measures how wrong the model is during training.

Examples include:

- mean squared error
- cross-entropy loss

Training attempts to reduce the loss.

---

## Optimization

A simplified training process is:

```text
Input
   ↓
Model
   ↓
Prediction
   ↓
Loss
   ↓
Gradient
   ↓
Update Parameters
```

You should understand this process before moving to PyTorch.

---

## Gradient Descent

Understand the basic idea:

\[
\theta_{new}
=
\theta_{old}
-
\eta \nabla_\theta L
\]

where:

- \(\theta\) represents model parameters
- \(L\) is the loss
- \(\nabla_\theta L\) is the gradient
- \(\eta\) is the learning rate

You do not need to derive every gradient manually.

You should understand what each term means.

---

## Overfitting

You should be able to explain this situation:

```text
Training accuracy: 99%
Validation accuracy: 75%
```

Think about:

- Why can this happen?
- Is higher training accuracy always better?
- What is generalization?

Use the sections on datasets, generalization, and overfitting from the Google Machine Learning Crash Course.

---

# Task 1C: Your First Machine Learning Experiment

Use the `sklearn.datasets.load_digits` dataset.

This is a small image dataset containing handwritten digits.

Use logistic regression from scikit-learn.

Complete:

```text
load data
↓
train/test split
↓
normalization
↓
train model
↓
predict
↓
calculate accuracy
↓
confusion matrix
```

You do not need to implement logistic regression yourself.

At this stage, focus on understanding the workflow.

---

## Week 1 Questions

You should be able to explain:

1. What is supervised learning?
2. What is unsupervised learning?
3. What is a feature?
4. What is a label?
5. What is the difference between training and testing?
6. What is a loss function?
7. What is gradient descent?
8. What is a learning rate?
9. What is overfitting?
10. Why do we need a validation set?

---

## Week 1 Deliverables

Submit:

```text
week1_numpy_practice.ipynb
week1_ml_basics.ipynb
```

Also write a one-page summary:

**“What happens when we train a machine learning model?”**

---

# 5. Week 2: PyTorch and Your First Neural Network

## Main Goal

Learn the PyTorch training pipeline and build an MLP for image classification.

---

# 5.1 PyTorch Fundamentals

Study:

- `torch.Tensor`
- tensor shape
- tensor dtype
- CPU and GPU
- `Dataset`
- `DataLoader`
- `nn.Module`
- `forward()`
- loss function
- optimizer
- automatic differentiation
- model parameters

Pay close attention to:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

You should understand why all three lines are needed.

---

# Task 2A: Tensor and Autograd Practice

Create:

`week2_pytorch_basics.ipynb`

Complete small exercises involving:

- creating tensors
- changing tensor shapes
- matrix multiplication
- moving tensors between devices
- creating tensors with gradients
- calculating a simple function
- calling `.backward()`
- inspecting `.grad`

For example, investigate:

\[
y = x^2 + 3x
\]

and compare the PyTorch gradient with the analytical derivative.

---

# 5.2 Neural Network Fundamentals

Before implementing the model, understand:

## Neuron

A simple neural unit performs something similar to:

\[
z = w^Tx+b
\]

followed by an activation function.

---

## Layer

A neural network contains multiple neurons arranged into layers.

---

## Activation Function

Learn:

- ReLU
- sigmoid
- softmax

Understand why a neural network needs nonlinear activation functions.

---

## Multilayer Perceptron

Your first neural network will look similar to:

```text
Image
 ↓
Flatten
 ↓
Linear Layer
 ↓
ReLU
 ↓
Linear Layer
 ↓
ReLU
 ↓
Output Layer
```

---

# Task 2B: Fashion-MNIST MLP

Use Fashion-MNIST.

Build a model similar to:

```text
784
↓
256
↓
ReLU
↓
128
↓
ReLU
↓
10
```

Use:

```text
CrossEntropyLoss
Adam or SGD
```

Record:

- training loss
- training accuracy
- test accuracy

Plot:

- training loss versus epoch
- test accuracy versus epoch

---

# Task 2C: Learning Rate Experiment

Train the same network using three learning rates.

For example:

```text
0.1
0.01
0.001
```

Do not change other settings.

Compare the training curves.

Answer:

1. What happens when the learning rate is too large?
2. What happens when it is too small?
3. Why should we change only one variable when running this experiment?

This is your first controlled machine learning experiment.

---

## Week 2 Questions

Explain:

- epoch
- batch
- mini-batch
- learning rate
- parameter
- gradient
- backpropagation
- optimizer
- activation function
- cross-entropy
- training mode
- evaluation mode

Also explain what happens during one training iteration:

```text
data
→ forward pass
→ prediction
→ loss
→ backward pass
→ parameter update
```

---

## Week 2 Deliverables

Submit:

```text
week2_pytorch_basics.ipynb
week2_mlp_fashion_mnist.ipynb
```

Include the learning-rate comparison.

---

# 6. Week 3: CNN, Image Classification, and Clustering

## Main Goal

Move from basic neural networks to image models and learn the difference between classification and clustering.

---

# 6.1 Convolutional Neural Networks

Study:

- convolution
- convolutional filters
- feature maps
- channels
- kernel size
- stride
- padding
- pooling

You do not need to understand advanced CNN architectures.

Focus on why CNNs work well with images.

---

# Task 3A: CNN Classification

Build a small CNN for Fashion-MNIST.

For example:

```text
Image
↓
Conv
↓
ReLU
↓
MaxPool
↓
Conv
↓
ReLU
↓
MaxPool
↓
Fully Connected
↓
10 Classes
```

Compare:

| Model | Test Accuracy |
|---|---:|
| MLP | |
| CNN | |

Answer:

1. Which model works better?
2. Why can convolution help with image data?
3. What does a convolutional filter learn?
4. What is a feature map?

---

# Task 3B: Kaggle Mini Project

Complete the Kaggle **Digit Recognizer** beginner competition.

[Kaggle Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer/overview)

The goal is **not** to obtain a very high leaderboard score.

Your goal is to complete a full image-classification workflow:

```text
data
↓
preprocessing
↓
training
↓
validation
↓
prediction
↓
submission
```

Use PyTorch and your own CNN.

You may read other solutions after making your first working version.

Submit at least one valid Kaggle result.

Record:

- network structure
- optimizer
- learning rate
- batch size
- epochs
- validation accuracy
- Kaggle score

---

# 6.2 Introduction to Clustering

Now move from supervised learning to unsupervised learning.

Study:

- clustering
- K-means
- cluster center
- Euclidean distance
- number of clusters \(K\)
- PCA

Use the scikit-learn clustering guide as a reference.

[scikit-learn Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html)

---

# Task 3C: Image Clustering

Use Fashion-MNIST.

Flatten every image:

```text
28 × 28
→
784-dimensional vector
```

Apply:

```text
K-means
K = 10
```

Then evaluate using:

- normalized mutual information (NMI)
- adjusted Rand index (ARI)

Do not expect clustering to match classification accuracy.

These are different problems.

---

# Task 3D: PCA + K-means

Run:

```text
Raw Pixels → K-means
```

and

```text
Raw Pixels → PCA → K-means
```

Try:

```text
PCA dimensions = 10, 20, 50
```

Create:

| Representation | Dimension | NMI | ARI |
|---|---:|---:|---:|
| Raw pixels | 784 | | |
| PCA | 50 | | |
| PCA | 20 | | |
| PCA | 10 | | |

---

## Week 3 Questions

Explain:

1. What is the difference between classification and clustering?
2. Why does clustering not use labels during training?
3. What does K mean in K-means?
4. What is a cluster center?
5. What does PCA do?
6. Why might raw pixels be poor features for clustering?
7. Why can dimensionality reduction help?
8. Why does cluster ID 0 not need to represent class 0?

---

## Week 3 Deliverables

Submit:

```text
week3_cnn.ipynb
week3_clustering.ipynb
kaggle_submission.csv
```

Record your Kaggle result.

---

# 7. Week 4: Representation Learning and Autoencoders

## Main Goal

Understand the idea of learning a representation instead of directly using raw pixels.

This week is very important for our later research.

---

# 7.1 What Is Representation Learning?

Suppose an image has:

```text
28 × 28 = 784 pixels
```

The raw 784 values may not be the best representation for learning.

Instead, we can learn:

\[
x \rightarrow z
\]

where:

- \(x\) is the original image
- \(z\) is a lower-dimensional latent representation

For example:

```text
784-dimensional image
→
20-dimensional latent representation
```

A good representation may contain useful information about:

- shape
- structure
- image category
- similarity between images

---

# 7.2 Autoencoder

An AE contains:

```text
Input
↓
Encoder
↓
Latent Representation
↓
Decoder
↓
Reconstruction
```

The encoder learns:

\[
z=f_\theta(x)
\]

The decoder learns:

\[
\hat{x}=g_\phi(z)
\]

Training tries to make:

\[
\hat{x} \approx x
\]

---

# Task 4A: Build an Autoencoder

Use Fashion-MNIST.

Start with a simple fully connected AE:

```text
784
↓
256
↓
64
↓
20
↓
64
↓
256
↓
784
```

Train it using reconstruction loss.

Plot:

- training loss
- original images
- reconstructed images

---

# Task 4B: Extract Latent Features

After training, do not use the decoder.

Pass the test images through only the encoder:

```text
Image
↓
Encoder
↓
Latent Representation
```

Save the latent representations.

If there are 10,000 images and your latent dimension is 20:

```text
Z.shape = (10000, 20)
```

---

# Task 4C: Clustering in the Latent Space

Apply K-means to:

### Method 1

Raw pixels

### Method 2

PCA features

### Method 3

AE latent features

Compare:

| Representation | Dimension | NMI | ARI |
|---|---:|---:|---:|
| Raw | 784 | | |
| PCA | 20 | | |
| AE | 20 | | |

---

# Task 4D: Latent Dimension Study

Train AEs using:

\[
d = 2,\;10,\;20,\;50
\]

Record:

- reconstruction loss
- NMI
- ARI

Think about the following question:

**Does better reconstruction always mean better clustering?**

Do not assume that the answer must be yes.

Use your results to answer.

---

# Task 4E: Visualization

For a two-dimensional latent space, directly visualize the latent features.

For higher-dimensional representations, use PCA for visualization.

Color the samples using ground-truth class labels.

Observe:

- whether similar classes are close
- whether clusters overlap
- which classes are hard to separate

---

## Week 4 Questions

Explain:

1. What is representation learning?
2. What is a latent representation?
3. What is an encoder?
4. What is a decoder?
5. What is reconstruction?
6. What is a bottleneck?
7. Why might AE features work better than raw pixels?
8. Can a good reconstruction model still have poor clustering features?
9. What happens when the latent dimension becomes very small?
10. What happens when it becomes very large?

---

## Week 4 Deliverables

Submit:

```text
week4_autoencoder.ipynb
week4_latent_clustering.ipynb
```

Also submit the AE comparison table.

---

# 8. Week 5: Variational Autoencoder and Final Mini Research Project

## Main Goal

Understand your first basic deep generative model and connect generation, representation learning, and clustering.

---

# 8.1 Variational Autoencoder

A VAE is related to an AE, but the latent representation is treated probabilistically.

Instead of directly producing one latent vector:

\[
z
\]

the encoder produces parameters such as:

\[
\mu
\]

and

\[
\log \sigma^2
\]

A latent variable is then sampled.

A simplified pipeline is:

```text
Image
↓
Encoder
↓
μ and log σ²
↓
Sample z
↓
Decoder
↓
Generated / Reconstructed Image
```

---

# 8.2 Concepts to Learn

Focus on understanding:

- latent variable
- probability distribution
- Gaussian distribution
- prior distribution
- posterior distribution at an introductory level
- reconstruction loss
- Kullback-Leibler divergence
- reparameterization trick
- sampling
- image generation

You do not need to understand the complete mathematical theory of variational inference in Week 5.

Your first goal is to understand what every part of the VAE is doing.

---

# PyTorch VAE Reference

After building your AE, read the official PyTorch VAE example.

[Official PyTorch VAE Example](https://github.com/pytorch/examples/blob/main/vae/main.py)

Do not copy the complete implementation without reading it.

Try to identify:

```text
encoder
mu
logvar
reparameterization
decoder
reconstruction loss
KL loss
training loop
sampling
```

---

# Task 5A: Implement a Basic VAE

Use MNIST first if Fashion-MNIST is difficult.

After the implementation works, run it on Fashion-MNIST.

Your model should support:

```text
encode(x)
reparameterize(mu, logvar)
decode(z)
forward(x)
```

---

# Task 5B: Reconstruction

Display:

```text
Original Image | VAE Reconstruction
```

Show at least 10 examples.

---

# Task 5C: Image Generation

Sample:

\[
z \sim N(0,I)
\]

Pass these samples through the decoder.

Generate at least 20 images.

Explain the difference between:

```text
reconstruction
```

and

```text
generation
```

---

# Task 5D: Latent Representation and Clustering

Extract latent representations.

Use the latent mean \(\mu\) as the representation for your first experiment.

Run K-means.

Compare:

| Method | Latent Dimension | NMI | ARI |
|---|---:|---:|---:|
| Raw Pixels | 784 | | |
| PCA | 20 | | |
| AE | 20 | | |
| VAE | 20 | | |

---

# Task 5E: Latent Space Visualization

If:

\[
d=2
\]

plot the latent space.

Color samples according to the true class.

Observe how different classes are distributed.

---

# Task 5F: Latent Interpolation

Select two images.

Encode them:

\[
x_1 \rightarrow z_1
\]

\[
x_2 \rightarrow z_2
\]

Create intermediate latent vectors:

\[
z(\alpha)
=
(1-\alpha)z_1+\alpha z_2
\]

for several values of \(\alpha\).

Decode the intermediate representations.

Display the image sequence.

Think about what this experiment tells you about the latent space.

---

# 9. Final Mini Research Project

## Title

**Learning Image Representations for Clustering with Autoencoders and Variational Autoencoders**

Use Fashion-MNIST as the main dataset.

Your main research question is:

> **How does the learned representation affect image clustering performance?**

Compare:

```text
Raw Pixels
      ↓
K-means
```

```text
PCA
 ↓
K-means
```

```text
AE Encoder
    ↓
Latent Features
    ↓
K-means
```

```text
VAE Encoder
     ↓
Latent Features
     ↓
K-means
```

---

# 10. Minimum Experimental Results

Your final report must contain at least:

| Representation | Dimension | NMI | ARI |
|---|---:|---:|---:|
| Raw Pixels | 784 | | |
| PCA | 20 | | |
| AE | 20 | | |
| VAE | 20 | | |

Also compare several latent dimensions:

\[
2,\;10,\;20,\;50
\]

---

# 11. Questions for the Final Analysis

Do not only report numbers.

Try to answer:

1. Does PCA improve clustering compared with raw pixels?
2. Does the AE improve clustering?
3. Does the VAE improve clustering?
4. How does latent dimension affect clustering?
5. How does latent dimension affect reconstruction?
6. Is lower reconstruction loss always related to higher NMI or ARI?
7. What differences do you observe between AE and VAE latent spaces?
8. Why might representation learning help clustering?
9. Which classes are difficult to separate?
10. What could be improved in the current models?

There is no requirement that AE or VAE must beat every baseline.

Unexpected or negative results are also useful if you can analyze them correctly.

---

# 12. Five-Week Schedule

| Week | Main Topics | Main Output |
|---|---|---|
| **Week 1** | Python, NumPy, ML basics, train/test split, loss, gradient descent, overfitting | NumPy notebook + basic ML experiment |
| **Week 2** | PyTorch, tensors, autograd, MLP, training loop | Fashion-MNIST MLP |
| **Week 3** | CNN, image classification, Kaggle, K-means, PCA | CNN + Kaggle submission + clustering |
| **Week 4** | Representation learning, AE, latent features | AE + latent clustering |
| **Week 5** | VAE, generation, latent representation, final comparison | VAE + final report + presentation |

---

# 13. Suggested Weekly Workflow

For each week:

### Step 1: Learn

Spend about 20–30% of your time reading or watching learning materials.

### Step 2: Implement

Spend about 40–50% of your time coding.

### Step 3: Experiment

Change one or two settings and observe what happens.

### Step 4: Analyze

Ask:

```text
What changed?
Why?
Was this expected?
```

### Step 5: Record

Keep your results instead of overwriting them.

---

# 14. Experiment Log

Starting from Week 2, maintain a table similar to:

| ID | Model | Dataset | LR | Batch | Epochs | Latent Dim. | Accuracy | NMI | ARI |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| E01 | MLP | Fashion-MNIST | 0.001 | 128 | 10 | N/A | | N/A | N/A |
| E02 | AE | Fashion-MNIST | 0.001 | 128 | 20 | 20 | N/A | | |
| E03 | VAE | Fashion-MNIST | 0.001 | 128 | 20 | 20 | N/A | | |

Also record the random seed.

This habit will become important when you start research.

---

# 15. Code Organization

Create a GitHub repository:

```text
research-preparation/
│
├── README.md
│
├── requirements.txt
│
├── notebooks/
│   ├── week1_numpy_practice.ipynb
│   ├── week1_ml_basics.ipynb
│   ├── week2_pytorch_basics.ipynb
│   ├── week2_mlp_fashion_mnist.ipynb
│   ├── week3_cnn.ipynb
│   ├── week3_clustering.ipynb
│   ├── week4_autoencoder.ipynb
│   └── week5_vae.ipynb
│
├── src/
│   ├── models.py
│   ├── train.py
│   ├── datasets.py
│   └── utils.py
│
├── results/
│   ├── figures/
│   └── experiments.csv
│
└── report/
    └── final_report.pdf
```

You do not need to create the full structure on Day 1.

Start with notebooks.

Move repeated code into `.py` files after you become comfortable with the workflow.

---

# 16. Final Deliverables

At the end of Week 5, submit the following.

## A. GitHub Repository

The repository should contain:

- all notebooks
- reusable Python code
- README
- experiment results
- figures

---

## B. Final Report

Length:

**approximately 4–6 pages**

Suggested sections:

### 1. Problem

What problem are you studying?

### 2. Dataset

What dataset did you use?

### 3. Methods

Briefly introduce:

- PCA
- K-means
- AE
- VAE

### 4. Experimental Setup

Report:

- architecture
- optimizer
- learning rate
- batch size
- number of epochs
- latent dimension

### 5. Results

Include tables and figures.

### 6. Analysis

Explain the results rather than only listing numbers.

### 7. Problems and Lessons

Describe:

- bugs
- failed experiments
- unexpected results
- what you learned

---

## C. Short Presentation

Prepare approximately **8–10 slides**.

Include:

1. problem
2. dataset
3. ML pipeline
4. AE
5. VAE
6. experiment setup
7. classification/clustering results
8. latent-space visualization
9. main observations
10. questions and possible next steps

---

# 17. Questions You Should Be Able to Answer After Five Weeks

## Machine Learning

What is supervised learning?

What is unsupervised learning?

What is the difference between classification and clustering?

What are training, validation, and test sets?

What is a loss function?

What is gradient descent?

What is overfitting?

What is generalization?

---

## PyTorch

What is a tensor?

What is `Dataset`?

What is `DataLoader`?

What does `nn.Module` do?

What happens during `forward()`?

What does `loss.backward()` do?

Why do we call `optimizer.zero_grad()`?

What does `optimizer.step()` do?

What is the difference between `model.train()` and `model.eval()`?

---

## Deep Learning

What is a neuron?

What is an activation function?

What is ReLU?

What is an MLP?

What is convolution?

Why does a CNN work well with image data?

What is a feature map?

---

## Representation Learning

What is a representation?

What is a latent representation?

Why may raw pixels be poor representations?

What is dimensionality reduction?

What is PCA?

Why can learned representations help clustering?

---

## Autoencoders

What is an encoder?

What is a decoder?

What is reconstruction loss?

What is a bottleneck?

What information should the latent representation contain?

---

## Variational Autoencoders

What is the difference between an AE and a VAE?

Why does the encoder output \(\mu\) and \(\log\sigma^2\)?

What is the role of the prior?

What does the KL term do?

Why do we need the reparameterization trick?

How can a VAE generate an image that was not in the training set?

---

# 18. Rules for Using Online Code and AI Tools

You may use:

- documentation
- tutorials
- GitHub
- Stack Overflow
- AI tools

for learning and debugging.

However, copying a complete implementation and making it run is not considered completion of the assignment.

For every important part of your code, you should be able to answer:

```text
What does this code do?

Why is it needed?

What is the input?

What is the output?

What would happen if I removed it?

What would happen if I changed it?
```

If you cannot answer these questions, continue studying that part.

---

# 19. Research Preparation Standard

At the end of these five weeks, you do not need to be an expert in machine learning or generative models.

You should, however, be able to independently:

```text
read a dataset
↓
build a model
↓
train it
↓
debug basic problems
↓
evaluate it
↓
visualize the results
↓
compare experiments
↓
explain what you observed
```

Once you can complete this workflow, we can move from tutorial-style tasks to a real research project involving **image representation learning, clustering, and deep generative models**.