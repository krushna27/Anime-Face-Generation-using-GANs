# Anime-Face-Generation-using-GANs
The goal of this project was to design and train a Generative Adversarial Network (GAN) capable of generating high-quality anime face images. Using the publicly available Anime Face Dataset from Kaggle, the model was trained to learn the underlying patterns of anime-style facial features and synthesize novel, realistic-looking faces.

 #  Dataset

- **Source**: [Kaggle - Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset)
- **Format**: 64×64 JPG images of anime-style faces
- **Usage**:
  - Images are resized to 64×64
  - Normalized to range `[-1, 1]` as required by the GAN model

GAN Architecture

### Generator
- Input: 100-dimensional latent noise vector
- Layers:
  - Dense → BatchNorm → ReLU
  - 3x Conv2DTranspose layers (upsample to 64×64)
  - Output: Tanh activation

### Discriminator
- Input: 64×64×3 RGB image
- Layers:
  - 3x Conv2D layers → LeakyReLU → Dropout
  - Flatten → Dense output (logit)
  - Output: No activation (from_logits = True)

### Training Details

- **Epochs**: 30
- **Batch Size**: 256
- **Optimizer**: Adam (lr = 1e-4, β₁ = 0.5)
- **Loss Function**: Binary Crossentropy
- **Noise Vector Size**: 100
- **Checkpoints**: Every 5 epochs
- **Output**: Generates sample images at each epoch and saves them to disk
