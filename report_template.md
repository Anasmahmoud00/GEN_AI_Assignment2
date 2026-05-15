# Report: Dates Generator (Assignment 2)
**Name:** [Your Name]
**ID:** [Your ID]

## 1. Problem Formulation
- **Objective:** Generate a date `dd-mm-yyyy` that satisfies four conditions: Day of week, Month, Leap year status, and Decade.
- **Input Encoding:** 22-dimensional vector (One-hot for day, month, leap; Normalised decade).
- **Output Encoding:** 3-dimensional vector (Normalised day, month, year).
- **Metric:** Condition Satisfaction Rate (CSR) — Accuracy is not suitable for this generative task.

## 2. Methodology (Models Implemented)

### Model 1: GAN (Course)
- **Type:** Conditional GAN.
- **Architecture:** MLP-based Generator and Discriminator.
- **Logic:** Generator creates dates from noise, Discriminator tries to tell real from fake.

### Model 2: VAE (Course)
- **Type:** Conditional VAE.
- **Architecture:** Encoder maps date to latent space; Decoder reconstructs it.
- **Loss:** MSE (Reconstruction) + KL Divergence.

### Model 3: MLP (Outside Course)
- **Type:** Supervised Regressor.
- **Logic:** Direct mapping from conditions to date using a deep neural network.

### Model 4: Normalizing Flow / RealNVP (Outside Course)
- **Type:** Invertible Generative Model.
- **Logic:** Learns a bijective mapping between Gaussian noise and the date distribution.

## 3. Analysis of Results
[After training, look at the plots in /plots/ and describe which model performed better.]

### Loss Curves
- GAN: [plots/gan_loss.png]
- VAE: [plots/vae_loss.png]
- MLP: [plots/mlp_loss.png]
- Flow: [plots/flow_loss.png]

## 4. Reflections & Conclusion
- **Successes:** [Which model was easiest to train?]
- **Failures:** [Describe any cases where the date didn't match the decade or weekday.]
- **Originality:** All code implemented from scratch using `tf.GradientTape()`.
