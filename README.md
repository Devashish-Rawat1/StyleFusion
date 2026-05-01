# StyleFusion

A web-based Neural Style Transfer application built with Flask and PyTorch. StyleFusion applies the artistic style of any reference image to a content photograph using Adaptive Instance Normalization (AdaIN), enabling real-time, arbitrary style transfer without retraining for each style.

---

## Overview

StyleFusion implements the AdaIN architecture introduced by Huang and Belongie (2017) in *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization*. The model separates content and style representations using a pretrained VGG encoder, aligns the statistical moments of the content feature maps to those of the style image via AdaIN, and reconstructs the output through a learned decoder network.

The application exposes this pipeline through a Flask web interface that accepts user-uploaded images and returns the stylized result, with a configurable alpha parameter to blend between the original content and the fully stylized output.

---

## Repository Structure

```
StyleFusion/                    # Core model and application code
│   ├── app.py                  # Flask application entry point
│   ├── models.py               # VGGEncoder and Decoder architecture
│   ├── utils.py                # AdaIN operation, transforms, dataset loader
│   ├── forms.py                # WTForms definitions for image upload
│   └── templates/
│       └── index.html          # Jinja2 frontend template
├── Demo_IO_Images/             # Example content, style, and output images
├── Research Papers Summary/    # Summaries of key NST research papers
├── code.ipynb                  # Experimentation and prototyping notebook
├── kaggle_train.py             # Two-phase training script (Kaggle-ready)
├── requirements.txt            # Python dependencies
├── Procfile.txt                # Gunicorn process definition for deployment
└── .gitignore
```

---

## Architecture

**Encoder** — A VGG-19 network pretrained on ImageNet, truncated at `relu4_1`, used to extract multi-scale feature representations. Weights are loaded from `vgg_normalised.pth` and kept frozen throughout training.

**AdaIN** — Adaptive Instance Normalization aligns the channel-wise mean and standard deviation of the content features to match those of the style features:

```
AdaIN(x, y) = sigma(y) * ((x - mu(x)) / sigma(x)) + mu(y)
```

**Decoder** — A mirror of the encoder (without pooling layers), trained from scratch to invert the AdaIN-transformed feature maps back into pixel space.

**Loss** — Content loss is computed as MSE between the decoded output features and the AdaIN target at `relu4_1`. Style loss is computed as MSE between the channel-wise means and standard deviations of the output and style features across all four VGG relu layers.

![AdaIN Style Transfer Architecture](adain_algo.png)
*Figure: An overview of the style transfer algorithm. A fixed VGG-19 encoder encodes both content and style images. The AdaIN layer aligns feature statistics in latent space. A learned decoder inverts the result back to pixel space. The same VGG encoder is reused to compute content loss (𝓛c) and style loss (𝓛s).*

---

## Training

Training was conducted on Kaggle using a two-phase curriculum to stabilize learning and improve high-resolution output quality.

**Phase 1**
- Epochs: 160
- Image size: 256 x 256
- Style weight: 5
- Batch size: 4

**Phase 2** (resumed from Phase 1 checkpoint)
- Epochs: 100
- Image size: 512 x 512
- Style weight: 10
- Batch size: 4

Both phases use Adam optimization with a learning rate of `1e-4` and a decay schedule of `lr / (1 + decay * epoch)`.

To run training:

```bash
# Edit the CONFIG block at the top of kaggle_train.py
# Set DATASET_PATH, CONTENT_DIR, STYLE_DIR, VGG_PATH, SAVE_DIR

python kaggle_train.py
```

Checkpoints for both the decoder weights and optimizer state are saved after every epoch. Sample output grids (content | style | stylized) are also saved for visual monitoring.

---

## Local Setup

**Requirements:** Python 3.9+, CUDA-capable GPU recommended

```bash
# Clone the repository
git clone https://github.com/Devashish-Rawat1/StyleFusion.git
cd StyleFusion

# Install dependencies
pip install -r requirements.txt
```

Place your trained decoder weights inside `NST_Code/` and ensure the VGG normalised weights are also accessible. Update any model path references in `app.py` accordingly.

```bash
# Run the development server
cd NST_Code
python app.py
```

The application will be available at `http://localhost:5000`.

---

## Dependencies

| Package | Version |
|---|---|
| Flask | 3.1.2 |
| Flask-WTF | 1.2.2 |
| Flask-Bootstrap | 3.3.7.1 |
| torch | 2.2.2 |
| torchvision | 0.17.2 |
| Pillow | 12.0.0 |
| numpy | >=1.24, <2.0 |
| Werkzeug | 3.1.4 |
| WTForms | 3.2.1 |
| gunicorn | latest |
| tqdm | 4.66.4 |

---

## Usage

1. Open the application in a browser.
2. Upload a **content image** — the photograph or image whose structure you want to preserve.
3. Upload a **style image** — the artwork whose visual texture and brushwork you want to apply.
4. Adjust the **alpha slider** to control the degree of stylization. A value of `1.0` applies the full style; lower values blend the output toward the original content.
5. Submit the form. The stylized image will be displayed and can be downloaded.

---

## Deployment

> ⚠️ **Important Note:** This Flask application **cannot be deployed on cloud service providers such as Render, Railway, or Heroku** due to the large memory and compute requirements of the PyTorch model. These free-tier platforms do not provide sufficient RAM or GPU access to run inference reliably.
>
> **For a live hosted demo, use the Hugging Face Spaces deployment instead:**
> 👉 [StyleFusion on Hugging Face Spaces](https://github.com/Devashish-Rawat1/StyleFusion-HF)

This repository includes a `Procfile.txt` configured for Gunicorn, intended for local or private server deployment only:

```
web: gunicorn --bind :$PORT app:app
```

---

## Research References

The `Research Papers Summary/` directory contains notes on the following papers:

- Gatys et al. (2015) — *A Neural Algorithm of Artistic Style* — the original optimization-based NST approach that this work builds upon.
- Johnson et al. (2016) — *Perceptual Losses for Real-Time Style Transfer* — introduced feed-forward networks for fast per-style transfer.
- Huang & Belongie (2017) — *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization* — the primary paper implemented in this project.

---

## Author

Devashish Rawat  
[GitHub](https://github.com/Devashish-Rawat1) · [Instagram](https://www.instagram.com/devashish__rawat)

---

## License

This project is released for academic and personal use. The VGG weights used for the encoder are subject to their original license terms from the Visual Geometry Group, University of Oxford.
