# ClearVision - Image Restoration with GANs

This repository provides:

* ✅ A ready-to-run **Streamlit application** for image deblurring and restoration
* ✅ Full **training pipeline** using a U-Net Generator and PatchGAN Discriminator
* ✅ Example **evaluation scripts** and image quality metrics
* ✅ Easy dataset replacement to train on your own images

---

##  Repository Structure

```
.
├── models/                # Files to load the model for the Streamlit Application
├── training_data/         # Training and evaluation related code
│   ├── data/mixed_dataset # Add your own clean dataset images here
│   ├── models/            # Has different model definitions like VAE , GANS ,etc.
│   ├── sample results/    # Example generated results from training
│   ├── utils/             # Dataset classes and helper functions
│   ├── evaluate.py        # Compute PSNR, SSIM, LPIPS metrics on results
│   ├── splitfor256.py     # Utility to split grid images into individual samples for evaluation
│   └── train256.py        # Main GAN training script
├── utils/img_uti.py       # Image preprocessing/postprocessing for app
├── app.py                 # Streamlit web app for real-time image restoration
└── README.md              # Project documentation (this file)
```

---

##  App Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

This will host a web interface where you can upload degraded images (blurred or noisy) and get restored results using your trained model.

---

##  Training Your Own Model

1. Add your clean, high-quality dataset to:

```
training_data/data/mixed_dataset/
```

2. Start training:

```bash
cd training_data
python train256.py
```

* Each epoch's generator and discriminator models are saved automatically in the `saved_models/` directory.

3. Evaluate performance:

```bash
python evaluate.py
```

The evaluation script computes:

* **PSNR** : Peak Signal-to-Noise Ratio
* **SSIM** : Structural Similarity Index
* **LPIPS** : Learned Perceptual Image Patch Similarity

---

##  Notes

* Currently focused on **blur** and **Gaussian noise** restoration tasks
* Designed for **256x256 images**, adjust `transforms.Resize()` if working with other sizes
* Easily extendable for more corruption types in the dataset class

---

**Contributions and improvements welcome!**
