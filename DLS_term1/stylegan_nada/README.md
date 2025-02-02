# DLS Final Project 2024 | StyleGAN-NADA Reimplementation

Optimize and convert: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KN9XO7adMwPIhdkBcBQfHbKaMoehfwZP?usp=sharing)

### Based on
1\. [StyleGAN2-ADA - Official PyTorch implementation](https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main)<br>
2\. [encoder4editing](https://github.com/omertov/encoder4editing)<br>
3\. [StyleGAN-NADA: CLIP-Guided Domain Adaptation of Image Generators](https://arxiv.org/pdf/2108.00946)<br>


### Setup
```.bash
git init final_project
cd final_project
git remote add -f origin https://github.com/kennythecanary/edu_projects.git
git config core.sparseCheckout true
echo "DLS_term1/stylegan_nada" >> .git/info/sparse-checkout
git pull origin main
cd DLS_term1/stylegan_nada
chmod u+x install.sh
./install.sh
```

### Usage via Console
```.bash
python app.py \
    --image_path 02000 \
    --source "photo" \
    --target "sketch" \
    --n_samples 30 \
    --batch_size 4 \
    --num_steps 10 \
    --lr 0.002 \
    --fmin_steps 5 \
    --fmin_evals 10 \
    --patience 5 \
    --z_samples 3 \
    --outdir /tmp/frames \
    --save_freq 1
```

### Usage via Local GUI
```.bash
python webapp.py
```