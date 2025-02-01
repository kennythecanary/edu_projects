# DLS Final Project 2024 | StyleGAN-NADA Reimplementation

Optimization: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KN9XO7adMwPIhdkBcBQfHbKaMoehfwZP?usp=sharing)

### Based on
1\. [StyleGAN2-ADA - Official PyTorch implementation](https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main)<br>
2\. [encoder4editing](https://github.com/omertov/encoder4editing)<br>
3\. [StyleGAN-NADA: CLIP-Guided Domain Adaptation of Image Generators](https://arxiv.org/pdf/2108.00946)<br>


### Setup
```.bash
git init stylegan
cd stylegan
git remote add -f origin https://github.com/kennythecanary/edu_projects.git
git config core.sparseCheckout true
echo "DLS_term1/stylegan_nada" >> .git/info/sparse-checkout
git pull origin main
cd DLS_term1/stylegan_nada
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
git clone https://github.com/omertov/encoder4editing.git
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
gdown 1O8OLrVNOItOJoNGMyQ8G8YRTeTYEfs0P
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2
rm -f shape_predictor_68_face_landmarks.dat.bz2
```

### Usage
```.bash
python app.py \
    --image_path /path/to/ffhq_dataset/dir
    --source "photo" \
    --target "sketch" \
    --batch_size 4 \
    --num_steps 10 \
    --lr 0.002 \
    --fmin_steps 5 \
    --fmin_evals 10 \
    --patience 5 \
    --z_samples 3 \
    --outdir /path/to/output/dir \
    --save_freq 1
```