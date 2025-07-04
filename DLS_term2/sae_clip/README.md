# DLS Final Project 2025 | SAE for CLIP

Stepic ID: 45681991

The goal of this project is to develop and train a Sparse Autoencoder (SEA) based on the CLIP model. It is used a model with an image encoder of ViT-B/16 as input and repack of the ImageNet-1k dataset for training. The app demonstrates some results.

<p align="center">
    <img src=teaser.png />
</p>



### Setup
```.bash
git init final_project
cd final_project
git remote add -f origin https://github.com/kennythecanary/edu_projects.git
git config core.sparseCheckout true
echo "DLS_term2/sae_clip" >> .git/info/sparse-checkout
git pull origin main
cd DLS_term2/sae_clip
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
chmod u+x fetch_models.sh
./fetch_models.sh
```

### Usage
```.bash
streamlit run app.py
```