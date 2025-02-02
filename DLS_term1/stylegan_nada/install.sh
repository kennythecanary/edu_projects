#!/bin/bash
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
gdown --folder --remaining-ok https://drive.google.com/drive/folders/1QXnHIoc_dmgzSdZQmQJ1I3QNOrRd7CSD
