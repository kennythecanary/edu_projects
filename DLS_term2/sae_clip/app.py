import streamlit as st
from model import *
from func import *
from metrics import *
import datasets
import numpy as np
import scipy as sp
import pandas as pd
from PIL import Image
import json



@st.cache_data
def fetch_processor(clip_model_name: str):
    return CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", use_fast=False)
    

@st.cache_data
def fetch_model(**kwargs):
    chpt_path = kwargs.pop("chpt_path")
    model = SAEonCLIP(**kwargs)
    model.sae.load_state_dict(torch.load(chpt_path))
    model.eval()
    return model


@st.cache_data
def fetch_dataset(dataset: str):
    return datasets.load_dataset(dataset, split="validation")


@st.cache_data
def fetch_image_activations(path: str):
    return sp.sparse.load_npz(path)


str2int = lambda x: ds.features["label"].names.index(x)
int2str = lambda x: ds.features["label"].names[x]



st.title("PatchSAE on CLIP ViT-B/16")

ds = fetch_dataset("benjamin-paine/imagenet-1k-256x256")

with open("models.json", "r") as inf:
    models = json.load(inf)

choice = st.radio("Model:", list(models.keys()))
params = models[choice]
DATA_DIR = params.pop("data_path")



tab_1, tab_2, tab_3 = st.tabs(["Image-level activations", "Patch-level activations", "Image Explorer"])


with tab_1:
    image_neurons = torch.load(f"{DATA_DIR}/image_neurons.pt")

    descending = st.checkbox("Descending", value=True)
    neurons, counts = image_neurons[:,1].unique(return_counts=True)
    neurons = neurons[counts.argsort(descending=descending)]
    counts = counts[counts.argsort(descending=descending)]
    neuron_selection = ["{} ({})".format(i[0].item(), i[1].item()) for i in zip(neurons, counts)]
        
    neuron = st.selectbox(
        "Neuron (number of images):", 
        neuron_selection,
        index=0
    )
    neuron = int(neuron.split(" ")[0])
    image_ids = image_neurons[image_neurons[:,1] == neuron][:,0]
        
    image_activations = fetch_image_activations(f"{DATA_DIR}/image_activations.npz")
    neuron_activations = torch.tensor(image_activations[image_ids.tolist(), neuron].toarray())
    
    size = len(neuron_activations)
    k = 16
    if size > 16:
        thr = st.slider(
            "TopK ceiling",
            min_value=1,
            max_value=size,
            value=size,
        )
        k += size - thr
    try:
        indices = image_ids[neuron_activations.squeeze(1).topk(k).indices]
    
    except RuntimeError as e:
        indices = image_ids[neuron_activations.squeeze(1).argsort()]
    
    batch = ds.select(indices[-16:])
    grid = make_grid(batch, nrow=4)

    st.subheader("Top highest activating images")
    st.image(grid)
    
    batch_activations = torch.tensor(image_activations[indices.tolist(), :].toarray())
    
    metadata = pd.DataFrame({
        "Log10 sparsity": [compute_sparsity(batch_activations).mean().log10().item()],
        "Mean activation value": [batch_activations.mean().item()],
        "Label entropy": [compute_label_entropy(batch_activations).sum().item()]
    })
    st.subheader("Summary statistics")

    styler = metadata.style.hide()
    st.write(styler.to_html(), unsafe_allow_html=True)    


with tab_2:
    class_label = st.selectbox(
        "Class Label:", 
        ds.features["label"].names[:-1],
        index=str2int("tabby, tabby cat")
    )
    image_ids = torch.where(torch.tensor(ds["label"]) == str2int(class_label))[0].tolist()
    image_id = st.select_slider(
        "Image ID:", 
        image_ids,
    )
    try:
        image = ds[image_id]["image"]
        H, W, _ = np.array(image).shape
        
        col_1, col_2 = st.columns(2)
        with col_1:
            st.image(image)
        
        inputs = ds.select([image_id])[0]
        inputs["label"] = int2str(inputs["label"])
        
        processor = fetch_processor("openai/clip-vit-base-patch16")
        masked_image = im_process(processor, inputs)
        
        top_neurons = torch.load(f"{DATA_DIR}/token_top_neurons.pt")[str2int(class_label)]
        
        top1_neurons = st.segmented_control(
            "Top-1 neurons:", 
            top_neurons[1:,0].unique().tolist(),
            selection_mode="multi"
        )
        top2_neurons = st.segmented_control(
            "Top-2 neurons:", 
            top_neurons[1:,1].unique().tolist(),
            selection_mode="multi"
        )
        top3_neurons = st.segmented_control(
            "Top-3 neurons:", 
            top_neurons[1:,2].unique().tolist(),
            selection_mode="multi"
        )
        for neuron in top1_neurons:
            masked_image = mask_image(processor, masked_image, top_neurons, k=1, neuron=neuron)
        for neuron in top2_neurons:
            masked_image = mask_image(processor, masked_image, top_neurons, k=2, neuron=neuron)
        for neuron in top3_neurons:
            masked_image = mask_image(processor, masked_image, top_neurons, k=3, neuron=neuron)
                
        with col_2:
            st.image(masked_image[0].permute(1,2,0).numpy(), width=W)

        activations_on = st.toggle("SAE activations")
        
        if activations_on:
            image_activations = fetch_image_activations(f"{DATA_DIR}/image_activations.npz")
            activations = image_activations[image_id,:].toarray()
            
            st.bar_chart(
                pd.DataFrame({"activation_value": activations[0]}).reset_index(), 
                x="index", 
                y="activation_value", 
                x_label="SAE Latent Index", 
                y_label="Activation Value"
            )
    
    except TypeError as e:
        pass
        

with tab_3:
    image = st.file_uploader("Upload an image", type="jpg")
    
    if image is not None:
        st.image(image)
        inputs = {"label": "none", "image": Image.open(image)}

        device_id = 1 if torch.cuda.is_available() else 0
        device = st.radio("Device:", ["cpu", "cuda"], index=device_id, horizontal=True)
        model = fetch_model(**params, device=device)
        with torch.no_grad():
            representation, activations, reconstruction = model.encode(inputs)
        
        st.bar_chart(
            pd.DataFrame({"activation_value": activations.mean(1)[0].detach().cpu()}).reset_index(), 
            x="index", 
            y="activation_value", 
            x_label="SAE Latent Index", 
            y_label="Activation Value"
        )