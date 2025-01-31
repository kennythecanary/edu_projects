import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms
import clip
import dlib
import pickle
from hyperopt import hp, fmin, tpe, STATUS_OK, early_stop
from functools import partial
import click
import os
import sys
sys.path.append("stylegan2-ada-pytorch")
sys.path.append("encoder4editing")
from utils.alignment import align_face
from utils.common import tensor2im
from models.psp import pSp
from argparse import Namespace
import numpy as np
from PIL import Image
from IPython.display import clear_output
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from tqdm.auto import tqdm
device = "cuda"



class GlobalCLIPLoss(torch.nn.Module):
    def __init__(self, model, stylegan_size=1024):
        super(GlobalCLIPLoss, self).__init__()
        self.model = model
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=stylegan_size//32)
        

    def forward(self, image, text):
        image = self.avg_pool(self.upsample(image))
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity



class DirectionalCLIPLoss(torch.nn.Module):
    def __init__(self, model, source_txt, target_txt, stylegan_size=1024):
        super(DirectionalCLIPLoss, self).__init__()
        self.model = model
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=stylegan_size//32)
        self.cos = torch.nn.CosineSimilarity()

        source_emb = self.model.encode_text(source_txt.to(device))
        target_emb = self.model.encode_text(target_txt.to(device))
        self.txt_direction = target_emb - source_emb


    def forward(self, source_img, target_img):
        source_img = self.avg_pool(self.upsample(source_img.to(device)))
        source_emb = self.model.encode_image(source_img)
        
        target_img = self.avg_pool(self.upsample(target_img.to(device)))
        target_emb = self.model.encode_image(target_img)
        
        img_direction = target_emb - source_emb
        similarity = 1 - self.cos(img_direction, self.txt_direction)
        return similarity



class FFHQDs(Dataset):
    def __init__(self, image_dir, n_samples=None, align=True):
        super().__init__()
        self.image_dir = image_dir
        files = [f for f in sorted(os.listdir(image_dir))]
        n_samples = len(files) if n_samples is None else n_samples
        self.files = files[:n_samples]
        self.len_ = len(self.files)
        self.align = align
        
    
    def __len__(self):
        return self.len_

    
    def _align_face(self, image_path):
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        return align_face(filepath=image_path, predictor=predictor)


    def __getitem__(self, index):
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        if self.align:
            aligned_img = self._align_face(os.path.join(self.image_dir, self.files[index]))
        else:
            aligned_img = Image.open(os.path.join(self.image_dir, self.files[index]))
            aligned_img = aligned_img.convert("RGB")
            
        return transform(aligned_img)



class MyStyleGANa:
    def __init__(self, ckpt_path="G_ckpt.pt"):
        encoder_path = "e4e_ffhq_encode.pt"
        ckpt = torch.load(encoder_path, map_location="cpu")
        opts = ckpt["opts"]
        opts["checkpoint_path"] = encoder_path
        opts = Namespace(**opts)
        self.encoder = pSp(opts)
        self.encoder.eval()
        self.encoder.to(device)
        
        print("Loading G_ema from checkpoint: ffhq.pkl")
        with open('ffhq.pkl', 'rb') as f:
            self.G = pickle.load(f)['G_ema'].to(device)
            
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.space = self._get_hyperopt_space()
        self.ckpt_path = ckpt_path


    def encode_images(self, inputs):
        transform = transforms.Compose([transforms.Resize((256, 256))])
        inputs = transform(inputs)
        images, latents = self.encoder(inputs.to(device), randomize_noise=False, return_latents=True)
        return latents


    def _get_hyperopt_space(self):
        modules = []
        for child in self.G.synthesis.named_children():
            for subchild in child[1].named_children():
                if subchild[0] != "torgb":
                    modules.append(".".join([child[0], subchild[0]]))

        space = {module: hp.choice(module, [False, True]) for module in modules}
        return space


    def _unfreeze_layers(self, model, layers):
        for param in model.parameters():
            param.requires_grad = False

        for child in model.synthesis.named_children():
            for subchild in child[1].named_children():
                if ".".join([child[0], subchild[0]]) in layers:
                    for param in subchild[1].parameters():
                        param.requires_grad = True
                    for param in subchild[1].affine.parameters():
                        param.requires_grad = False
        return model


    def _hyperopt_objective(self, params, target, z_samples, lr, fmin_steps):
        """ Function to minimize at the layer selection phase """
        running_loss = 0.
        
        for _ in range(z_samples):
            conv_blocks = [k for k, v in params.items() if v]
            self.G = self._unfreeze_layers(self.G, conv_blocks)
    
            z = torch.randn([1, self.G.z_dim], device=device, requires_grad=True)
            c = None
            criterion = GlobalCLIPLoss(self.clip_model)
            optimizer = optim.Adam([z], lr=lr)
    
            for step in range(fmin_steps):
                generated_img = self.G(z, c)
                loss = criterion(generated_img, target)
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
            
            self.G.load_state_dict(torch.load(self.ckpt_path, weights_only=True))
            running_loss += loss.item()
        
        return {'loss': running_loss / z_samples, 
                'status': STATUS_OK}
        
        
    def _fit_epoch(self, train_loader, criterion, optimizer):
        running_loss = 0.
        
        for images in tqdm(train_loader):
            with torch.no_grad():
                w = self.encode_images(images).requires_grad_()
            
            optimizer.zero_grad()
            generated_img = self.G.synthesis(w, noise_mode="const", force_fp32=True)
            loss = criterion(images, generated_img).mean()
            loss.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss.item() * len(images)
            torch.cuda.empty_cache()
        
        return running_loss / len(train_loader)


    def train(self, train_loader, source, target, num_steps=10, lr=0.01, fmin_steps=5, fmin_evals=10, patience=5, z_samples=3, 
        display=True, outdir=None, save_freq=1,
    ):
        source, target = [torch.cat([clip.tokenize(prompt)]).to(device) for prompt in [source, target]]
        
        criterion = DirectionalCLIPLoss(self.clip_model, source, target)
        optimizer = optim.Adam(self.G.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        if display:
            source_img = next(iter(train_loader))[0].detach().cpu()
        
        for step in range(num_steps):
            fmin_objective = partial(
                self._hyperopt_objective, 
                target=target, 
                z_samples=z_samples, 
                lr=lr, 
                fmin_steps=fmin_steps
            )
            torch.save(self.G.state_dict(), self.ckpt_path)
            best = fmin(fmin_objective, space=self.space, algo=tpe.suggest, max_evals=fmin_evals, 
                early_stop_fn=early_stop.no_progress_loss(patience)
            )
            most_relevant_layers = [k for k,v in best.items() if v]
            self.G = self._unfreeze_layers(self.G, most_relevant_layers)
            
            dc_loss = self._fit_epoch(train_loader, criterion, optimizer)
            scheduler.step()
            
            output = "Step [%d/%d] Unfreezed: %d > Loss: %f" % (step+1, num_steps, len(most_relevant_layers), dc_loss)
            if display:
                self._display_output(source_img)
                plt.axis("off")
                plt.title(output)
                plt.show()
            else:
                print(output)

            if outdir is not None and not step % save_freq:
                self._save_images(train_loader, step, outdir)


    def _concat_images(self, result_image, source_image, resize_dims=(256, 256)):
        res = np.concatenate([np.array(result_image.resize(resize_dims)),
                              np.array(source_image.resize(resize_dims))], axis=1)
        return Image.fromarray(res)

    
    def _display_output(self, image):
        w = self.encode_images(image.unsqueeze(0))
        generated_img = self.G.synthesis(w, noise_mode="const", force_fp32=True).detach().cpu().squeeze(0)
        display_img = self._concat_images(tensor2im(generated_img), tensor2im(image))
        clear_output(wait=True)
        plt.imshow(display_img)


    def _save_images(self, train_loader, step, outdir):
        for d, images in enumerate(tqdm(train_loader, desc="Saving images", leave=False)):
            with torch.no_grad():
                w = self.encode_images(images)
            generated_img = self.G.synthesis(w, noise_mode="const", force_fp32=True)
            for i, image in enumerate(generated_img):
                path = os.path.join(outdir, "{:05d}".format(d))
                os.makedirs(path, exist_ok=True)
                image = tensor2im(image)
                f = os.path.join(path, "{:03d}_{:03d}.png".format(i, step))
                image.save(f)



@click.command()
@click.option("--image_path", type=str, help="Path to input images", metavar="DIR", required=True)
@click.option("--source", type=str, help="Source domain text", required=True)
@click.option("--target", type=str, help="Target domain text", required=True)
@click.option("--n_samples", type=int, help="Number of training examples", default=None)
@click.option("--batch_size", type=int, help="Batch size", default=4)
@click.option("--num_steps", type=int, help="Number of iterations at the 2nd phase", default=10)
@click.option("--lr", type=float, help="Learning rate", default=0.01)
@click.option("--fmin_steps", type=int, help="Number of iterations at the 1st phase", default=5)
@click.option("--fmin_evals", type=int, help="Maximum of evaluations at the 1st phase", default=10)
@click.option("--patience", type=int, help="Early stopping rounds at the 1st phase", default=5)
@click.option("--z_samples", type=int, help="Number of samples at the 1st phase", default=3)
@click.option("--display", type=bool, help="Display generated and source images during training", default=False)
@click.option("--outdir", type=str, help="Path to output images", metavar="DIR", default=None)
@click.option("--save_freq", type=int, help="How often to save output images", default=1)
@click.option("--ckpt_path", type=str, help="Checkpoint path", default="G_ckpt.pt")

def generate(
	image_path, source, target, n_samples, batch_size, num_steps, lr, fmin_steps, fmin_evals, patience, 
	z_samples, display, outdir, save_freq, ckpt_path
):
	""" Generate images based on source image and text domains """
	dataset = FFHQDs(image_path, n_samples)
	dataloader = DataLoader(dataset, batch_size=batch_size)
	
	model = MyStyleGANa(ckpt_path)
	model.train(dataloader, source, target, num_steps, lr, fmin_steps, fmin_evals, patience, z_samples, 
	    display, outdir, save_freq
	)



if __name__ == "__main__":
    generate()
    
    
