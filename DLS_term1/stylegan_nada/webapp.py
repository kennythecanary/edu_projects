import gradio as gr
from app import *


def generate(
    source, target, image, lr,
    ds_path=None, 
    n_samples=None,
    batch_size=None, 
    num_steps=None, 
    z_samples=None,
    layer_selection_steps=None, 
    layer_selection_evals=None, 
    patience=None,
    ckpt_path=None
):
    dataset = FFHQDs(ds_path, n_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    model = MyStyleGANa(ckpt_path)

    source, target = [torch.cat([clip.tokenize(prompt)]).to(device) for prompt in [source, target]]

    criterion = DirectionalCLIPLoss(model.clip_model, source, target)
    optimizer = optim.Adam(model.G.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    aligned_img = dataset._align_face(image)
    input_img = transform(aligned_img)

    for step in range(num_steps):
        fmin_objective = partial(model._hyperopt_objective, target=target, z_samples=z_samples, lr=lr, 
            fmin_steps=layer_selection_steps)
        
        torch.save(model.G.state_dict(), model.ckpt_path)
        best = fmin(fmin_objective, space=model.space, algo=tpe.suggest, max_evals=layer_selection_evals, 
            early_stop_fn=early_stop.no_progress_loss(patience))
        
        most_relevant_layers = [k for k,v in best.items() if v]
        model.G = model._unfreeze_layers(model.G, most_relevant_layers)
        dc_loss = model._fit_epoch(dataloader, criterion, optimizer)
        scheduler.step()
        output_loss = "Step [%d/%d] Unfreezed: %d > Loss: %f" % (step+1, num_steps, len(most_relevant_layers), dc_loss)
        
        with torch.no_grad():
            w = model.encode_images(input_img.unsqueeze(0))
        
        generated_img = model.G.synthesis(w, noise_mode="const", force_fp32=True).detach().cpu().squeeze(0)
        
        yield output_loss, tensor2im(generated_img)


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox("Photo"),
        gr.Textbox("Sketch"),
        gr.Image(type="filepath"),
        gr.Number(0.002),
    ],
    additional_inputs=[
        gr.Textbox("02000"),
        gr.Number(50),
        gr.Number(4),
        gr.Number(10),
        gr.Number(3),
        gr.Number(5),
        gr.Number(10),
        gr.Number(5),
        gr.Textbox("/tmp/G_chpt.pt")
    ],
    outputs=["text", "image"],
    title="DLS Final Project 2024 | StyleGAN-NADA Reimplementation"
)
demo.launch()