import os, re
import torch
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from itertools import islice
from einops import rearrange
import time
from pytorch_lightning import seed_everything
from contextlib import nullcontext
from ldm.util import instantiate_from_config
from optimizedSD.optimUtils import split_weighted_subprompts
from transformers import logging
logging.set_verbosity_error()

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    return sd

def generate_image(prompt, batch_size=1, H=512, W=512, for_waifu=False):
    config = "optimizedSD/v1-inference.yaml"
    ckpt = 'models/ldm/waifu-diffusion-v1/model.ckpt' if for_waifu else "models/ldm/stable-diffusion-v1/model.ckpt"

    outpath = 'outputs/txt2img-samples'
    ddim_steps = 50
    C = 4
    f = 8
    scale = 7.5

    seed = randint(0, 1000000)
    seed_everything(seed)

    sd = load_model_from_config(f"{ckpt}")
    li, lo = [], []
    for key, value in sd.items():
        sp = key.split(".")
        if (sp[0]) == "model":
            if "input_blocks" in sp:
                li.append(key)
            elif "middle_block" in sp:
                li.append(key)
            elif "time_embed" in sp:
                li.append(key)
            else:
                lo.append(key)
    for key in li:
        sd["model1." + key[6:]] = sd.pop(key)
    for key in lo:
        sd["model2." + key[6:]] = sd.pop(key)

    config = OmegaConf.load(f"{config}")

    model = instantiate_from_config(config.modelUNet)
    _, _ = model.load_state_dict(sd, strict=False)
    model.eval()
    model.unet_bs = 1
    model.cdevice = 'cuda'

    modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.eval()
    modelCS.cond_stage_model.device = 'cuda'

    modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = modelFS.load_state_dict(sd, strict=False)
    modelFS.eval()
    del sd

    start_code = None
    data = [batch_size * [prompt]]
    precision_scope = nullcontext

    with torch.no_grad():
        for prompts in tqdm(data, desc="data"):
            with precision_scope('cuda'):
                modelCS.to('cuda')
                uc = modelCS.get_learned_conditioning(batch_size * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)

                subprompts, weights = split_weighted_subprompts(prompts[0])
                if len(subprompts) > 1:
                    c = torch.zeros_like(uc)
                    totalWeight = sum(weights)
                    # normalize each "sub prompt" and add it
                    for i in range(len(subprompts)):
                        weight = weights[i]
                        # if not skip_normalize:
                        weight = weight / totalWeight
                        c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                else:
                    c = modelCS.get_learned_conditioning(prompts)

                shape = [batch_size, C, H // f, W // f]

                mem = torch.cuda.memory_allocated() / 1e6
                modelCS.to('cpu')
                while torch.cuda.memory_allocated() / 1e6 >= mem:
                    time.sleep(1)

                samples_ddim = model.sample(
                    S=ddim_steps,
                    conditioning=c,
                    seed=seed,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc,
                    eta=0.0,
                    x_T=start_code,
                )

                modelFS.to('cuda')

                images = []

                for i in range(batch_size):
                    x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                    x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), 'c h w -> h w c')
                    images.append(Image.fromarray(x_sample.astype(np.uint8)))

                mem = torch.cuda.memory_allocated() / 1e6
                modelFS.to('cpu')
                while torch.cuda.memory_allocated() / 1e6 >= mem:
                    time.sleep(1)
                del samples_ddim

                return images, seed
