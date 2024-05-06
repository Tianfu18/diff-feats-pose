import numpy as np
from PIL import Image
import PIL
import torch
from diffusers import StableDiffusionPipeline

from .resnet import collect_feats


def get_tokens_embedding(clip_tokenizer, clip, device, prompt):
    tokens = clip_tokenizer(
        prompt,
        padding="max_length",
        max_length=clip_tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
        return_overflowing_tokens=True,
    )
    input_ids = tokens.input_ids.to(device)
    embedding = clip(input_ids).last_hidden_state
    return tokens, embedding


def generalized_step(x, model, **kwargs):
    """
    Performs either the generation or inversion diffusion process.
    """
    t = kwargs.get("time_step", 0)

    with torch.no_grad():
        n = x.size(0)
        xs = [x]
        t = (torch.ones(n) * t).to(x.device)
        xt = xs[-1].to(x.device)
        cond = kwargs["conditional"]
        et = model(xt, t, encoder_hidden_states=cond).sample
        return et


def freeze_weights(weights):
    for param in weights.parameters():
        param.requires_grad = False


def init_models(
        device="cuda",
        model_id="runwayml/stable-diffusion-v1-5",
        freeze=True
):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        # revision="fp16",
        # local_files_only=True
    )
    unet = pipe.unet
    vae = pipe.vae
    clip = pipe.text_encoder
    clip_tokenizer = pipe.tokenizer
    unet.to(device)
    vae.to(device)
    clip.to(device)
    if freeze:
        freeze_weights(unet)
        freeze_weights(vae)
        freeze_weights(clip)
    return unet, vae, clip, clip_tokenizer


def collect_and_resize_feats(unet, idxs, resolution=-1):
    latent_feats = collect_feats(unet, idxs=idxs)
    latent_feats = [feat[None].float() for feat in latent_feats]

    if resolution > 0:
        latent_feats = [torch.nn.functional.interpolate(latent_feat, size=resolution, mode="bilinear")
                        for latent_feat in latent_feats]
    return latent_feats
