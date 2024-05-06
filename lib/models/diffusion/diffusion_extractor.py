import torch

from .stable_diffusion.diffusion import (
    init_models,
    get_tokens_embedding,
    generalized_step,
    collect_and_resize_feats
)
from .stable_diffusion.resnet import init_resnet_func


class DiffusionExtractor:
    def __init__(self, config, device):
        self.device = device
        self.timestep = config.model.timestep
        self.generator = torch.Generator(self.device).manual_seed(0)
        self.batch_size = config.train.batch_size

        self.unet, self.vae, self.clip, self.clip_tokenizer = init_models(
            device=self.device, model_id=config.model.diffusion_id
        )
        self.prompt = ""
        self.negative_prompt = ""

        self.idxs = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
        self.output_resolution = config.model.output_resolution

    def change_cond(self, prompt, cond_type="cond", ):
        with torch.no_grad():
            with torch.autocast("cuda"):
                _, new_cond = get_tokens_embedding(self.clip_tokenizer, self.clip, self.device, prompt)
                new_cond = new_cond.expand((self.batch_size, *new_cond.shape[1:]))
                new_cond = new_cond.to(self.device)
                if cond_type == "cond":
                    self.cond = new_cond
                    self.prompt = prompt
                elif cond_type == "uncond":
                    self.uncond = new_cond
                    self.negative_prompt = prompt
                else:
                    raise NotImplementedError

    def run(self, latent):
        xs = generalized_step(
            latent,
            self.unet,
            time_step=self.timestep,
            conditional=self.cond,
            unconditional=self.uncond,
        )
        return xs

    def get_feats(self, latents, extractor_fn, preview_mode=False):
        if not preview_mode:
            init_resnet_func(self.unet, save_hidden=True, reset=True, idxs=self.idxs)
        out_feat = extractor_fn(latents)
        if not preview_mode:
            timestep_feats = collect_and_resize_feats(self.unet, self.idxs, self.output_resolution)
            feats = timestep_feats
            init_resnet_func(self.unet, reset=True)
        else:
            feats = None
        return feats

    def forward(self, images, preview_mode=False):
        self.batch_size = images.size(0)
        self.change_cond(self.prompt, "cond")
        self.change_cond(self.negative_prompt, "uncond")
        images = torch.nn.functional.interpolate(images, size=512, mode="bilinear")
        latents = self.vae.encode(images).latent_dist.sample(generator=None) * 0.18215
        extractor_fn = lambda latents: self.run(latents)
        with torch.no_grad():
            with torch.autocast("cuda"):
                return self.get_feats(latents, extractor_fn, preview_mode=preview_mode)
