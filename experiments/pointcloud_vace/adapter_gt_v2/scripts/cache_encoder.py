"""Frozen Wan VAE execution for the fixed GT-video windows."""
from pathlib import Path
from types import SimpleNamespace

import torch

from common import ROOT, sha
import worker


def load_vae(run):
    from diffsynth.models.wan_video_vae import WanVideoVAE
    from diffsynth.utils.state_dict_converters.wan_video_vae import WanVideoVAEStateDictConverter
    from safetensors.torch import load_file

    config = worker.read_config(run)
    for name, expected in config["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Run source identity changed: {name}")
    weights = Path(config["models"]["paths"]["vae"])
    if sha(weights) != config["models"]["expected_sha256"]["vae"]:
        raise ValueError("VAE file differs from the frozen run")
    torch.set_num_threads(4)
    vae = WanVideoVAE()
    vae.load_state_dict(WanVideoVAEStateDictConverter(load_file(str(weights))), strict=True, assign=True)
    vae.eval().requires_grad_(False)
    return vae.to(device="cuda", dtype=torch.bfloat16)


def normalized(gray):
    frames = torch.from_numpy(gray).to(device="cuda", dtype=torch.bfloat16)
    return (frames * (2 / 255) - 1)[None, None].expand(-1, 3, -1, -1, -1).contiguous()


class TileEncoder:
    """Replay one original batch-one, 17-frame tile with fresh input storage."""

    def __init__(self, vae, capture=True):
        self.vae = vae
        self.capture = capture
        self.scale = [s.to(device="cuda", dtype=torch.bfloat16) for s in vae.scale]
        self.input = torch.zeros((1, 3, 17, 240, 416), device="cuda", dtype=torch.bfloat16)
        if capture:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    vae.model.encode(self.input, self.scale)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.encoded = vae.model.encode(self.input, self.scale)
        dummy = torch.empty(1, 16, 5, 30, 52)
        self.tiles = []
        self.weight = torch.zeros((1, 1, 5, 60, 104), dtype=torch.bfloat16, device="cuda")
        for y in (0, 15, 30):
            for x in (0, 26, 52):
                mask = vae.build_mask(dummy, (y == 0, y == 30, x == 0, x == 52), (15, 26))
                mask = mask.to(device="cuda", dtype=torch.bfloat16)
                self.tiles.append((y, x, mask))
                self.weight[..., y:y + 30, x:x + 52] += mask

    def __call__(self, gray):
        video = normalized(gray)
        output = torch.zeros((1, 16, 5, 60, 104), device="cuda", dtype=torch.bfloat16)
        for y, x, mask in self.tiles:
            self.input.copy_(video[..., y * 8:y * 8 + 240, x * 8:x * 8 + 416])
            if self.capture:
                self.graph.replay()
                encoded = self.encoded
            else:
                encoded = self.vae.model.encode(self.input, self.scale)
            output[..., y:y + 30, x:x + 52] += encoded * mask
        return (output / self.weight).cpu()


def make_encoder(vae, variant):
    if variant == "reference":
        pipe = SimpleNamespace(device="cuda", vae=vae)
        return lambda raw: worker.encode_raw(pipe, raw)
    if variant == "whole_frame":
        scale = [s.to(device="cuda", dtype=torch.bfloat16) for s in vae.scale]
        encode = lambda gray: vae.model.encode(normalized(gray), scale).cpu()
    else:
        if variant == "channels_last_graph":
            torch.nn.utils.convert_conv3d_weight_memory_format(vae.model.encoder, torch.channels_last_3d)
        if variant not in ("tile_graph", "channels_last_graph"):
            raise ValueError(f"Unknown encoder: {variant}")
        encode = TileEncoder(vae)

    def pair(raw):
        return dict(input_latents=encode(raw["target"]), reactive=encode(raw["partial"]),
                    occupancy=torch.from_numpy(raw["occupancy"]).unsqueeze(0).float(),
                    id=raw["id"], frame_ids=raw["frame_ids"], cache_key=raw["cache_key"])

    return pair
