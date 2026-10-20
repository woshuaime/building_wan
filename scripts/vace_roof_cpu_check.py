"""Exercise local VACE LoRA and mask interfaces without loading model weights."""

import importlib.metadata
from pathlib import Path


def check_interfaces(config):
    import numpy as np
    import torch
    import diffsynth
    from PIL import Image

    from diffsynth.diffusion.training_module import DiffusionTrainingModule
    from diffsynth.models.wan_video_vace import VaceWanModel
    from diffsynth.pipelines.wan_video import WanVideoPipeline, WanVideoUnit_VACE

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU interface check must not initialize CUDA")
    settings = config["training"]
    version = importlib.metadata.version("diffsynth")
    if version != "2.1.5":
        raise ValueError(f"Expected local DiffSynth 2.1.5, got {version}")
    expected = Path(__file__).resolve().parents[1] / "code/DiffSynth-Studio/diffsynth/__init__.py"
    if Path(diffsynth.__file__).resolve() != expected.resolve():
        raise ValueError("DiffSynth must be the existing project vendor copy")
    with torch.device("meta"):
        model = VaceWanModel()
        model.requires_grad_(False)
        model = DiffusionTrainingModule().add_lora_to_model(
            model,
            target_modules=settings["lora_target_modules"],
            lora_rank=settings["lora_rank"],
            lora_alpha=settings["lora_alpha"],
        )
    parameters = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    if not all(".lora_A." in name or ".lora_B." in name for name, _ in parameters):
        raise ValueError("Non-LoRA parameters would be trainable")
    layer_names = {name.split(".lora_")[0] for name, _ in parameters}
    parameter_count = sum(value.numel() for _, value in parameters)
    if (len(layer_names), len(parameters), parameter_count) != (150, 300, 10_936_320):
        raise ValueError("VACE rank-16 target-module match differs from the frozen experiment")
    scales = [
        module.scaling["default"]
        for module in model.modules()
        if hasattr(module, "lora_A") and hasattr(module, "scaling")
    ]
    if len(scales) != 150 or any(scale != 1.0 for scale in scales):
        raise ValueError("Training alpha/rank must equal the frozen inference multiplier 1.0")
    del model, parameters

    class ShapeOnlyVAE(torch.nn.Module):
        def encode(self, images, **kwargs):
            channels = images.mean(dim=1, keepdim=True).repeat(1, 16, 1, 1, 1)
            return torch.nn.functional.interpolate(channels, size=(5, 2, 2), mode="nearest")

    pipe = WanVideoPipeline(device="cpu", torch_dtype=torch.float32)
    pipe.vae = ShapeOnlyVAE()
    conditions = [Image.new("RGB", (16, 16), (127, 127, 127)) for _ in range(17)]
    pixels = np.zeros((16, 16, 3), dtype=np.uint8)
    pixels[:, 8:] = 255
    masks = [Image.fromarray(pixels) for _ in range(17)]
    unit = WanVideoUnit_VACE()
    context = unit.process(
        pipe, conditions, masks, None, 1.0, 16, 16, 17, False, (30, 52), (15, 26)
    )["vace_context"]
    if list(context.shape) != [1, 96, 5, 2, 2]:
        raise ValueError("Actual VACE unit returned unexpected conditioning dimensions")
    if not (torch.all(context[:, 32:, :, :, :1] == 0)
            and torch.all(context[:, 32:, :, :, 1:] == 1)):
        raise ValueError("Actual VACE unit did not preserve binary white-generation mask semantics")
    black_masks = [Image.new("RGB", (16, 16), (0, 0, 0)) for _ in range(17)]
    black_context = unit.process(
        pipe, conditions, black_masks, None, 1.0, 16, 16, 17, False, (30, 52), (15, 26)
    )["vace_context"]
    if not torch.all(black_context[:, 32:] == 0):
        raise ValueError("Actual VACE unit changed an all-black mask")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU interface check unexpectedly initialized CUDA")
    return {
        "version": version,
        "matched_layers": len(layer_names),
        "lora_tensors": 300,
        "trainable_parameters": parameter_count,
        "loader_mask_verified": True,
        "cuda_initialized": False,
    }
