"""Preserve video geometry while leaving native reference-prefix residuals zero."""


def align_reference_geometry(tokens, mask, video_latent_frames, reference_latents):
    import torch
    if tokens.shape[1] % video_latent_frames or mask.shape != (*tokens.shape[:2], 1):
        raise ValueError("Unexpected video geometry layout")
    prefix = reference_latents * (tokens.shape[1] // video_latent_frames)
    if prefix:
        tokens = torch.cat((tokens.new_zeros(tokens.shape[0], prefix, tokens.shape[2]), tokens), dim=1)
        # Gate after the biased projection: zero features alone are insufficient.
        mask = torch.cat((mask.new_zeros(mask.shape[0], prefix, 1), mask), dim=1)
    return tokens, mask


class ReferenceGeometryModelFn:
    def __init__(self, tokens, mask, projection, reference_latents):
        self.tokens, self.mask = align_reference_geometry(tokens, mask, 5, reference_latents)
        self.projection = projection
        self.reference_latents = reference_latents
        self.calls = 0
        self.last_layout = None

    def __call__(self, *args, **kwargs):
        from integration import model_fn_wan_video
        context, latents = kwargs["vace_context"], kwargs["latents"]
        if context.shape[2] != 5 + self.reference_latents or latents.shape[2] != context.shape[2]:
            raise ValueError("Native reference/video latent layout differs from geometry")
        self.calls += 1
        self.last_layout = dict(vace_context=list(context.shape), latents=list(latents.shape),
                                structure_tokens=list(self.tokens.shape), structure_gate=list(self.mask.shape),
                                reference_latents=self.reference_latents,
                                reference_geometry_residual_zero=True)
        return model_fn_wan_video(*args, structure_tokens=self.tokens, structure_mask=self.mask,
                                  structure_projection=self.projection, **kwargs)
