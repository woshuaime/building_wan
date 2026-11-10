"""Use one immutable geometry condition for both classifier-free guidance branches."""


class FixedGeometryModelFn:
    def __init__(self, tokens, mask, projection):
        self.tokens, self.mask, self.projection = tokens, mask, projection
        self.calls = 0

    def __call__(self, *args, **kwargs):
        from integration import model_fn_wan_video
        self.calls += 1
        return model_fn_wan_video(*args, structure_tokens=self.tokens, structure_mask=self.mask,
                                  structure_projection=self.projection, **kwargs)
