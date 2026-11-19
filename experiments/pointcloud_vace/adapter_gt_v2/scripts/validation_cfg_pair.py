"""Reuse identical first-block self-attention between the two sequential CFG branches."""

from contextlib import ExitStack, contextmanager
from functools import partial

import torch


class FirstAttentionPair:
    """Keep only this timestep's two first-block results in inference mode.

    In this Wan model the first DiT and VACE self-attentions both precede
    the first text cross-attention. Check the actual inputs before reuse.
    """

    def __init__(self, dit_attention, vace_attention):
        if dit_attention is vace_attention:
            raise ValueError("Expected independent DiT and VACE attention modules")
        self._modules = (dit_attention, vace_attention)
        self._stack = None
        self._negative = None
        self._key = None
        self._saved = {}
        self.hits = self.misses = 0

    def __enter__(self):
        if self._stack is not None:
            raise RuntimeError("CFG attention context is already installed")
        with ExitStack() as stack:
            for module in self._modules:
                original = module.forward
                if "forward" in vars(module):
                    stack.callback(setattr, module, "forward", original)
                else:
                    stack.callback(delattr, module, "forward")
                module.forward = partial(self._forward, module, original)
            self._stack = stack.pop_all()
        return self

    def __exit__(self, *exc):
        self._stack.close()
        self._stack = None
        self._negative = self._key = None
        self._saved.clear()

    @contextmanager
    def branch(self, latents, timestep, negative):
        if self._stack is None or self._negative is not None or torch.is_grad_enabled():
            raise RuntimeError("CFG pair requires installed, non-nested inference branches")
        if negative:
            if self._key is None or self._key[0] is not latents or self._key[1] is not timestep:
                self._key = None
                self._saved.clear()
                raise RuntimeError("Negative CFG branch must use its positive branch's latents and timestep")
        else:
            self._saved.clear()
            self._key = (latents, timestep)
        self._negative = negative
        try:
            yield
            expected = 0 if negative else len(self._modules)
            if len(self._saved) != expected:
                raise RuntimeError("Expected one call to each first-block attention per CFG branch")
        except BaseException:
            self._key = None
            self._saved.clear()
            raise
        finally:
            self._negative = None
            if negative:
                self._key = None
                self._saved.clear()

    def _forward(self, module, original, x, freqs):
        if self._negative is None:
            return original(x, freqs)
        if not self._negative:
            if module in self._saved:
                raise RuntimeError("Repeated first-block attention in positive CFG branch")
            result = original(x, freqs)
            self._saved[module] = (x, freqs, result)
            return result

        previous_x, previous_freqs, result = self._saved.pop(module)
        if (x.dtype == previous_x.dtype and freqs.dtype == previous_freqs.dtype
                and torch.equal(x, previous_x) and torch.equal(freqs, previous_freqs)):
            self.hits += 1
            return result
        self.misses += 1
        return original(x, freqs)
