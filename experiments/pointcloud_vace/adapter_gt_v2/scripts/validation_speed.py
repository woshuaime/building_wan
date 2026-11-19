"""Qualify faster inference on a real CFG pair, before advancing its scheduler."""
from contextlib import ExitStack
from time import perf_counter

import torch
from validation_acceleration import accelerated
from validation_cfg_pair import FirstAttentionPair


class ValidationSpeed:
    def __init__(self, pipe, core, policy, save):
        self.pipe, self.core = pipe, core
        self.policy, self.save = policy, save
        self.active = ExitStack()
        self.pair = None
        self.branch_count = 0
        self.inputs = []
        self.reference = []
        self.baseline_seconds = []
        self.candidate_seconds = []

    def __enter__(self):
        if self.policy.get("selected") == "accelerated":
            self.enable()
        return self

    def __exit__(self, *exc):
        self.active.close()

    def enable(self):
        self.active.enter_context(accelerated(self.pipe))
        self.pair = self.active.enter_context(FirstAttentionPair(
            self.pipe.dit.blocks[0].self_attn, self.core.vace_blocks[0].self_attn))

    def disable(self):
        self.active.close()
        self.pair = None

    def decide(self, use, reason):
        self.disable()
        self.policy.update(selected="accelerated" if use else "original", reason=reason,
                           baseline_pair_seconds=self.baseline_seconds,
                           candidate_pair_seconds=self.candidate_seconds)
        self.save(self.policy)
        print(f"验证速度选型 {self.policy['selected']} · {reason}", flush=True)
        if use:
            self.enable()

    def candidate(self, forward, inputs, negative):
        with self.pair.branch(inputs["latents"], inputs["timestep"], negative=negative):
            return forward(**inputs)

    def forward(self, forward, inputs):
        step = self.branch_count // 2 + 1
        negative = bool(self.branch_count % 2)
        self.branch_count += 1
        if self.policy.get("selected") == "accelerated":
            return self.candidate(forward, inputs, negative)
        if self.policy.get("selected") == "original" or step == 1:
            return forward(**inputs)

        # Steps 2 and 3 feed their reference outputs to the real scheduler. Each
        # candidate sees the exact same two inputs, including its timestep and text.
        if not negative:
            torch.cuda.synchronize()
            self.start = perf_counter()
        output = forward(**inputs)
        self.inputs.append(inputs)
        self.reference.append(output)
        if not negative:
            return output
        torch.cuda.synchronize()
        self.baseline_seconds.append(perf_counter() - self.start)
        try:
            self.enable()
            start = perf_counter()
            pos = self.candidate(forward, self.inputs[0], False)
            neg = self.candidate(forward, self.inputs[1], True)
            torch.cuda.synchronize()
            self.candidate_seconds.append(perf_counter() - start)
            equal = torch.equal(pos, self.reference[0]) and torch.equal(neg, output)
            hits = self.pair.hits
            del pos, neg
        except (ValueError, RuntimeError, ImportError) as error:
            self.decide(False, f"候选核对未通过 · {error}")
            return output
        finally:
            self.disable()
            self.inputs.clear()
            self.reference.clear()
        print(f"测速 {step - 1}/2 · 同步双分支 {self.baseline_seconds[-1]:.3f}s → "
              f"{self.candidate_seconds[-1]:.3f}s · 逐值相同={equal} · 复用 {hits}/2", flush=True)
        if not equal or hits != 2:
            self.decide(False, "原速度路径保留，双分支核对未通过")
        elif step == 3:
            # First candidate invocation warms the CUDA kernel. Use the second
            # matched timestep for timing, and both timesteps for exactness.
            base, fast = self.baseline_seconds[-1], self.candidate_seconds[-1]
            self.decide(fast < base * 0.98, f"同一步双分支 {base:.3f}s → {fast:.3f}s，两步输出逐值相同")
        return output
