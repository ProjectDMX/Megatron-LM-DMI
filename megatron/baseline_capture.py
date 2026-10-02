"""Native NNsight trace, immediate and optimizer-iteration-deferred."""
from nnsight import NNsight
from .baseline_sites import CaptureBase, HOOKS

class Capture(CaptureBase):
    def __init__(self, model, selected=HOOKS, mode="immediate", *, vocab_topk=256, source_sampling=None):
        super().__init__(model, selected, mode, vocab_topk=vocab_topk, source_sampling=source_sampling)
        if mode not in ("immediate", "deferred"):
            raise ValueError(mode)
        self.traced = NNsight(model)
        self.weight_traced = NNsight(self.weight_model) if self.weight_model is not None else None
        self.active_points = self.activation_points
        self.envoys = []
        for path, point in self.activation_points:
            envoy = self.traced
            for part in path.split("."):
                envoy = getattr(envoy, part)
            self.envoys.append(envoy)

    def _native_forward(self, *args, **kwargs):
        # A new trace for each scheduled microbatch, without replaying its forward.
        # The list preserves ALL observations, not only the final microbatch.
        saved = []
        deferred = self.mode == "deferred"
        with self.traced.trace(*args, **kwargs):
            for envoy in self.envoys:
                if deferred:
                    value = envoy.output.detach().save()
                    arrived_ns = None
                else:
                    payload = envoy.output.detach().cpu()
                    arrived_ns = self.now_ns()
                    value = payload.save()
                saved.append((value, arrived_ns))
            result = self.traced.output.save()
        for (_, point), (tensor, arrived_ns) in zip(self.active_points, saved, strict=True):
            self.accept(point, tensor, deferred=deferred, t_arrive_ns=arrived_ns)
        return result

    def forward(self, *args, **kwargs):
        if not self.activation_points:
            return self.model(*args, **kwargs)
        return self._native_forward(*args, **kwargs)

    def capture_weights(self):
        if not self.weight_points:
            return
        old = self.traced, self.envoys, self.active_points, self.microbatch
        self.traced, self.active_points, self.microbatch = self.weight_traced, self.weight_points, -1
        self.envoys = []
        for path, _ in self.weight_points:
            envoy = self.weight_traced
            for part in path.split('.'):
                envoy = getattr(envoy, part)
            self.envoys.append(envoy)
        try:
            self._native_forward(__import__('torch').empty(0))
        finally:
            self.traced, self.envoys, self.active_points, self.microbatch = old
