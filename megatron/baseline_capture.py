"""Native NNsight trace, immediate and optimizer-iteration-deferred."""
from nnsight import NNsight
from .baseline_sites import CaptureBase, HOOKS

class Capture(CaptureBase):
    def __init__(self, model, selected=HOOKS, mode="immediate", *, vocab_topk=256):
        super().__init__(model, selected, mode, vocab_topk=vocab_topk)
        if mode not in ("immediate", "deferred"):
            raise ValueError(mode)
        self.traced = NNsight(model)
        self.envoys = []
        for path, point in self.points:
            envoy = self.traced
            for part in path.split("."):
                envoy = getattr(envoy, part)
            self.envoys.append(envoy)

    def forward(self, *args, **kwargs):
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
        for (_, point), (tensor, arrived_ns) in zip(self.points, saved, strict=True):
            self.accept(point, tensor, deferred=deferred, t_arrive_ns=arrived_ns)
        return result
