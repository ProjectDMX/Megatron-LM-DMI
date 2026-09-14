"""Ordinary PyTorch forward callback with blocking D2H."""
from .baseline_sites import CaptureBase, HOOKS

class Capture(CaptureBase):
    def __init__(self, model, selected=HOOKS, mode=None, *, vocab_topk=256):
        super().__init__(model, selected, mode, vocab_topk=vocab_topk)
        if mode not in (None, "immediate"):
            raise ValueError(mode)
        for _, point in self.points:
            self.handles.append(point.register_forward_hook(self._copy))

    def _copy(self, module, args, output):
        payload = output.detach().cpu()
        arrived_ns = self.now_ns()
        self.accept(module, payload, t_arrive_ns=arrived_ns)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
