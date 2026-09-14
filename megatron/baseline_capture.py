"""Native TorchLens predicate operation recording with cpu_async."""
from queue import Queue
from threading import Thread
import torch
import torchlens
from torchlens.fastlog import CaptureSpec
from .baseline_sites import CaptureBase, HOOKS

class _TraceRoot(torch.nn.Module):
    """Expose nn.Module.buffers() without changing Megatron DDP's buffers list.

    The registered child keeps the same 'module.*' addresses as DDP. Actual
    execution still enters the ORIGINAL DDP.forward, including its pre-forward
    synchronization; this is not a bypass of distributed training.
    """
    def __init__(self, ddp):
        super().__init__()
        self.module = ddp.module
        object.__setattr__(self, '_ddp_call', ddp)

    def forward(self, *args, **kwargs):
        return self._ddp_call(*args, **kwargs)

class Capture(CaptureBase):
    def __init__(self, model, selected=HOOKS, mode=None, *, vocab_topk=256):
        super().__init__(model, selected, mode, vocab_topk=vocab_topk)
        if mode not in (None, 'immediate'):
            raise ValueError(mode)
        self.trace_root = model if callable(model.buffers) else _TraceRoot(model)
        self._ready_queue = Queue()
        self._ready_errors = []
        self._ready_worker = Thread(target=self._observe_ready, daemon=True,
                                    name="torchlens-cpu-ready")
        self._ready_worker.start()
        for _, point in self.points:
            self.handles.append(point.register_forward_hook(self._copy_enqueued))

    def _copy_enqueued(self, point, args, output):
        # TorchLens copies the selected view_as output inline, before this
        # observation module returns. Its native cpu_async copy uses the current
        # stream. An event here follows that copy, but not later model work.
        meta = self._fired[point][-1]
        if output.is_cuda:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(output.device))
            self._ready_queue.put((event, meta))
        else:
            meta["t_arrive_ns"] = self.now_ns()

    def _observe_ready(self):
        # Do not block the model at each hook. Timestamp host-observed readiness
        # individually, not at the end-of-forward synchronization. This includes
        # host observer scheduling delay; it is not an exact GPU timestamp.
        while True:
            item = self._ready_queue.get()
            try:
                if item is None:
                    return
                event, meta = item
                event.synchronize()
                meta["t_arrive_ns"] = self.now_ns()
            except Exception as exc:
                self._ready_errors.append(exc)
            finally:
                self._ready_queue.task_done()

    def _select(self, ctx):
        if (ctx.kind == "op" and ctx.address in self.by_path
                and ctx.func_name == "view_as"):
            return CaptureSpec(save_mode="cpu_async", keep_grad=False)
        return False

    def forward(self, *args, **kwargs):
        output, recording = torchlens.record(
            self.trace_root, args, input_kwargs=kwargs, save=self._select,
            return_output=True, on_predicate_error="fail-fast")
        # cpu_async only enqueues; do not expose incomplete CPU payloads.
        # This wait is part of the capture cost, not an omitted sink cost.
        torch.cuda.current_stream().synchronize()
        self._ready_queue.join()
        if self._ready_errors:
            raise RuntimeError("CPU-ready observer failed") from self._ready_errors[0]
        seen = set()
        for record in recording.records:
            path = record.ctx.address
            if record.ram_payload is None:
                raise AssertionError(f"Missing native TorchLens payload for {path}")
            self.accept(self.by_path[path], record.ram_payload)
            seen.add(path)
        if seen != set(self.by_path):
            raise AssertionError(f"Missing sites: {set(self.by_path) - seen}")
        return output

    def close(self):
        self._ready_queue.put(None)
        self._ready_queue.join()
        self._ready_worker.join()
        super().close()
