"""Baseline observation sites and iteration-scoped record bookkeeping.

This module has no DMI imports. Sites expose a zero-copy view, do not substitute
the original tensor in Megatron. Optional GPU top-k and sample-loss reductions
feed separate sites; the original raw selections perform no preprocessing.
"""
from collections import Counter, defaultdict, deque
from time import perf_counter_ns
import torch

HOOKS = ("hidden_states", "router_logits", "moe_inverse_map",
         "moe_packed_weighted_output", "resid_final", "vocab_logits")
ORDER = {name: index for index, name in enumerate(HOOKS)}
FEATURES = ("vocab_topk", "loss_summary", "grad_norm")
PREPARED_HOOKS = ("vocab_topk_values", "vocab_topk_indices",
                  "lm_per_sample_loss", "lm_per_sample_loss_token_count")
ORDER.update({name: len(ORDER) + i for i, name in enumerate(PREPARED_HOOKS)})


def parallel_coordinates():
    from megatron.core import parallel_state as ps
    if not ps.model_parallel_is_initialized():
        return dict(tp_rank=0, tp_size=1, pp_rank=0, pp_size=1, dp_rank=0)
    return dict(tp_rank=ps.get_tensor_model_parallel_rank(),
                tp_size=ps.get_tensor_model_parallel_world_size(),
                pp_rank=ps.get_pipeline_model_parallel_rank(),
                pp_size=ps.get_pipeline_model_parallel_world_size(),
                dp_rank=ps.get_data_parallel_rank())

class Observation(torch.nn.Module):
    def __init__(self, name, layer):
        super().__init__()
        self.name = name
        self.layer = layer

    def forward(self, tensor):
        return tensor.view_as(tensor)


class VocabTopK(torch.nn.Module):
    """Detached local top-k; native capture sees only the two selected outputs."""
    def __init__(self, coordinates, local_vocab_size, k=256):
        super().__init__()
        self.local_vocab_size = int(local_vocab_size)
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= self.local_vocab_size:
            raise ValueError(f"vocab_topk must be an integer in [1, {self.local_vocab_size}], got {k!r}")
        self.k = k
        self.values = Observation("vocab_topk_values", -1)
        self.indices = Observation("vocab_topk_indices", -1)
        for point in (self.values, self.indices):
            point.extra_metadata = dict(
                **coordinates, local_vocab_size=self.local_vocab_size,
                vocab_start=coordinates["tp_rank"] * self.local_vocab_size,
                vocab_end=(coordinates["tp_rank"] + 1) * self.local_vocab_size,
                sample_axis=0, token_axis=1, index_scope="tp_local", topk=self.k)

    def forward(self, logits):
        if logits.ndim != 3 or logits.shape[-1] != self.local_vocab_size:
            raise ValueError("vocab_topk requires [S,B,V_local] ungathered logits")
        values, indices = torch.topk(logits.detach(), self.k, dim=-1, largest=True, sorted=True)
        self.values(values.transpose(0, 1))
        self.indices(indices.to(torch.int32).transpose(0, 1))


class SampleLoss(torch.nn.Module):
    """DMI-matched dense per-sample mean and loss-token count, not a new loss."""
    def __init__(self):
        super().__init__()
        self.mean = Observation("lm_per_sample_loss", -1)
        self.count = Observation("lm_per_sample_loss_token_count", -1)
        for point in (self.mean, self.count):
            point.extra_metadata = dict(sample_axis=0)

    def forward(self, token_loss, loss_mask):
        if not isinstance(token_loss, torch.Tensor) or token_loss.ndim != 2:
            raise ValueError("loss_summary requires dense [B,S] token losses")
        if loss_mask is None:
            loss_mask = torch.ones_like(token_loss)
        if loss_mask.ndim != 2 or loss_mask.shape != token_loss.shape:
            raise ValueError("loss_summary requires a matching [B,S] loss mask")
        mask = loss_mask.detach().float()
        count = mask.sum(dim=1)
        mean = (token_loss.detach().float() * mask).sum(dim=1) / count.clamp_min(1)
        self.mean(mean[:, None])
        self.count(count.to(torch.int64)[:, None])


def install_sites(model, selected, *, vocab_topk=256):
    selected = set(selected)
    unknown = selected.difference((*HOOKS, *FEATURES))
    if unknown:
        raise ValueError(f"Unsupported raw hooks: {sorted(unknown)}")
    def add(module, name, layer):
        if name not in selected:
            return None
        attr = "baseline_" + name
        if hasattr(module, attr):
            return getattr(module, attr)
        point = Observation(name, layer)
        module.add_module(attr, point)
        return point
    # Freeze discovery before adding child modules.
    for module in list(model.modules()):
        cls = type(module).__name__
        if cls == "TransformerLayer":
            add(module, "hidden_states", int(module.layer_number) - 1)
        elif cls == "TopKRouter":
            add(module, "router_logits", int(module.layer_number) - 1)
        elif cls == "MoELayer":
            layer = int(module.layer_number) - 1
            add(module, "moe_packed_weighted_output", layer)
            point = add(module, "moe_inverse_map", layer)
            if point is not None:
                if type(module.token_dispatcher).__name__ != "MoEAlltoAllTokenDispatcher":
                    raise NotImplementedError("Inverse map requires alltoall dispatcher")
                module.token_dispatcher.baseline_moe_inverse_map = point
        elif cls == "TransformerBlock" and module.final_layernorm is not None:
            add(module, "resid_final", -1)
        elif cls == "GPTModel" and module.post_process:
            add(module, "vocab_logits", -1)
            coordinates = parallel_coordinates()
            if "vocab_topk" in selected:
                if not module.parallel_output:
                    raise ValueError("vocab_topk requires parallel_output=True")
                module.add_module("baseline_vocab_topk", VocabTopK(
                    coordinates, module.output_layer.output_size_per_partition, k=vocab_topk))
            if "loss_summary" in selected and coordinates["tp_rank"] == 0:
                module.add_module("baseline_loss_summary", SampleLoss())
    points = [(path, mod) for path, mod in model.named_modules()
              if isinstance(mod, Observation)]
    return sorted(points, key=lambda p: (p[1].layer if p[1].layer >= 0 else 10**9,
                                          ORDER[p[1].name]))

class CaptureBase:
    def __init__(self, model, selected=HOOKS, mode=None, *, vocab_topk=256):
        # TE modules are supported; TE/Megatron CUDA-graph replay is not.
        # Reject graph mode before attaching sites instead of silently missing
        # observations when Python hooks are bypassed on replay.
        for module in model.modules():
            config = getattr(module, "config", None)
            if (getattr(config, "cuda_graph_impl", "none") != "none"
                    or getattr(config, "enable_cuda_graph", False)
                    or getattr(config, "external_cuda_graph", False)):
                raise ValueError("Raw baselines require eager execution: set "
                                 "cuda_graph_impl='none', enable_cuda_graph=False "
                                 "and external_cuda_graph=False. TE modules remain supported.")
        if isinstance(vocab_topk, bool) or not isinstance(vocab_topk, int) or vocab_topk < 1:
            raise ValueError("vocab_topk must be a positive integer")
        self.model = model
        self.selected = frozenset(selected)
        self.coordinates = parallel_coordinates()
        self._grad_norm_iteration = None
        self._grad_point = Observation("grad_norm", -1)
        if "grad_norm" in self.selected:
            if hasattr(model, "_baseline_capture_controller"):
                raise RuntimeError("A baseline capture is already attached to this model")
            model._baseline_capture_controller = self
        self.points = install_sites(model, selected, vocab_topk=vocab_topk)
        self.by_path = dict(self.points)
        self.records = []
        self.pending = []
        self.handles = []
        self.iteration = None
        self.microbatch = None
        self.mode = mode
        self.pending_peak_bytes = 0
        self.iteration_audits = []
        self.occurrences = Counter()
        self.first_iteration_start_ns = None
        self.iteration_start_ns = None
        self._fired = defaultdict(deque)
        for _, point in self.points:
            self.handles.append(point.register_forward_pre_hook(self._hook_fire))

    def begin_iteration(self, iteration, *, start_ns=None):
        if self.pending or any(self._fired.values()):
            raise RuntimeError("Previous iteration has unconsumed observations")
        start_ns = perf_counter_ns() if start_ns is None else int(start_ns)
        if self.first_iteration_start_ns is None:
            self.first_iteration_start_ns = start_ns
        if start_ns < self.first_iteration_start_ns:
            raise ValueError("Iteration start precedes the clock origin")
        self.iteration_start_ns = start_ns - self.first_iteration_start_ns
        self.iteration = int(iteration)
        self.occurrences.clear()

    def now_ns(self):
        if self.first_iteration_start_ns is None:
            raise RuntimeError("Call begin_iteration before capturing")
        return perf_counter_ns() - self.first_iteration_start_ns

    def _hook_fire(self, point, args):
        # Same host observation boundary for every backend; before copy work.
        fired_ns = self.now_ns()
        self._fired[point].append(dict(self.metadata(point, args[0]),
                                       t_hook_fire_ns=fired_ns))

    def metadata(self, point, tensor):
        key = (self.microbatch, point.name, point.layer)
        occurrence = self.occurrences[key]
        self.occurrences[key] += 1
        return dict(iteration=self.iteration, microbatch=self.microbatch,
                    hook=point.name, layer=point.layer, occurrence=occurrence,
                    shape=list(tensor.shape), dtype=str(tensor.dtype),
                    bytes=tensor.numel() * tensor.element_size(),
                    **dict(self.coordinates, **getattr(point, "extra_metadata", {})))

    def record_grad_norm(self, value, *, iteration):
        """Observe the already CPU-ready reduced optimizer statistic; no new reduction."""
        if "grad_norm" not in self.selected:
            return
        coords = self.coordinates
        if coords["pp_rank"] != coords["pp_size"] - 1 or coords["tp_rank"] or coords["dp_rank"]:
            return
        if iteration != self.iteration or self._grad_norm_iteration == iteration:
            raise RuntimeError("grad_norm must be emitted once for the current iteration")
        fired = self.now_ns()
        if isinstance(value, torch.Tensor):
            if value.is_cuda or value.numel() != 1:
                raise ValueError("Expected Megatron's existing host gradient-norm scalar")
            value = value.item()
        tensor = torch.tensor([-1.0 if value is None else value], dtype=torch.float32)
        previous_microbatch = self.microbatch
        self.microbatch = -1
        try:
            meta = self.metadata(self._grad_point, tensor)
        finally:
            self.microbatch = previous_microbatch
        self.records.append(dict(**meta, tensor=tensor, source_available=value is not None,
                                 t_hook_fire_ns=fired, t_arrive_ns=self.now_ns()))
        self._grad_norm_iteration = iteration

    def accept(self, point, tensor, deferred=False, *, t_arrive_ns=None):
        if not self._fired[point]:
            raise RuntimeError("Received a payload without its hook-fire timestamp")
        meta = self._fired[point].popleft()
        if deferred:
            if not tensor.is_cuda:
                raise AssertionError("Deferred payload must remain on GPU")
            self.pending.append((meta, tensor))
            self.pending_peak_bytes = max(self.pending_peak_bytes,
                                          sum(m["bytes"] for m, _ in self.pending))
        else:
            if tensor.is_cuda:
                raise AssertionError("Capture backend did not deliver a CPU tensor")
            if t_arrive_ns is not None:
                meta["t_arrive_ns"] = int(t_arrive_ns)
            if "t_arrive_ns" not in meta or meta["t_arrive_ns"] < meta["t_hook_fire_ns"]:
                raise AssertionError("Missing or invalid CPU-ready timestamp")
            self.records.append(dict(**meta, tensor=tensor))

    def end_iteration(self):
        flush_start_ns = self.now_ns()
        held = len(self.pending)
        for meta, tensor in self.pending:
            cpu_tensor = tensor.cpu()
            arrived_ns = self.now_ns()
            self.records.append(dict(**meta, tensor=cpu_tensor, t_arrive_ns=arrived_ns))
        self.pending.clear()
        if any(self._fired.values()):
            raise RuntimeError("Hook observations were not consumed")
        if torch.cuda.is_initialized():
            torch.cuda.current_stream().synchronize()
        self.iteration_audits.append(dict(iteration=self.iteration,
                                         t_iteration_start_ns=self.iteration_start_ns,
                                         t_flush_start_ns=flush_start_ns,
                                         t_capture_end_ns=self.now_ns(),
                                         deferred_records=held,
                                         pending_after=0))

    def close(self):
        if self.pending or any(self._fired.values()):
            raise RuntimeError("Cannot close with uncopied payloads")
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if getattr(self.model, "_baseline_capture_controller", None) is self:
            del self.model._baseline_capture_controller

class ReferenceCapture(CaptureBase):
    """Untimed source-value oracle; deliberately not an evaluated baseline."""
    def __init__(self, model, selected=HOOKS, mode=None, *, vocab_topk=256):
        super().__init__(model, selected, mode, vocab_topk=vocab_topk)
        for _, point in self.points:
            self.handles.append(point.register_forward_hook(self._copy))

    def _copy(self, module, args, output):
        payload = args[0].detach().cpu()
        arrived_ns = self.now_ns()
        self.accept(module, payload.clone(), t_arrive_ns=arrived_ns)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
