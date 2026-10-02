"""Native NNsight 0.7.0 CPU cache, with event-based arrival measurement.

NNsight owns the detach/to(cpu, non_blocking=True) operation. The observer below
only fences and timestamps that operation; it does not implement transfers.
"""
import json
from pathlib import Path
from importlib.metadata import version
from queue import Queue
from threading import Lock, Thread

import torch
from nnsight import NNsight
from nnsight.intervention.hooks import add_ordered_hook

from .baseline_sites import CaptureBase, HOOKS


class Capture(CaptureBase):
    def __init__(self, model, selected=HOOKS, mode="cache_async", *, vocab_topk=256):
        if mode != "cache_async":
            raise ValueError(f"This checkout requires mode='cache_async', got {mode!r}")
        if version("nnsight") != "0.7.0":
            raise RuntimeError("This baseline is validated for NNsight 0.7.0's native nonblocking cache")
        super().__init__(model, selected, mode, vocab_topk=vocab_topk)
        self.traced = NNsight(model)
        self.envoys = []
        for path, _ in self.points:
            envoy = self.traced
            for part in path.split("."):
                envoy = getattr(envoy, part)
            self.envoys.append(envoy)
        self._ready_queue = Queue()
        self._ready_errors = []
        self._pending_lock = Lock()
        self._pending_bytes = 0
        self._queued_records = 0
        self._observer_handles = []
        self._active_cache = None
        self._closed = False
        self._init_diagnostics()
        self._cuda_device = next((p.device for p in model.parameters() if p.is_cuda), None)
        self._ready_worker = Thread(target=self._observe_ready, daemon=True,
                                    name="nnsight-cache-cpu-ready")
        self._ready_worker.start()

    def _init_diagnostics(self):
        self._native_handles = []
        self._native_cache = None
        self._diagnostics_path = None
        self._diagnostics_rank = None
        self._cuda_device = None
        self._trace_baseline = None
        self._reset_diagnostics()

    def _reset_diagnostics(self):
        self._native_calls = self._native_successes = 0
        self._native_d2h_calls = self._native_d2h_bytes = 0
        self._completed_records = self._iteration_pending_peak = 0
        self._trace_rows = []
        self._trace_cache_bytes = 0
        self._trace_cache_peak = 0

    def configure_diagnostics(self, output, *, rank):
        self._diagnostics_path = Path(output) / "async_diagnostics.jsonl"
        self._diagnostics_rank = int(rank)
        with self._diagnostics_path.open("x") as stream:
            stream.write(json.dumps(dict(
                schema_version=1, event="configuration", rank=int(rank),
                implementation="native-cache-explicit-cleanup-v2",
                memory_peaks="process lifetime; counters are not reset",
                memory_sampling="after measured iteration end",
                pending_and_cache_bytes="overlapping ownership; do not sum")) + "\n")

    def _memory_snapshot(self):
        result = {"rss_bytes": None, "rss_peak_bytes": None,
                  "cuda_device": str(self._cuda_device) if self._cuda_device is not None else None, "gpu": None,
                  "pinned_host": None, "errors": []}
        try:
            status = Path("/proc/self/status").read_text().splitlines()
            for line in status:
                key = line.split(":", 1)[0]
                if key in ("VmRSS", "VmHWM"):
                    result["rss_bytes" if key == "VmRSS" else "rss_peak_bytes"] = int(line.split()[1]) * 1024
        except Exception as exc:
            result["errors"].append("rss: " + str(exc))
        if self._cuda_device is not None:
            for label, read, keys in (
                ("gpu", lambda: torch.cuda.memory_stats(self._cuda_device),
                 [f"{kind}_bytes.all.{stat}" for kind in ("allocated", "reserved")
                  for stat in ("current", "peak")]),
                ("pinned_host", lambda: torch.cuda.memory.host_memory_stats(),
                 [f"{kind}.{stat}" for kind in ("allocated_bytes", "active_bytes", "active_requests")
                  for stat in ("current", "peak")]),
            ):
                try:
                    stats = read()
                    result[label] = {key: stats.get(key) for key in keys}
                except Exception as exc:
                    result["errors"].append(label + ": " + str(exc))
        return result

    def _append_diagnostics(self, event, **extra):
        row = dict(schema_version=1, event=event, rank=self._diagnostics_rank,
                   iteration=self.iteration + 1, native_cache_calls=self._native_calls,
                   native_cache_successes=self._native_successes,
                   native_d2h_calls=self._native_d2h_calls,
                   native_d2h_bytes=self._native_d2h_bytes,
                   queued_records=self._queued_records, completed_records=self._completed_records,
                   pending_cpu_bytes=self._pending_bytes,
                   pending_cpu_bytes_peak=self._iteration_pending_peak,
                   trace_cache_bytes_peak=self._trace_cache_peak,
                   trace_cache_entries_after=len(self._active_cache) if self._active_cache is not None else 0,
                   native_handles_after=len(self._native_handles),
                   observer_handles_after=len(self._observer_handles),
                   traces=list(self._trace_rows), memory=self._memory_snapshot(), **extra)
        if self._diagnostics_path is not None:
            with self._diagnostics_path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        return row

    def _create_native_cache(self, tracer):
        # NNsight 0.7.0 can discard the completed mediator before its persistent
        # hooks are removed. Keep only the handles installed by THIS cache call.
        owner = self.traced.interleaver.current
        first = len(owner.hooks)
        try:
            cache = tracer.cache(modules=self.envoys, device=torch.device("cpu"),
                                 detach=True, include_output=True, include_inputs=False)
        finally:
            # Also own any handles registered before a partial setup failure.
            self._native_handles.extend(owner.hooks[first:])
        self._active_cache = cache
        native = owner.user_cache[-1]
        if native.cache is not cache or "add" in vars(native):
            raise RuntimeError("Unexpected NNsight native cache ownership")
        self._native_cache = native
        original_add = native.add

        def counted_add(path, key, value):
            self._native_calls += 1
            result = original_add(path, key, value)
            self._native_successes += 1
            size = value.numel() * value.element_size()
            self._trace_cache_bytes += size
            self._trace_cache_peak = max(self._trace_cache_peak, self._trace_cache_bytes)
            if value.is_cuda:
                self._native_d2h_calls += 1
                self._native_d2h_bytes += size
            return result

        # Instance-local instrumentation; native copy implementation is unchanged.
        native.add = counted_add
        return cache

    def begin_iteration(self, iteration, *, start_ns=None):
        if self._ready_queue.unfinished_tasks or self._pending_bytes:
            raise RuntimeError("Previous iteration still has pending CPU cache copies")
        if self._ready_errors:
            raise RuntimeError("CPU-ready observer failed") from self._ready_errors[0]
        super().begin_iteration(iteration, start_ns=start_ns)
        self._queued_records = 0
        self._reset_diagnostics()

    def _install_ready_observers(self, cache):
        self._active_cache = cache
        for (_, point), envoy in zip(self.points, self.envoys, strict=True):
            def after_cache(module, args, output, *, path=envoy.path):
                # Native cache hook ran immediately before this observer. Read
                # only tensor metadata here, never the pending CPU payload.
                entry = cache[path]
                if isinstance(entry, list):
                    entry = entry[-1]
                self._copy_enqueued(module, output, entry.output)

            # Same ordering key as native cache, registered AFTER it. NNsight's
            # stable ordered insertion keeps all intervention hooks before both.
            after_cache.mediator_idx = float("inf")
            self._observer_handles.append(add_ordered_hook(point, after_cache, "output"))

    def _copy_enqueued(self, point, output, cpu_tensor):
        if cpu_tensor.is_cuda:
            raise AssertionError("Native cache did not return a CPU tensor")
        if not self._fired[point]:
            raise RuntimeError("Native cache copy has no hook-fire metadata")
        meta = self._fired[point].popleft()
        if output.is_cuda:
            self._cuda_device = output.device
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(output.device))
        else:
            event = None
        with self._pending_lock:
            self._pending_bytes += meta["bytes"]
            self.pending_peak_bytes = max(self.pending_peak_bytes, self._pending_bytes)
            self._iteration_pending_peak = max(self._iteration_pending_peak, self._pending_bytes)
        self._queued_records += 1
        self._ready_queue.put((event, meta, cpu_tensor))

    def _observe_ready(self):
        while True:
            item = self._ready_queue.get()
            try:
                if item is None:
                    return
                event, meta, cpu_tensor = item
                if event is not None:
                    event.synchronize()
                # Host-observed completion: includes observer scheduling delay,
                # as with the existing TorchLens observer. Not a GPU timestamp.
                arrived_ns = self.now_ns()
                if arrived_ns < meta["t_hook_fire_ns"]:
                    raise AssertionError("CPU arrival precedes hook fire")
                self.records.append(dict(**meta, tensor=cpu_tensor, t_arrive_ns=arrived_ns))
                self._completed_records += 1
            except Exception as exc:
                self._ready_errors.append(exc)
            finally:
                if item is not None:
                    with self._pending_lock:
                        self._pending_bytes -= item[1]["bytes"]
                    # Do not retain the last payload while waiting for more work.
                    event = meta = cpu_tensor = item = None
                self._ready_queue.task_done()

    def _release_trace(self):
        native_count, observer_count = len(self._native_handles), len(self._observer_handles)
        for handles in (self._observer_handles, self._native_handles):
            for handle in handles:
                handle.remove()
            handles.clear()
        if self._native_cache is not None:
            # Break the instrumentation closure's reference to the native cache.
            del self._native_cache.add
            self._native_cache = None
        if self._active_cache is not None:
            # The completion queue still owns any CPU tensors in flight.
            self._active_cache.clear()
            self._active_cache = None
        if self._trace_baseline is not None:
            baseline, calls_before, queued_before = self._trace_baseline
            residual = sum(len(set(point._forward_hooks) - ids) for point, ids in baseline)
            missing = sum(len(ids - set(point._forward_hooks)) for point, ids in baseline)
            self._trace_rows.append(dict(
                microbatch=self.microbatch, native_hooks_removed=native_count,
                observer_hooks_removed=observer_count, residual_hooks=residual,
                missing_preexisting_hooks=missing,
                native_cache_calls=self._native_calls - calls_before,
                observed_copies=self._queued_records - queued_before,
                cache_payload_bytes=self._trace_cache_bytes))
            self._trace_baseline = None
            if residual or missing:
                raise RuntimeError("Native trace cleanup did not restore observation hooks")
        self._trace_cache_bytes = 0

    def forward(self, *args, **kwargs):
        if self._ready_errors:
            raise RuntimeError("CPU-ready observer failed") from self._ready_errors[0]
        self._trace_baseline = ([(point, set(point._forward_hooks)) for _, point in self.points],
                                self._native_calls, self._queued_records)
        try:
            try:
                with self.traced.trace(*args, **kwargs) as tracer:
                    cache = self._create_native_cache(tracer)
                    self._install_ready_observers(cache)
                    result = self.traced.output.save()
                return result
            finally:
                self._release_trace()
        except BaseException as exc:
            # A failed step has no valid timing. Preserve diagnostics immediately.
            try:
                self._append_diagnostics("trace_failure", error=repr(exc))
            except Exception as diagnostic_error:
                exc.add_note("Could not append async diagnostics: " + repr(diagnostic_error))
            raise

    def end_iteration(self):
        flush_start_ns = self.now_ns()
        # One host completion boundary per optimizer iteration, inside the
        # existing measured interval. No stream/device-wide synchronize.
        self._ready_queue.join()
        if self._ready_errors:
            self._append_diagnostics("completion_failure", error=repr(self._ready_errors[0]))
            raise RuntimeError("CPU-ready observer failed") from self._ready_errors[0]
        super().end_iteration()
        self.iteration_audits[-1].update(
            t_flush_start_ns=flush_start_ns, async_records=self._queued_records,
            completion_policy="copy events; drain at optimizer-iteration end",
            pending_cpu_bytes_after=self._pending_bytes)
        # Queries and file I/O happen AFTER CaptureBase records t_capture_end_ns.
        expected = len(self.points) * len(self._trace_rows)
        valid = (self._native_calls == self._native_successes == self._queued_records
                 == self._completed_records == expected and self._pending_bytes == 0
                 and all(row["native_cache_calls"] == row["observed_copies"] == len(self.points)
                         for row in self._trace_rows))
        self.iteration_audits[-1]["async_diagnostics"] = self._append_diagnostics(
            "iteration_end", expected_native_cache_calls=expected, valid=valid,
            t_capture_end_ns=self.iteration_audits[-1]["t_capture_end_ns"])
        if not valid:
            raise RuntimeError("Native cache copy-count/cleanup validation failed")

    def close(self):
        if self._closed:
            return
        self._release_trace()
        self._ready_queue.put(None)
        self._ready_queue.join()
        self._ready_worker.join()
        self._closed = True
        super().close()
        if self._ready_errors:
            raise RuntimeError("CPU-ready observer failed") from self._ready_errors[0]
