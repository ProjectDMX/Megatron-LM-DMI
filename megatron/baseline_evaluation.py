"""Opt-in hidden-state capture in native training; unbounded metadata, no payload files."""
import json
import os
from pathlib import Path
from time import perf_counter_ns


class MetadataRecords(list):
    """Discard completed CPU payloads immediately; retain metadata until shutdown."""
    def append(self, record):
        if record['tensor'].is_cuda:
            raise RuntimeError('Baseline sink received a GPU payload')
        super().append({k: v for k, v in record.items() if k != 'tensor'})


class HiddenStateEvaluation:
    def __init__(self, model, args, output_dir, mode):
        from megatron.baseline_capture import Capture
        self.model, self.args = model, args
        self.capture = Capture(model, selected=['hidden_states'], mode=mode)
        self.capture.records = MetadataRecords()
        self.coords = self.capture.coordinates
        self.layers = sorted(point.layer for _, point in self.capture.points)
        count = args.num_layers // self.coords['pp_size']
        first = self.coords['pp_rank'] * count
        if self.layers != list(range(first, first + count)):
            raise RuntimeError(f'Unexpected hidden-state sites: {self.layers}')
        self.microbatches = args.global_batch_size // (args.micro_batch_size * args.data_parallel_size)
        self.shape = [args.seq_length // args.tensor_model_parallel_size,
                      args.micro_batch_size, args.hidden_size]
        self.bytes = 2
        for dim in self.shape:
            self.bytes *= dim
        self.output = Path(output_dir) / f'rank_{args.rank:05d}'
        self.output.mkdir(parents=True, exist_ok=False)
        self.timings = []
        self.active = self.closed = False
        model._baseline_hidden_evaluation = self

    def begin_iteration(self, iteration):
        start = perf_counter_ns()
        if self.active:
            raise RuntimeError('Previous baseline iteration is active')
        self.capture.begin_iteration(iteration, start_ns=start)
        self.record_start = len(self.capture.records)
        self.next_microbatch = 0
        self.active = True

    def forward(self, *args, **kwargs):
        if not self.active:
            raise RuntimeError('Capture forward outside a timed iteration')
        self.capture.microbatch = self.next_microbatch
        self.next_microbatch += 1
        return self.capture.forward(*args, **kwargs)

    def end_iteration(self, skipped_iter):
        # Includes deferred .cpu() calls; adds no CUDA wait or collective.
        self.capture.end_iteration()
        audit = self.capture.iteration_audits[-1]
        start, end = audit['t_iteration_start_ns'], audit['t_capture_end_ns']
        self.timings.append(dict(iteration=self.capture.iteration + 1,
                                 t_iteration_start_ns=start, t_iteration_end_ns=end,
                                 duration_ns=end-start))
        self.active = False
        # Delivery checks occur after the recorded timing boundary.
        records = self.capture.records[self.record_start:]
        expected = {(mb, layer) for mb in range(self.microbatches) for layer in self.layers}
        actual = [(r['microbatch'], r['layer']) for r in records]
        if self.next_microbatch != self.microbatches or len(actual) != len(expected) or set(actual) != expected:
            raise RuntimeError(f'Incomplete hidden-state capture: {len(actual)} records; expected {len(expected)}')
        for r in records:
            if (r['hook'] != 'hidden_states' or r['iteration'] != self.capture.iteration
                    or r['occurrence'] != 0 or r['shape'] != self.shape
                    or r['dtype'] != 'torch.bfloat16' or r['bytes'] != self.bytes
                    or not start <= r['t_hook_fire_ns'] <= r['t_arrive_ns'] <= end):
                raise RuntimeError(f'Invalid hidden-state record: {r}')
        audit.update(skipped_iter=int(skipped_iter), records=len(records),
                     payload_bytes=sum(r['bytes'] for r in records))

    def close(self):
        if self.closed:
            return
        if self.active:
            raise RuntimeError('Cannot close an active iteration')
        self.capture.close()
        complete = [r['iteration'] for r in self.timings] == list(range(1, self.args.train_iters + 1))
        outputs = {
            'iterations.json': self.timings,
            'iteration_audits.json': [dict(a, iteration=a['iteration']+1) for a in self.capture.iteration_audits],
            'summary.json': dict(complete=complete, rank=self.args.rank, coordinates=self.coords,
                                iterations=len(self.timings), records=len(self.capture.records),
                                mode=self.capture.mode, layers=self.layers,
                                expected_records_per_iteration=len(self.layers)*self.microbatches,
                                expected_shape=self.shape, pending_peak_bytes=self.capture.pending_peak_bytes,
                                clock='rank-local first training-step start; CPU nanoseconds',
                                payload_files=False, metadata_storage='unbounded lists; write at training end'),
        }
        for name, data in outputs.items():
            with (self.output/name).open('x') as handle:
                json.dump(data, handle)
                handle.write('\n')
        with (self.output/'events.jsonl').open('x') as handle:
            for record in self.capture.records:
                handle.write(json.dumps(dict(record, iteration=record['iteration']+1))+'\n')
        del self.model._baseline_hidden_evaluation
        self.closed = True
        if not complete:
            raise RuntimeError('Baseline did not complete the requested iterations')
        print(f'BASELINE_HIDDEN_COMPLETE rank={self.args.rank} iterations={len(self.timings)} records={len(self.capture.records)} output={self.output}', flush=True)


def setup_hidden_state_evaluation(model, args):
    output = os.environ.get('BASELINE_HIDDEN_METRICS_DIR')
    if not output:
        if os.environ.get('BASELINE_EVAL_WORKLOAD'):
            raise ValueError('Requested baseline workload requires BASELINE_HIDDEN_METRICS_DIR')
        return None
    workload = os.environ.get('BASELINE_EVAL_WORKLOAD', 'hidden_states')
    if args.skip_train and workload != 'validation_quality':
        raise ValueError('--skip-train capture requires the validation_quality workload')
    if workload == 'validation_quality' and (args.eval_iters <= 0 or not args.do_valid):
        raise ValueError('validation_quality capture requires an enabled validation pass')
    if (len(model) != 1 or args.use_legacy_models or args.perform_rl_step
            or args.cuda_graph_impl != 'none'
            or args.virtual_pipeline_model_parallel_size is not None
            or args.context_parallel_size != 1
            or not args.bf16 or args.recompute_granularity is not None
            or args.overlap_moe_expert_parallel_comm):
        raise ValueError('Capture requires the frozen non-interleaved BF16 setup without CUDA graphs')
    if workload != 'hidden_states':
        from .baseline_workload_evaluation import WorkloadEvaluation
        return WorkloadEvaluation(model[0], args, output,
                                  os.environ.get('BASELINE_CAPTURE_MODE', 'immediate'), workload)
    if args.eval_iters != 0:
        raise ValueError('Use a phase-aware workload for validation capture')
    return HiddenStateEvaluation(model[0], args, output, os.environ.get('BASELINE_CAPTURE_MODE', 'immediate'))
