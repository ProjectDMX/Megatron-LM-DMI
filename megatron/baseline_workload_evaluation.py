"""Opt-in native evaluation for the EP, multi-signal, and raw-vocab workloads."""
import json
import math
from pathlib import Path
from .baseline_evaluation import HiddenStateEvaluation, MetadataRecords

WORKLOADS = {
    'ep_outputs': ('moe_inverse_map', 'moe_packed_weighted_output'),
    'multi_signal': ('hidden_states', 'router_logits', 'vocab_topk', 'loss_summary',
                     'grad_norm', 'moe_inverse_map', 'moe_packed_weighted_output'),
    'vocab_raw': ('vocab_logits',),
}


class WorkloadEvaluation(HiddenStateEvaluation):
    # Reuse the existing start/forward boundaries. Do not wrap train_step.
    def __init__(self, model, args, output_dir, mode, workload):
        from .baseline_capture import Capture
        self.model, self.args, self.workload = model, args, workload
        selected = WORKLOADS[workload]
        if workload != 'vocab_raw' and (
                args.moe_token_dispatcher_type != 'alltoall'
                or args.moe_permute_fusion
                or args.moe_expert_capacity_factor is not None
                or args.moe_router_padding_for_quantization):
            raise ValueError('EP capture requires dropless unfused alltoall without quantization padding')
        self.capture = Capture(model, selected=selected, mode=mode, vocab_topk=256)
        self.capture.records = MetadataRecords()
        self.coords = self.capture.coordinates
        self.microbatches = args.global_batch_size // (args.micro_batch_size * args.data_parallel_size)
        count = args.num_layers // self.coords['pp_size']
        first = self.coords['pp_rank'] * count
        self.layers = list(range(first, first + count))
        self.sites = {(point.name, point.layer) for _, point in self.capture.points}
        expected = set()
        for hook in ('hidden_states', 'router_logits', 'moe_inverse_map', 'moe_packed_weighted_output'):
            if hook in selected:
                expected.update((hook, layer) for layer in self.layers)
        last = self.coords['pp_rank'] == self.coords['pp_size'] - 1
        if last:
            if 'vocab_logits' in selected:
                expected.add(('vocab_logits', -1))
            if 'vocab_topk' in selected:
                expected.update((h, -1) for h in ('vocab_topk_values', 'vocab_topk_indices'))
            if 'loss_summary' in selected and self.coords['tp_rank'] == 0:
                expected.update((h, -1) for h in ('lm_per_sample_loss', 'lm_per_sample_loss_token_count'))
        if self.sites != expected or len(self.capture.points) != len(expected):
            raise RuntimeError(f'Unexpected {workload} sites: {self.sites ^ expected}')
        self.grad_owner = ('grad_norm' in selected and last
                           and self.coords['tp_rank'] == self.coords['dp_rank'] == 0)
        self.output = Path(output_dir) / f'rank_{args.rank:05d}'
        self.output.mkdir(parents=True, exist_ok=False)
        self.timings = []
        self.active = self.closed = False
        model._baseline_hidden_evaluation = self

    def validate_shape(self, record):
        a = self.args
        s, b, h = a.seq_length, a.micro_batch_size, a.hidden_size
        local_s = s // a.tensor_model_parallel_size
        name, shape, dtype = record['hook'], record['shape'], record['dtype']
        expected = {
            'hidden_states': ([local_s, b, h], 'torch.bfloat16'),
            'moe_inverse_map': ([local_s * b * a.moe_router_topk], 'torch.int64'),
            'vocab_logits': ([s, b, a.padded_vocab_size // a.tensor_model_parallel_size], 'torch.bfloat16'),
            'vocab_topk_values': ([b, s, 256], 'torch.bfloat16'),
            'vocab_topk_indices': ([b, s, 256], 'torch.int32'),
            'lm_per_sample_loss': ([b, 1], 'torch.float32'),
            'lm_per_sample_loss_token_count': ([b, 1], 'torch.int64'),
            'grad_norm': ([1], 'torch.float32'),
        }
        if name == 'moe_packed_weighted_output':
            valid = len(shape) == 2 and shape[0] >= 0 and shape[1] == h and dtype == 'torch.bfloat16'
        elif name == 'router_logits':
            valid = (shape in ([local_s, b, a.num_experts], [local_s*b, a.num_experts])
                     and dtype in ('torch.bfloat16', 'torch.float32'))
        else:
            valid = name in expected and (shape, dtype) == expected[name]
        sizes = {'torch.bfloat16': 2, 'torch.float32': 4, 'torch.int32': 4, 'torch.int64': 8}
        if not valid or record['bytes'] != math.prod(shape) * sizes.get(dtype, 0):
            raise RuntimeError(f'Unexpected prepared shape/dtype/bytes: {record}')

    def end_iteration(self, skipped_iter):
        self.capture.end_iteration()
        audit = self.capture.iteration_audits[-1]
        start, end = audit['t_iteration_start_ns'], audit['t_capture_end_ns']
        self.timings.append(dict(iteration=self.capture.iteration+1,
                                 t_iteration_start_ns=start, t_iteration_end_ns=end,
                                 duration_ns=end-start))
        self.active = False
        records = self.capture.records[self.record_start:]
        expected = {(mb, hook, layer) for mb in range(self.microbatches) for hook, layer in self.sites}
        if self.grad_owner:
            expected.add((-1, 'grad_norm', -1))
        actual = [(r['microbatch'], r['hook'], r['layer']) for r in records]
        if self.next_microbatch != self.microbatches or len(actual) != len(expected) or set(actual) != expected:
            raise RuntimeError(f'Incomplete {self.workload}: {len(actual)} records; expected {len(expected)}; missing {expected-set(actual)}')
        for record in records:
            self.validate_shape(record)
            if (record['iteration'] != self.capture.iteration or record['occurrence'] != 0
                    or any(record[k] != v for k, v in self.coords.items())
                    or not start <= record['t_hook_fire_ns'] <= record['t_arrive_ns'] <= end):
                raise RuntimeError(f'Invalid delivered record: {record}')
        audit.update(skipped_iter=int(skipped_iter), records=len(records),
                     payload_bytes=sum(r['bytes'] for r in records))
        if skipped_iter:
            raise RuntimeError(f'Optimizer step skipped in {self.workload}')

    def close(self):
        if self.closed:
            return
        if self.active:
            raise RuntimeError('Cannot close an active iteration')
        self.capture.close()
        complete = [r['iteration'] for r in self.timings] == list(range(1, self.args.train_iters+1))
        outputs = {
            'iterations.json': self.timings,
            'iteration_audits.json': [dict(a, iteration=a['iteration']+1) for a in self.capture.iteration_audits],
            'summary.json': dict(complete=complete, rank=self.args.rank, coordinates=self.coords,
                                workload=self.workload, selected_hooks=list(WORKLOADS[self.workload]),
                                iterations=len(self.timings), records=len(self.capture.records),
                                mode=self.capture.mode, sites=sorted(self.sites),
                                expected_records_per_iteration=len(self.sites)*self.microbatches+int(self.grad_owner),
                                pending_peak_bytes=self.capture.pending_peak_bytes,
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
        print(f'BASELINE_WORKLOAD_COMPLETE workload={self.workload} rank={self.args.rank} iterations={len(self.timings)} records={len(self.capture.records)} output={self.output}', flush=True)
