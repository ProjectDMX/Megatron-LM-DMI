"""Native baseline workloads, with phase-specific forward and weight observations."""
import json
import math
import os
from pathlib import Path
from .baseline_evaluation import HiddenStateEvaluation, MetadataRecords
from .baseline_sampling import read_source_sampling

EP = ('moe_inverse_map', 'moe_packed_weighted_output', 'selected_expert_ids')
WORKLOADS = {
    'ep_outputs': ('moe_inverse_map', 'moe_packed_weighted_output'),
    'multi_signal': ('hidden_states', 'router_logits', 'vocab_topk', 'loss_summary',
                     'grad_norm', 'moe_inverse_map', 'moe_packed_weighted_output'),
    'vocab_raw': ('vocab_logits',),
    'router_logits': ('router_logits',),
    'qk_weights': ('qk_weights',),
    'ep_full': EP,
    'ep_sampled': EP,
    'training_health_full': ('qk_weights', *EP, 'routing_weights'),
    'training_health_sampled': ('qk_weights', *EP, 'routing_weights'),
    'validation_quality': ('resid_final', 'router_logits', 'loss_summary'),
    'final_hidden_states': ('resid_final',),
}


class WorkloadEvaluation(HiddenStateEvaluation):
    def __init__(self, model, args, output_dir, mode, workload):
        from .baseline_capture import Capture
        self.model, self.args, self.workload = model, args, workload
        selected = WORKLOADS[workload]
        ep = 'moe_packed_weighted_output' in selected or 'moe_inverse_map' in selected
        if ep and (args.moe_token_dispatcher_type != 'alltoall' or args.moe_permute_fusion
                   or args.moe_expert_capacity_factor is not None
                   or args.moe_router_padding_for_quantization):
            raise ValueError('EP capture requires dropless unfused alltoall without quantization padding')
        policy = read_source_sampling(os.environ.get('BASELINE_HOOK_CONFIG'), ep_enabled=ep)
        if workload.endswith('_sampled') and policy is None:
            raise ValueError('Sampled workload requires source_sampling in BASELINE_HOOK_CONFIG')
        if workload.endswith('_full') and policy is not None:
            raise ValueError('Full-capture workload must not configure source sampling')
        self.capture = Capture(model, selected=selected, mode=mode, source_sampling=policy)
        self.capture.records = MetadataRecords()
        self.coords = self.capture.coordinates
        self.microbatches = args.global_batch_size // (args.micro_batch_size * args.data_parallel_size)
        self.sites = {(p.name, p.layer) for _, p in self.capture.activation_points}
        self.weight_sites = {(p.name, p.layer) for _, p in self.capture.weight_points}
        if len(self.sites) != len(self.capture.activation_points):
            raise RuntimeError('Duplicate activation sites')
        self.grad_owner = ('grad_norm' in selected and self.coords['pp_rank'] == self.coords['pp_size']-1
                           and self.coords['tp_rank'] == self.coords['dp_rank'] == 0)
        self.output = Path(output_dir) / f'rank_{args.rank:05d}'
        self.output.mkdir(parents=True, exist_ok=False)
        if hasattr(self.capture, "configure_diagnostics"):
            self.capture.configure_diagnostics(self.output, rank=args.rank)
        self.timings = []
        self.active = self.closed = False
        self.phase = 'train'
        self.pass_id = 0
        self.capture_phase = 'validation' if workload == 'validation_quality' else 'train'
        self.training_start = None
        self.completed_training = []
        model._baseline_hidden_evaluation = self
        if self.capture.weight_model is not None:
            self.capture.phase = 'initial'
            self.capture.begin_iteration(args.iteration - 1)
            self.capture.capture_weights()
            self.capture.end_iteration()
        self.capture.set_enabled(False)

    def begin_iteration(self, iteration, *, phase='train', microbatches=None, pass_id=0):
        from megatron.core.num_microbatches_calculator import get_num_microbatches
        if self.active:
            raise RuntimeError('Previous baseline iteration is active')
        if phase == 'train':
            if self.training_start is None:
                self.training_start = iteration
            self.microbatches = get_num_microbatches()
        else:
            self.microbatches = microbatches
        self.phase, self.pass_id = phase, pass_id
        self.capture.set_enabled(phase == self.capture_phase)
        if phase != self.capture_phase:
            return
        self.capture.phase, self.capture.pass_id = phase, pass_id
        super().begin_iteration(iteration)

    def forward(self, *args, **kwargs):
        if not self.active:
            return self.model(*args, **kwargs)
        return super().forward(*args, **kwargs)

    def before_optimizer(self):
        if self.active and self.phase == 'train':
            self.capture.capture_weights()

    def validate_shape(self, record):
        a = self.args
        s, b, h = a.seq_length, a.micro_batch_size, a.hidden_size
        local_s = s // a.tensor_model_parallel_size if a.sequence_parallel else s
        name, shape, dtype = record['hook'], record['shape'], record['dtype']
        expected = {
            'hidden_states': ([local_s, b, h], 'torch.bfloat16'),
            'resid_final': ([local_s, b, h], 'torch.bfloat16'),
            'moe_inverse_map': ([local_s * b * a.moe_router_topk], 'torch.int64'),
            'vocab_logits': ([s, b, a.padded_vocab_size // a.tensor_model_parallel_size], 'torch.bfloat16'),
            'vocab_topk_values': ([b, s, 256], 'torch.bfloat16'),
            'vocab_topk_indices': ([b, s, 256], 'torch.int32'),
            'lm_per_sample_loss': ([b, 1], 'torch.float32'),
            'lm_per_sample_loss_token_count': ([b, 1], 'torch.int64'),
            'grad_norm': ([1], 'torch.float32'),
            'selected_expert_ids': ([b, local_s, a.moe_router_topk], 'torch.int64'),
        }
        if name in ('query_projection_weight', 'key_projection_weight'):
            valid = (len(shape) == 1 and dtype == 'torch.uint8'
                     and shape[0] == sum(span[2] for span in record['weight_layout']['fragments']))
        elif name == 'moe_packed_weighted_output':
            valid = len(shape) == 2 and shape[0] >= 0 and shape[1] == h and dtype == 'torch.bfloat16'
        elif name == 'routing_weights':
            valid = shape == [b, local_s, a.moe_router_topk] and dtype in ('torch.float32', 'torch.bfloat16')
        elif name == 'router_logits':
            valid = shape in ([local_s,b,a.num_experts], [local_s*b,a.num_experts]) and dtype in ('torch.bfloat16','torch.float32')
        else:
            valid = name in expected and (shape, dtype) == expected[name]
        sizes = {'torch.uint8':1, 'torch.bfloat16':2, 'torch.float32':4, 'torch.int32':4, 'torch.int64':8}
        if not valid or record['bytes'] != math.prod(shape) * sizes.get(dtype, 0):
            raise RuntimeError(f'Unexpected prepared shape/dtype/bytes: {record}')

    def end_iteration(self, skipped_iter):
        if self.phase == 'train':
            self.completed_training.append(self.training_start + len(self.completed_training))
        if not self.active:
            return
        self.capture.end_iteration()
        audit = self.capture.iteration_audits[-1]
        start, end = audit['t_iteration_start_ns'], audit['t_capture_end_ns']
        self.timings.append(dict(iteration=self.capture.iteration+1, phase=self.phase, pass_id=self.pass_id,
                                 t_iteration_start_ns=start, t_iteration_end_ns=end, duration_ns=end-start))
        self.active = False
        self.capture.set_enabled(False)
        records = self.capture.records[self.record_start:]
        expected = {(mb, hook, layer) for mb in range(self.microbatches) for hook, layer in self.sites}
        if self.phase == 'train':
            expected.update((-1,hook,layer) for hook,layer in self.weight_sites)
            if self.grad_owner:
                expected.add((-1, 'grad_norm', -1))
        actual = [(r['microbatch'],r['hook'],r['layer']) for r in records]
        if self.next_microbatch != self.microbatches or len(actual) != len(expected) or set(actual) != expected:
            raise RuntimeError(f'Incomplete {self.workload}: {len(actual)} records; expected {len(expected)}; missing {expected-set(actual)}')
        for record in records:
            self.validate_shape(record)
            if (record['iteration'] != self.capture.iteration or record['occurrence'] != 0
                    or record['phase'] != self.phase or record['pass_id'] != self.pass_id
                    or any(record[k] != v for k,v in self.coords.items())
                    or not start <= record['t_hook_fire_ns'] <= record['t_arrive_ns'] <= end):
                raise RuntimeError(f'Invalid delivered record: {record}')
        audit.update(phase=self.phase, pass_id=self.pass_id, skipped_iter=int(skipped_iter), records=len(records),
                     payload_bytes=sum(r['bytes'] for r in records))

    def close(self):
        if self.closed:
            return
        if self.active:
            raise RuntimeError('Cannot close an active iteration')
        self.capture.close()
        complete = len(self.completed_training) == self.args.train_iters - (self.training_start or 0)
        if self.workload == 'validation_quality':
            complete = complete and bool(self.timings)
        outputs = {
            'iterations.json': self.timings,
            'iteration_audits.json': [dict(a, iteration=a['iteration']+1)
                                      for a in self.capture.iteration_audits],
            'summary.json': dict(complete=complete, rank=self.args.rank, coordinates=self.coords,
                                workload=self.workload, selected_hooks=list(WORKLOADS[self.workload]),
                                iterations=len(self.timings), records=len(self.capture.records), mode=self.capture.mode,
                                sites=sorted(self.sites), weight_sites=sorted(self.weight_sites),
                                expected_records_per_iteration=(len(self.sites)*self.microbatches
                                    +len(self.weight_sites)+int(self.grad_owner)),
                                metadata_storage='unbounded lists; write at shutdown',
                                pending_peak_bytes=self.capture.pending_peak_bytes,
                                clock='rank-local capture origin; CPU nanoseconds', payload_files=False),
        }
        if self.capture.weight_model is not None:
            outputs['weight_layouts.json'] = self.capture.weight_model.layouts
        for name,data in outputs.items():
            (self.output/name).write_text(json.dumps(data)+'\n')
        with (self.output/'events.jsonl').open('x') as handle:
            for record in self.capture.records:
                handle.write(json.dumps(dict(record, iteration=record['iteration']+1))+'\n')
        del self.model._baseline_hidden_evaluation
        self.closed = True
        if not complete:
            raise RuntimeError('Baseline did not complete the requested iterations/phases')
        print(f'BASELINE_WORKLOAD_COMPLETE workload={self.workload} rank={self.args.rank} records={len(self.capture.records)}', flush=True)
